import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from smplx import SMPLX

from vitpose.estimator import load_vit_model, estimate_kp2ds_from_bbox_vitpose
from pipeline.joints import COCO_17, CONTACT_JOINT_IDS, OPENPOSE_25, VITPOSE_25, reorder
from pipeline.video import load_video
from spec import run_cam_calib
from pipeline.tracking import detect_segment_track
from pipeline.odometry import ODOMETRIES
from pipeline.hmr import HMR_BACKBONES
from pipeline.hmr.static_conf import save_static_conf, load_static_conf
from pipeline.world import world_hps_estimation
from pipeline.drift import stabilize_world_drift
from pipeline.interpolation import pose_interpolation
from pipeline.post_optimization import post_optimization
from pipeline.export import save_results_npz
from pipeline import log


NUM_STEPS = 9


class Pipeline:
    """
    World-space 3D humans from a monocular video:
      1. SPEC camera pitch and roll           (gravity)
      2. detect, segment and track people     (tracking.py: YOLO11-pose + SAM 2.1)
      3. camera trajectory and intrinsics     (odometry.py: ViPE)
      4. 2D keypoints                         (ViTPose)
      5. camera-frame SMPL-X                  (hmr/: PromptHMR-Vid, GVHMR or TRAM)
      6. lift to a gravity-aligned world      (world.py)
      7. drift correction                     (drift.py)
      8. fill undetected frames               (interpolation.py)
      9. post-optimization (PGO)              (post_optimization.py)
    """

    def __init__(self, cfg_path="pipeline/config.yaml", **overrides):
        self.cfg = OmegaConf.merge(OmegaConf.load(cfg_path), overrides)
        self.smplx = SMPLX('data/body_models/smplx/SMPLX_NEUTRAL.npz',
                           use_pca=False, flat_hand_mean=True, num_betas=10)
        self.images = None

    def __call__(self, input_video, output_folder):
        """Run every step on one video and write results.npz to output_folder."""
        cfg = self.cfg
        start = time.time()
        self.images, self.fps, (video_width, video_height) = load_video(
            input_video, output_folder, cfg.max_height, cfg.max_fps)
        num_frames, height, width = self.images.shape[:3]
        resized = f", processed at {width}x{height}" if (width, height) != (video_width, video_height) else ""
        print(f"{input_video}: {num_frames} frames at {self.fps:g} fps, "
              f"{video_width}x{video_height}{resized}")
        print(f"Output: {output_folder}")
        self.results = {'people': {}}
        odometry_name, odometry = ODOMETRIES[cfg.odometry]
        step = lambda i, title: log.step(i, NUM_STEPS, title)

        with step(1, "Estimating gravity with SPEC"):
            self.results['spec_calib'] = run_cam_calib(
                self.images, out_folder=os.path.join(output_folder, 'spec_calib'), save_res=True,
                stride=max(len(self.images) // 30, 1), method='spec', first_frame_idx=0)
            first = self.results['spec_calib']['first_frame']
            log.detail(f"first frame: camera pitch {np.degrees(first['pitch']):.1f} deg, "
                       f"roll {np.degrees(first['roll']):.1f} deg")

        with step(2, "Detecting, segmenting and tracking people with YOLO11-pose and SAM 2.1"):
            self.results['people'] = detect_segment_track(self.images, cfg)
            log.detail(f"{len(self.results['people'])} people tracked")

        with step(3, f"Estimating the camera with {odometry_name}"):
            self.results['camera'] = odometry(input_video, output_folder, cfg)
            focal = float(self.results['camera']['img_focal'])
            log.detail(f"focal length {focal:.1f} px at {video_width}x{video_height} (vertical field "
                       f"of view {np.degrees(2 * np.arctan(video_height / (2 * focal))):.1f} deg)")

        with step(4, "Estimating 2D keypoints with ViTPose"):
            self.estimate_2d_keypoints()

        with step(5, f"Estimating camera-frame humans with {HMR_BACKBONES[cfg.hmr_type][0]}"):
            self.estimate_humans(output_folder)

        with step(6, "Lifting humans to world coordinates"):
            self.results = world_hps_estimation(cfg, self.results, self.smplx)
            to_numpy(self.results)

        with step(7, "Correcting drift"):
            self.results = stabilize_world_drift(self.results, self.smplx, cfg)
            drift_found = self.results['camera_world']['drift']['mode'] != 'none'

        with step(8, "Interpolating poses over undetected frames"):
            self.results = pose_interpolation(self.results)
            filled = sum(int(np.sum(p['gaps'])) for p in self.results['people'].values())
            log.detail(f"{filled} person-frames filled in")

        with step(9, "Running post-optimization"):
            self.results = post_optimization(cfg, self.results, self.images, self.smplx, opt_drift=drift_found)

        path = os.path.join(output_folder, 'results.npz')
        save_results_npz(self.results, path)
        print(f"Saved {path}: {len(self.results['people'])} people, {num_frames} frames, "
              f"{log.duration(time.time() - start)} in total")
        return self.results

    def estimate_2d_keypoints(self):
        model = load_vit_model(model_path='data/pretrain/vitpose-h-coco_25.pth')
        for k, v in log.progress(self.results['people'].items(), 'people'):
            kpts_2d = estimate_kp2ds_from_bbox_vitpose(model, self.images, v['bboxes'], k, v['frames'])
            kpts_2d = reorder(kpts_2d, VITPOSE_25, OPENPOSE_25)
            v['keypoints_2d'] = kpts_2d
            v['vitpose'] = reorder(kpts_2d, OPENPOSE_25, COCO_17)   # what PromptHMR-Vid reads
        del model

    def estimate_humans(self, output_folder):
        """Camera-frame SMPL-X per person into results['people'][k]['smplx_cam'], plus contact
        confidence. A phmr run saves its contact confidence to static_conf.pkl; a run with
        static_conf_dir set replaces its own with the one saved there."""
        cfg = self.cfg
        _, backbone = HMR_BACKBONES[cfg.hmr_type]
        self.results = backbone(self.images, self.results, self.smplx, cfg, output_folder)
        self.results['contact_joint_ids'] = CONTACT_JOINT_IDS

        if cfg.hmr_type == 'phmr':
            save_static_conf(self.results, os.path.join(output_folder, 'static_conf.pkl'), source=cfg.hmr_type)
        elif cfg.static_conf_dir:
            video_name = os.path.basename(os.path.normpath(output_folder))
            load_static_conf(self.results, os.path.join(cfg.static_conf_dir, video_name, 'static_conf.pkl'))

    def create_world4d(self, total=None, step=1):
        """Per-frame people and camera in world coordinates, for the viser viewer."""
        results = self.results
        num_frames = len(results['camera_world']['Rwc'])
        total = num_frames if total is None else min(total, num_frames)

        world4d = {}
        for i in range(0, total, step):
            pose, shape, transl, track_id = [], [], [], []
            for person in results['people'].values():
                in_frame = np.where(person['frames'] == i)[0]
                if len(in_frame) == 1:
                    pose.append(person['smplx_world']['pose'][in_frame])
                    shape.append(person['smplx_world']['shape'][in_frame])
                    transl.append(person['smplx_world']['trans'][in_frame])
                    track_id.append(person['track_id'])

            camera = np.eye(4)
            camera[:3, :3] = results['camera_world']['Rwc'][i]
            camera[:3, 3] = results['camera_world']['Twc'][i]

            if len(track_id) > 0:
                world4d[i] = {'pose': torch.tensor(np.concatenate(pose)).float().reshape(len(track_id), -1, 3),
                              'shape': torch.tensor(np.concatenate(shape)).float(),
                              'trans': torch.tensor(np.concatenate(transl)).float(),
                              'track_id': torch.tensor(np.array(track_id)) - 1,
                              'camera': camera}
            else:
                world4d[i] = {'track_id': np.array([]), 'camera': camera}
        return world4d


def to_numpy(d):
    for k, v in d.items():
        if isinstance(v, dict):
            to_numpy(v)
        elif isinstance(v, torch.Tensor):
            d[k] = v.detach().cpu().numpy()
