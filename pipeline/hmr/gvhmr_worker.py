"""
GVHMR on the pipeline's tracks, run inside GVHMR's conda env with GVHMR's checkout as the
working directory (gvhmr.py does this).
Usage: python /abs/path/to/gvhmr_worker.py --input in.pt --output out.pt
"""
import argparse
import importlib.util
import os
import torch
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import hydra
from hydra import initialize_config_module, compose
from hmr4d.configs import register_store_gvhmr

from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
from hmr4d.utils.preproc.vitfeat_extractor import Extractor, get_batch
from pytorch3d.transforms import axis_angle_to_matrix

# pipeline/joints.py by path: this runs in GVHMR's env, without the pipeline package
_spec = importlib.util.spec_from_file_location(
    'joints', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'joints.py'))
joints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(joints)
OPENPOSE_TO_COCO = joints.indices(joints.OPENPOSE_25, joints.COCO_17)
CKPT_PATH = 'inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    data = torch.load(args.input, map_location='cpu')
    # images were saved as tensors to avoid numpy cross-version issues
    images = {f: v.numpy() for f, v in data['images'].items()}  # {frame_idx: np.ndarray}
    tracks = data['tracks']       # {track_id: {frames, bboxes(tensor), keypoints_2d(tensor)}}
    img_focal = data['img_focal']
    img_center = data['img_center']
    rwc = data['pred_cam_R']       # (N, 3, 3) camera-to-world rotation from the odometry

    cam_intrinsic = torch.eye(3)
    cam_intrinsic[0, 0] = float(img_focal)
    cam_intrinsic[1, 1] = float(img_focal)
    cam_intrinsic[0, 2] = float(img_center[0])
    cam_intrinsic[1, 2] = float(img_center[1])

    # Load original GVHMR model using same mechanism as demo.py.
    # initialize_config_module + register_store_gvhmr is required: the old Hydra
    # version in this env fails to resolve 'module.ClassName' _target_ strings
    # via plain import_module, but structured configs (registered by
    # register_store_gvhmr) carry actual class references that bypass that path.
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=["video_name=placeholder", "static_cam=True"])

    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(CKPT_PATH)
    model = model.eval().cuda()

    extractor = Extractor()

    results = {}
    for k, track in tracks.items():
        frames = track['frames']
        seqlen = len(frames)

        # Smooth bboxes (bboxes already a tensor)
        bbx_xys = get_bbx_xys_from_xyxy(track['bboxes'].float()).float()
        bbx_np = bbx_xys.numpy()
        smoothed = np.array([signal.medfilt(param, 11) for param in bbx_np.T]).T
        bbx_np = np.array([gaussian_filter1d(traj, 3) for traj in smoothed.T]).T
        bbx_xys = torch.from_numpy(bbx_np).float()

        # Extract HMR2 ViT features
        person_imgs = np.stack([images[f] for f in frames])  # (L, H, W, 3) RGB
        imgs_tensor, _ = get_batch(person_imgs, bbx_xys, img_ds=1.0, path_type="np")
        f_imgseq = extractor.extract_video_features(imgs_tensor, bbx_xys)  # (L, 1024)

        # Keypoints: OP25 -> COCO17 (keypoints_2d already a tensor)
        kp2d_op25 = track['keypoints_2d'].float()  # (L, 25+, 3)
        kp2d = kp2d_op25[:, OPENPOSE_TO_COCO, :]  # (L, 17, 3)

        rcw = rwc[frames].transpose(-1, -2).float()    # (L, 3, 3) world-to-camera
        K_fullimg = cam_intrinsic[None].repeat(seqlen, 1, 1)
        cam_angvel = compute_cam_angvel(rcw)
        static_cam = torch.allclose(rcw, torch.eye(3).expand_as(rcw), atol=1e-3)

        input_data = {
            "length": torch.tensor(seqlen),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq,  # (L, 1024) — original GVHMR format
        }

        with torch.amp.autocast('cuda'):
            pred = model.predict(input_data, static_cam=static_cam)

        def build_pose_rotmat(global_orient, body_pose):
            device = global_orient.device
            L = global_orient.shape[0]
            pose_aa = torch.cat(
                [global_orient, body_pose, torch.zeros(L, 99).to(device)],
                dim=-1,
            )
            rotmat = axis_angle_to_matrix(pose_aa[:, :75].reshape(-1, 3)).reshape(-1, 25, 3, 3)
            hand_rotmat = torch.eye(3).repeat(rotmat.shape[0], 30, 1, 1).to(rotmat)
            return pose_aa, torch.cat([rotmat, hand_rotmat], dim=1)

        # Incam
        pose_aa_c, rotmat_c = build_pose_rotmat(
            pred['smpl_params_incam']['global_orient'],
            pred['smpl_params_incam']['body_pose'],
        )

        static_conf_logits = pred['net_outputs']['model_output']['static_conf_logits'][0]
        contact_label = static_conf_logits.sigmoid() > joints.CONTACT_THRESHOLD

        results[k] = {
            'rotmat': rotmat_c.cpu().float().numpy(),
            'pose': pose_aa_c.cpu().numpy(),
            'shape': pred['smpl_params_incam']['betas'].cpu().float().numpy(),
            'trans': pred['smpl_params_incam']['transl'].cpu().float().numpy(),
            'contact': contact_label.cpu().numpy(),
            'static_conf_logits': static_conf_logits.cpu().numpy(),
        }

    torch.save(results, args.output)


if __name__ == '__main__':
    main()
