"""
Odometry: the camera's motion and intrinsics on every frame of the video.

An odometry is a function (video_path, output_folder, cfg) -> dict with

    pred_cam_R   (T,3,3)  camera-to-world rotation
    pred_cam_T   (T,3)    camera position in the world, in meters
    img_focal    float    focal length, in pixels of the input video
    img_center   (2,)     principal point, in pixels of the input video

for all T frames of the video. The world is the first frame's camera (OpenCV axes: x right,
y down, z forward), since the pipeline levels it with SPEC's pitch and roll of that frame.
Translations must be metric: people are placed at the depths the HMR model predicts.

To add one, write such a function, add it to ODOMETRIES with the name shown in the output,
and set `odometry: <key>` in config.yaml. Its own settings go in config.yaml too.
"""
import os
import subprocess

import numpy as np

from pipeline import log


def run_vipe(video_path, output_folder, cfg):
    """
    ViPE, run in its own conda env (cfg.vipe_env) with the cfg.vipe_pipeline config. Its
    output goes to cfg.vipe_dir (default <output_dir>/vipe) and is reused when it exists.
    """
    vipe_dir = cfg.vipe_dir or os.path.join(os.path.dirname(os.path.normpath(output_folder)), 'vipe')
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    pose_path = os.path.join(vipe_dir, 'pose', f'{video_name}.npz')
    if cfg.vipe_rerun or not os.path.exists(pose_path):
        log_path = os.path.join(vipe_dir, 'logs', f'{video_name}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log.detail(f'running ViPE, its output in {log_path}')
        cmd = ['conda', 'run', '-n', cfg.vipe_env, '--no-capture-output',
               'vipe', 'infer', video_path, '--output', vipe_dir, '--pipeline', cfg.vipe_pipeline]
        with open(log_path, 'w') as f:
            if subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT) != 0:
                with open(log_path) as g:
                    print(''.join(g.readlines()[-20:]), end='')
                raise RuntimeError(f'ViPE failed, see {log_path}')
    else:
        log.detail(f'reusing {pose_path}')

    with np.load(pose_path) as p:
        matrices = p['data']  # (T,4,4) camera-to-world
    with np.load(os.path.join(vipe_dir, 'intrinsics', f'{video_name}.npz')) as i:
        intrinsics = i['data'].mean(axis=0)  # fx, fy, cx, cy
    return {'pred_cam_R': matrices[:, :3, :3], 'pred_cam_T': matrices[:, :3, 3],
            'img_focal': intrinsics[0], 'img_center': intrinsics[2:]}


def run_da3(video_path, output_folder, cfg):
    """
    Not included in this release. Depth Anything 3, compared in the paper: would read the
    camera poses and intrinsics DA3 exports for the video, scaled to meters.
    """
    raise NotImplementedError('DA3 odometry is not included in this release; see pipeline/odometry.py')


def run_droid_slam_masked(video_path, output_folder, cfg):
    """
    Not included in this release. DROID-SLAM with people masked out, compared in the paper:
    would need DROID-SLAM in its own env, the person masks from the tracking step, and a
    metric scale.
    """
    raise NotImplementedError('masked DROID-SLAM odometry is not included in this release; '
                              'see pipeline/odometry.py')


# key in config.yaml -> (name shown in the output, function)
ODOMETRIES = {
    'vipe': ('ViPE', run_vipe),
    'da3': ('DA3', run_da3),
    'droid_slam_masked': ('masked DROID-SLAM', run_droid_slam_masked),
}
