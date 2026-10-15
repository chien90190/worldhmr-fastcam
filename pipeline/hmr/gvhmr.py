import os
import subprocess
import tempfile
import torch
import numpy as np

_WORKER_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gvhmr_worker.py'))


class GVHMR_Video():
    """
    Camera-frame SMPL-X from GVHMR, run by gvhmr_worker.py inside GVHMR's own checkout and
    conda env. The tracks, ViTPose keypoints and camera intrinsics come from this pipeline,
    so GVHMR sees the same people and camera as PromptHMR-Vid.
    """

    def __init__(self, gvhmr_dir, conda_env='gvhmr'):
        self.gvhmr_dir = os.path.abspath(gvhmr_dir)
        self.conda_env = conda_env

    def run(self, images, results):
        tracks = results['people']
        camera = results['camera']

        # The odometry's intrinsics are at the video's own resolution; `images` may be resized.
        img_center = np.asarray(camera['img_center'], dtype=np.float64)
        scale = images[0].shape[0] / (float(img_center[1]) * 2.0)
        img_focal = float(camera['img_focal']) * scale
        img_center = img_center * scale

        needed_frames = set()
        for track in tracks.values():
            needed_frames.update(int(f) for f in track['frames'])

        # Tensors and plain Python types only, so the worker can load them under any numpy.
        input_data = {
            'images': {int(f): torch.from_numpy(np.asarray(images[f])) for f in needed_frames},
            'tracks': {
                k: {
                    'frames': [int(f) for f in track['frames']],
                    'bboxes': torch.from_numpy(np.asarray(track['bboxes'])),
                    'keypoints_2d': torch.from_numpy(np.asarray(track['keypoints_2d'])),
                }
                for k, track in tracks.items()
            },
            'img_focal': img_focal,
            'img_center': [float(img_center[0]), float(img_center[1])],
            'pred_cam_R': torch.from_numpy(np.asarray(camera['pred_cam_R'])).float(),  # (N, 3, 3) camera-to-world
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, 'gvhmr_input.pt')
            output_path = os.path.join(tmp, 'gvhmr_output.pt')
            torch.save(input_data, input_path)
            # GVHMR's own output is shown only if it fails
            run = subprocess.run([
                'conda', 'run', '-n', self.conda_env, '--no-capture-output',
                '--cwd', self.gvhmr_dir,
                'python', _WORKER_SCRIPT, '--input', input_path, '--output', output_path,
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if run.returncode != 0:
                print(''.join(run.stdout.splitlines(keepends=True)[-30:]), end='')
                raise RuntimeError(f'GVHMR failed in conda env {self.conda_env}')
            gvhmr_results = torch.load(output_path, map_location='cpu')

        for k, smplx_cam in gvhmr_results.items():
            results['people'][k]['smplx_cam'] = smplx_cam
            results['people'][k]['smplx_pose'] = torch.from_numpy(smplx_cam['rotmat'])    # (L, 55, 3, 3)
            results['people'][k]['smplx_transl'] = torch.from_numpy(smplx_cam['trans'])   # (L, 3)
            results['people'][k]['smplx_betas'] = torch.from_numpy(smplx_cam['shape'])    # (L, 10)
        return results
