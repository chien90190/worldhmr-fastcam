"""
Turn other methods' outputs into results.npz, the format evaluate.py reads.

    python evaluation/convert.py --input <method output dir> --output <predictions dir>

Each <input>/<clip>/ holds one of

  results.pkl              a PromptHMR-style result: camera_world, and per person frames,
                           detected, smplx_world and, if poses were filled in, gaps
  camera.npy + hps/*.npy   TRAM's output: camera-frame SMPL and the camera-to-world motion

and becomes <output>/<clip>/results.npz with the keys evaluate.py reads: camera/intrinsics,
camera/extrinsics, track_id, detected, body_model and the world-space body: smplx/* for
SMPL-X, or smpl/* for TRAM's SMPL (body_pose (P,T,69)). TRAM's world is shifted so that the lowest
vertex of all people over the clip is at y = 0.
"""
import glob
import os
import sys

import joblib
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bodies import BodyModels  # noqa: E402
from pipeline import export  # noqa: E402


def convert_results_pkl(path):
    results = joblib.load(path)
    cam = results['camera_world']
    num_frames = len(cam['Rcw'])
    out = {'body_model': np.array('smplx'),
           'camera/intrinsics': export.intrinsics_matrix(cam['img_focal'], cam['img_center']),
           'camera/extrinsics': export.extrinsics_matrix(cam['Rcw'], cam['Tcw'])}
    cols = {k: [] for k in ['detected', 'smplx/betas', 'smplx/global_orient', 'smplx/body_pose', 'smplx/transl']}
    ids = sorted(results['people'])
    for tid in ids:
        p = results['people'][tid]
        w = p['smplx_world']
        frames = np.asarray(p['frames'])
        gaps = np.asarray(p['gaps'], bool) if 'gaps' in p else np.zeros(len(frames), bool)
        betas = np.asarray(w['shape'], np.float32)
        assert np.allclose(betas, betas[0], atol=1e-5), (path, tid, 'betas vary over frames')
        pose = np.asarray(w['pose'], np.float32)
        assert len(pose) == len(frames), (path, tid, 'world pose is not parallel to frames')
        # detected is parallel to the frames that are not filled in
        cols['detected'].append(export.spread(np.asarray(p['detected'], bool), frames[~gaps], num_frames, False))
        cols['smplx/betas'].append(betas[0])
        cols['smplx/global_orient'].append(export.spread(pose[:, :3], frames, num_frames))
        cols['smplx/body_pose'].append(export.spread(pose[:, 3:66], frames, num_frames))
        cols['smplx/transl'].append(export.spread(np.asarray(w['trans'], np.float32), frames, num_frames))
    out['track_id'] = np.array(ids, np.int32)
    out.update({k: np.stack(v) for k, v in cols.items()})
    return out


@torch.no_grad()
def convert_tram(clip_dir, body_models):
    device = body_models.device
    cam = np.load(os.path.join(clip_dir, 'camera.npy'), allow_pickle=True).item()
    R_wc = np.asarray(cam['world_cam_R'], np.float32)   # camera -> world
    T_wc = np.asarray(cam['world_cam_T'], np.float32)
    num_frames = len(R_wc)
    smpl = body_models.smpl

    tracks = []
    for path in glob.glob(os.path.join(clip_dir, 'hps', 'hps_track_*.npy')):
        hps = np.load(path, allow_pickle=True).item()
        frames = np.asarray(hps['frame'])
        rotmat = hps['pred_rotmat'].float().to(device)                 # (N,24,3,3) camera frame
        betas = hps['pred_shape'].float().to(device).mean(0)            # one shape per track
        transl = hps['pred_trans'].float().to(device).reshape(-1, 3)
        n = len(frames)
        verts = smpl(betas=betas[None].repeat(n, 1), transl=transl, global_orient=rotmat[:, [0]],
                     body_pose=rotmat[:, 1:], pose2rot=False).vertices
        R, T = torch.as_tensor(R_wc[frames], device=device), torch.as_tensor(T_wc[frames], device=device)
        lowest = (torch.einsum('bij,bnj->bni', R, verts) + T[:, None])[..., 1].min().item()
        # SMPL rotates about the rest-pose pelvis j0, so moving the body to the world frame
        # takes transl_w = R (j0 + transl_c) + T - j0
        j0 = smpl(betas=betas[None]).joints[0, 0]
        transl_w = torch.einsum('bij,bj->bi', R, j0 + transl) + T - j0
        orient_w = torch.einsum('bij,bjk->bik', R, rotmat[:, 0])
        track_id = int(os.path.basename(path)[len('hps_track_'):-len('.npy')])
        tracks.append((track_id, frames, betas.cpu().numpy(), orient_w.cpu().numpy(),
                       rotmat[:, 1:].cpu().numpy(), transl_w.cpu().numpy(), lowest))

    ground = min(t[-1] for t in tracks)
    up = np.array([0, ground, 0], np.float32)
    R_cw = R_wc.transpose(0, 2, 1)
    out = {'body_model': np.array('smpl'),
           'camera/intrinsics': export.intrinsics_matrix(cam['img_focal'], cam['img_center']),
           'camera/extrinsics': export.extrinsics_matrix(R_cw, np.einsum('bij,bj->bi', R_cw, up - T_wc))}
    tracks.sort(key=lambda t: t[0])
    as_axis_angle = lambda m: Rotation.from_matrix(m.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    out['track_id'] = np.array([t[0] for t in tracks], np.int32)
    out['detected'] = np.stack([export.spread(np.ones(len(t[1]), bool), t[1], num_frames, False) for t in tracks])
    out['smpl/betas'] = np.stack([t[2] for t in tracks])
    out['smpl/global_orient'] = np.stack([export.spread(as_axis_angle(t[3]), t[1], num_frames) for t in tracks])
    out['smpl/body_pose'] = np.stack([export.spread(as_axis_angle(t[4]).reshape(len(t[1]), 69), t[1], num_frames)
                                      for t in tracks])
    out['smpl/transl'] = np.stack([export.spread(t[5] - up, t[1], num_frames) for t in tracks])
    return out


def main(input: str, output: str, clips: tuple[str, ...] = (), device: str = 'cuda'):
    """
    input   a method's output folder, one subfolder per clip
    output  where <clip>/results.npz go
    clips   convert only these subfolders
    """
    names = list(clips) or sorted(d for d in os.listdir(input) if os.path.isdir(os.path.join(input, d)))
    body_models = None
    for name in names:
        clip_dir = os.path.join(input, name)
        if os.path.isfile(os.path.join(clip_dir, 'camera.npy')) and os.path.isdir(os.path.join(clip_dir, 'hps')):
            if body_models is None:
                body_models = BodyModels(torch.device(device if torch.cuda.is_available() else 'cpu'))
            arrays = convert_tram(clip_dir, body_models)
        elif os.path.isfile(os.path.join(clip_dir, 'results.pkl')):
            arrays = convert_results_pkl(os.path.join(clip_dir, 'results.pkl'))
        else:
            raise SystemExit(f'{clip_dir}: neither results.pkl nor camera.npy + hps/')
        os.makedirs(os.path.join(output, name), exist_ok=True)
        np.savez_compressed(os.path.join(output, name, 'results.npz'), **arrays)
        print(f'{name}: {len(arrays["track_id"])} people', flush=True)


if __name__ == '__main__':
    tyro.cli(main)
