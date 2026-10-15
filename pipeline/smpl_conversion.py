"""SMPL bodies as SMPL-X: shape fitted, translation re-referenced (poses are copied by the caller)."""
import pickle

import numpy as np
import torch
from smplx import SMPL, SMPLX

SMPL_DIR = 'data/body_models/smpl'
SMPLX_PATH = 'data/body_models/smplx/SMPLX_NEUTRAL.npz'
SMPLX2SMPL_PATH = 'data/body_models/smplx2smpl.pkl'
NUM_BETAS = 10


@torch.no_grad()
def _rest_basis(model, **extra_dims):
    """Rest-pose vertices and pelvis as linear functions of the betas: (v0, dv, j0, dj) with
    v(betas) = v0 + betas @ dv and j(betas) = j0 + betas @ dj."""
    unit = torch.cat([torch.zeros(1, NUM_BETAS), torch.eye(NUM_BETAS)])
    zeros = lambda dim: torch.zeros((len(unit), dim))
    out = model(betas=unit, global_orient=zeros(3), body_pose=zeros(model.NUM_BODY_JOINTS * 3),
                **{k: zeros(dim) for k, dim in extra_dims.items()})
    v, j = out.vertices.double().numpy(), out.joints[:, 0].double().numpy()
    return v[0], v[1:] - v[:1], j[0], j[1:] - j[:1]


class SmplToSmplx:
    """
    SMPL-X betas fitted so the SMPL-X rest mesh, mapped to the SMPL topology, matches the SMPL
    rest mesh up to an offset d (least squares). Both rest meshes are linear in the betas, so
    the fit is one linear map, computed once.
    """

    def __init__(self):
        smpl = SMPL(model_path=SMPL_DIR, gender='neutral')
        smplx = SMPLX(SMPLX_PATH, use_pca=False, flat_hand_mean=True, num_betas=NUM_BETAS)
        with open(SMPLX2SMPL_PATH, 'rb') as f:
            smplx2smpl = pickle.load(f)['matrix']                               # (6890, 10475)
        vs, dvs, self.js, self.djs = _rest_basis(smpl)
        vx, dvx, self.jx, self.djx = _rest_basis(
            smplx, jaw_pose=3, leye_pose=3, reye_pose=3, left_hand_pose=45, right_hand_pose=45, expression=10)
        vx = smplx2smpl @ vx
        dvx = np.einsum('ij,bjk->bik', smplx2smpl, dvx)
        # unknowns [betas_x (10), d (3)]; one row per SMPL vertex coordinate
        A = np.concatenate([dvx.transpose(1, 2, 0), np.broadcast_to(np.eye(3), (len(vx), 3, 3))], -1)
        pinv = np.linalg.pinv(A.reshape(-1, NUM_BETAS + 3))
        self.offset = pinv @ (vs - vx).reshape(-1)                              # (13,)
        self.gain = pinv @ dvs.reshape(NUM_BETAS, -1).T                         # (13, 10)

    def fit(self, betas):
        """SMPL betas (N,10) -> SMPL-X betas (N,10) and offsets d (N,3)."""
        x = np.asarray(betas, np.float64) @ self.gain.T + self.offset
        return x[:, :NUM_BETAS], x[:, NUM_BETAS:]

    def __call__(self, betas, global_orient, transl):
        """
        One SMPL body per row: betas (N,10), global_orient (N,3,3) rotation matrices,
        transl (N,3). Returns SMPL-X betas (N,10) and transl (N,3) that put the SMPL-X mesh,
        posed with the same global_orient and body pose, on the SMPL mesh. Each model rotates
        about its own rest pelvis (j_s, j_x), so transl_x = transl + j_s - j_x + R (d + j_x - j_s).
        """
        betas = np.asarray(betas, np.float64)
        betas_x, d = self.fit(betas)
        j_s = self.js + betas @ self.djs
        j_x = self.jx + betas_x @ self.djx
        transl_x = transl + j_s - j_x + np.einsum('nij,nj->ni', global_orient, d + j_x - j_s)
        return betas_x.astype(np.float32), transl_x.astype(np.float32)
