"""Body models and reading a results.npz into per-person meshes."""
import pickle

import numpy as np
import torch
from smplx import SMPL, SMPLX

SMPLX_PATH = 'data/body_models/smplx/SMPLX_NEUTRAL.npz'
SMPL_DIR = 'data/body_models/smpl'
SMPLX2SMPL_PATH = 'data/body_models/smplx2smpl.pkl'

# COCO-17 keypoints as body model joints: SMPL-X's face and foot joints, or SMPL's extra
# vertex joints 24-28 for the face.
SMPLX_COCO = [55, 58, 59, 56, 57, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
SMPL_COCO = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
NUM_SMPLX_BODY_JOINTS = 22


class BodyModels:
    def __init__(self, device):
        self.device = device
        self.smplx = SMPLX(SMPLX_PATH, use_pca=False, flat_hand_mean=True, num_betas=10).to(device)
        self.smpl = SMPL(model_path=SMPL_DIR, gender='neutral', batch_size=1).to(device)
        with open(SMPLX2SMPL_PATH, 'rb') as f:
            self.smplx2smpl = torch.from_numpy(pickle.load(f)['matrix']).float().to(device)
        self.smpl_regressor = self.smpl.J_regressor[:24].float()

    def faces(self, model):
        return (self.smplx if model == 'smplx' else self.smpl).faces

    @torch.no_grad()
    def vertices(self, person, frames):
        """The person's own mesh (SMPL-X or SMPL) on the given frames, (N,V,3)."""
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x[frames]), dtype=torch.float32, device=self.device)
        n = len(frames)
        betas = torch.as_tensor(person.betas, device=self.device).float()[None].repeat(n, 1)
        if person.model == 'smplx':
            zeros3 = torch.zeros((n, 3), device=self.device)
            return self.smplx(
                global_orient=t(person.global_orient), body_pose=t(person.body_pose), betas=betas,
                transl=t(person.transl), jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                left_hand_pose=torch.zeros((n, 45), device=self.device),
                right_hand_pose=torch.zeros((n, 45), device=self.device),
                expression=torch.zeros((n, 10), device=self.device),
            ).vertices
        return self.smpl(global_orient=t(person.global_orient), body_pose=t(person.body_pose),
                         betas=betas, transl=t(person.transl)).vertices

    @torch.no_grad()
    def forward(self, person, frames):
        """
        Posed body of one person on the given frames.

        Returns (smpl_vertices (N,6890,3), smpl_joints (N,24,3), coco_joints (N,17,3),
        body_joints (N,22,3) or None). SMPL-X meshes are mapped to the SMPL topology, so
        every metric runs on the same 24 joints and foot vertices whichever model made
        the prediction; body_joints are SMPL-X's own and exist only for SMPL-X.
        """
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x[frames]), dtype=torch.float32, device=self.device)
        n = len(frames)
        if person.model == 'smplx':
            zeros3 = torch.zeros((n, 3), device=self.device)
            out = self.smplx(
                global_orient=t(person.global_orient), body_pose=t(person.body_pose),
                betas=torch.as_tensor(person.betas, device=self.device).float()[None].repeat(n, 1),
                transl=t(person.transl), jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                left_hand_pose=torch.zeros((n, 45), device=self.device),
                right_hand_pose=torch.zeros((n, 45), device=self.device),
                expression=torch.zeros((n, 10), device=self.device),
            )
            verts = torch.einsum('ij,bjk->bik', self.smplx2smpl, out.vertices)
            joints = torch.einsum('ij,bjk->bik', self.smpl_regressor, verts)
            return verts, joints, out.joints[:, SMPLX_COCO], out.joints[:, :NUM_SMPLX_BODY_JOINTS]

        out = self.smpl(
            global_orient=t(person.global_orient), body_pose=t(person.body_pose),
            betas=torch.as_tensor(person.betas, device=self.device).float()[None].repeat(n, 1),
            transl=t(person.transl),
        )
        return out.vertices, out.joints[:, :24], out.joints[:, SMPL_COCO], None


class Person:
    """One track of a results.npz: world-space body parameters over all T frames (NaN where
    the person has no pose) and which frames come from a detection."""

    def __init__(self, data, k, model):
        self.model = model
        self.track_id = int(data['track_id'][k])
        self.betas = data[f'{model}/betas'][k]
        self.global_orient = data[f'{model}/global_orient'][k]
        self.body_pose = data[f'{model}/body_pose'][k]
        self.transl = data[f'{model}/transl'][k]
        self.present = ~np.isnan(self.transl).any(-1)          # (T,) has a pose
        self.detected = data['detected'][k] & self.present      # (T,) pose from a detection


def load_people(path):
    """(people, camera) from a results.npz; camera holds extrinsics (T,4,4) and intrinsics."""
    data = np.load(path)
    model = str(data['body_model'])
    assert f'{model}/betas' in data.files, (path, f'body_model is {model} but there is no {model}/betas')
    people = [Person(data, k, model) for k in range(len(data['track_id']))]
    camera = {'extrinsics': data['camera/extrinsics'], 'intrinsics': data['camera/intrinsics']}
    return people, camera
