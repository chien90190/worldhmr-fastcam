"""Jitter and foot skating of world-space bodies, on the frames where a person is detected."""
import numpy as np
import torch

FPS = 24
# SMPL heel and toe vertices, left then right
FOOT_VERTICES = [3216, 3387, 6617, 6787]
CONTACT_HEIGHT_MM = 50.0


def track_motion(body_models, person):
    """
    (jitter, skating) samples of one person.

    jitter: one value per window of four consecutive detected frames, the mean over the 24
    SMPL joints of the third finite difference times FPS^3, in 10 m/s^3.
    skating: one value per foot vertex and pair of consecutive detected frames where the
    vertex is below CONTACT_HEIGHT_MM on the first frame, its displacement in mm.
    """
    frames = np.flatnonzero(person.detected)
    if len(frames) == 0:
        return np.zeros(0), np.zeros(0)
    verts, joints, _, _ = body_models.forward(person, frames)
    consecutive = np.diff(frames) == 1

    jerk = (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * FPS ** 3
    jitter = jerk.norm(dim=2).mean(dim=-1).cpu().numpy() / 10.0
    jitter = jitter[consecutive[:-2] & consecutive[1:-1] & consecutive[2:]]

    feet = verts[:, FOOT_VERTICES]
    contact = feet[:, :, 1] * 1000 < CONTACT_HEIGHT_MM
    disp = (feet[1:] - feet[:-1]).norm(dim=-1)
    mask = contact[:-1] & torch.as_tensor(consecutive, device=feet.device)[:, None]
    skating = disp[mask].cpu().numpy() * 1000
    return jitter, skating
