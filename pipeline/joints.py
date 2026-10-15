"""Joint and keypoint conventions shared by the pipeline's steps. Defined here only."""
import numpy as np

# SMPL-X joints the HMR backbones predict contact for, in the column order of static_conf_logits
CONTACT_JOINTS = {7: 'left_ankle', 10: 'left_foot', 8: 'right_ankle', 11: 'right_foot',
                  20: 'left_wrist', 21: 'right_wrist'}
CONTACT_JOINT_IDS = list(CONTACT_JOINTS)
FOOT_JOINT_IDS = CONTACT_JOINT_IDS[:4]      # the first four columns: ankles and feet
# sigmoid(static_conf_logits) above this is a planted joint in the results
CONTACT_THRESHOLD = 0.8

# OpenPose BODY_25, the order of results' keypoints_2d
OPENPOSE_25 = [
    'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
    'left_wrist', 'mid_hip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
    'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe',
    'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel',
]

# ViTPose's 25 keypoints (COCO-17 + neck, mid hip and feet)
VITPOSE_25 = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'neck', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
    'right_hip', 'mid_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel',
]

# COCO-17, the order PromptHMR-Vid and GVHMR read
COCO_17 = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# SMPL-X joints in OpenPose BODY_25 order (55-65 are SMPL-X's extra face and foot joints)
SMPLX_TO_OPENPOSE_25 = np.array([
    55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
    8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65,
], dtype=np.int32)


def reorder(keypoints, src, dst):
    """(N,len(src),C) keypoints in the order src -> (N,len(dst),C) in the order dst; joints
    dst has and src lacks are zero."""
    out = np.zeros((keypoints.shape[0], len(dst), keypoints.shape[-1]))
    for i, name in enumerate(dst):
        if name in src:
            out[:, i] = keypoints[:, src.index(name)]
    return out


def indices(src, dst):
    """Index into src of every name of dst."""
    return [src.index(name) for name in dst]
