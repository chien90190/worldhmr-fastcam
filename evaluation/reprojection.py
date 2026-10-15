"""2D PCK of the projected world-space bodies against YOLO11n-pose keypoints."""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

DET_SCORE = 0.35        # YOLO detections below this score are ignored
JOINT_CONF = 0.3        # YOLO keypoints at or below this confidence are not scored
PCK_THRESHOLD = 0.15    # a joint is correct within this fraction of the YOLO box height


def project(points, intrinsics, extrinsics):
    """(N,J,3) world points seen by per-frame (N,3,3) intrinsics and (N,4,4) extrinsics -> (N,J,2) pixels."""
    cam = np.einsum('nij,nkj->nki', extrinsics[:, :3, :3], points) + extrinsics[:, None, :3, 3]
    x = intrinsics[:, None, 0, 0] * cam[..., 0] / cam[..., 2] + intrinsics[:, None, 0, 2]
    y = intrinsics[:, None, 1, 1] * cam[..., 1] / cam[..., 2] + intrinsics[:, None, 1, 2]
    return np.stack([x, y], axis=-1)


def per_frame_intrinsics(intrinsics, num_frames):
    return intrinsics if intrinsics.ndim == 3 else np.broadcast_to(intrinsics, (num_frames, 3, 3))


def visible(keypoints):
    mask = keypoints[:, 2] > JOINT_CONF
    return mask if mask.any() else np.ones(len(keypoints), bool)


def clip_pck(body_models, people, camera, yolo):
    """
    (correct joints, scored joints, matched people) of one clip.

    On every frame, the projected COCO-17 joints of the detected people are matched one to
    one to the YOLO detections by Hungarian on the mean distance over the visible YOLO
    keypoints; every matched pair then scores its visible keypoints.
    """
    num_frames = len(camera['extrinsics'])
    intrinsics = per_frame_intrinsics(camera['intrinsics'], num_frames)
    pred_by_frame = defaultdict(list)
    for person in people:
        frames = np.flatnonzero(person.detected)
        if len(frames) == 0:
            continue
        _, _, coco, _ = body_models.forward(person, frames)
        uv = project(coco.cpu().numpy(), intrinsics[frames], camera['extrinsics'][frames])
        for t, joints in zip(frames, uv):
            pred_by_frame[t].append(joints)

    correct = total = matches = 0
    for t in range(len(yolo['score'])):
        keep = yolo['score'][t] >= DET_SCORE
        gt_kps, boxes = yolo['keypoints_2d'][t][keep], yolo['bbox_xyxy'][t][keep]
        pred_kps = pred_by_frame.get(t, [])
        if not pred_kps or not len(gt_kps):
            continue

        cost = np.zeros((len(pred_kps), len(gt_kps)))
        for i, pred in enumerate(pred_kps):
            for j, gt in enumerate(gt_kps):
                v = visible(gt)
                cost[i, j] = np.mean(np.linalg.norm(pred[v] - gt[v, :2], axis=1))
        for i, j in zip(*linear_sum_assignment(cost)):
            height = boxes[j][3] - boxes[j][1]
            if height <= 0:
                continue
            v = visible(gt_kps[j])
            dist = np.linalg.norm(pred_kps[i][v] - gt_kps[j][v, :2], axis=1)
            correct += int(np.sum(dist / height < PCK_THRESHOLD))
            total += len(dist)
            matches += 1
    return correct, total, matches
