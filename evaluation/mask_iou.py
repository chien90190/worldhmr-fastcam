"""
Overlap of the rendered body meshes with person masks supplied by the user.

Masks file, one per clip, <masks dir>/<clip>.npz:

    masks   (N,H,W) bool   one person mask per row, at the clip's resolution
    frame   (N,)    int    the clip frame each mask belongs to

Every person with a pose on a frame is rendered on its own, matched one to one to that
frame's masks by Hungarian on box IoU (boxes = mask extents, pairs below MIN_BOX_IOU are
dropped), and each match scores |mesh & mask| / |mesh|.
"""
import os

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import pyrender
import trimesh
from scipy.optimize import linear_sum_assignment

MIN_BOX_IOU = 0.1
# OpenCV camera (x right, y down, z forward) to OpenGL (y up, z backward)
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 < x0 or y1 < y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def clip_mask_iou(body_models, people, camera, masks):
    """(sum of scores, matches, rendered people, masks) of one clip."""
    frame_of_mask, masks = masks['frame'], masks['masks']
    height, width = masks.shape[1:]
    num_frames = len(camera['extrinsics'])
    intrinsics = camera['intrinsics']

    meshes = {}
    for person in people:
        frames = np.flatnonzero(person.present)
        if len(frames) == 0:
            continue
        verts = body_models.vertices(person, frames).cpu().numpy()
        faces = body_models.faces(person.model)
        for t, v in zip(frames, verts):
            meshes.setdefault(t, []).append(trimesh.Trimesh(v, faces, process=False))

    renderer = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene()
    camera_node = scene.add(pyrender.PerspectiveCamera(yfov=1.0))
    score_sum, matches, num_meshes, num_masks = 0.0, 0, 0, 0
    for t in range(num_frames):
        focal = intrinsics[t, 0, 0] if intrinsics.ndim == 3 else intrinsics[0, 0]
        camera_node.camera.yfov = 2 * np.arctan(height / (2 * float(focal)))
        scene.set_pose(camera_node, np.linalg.inv(camera['extrinsics'][t]).astype(np.float32) @ CV_TO_GL)

        mesh_masks = []
        for mesh in meshes.get(t, []):
            node = scene.add(pyrender.Mesh.from_trimesh(mesh))
            depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            scene.remove_node(node)
            rendered = np.isfinite(depth) & (depth > 0)
            if rendered.any():
                mesh_masks.append(rendered)
        frame_masks = masks[frame_of_mask == t]
        num_meshes += len(mesh_masks)
        num_masks += len(frame_masks)
        if not mesh_masks or not len(frame_masks):
            continue

        mesh_boxes = [mask_box(m) for m in mesh_masks]
        boxes = [mask_box(m) for m in frame_masks]
        cost = np.array([[1.0 - box_iou(a, b) for b in boxes] for a in mesh_boxes])
        for i, j in zip(*linear_sum_assignment(cost)):
            if 1.0 - cost[i, j] < MIN_BOX_IOU:
                continue
            score_sum += np.logical_and(mesh_masks[i], frame_masks[j]).sum() / mesh_masks[i].sum()
            matches += 1
    renderer.delete()
    return score_sum, matches, num_meshes, num_masks
