"""
Render a results.npz as a video: the input frames with the meshes on top | the people and the
camera in the world.

    python scripts/render.py --results results/dance/results.npz --video dance.mp4 --output dance_render.mp4

Left: the world-space bodies seen through the estimated camera. Right: the stage from a fixed
viewpoint, chosen per clip so the whole camera path and every person stay in view, with the
camera's path, short floor trails behind each person, and the camera as a small pyramid (an
arrow at the edge when it is out of view). Each person keeps one colour, ordered left to right
where they first appear; frames where a person is not detected are drawn gray.
"""
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import imageio
import numpy as np
import torch
import trimesh
import tyro
from tqdm import tqdm

np.infty = np.inf  # pyrender reads the NumPy 1 alias
import pyrender  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation'))
from bodies import BodyModels, load_people  # noqa: E402

OUT_W, OUT_H, SUPERSAMPLE = 1280, 720, 2   # overlay panel
WORLD_W = 720                              # world panel width (square)
WORLD_UP = np.array([0, 1, 0], dtype='f')
# people are coloured left to right in the image, sampled evenly from this ramp
PALETTE = ['#d1495b', '#ed7d3a', '#e8b04b', '#7fb069', '#2a9d8f', '#3d8fd1', '#6c63c7', '#b8336a']
BG = [0.965, 0.968, 0.975]
FLOOR_COLORS = ([0.925, 0.93, 0.94], [0.885, 0.893, 0.908])
UNDETECTED = '#8a8f98'
UNDETECTED_ALPHA = 0.45
CAMERA_BODY = [0.16, 0.17, 0.2]
CAMERA_PATH = '#b8336a'
EDGE_MARKER = 18                   # px: size of the arrow standing in for an out-of-view camera
TRAIL_FRAMES = 36
TRAIL_WIDTH = 0.07                 # share of body height
WORLD_FOV = np.radians(40)
WORLD_MARGIN = 1.25                # room around the framed content
# viewpoints searched: azimuth 0 is behind the input camera, elevation above the floor
VIEW_AZIMUTHS = np.radians(np.arange(-90, 91, 15))
VIEW_ELEVATIONS = np.radians([30, 40, 50, 60, 70])
MIN_PERSON_PX = 70                 # median person height in the world panel the framing keeps
GROUND_PCT = 25                    # percentile of a person's lowest vertex over its frames, as its floor contact
FLOOR_SEE_THROUGH = 0.55           # floor opacity when the camera path dips below it


# ------------------------------------------------------------------------------------
# scene
# ------------------------------------------------------------------------------------

def load_scene(path, body_models):
    """Tracks {frames, verts, root, detected} in world space, mesh faces and the camera."""
    people, camera = load_people(path)
    E, K = camera['extrinsics'], camera['intrinsics']
    Rcw, Tcw = E[:, :3, :3], E[:, :3, 3]
    Rwc = np.transpose(Rcw, (0, 2, 1))
    cam = {'Rcw': Rcw, 'Tcw': Tcw, 'Rwc': Rwc, 'Twc': -np.einsum('tij,tj->ti', Rwc, Tcw),
           'focal': K[..., 0, 0], 'center': K[..., :2, 2]}
    tracks = []
    for person in people:
        frames = np.flatnonzero(person.present)
        if len(frames) == 0:
            continue
        tracks.append({'frames': frames, 'verts': body_models.vertices(person, frames).cpu().numpy(),
                       'root': person.transl[frames], 'detected': person.detected[frames]})
    return tracks, body_models.faces(people[0].model), cam


def camera_pose_gl(cam, f):
    """Camera-to-world pose of frame f with OpenGL axes, as pyrender wants it."""
    pose = np.eye(4, dtype='f')
    pose[:3, :3], pose[:3, 3] = cam['Rwc'][f], cam['Twc'][f]
    return pose @ np.diag([1, -1, -1, 1]).astype('f')


def look_at(eye, target, up):
    z = (eye - target) / (np.linalg.norm(eye - target) + 1e-9)
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-9
    m = np.eye(4, dtype='f')
    m[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    m[:3, 3] = eye
    return m


def rgb(hex_color):
    h = hex_color.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def material(color, alpha=1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=list(color ** 1.2) + [alpha], metallicFactor=0.0, roughnessFactor=0.65,
        alphaMode='BLEND' if alpha < 1 else 'OPAQUE')


def assign_colors(tracks, cam):
    """Colour by left-to-right position in the image where each track first appears."""
    keys = []
    for t in tracks:
        f = int(t['frames'][0])
        p = cam['Rcw'][f] @ t['root'][0] + cam['Tcw'][f]
        keys.append(p[0] / max(p[2], 1e-6))
    colors = [None] * len(tracks)
    for rank, i in enumerate(np.argsort(keys)):
        pos = rank / max(len(tracks) - 1, 1) * (len(PALETTE) - 1)
        colors[i] = rgb(PALETTE[int(round(pos))])
    return colors


# ------------------------------------------------------------------------------------
# world viewpoint
# ------------------------------------------------------------------------------------

def floor_level(tracks):
    """Median over people of each one's floor contact (lowest vertex, GROUND_PCT percentile over its frames)."""
    return float(np.median([np.percentile(t['verts'][:, :, 1].min(1), GROUND_PCT) for t in tracks]))


def horizontal_toward(point, center):
    d = point - center
    d[1] = 0.0
    return d / max(np.linalg.norm(d), 1e-6)


def view_axes(toward_cam, azimuth, elevation):
    """Viewer axes (back: target -> eye, right, up), swung `azimuth` about the vertical from the
    input camera's side and raised by `elevation`."""
    c, s = np.cos(azimuth), np.sin(azimuth)
    direction = np.array([c * toward_cam[0] + s * toward_cam[2], 0.0, -s * toward_cam[0] + c * toward_cam[2]])
    back = np.cos(elevation) * direction + np.sin(elevation) * WORLD_UP
    right = np.cross(WORLD_UP, back)
    right /= np.linalg.norm(right)
    return back, right, np.cross(back, right)


def fit_view(keep, center, back, right, up, height, margin):
    """Centre the projected bounding box of `keep` and set the distance so every point fits."""
    rel = keep - center
    x, y = rel @ right, rel @ up
    target = center + right * (x.max() + x.min()) / 2 + up * (y.max() + y.min()) / 2
    half_x = max((x.max() - x.min()) / 2, height)
    half_y = max((y.max() - y.min()) / 2, height)
    tan_x = np.tan(np.arctan(np.tan(WORLD_FOV / 2) * WORLD_W / OUT_H)) / margin
    tan_y = np.tan(WORLD_FOV / 2) / margin
    dist = max(half_x / tan_x, half_y / tan_y)
    for _ in range(100):
        v = keep - (target + dist * back)
        depth = -(v @ back)
        if np.any(depth <= 0.05 * dist):
            dist *= 1.5
            continue
        sx, sy = (v @ right) / depth, (v @ up) / depth
        mx, my = (sx.max() + sx.min()) / 2, (sy.max() + sy.min()) / 2
        target = target + (right * mx + up * my) * dist
        scale = max((sx.max() - sx.min()) / 2 / tan_x, (sy.max() - sy.min()) / 2 / tan_y)
        dist *= scale
        if abs(scale - 1) < 1e-4 and max(abs(mx) / tan_x, abs(my) / tan_y) < 1e-4:
            break
    for _ in range(400):
        v = keep - (target + dist * back)
        depth = -(v @ back)
        if np.all(depth > 0) and np.all(np.abs(v @ right) / depth <= tan_x * 1.0001) \
                and np.all(np.abs(v @ up) / depth <= tan_y * 1.0001):
            break
        dist *= 1.005
    return target, dist, max(half_x, half_y)


def screen(points, eye, back, right, up):
    v = points - eye
    depth = -(v @ back)
    return np.stack([v @ right, v @ up], 1) / depth[:, None], depth


def score_view(feet, heads, cam_keep, eye, back, right, up):
    """Median person height in px, the share of the camera path in front of the people, and how
    wide the camera path reads on screen."""
    f_px = OUT_H / 2 / np.tan(WORLD_FOV / 2)
    sf, df = screen(feet, eye, back, right, up)
    sh, _ = screen(heads, eye, back, right, up)
    person_px = float(np.median(np.abs(sh[:, 1] - sf[:, 1]))) * f_px
    if len(cam_keep) == 0:
        return person_px, 0.0, 1.0
    sc, dc = screen(cam_keep, eye, back, right, up)
    lo, hi = np.minimum(sf.min(0), sh.min(0)), np.maximum(sf.max(0), sh.max(0))
    occludes = float(np.mean(np.all((sc >= lo) & (sc <= hi), 1) & (dc < np.median(df))))
    extent = np.ptp(sc, 0) * f_px / np.array([WORLD_W, OUT_H])
    return person_px, occludes, float(np.clip(extent.max(), 0, 1))


def best_view(feet, heads, cam_keep, center, toward_cam, height, body, glyphs):
    """The candidate viewpoint with the largest people, penalizing a camera path that crosses in
    front of them, reads small, or a view far from the input camera's side."""
    keep = np.concatenate([feet, heads, cam_keep, body, glyphs])
    best = None
    for az in VIEW_AZIMUTHS:
        for el in VIEW_ELEVATIONS:
            back, right, up = view_axes(toward_cam, az, el)
            target, dist, radius = fit_view(keep, center, back, right, up, height, WORLD_MARGIN)
            px, occ, wide = score_view(feet, heads, cam_keep, target + dist * back, back, right, up)
            score = px * (1 - occ) * (0.8 + 0.2 * np.cos(az)) * (0.6 + 0.4 * wide)
            if best is None or score > best['score']:
                best = {'score': score, 'px': px, 'target': target, 'dist': dist, 'radius': radius, 'back': back}
    return best


def world_view(tracks, cam):
    """A fixed viewpoint keeping every person over the whole clip and as much of the camera path
    as possible in view, people at least MIN_PERSON_PX tall."""
    floor_y = floor_level(tracks)
    opening = [(t, k) for t in tracks for k in np.where(t['frames'] < 12)[0]] or [(t, 0) for t in tracks]
    height = float(np.median([np.ptp(t['verts'][k][:, 1]) for t, k in opening]))
    feet = np.concatenate([t['root'][::2] for t in tracks])
    feet[:, 1] = floor_y
    heads = feet + WORLD_UP * height
    center = feet.mean(0)

    body = np.concatenate([t['verts'][::3, ::50].reshape(-1, 3) for t in tracks])
    cam_path = cam['Twc']
    glyphs = np.stack([glyph_vertices(camera_pose_gl(cam, f), 0.32 * height) for f in range(len(cam_path))])
    toward_cam = horizontal_toward(cam_path[0].copy(), center)
    offsets = np.linalg.norm((cam_path - center)[:, [0, 2]], axis=1)
    # drop the far end of the camera path only as far as needed to keep people large enough
    for reach in [offsets.max()] + list(np.quantile(offsets, np.linspace(0.95, 0.05, 19))):
        near = offsets <= reach
        best = best_view(feet, heads, cam_path[near], center, toward_cam, height, body,
                         glyphs[near][::2].reshape(-1, 3))
        if best['px'] >= MIN_PERSON_PX:
            break

    target, back, radius = best['target'], best['back'], best['radius']
    eye = target + best['dist'] * back
    d = back * np.array([1.0, 0.0, 1.0])
    d /= max(np.linalg.norm(d), 1e-6)
    az = np.radians(32)
    c, s = np.cos(az), np.sin(az)
    t = np.array([c * d[0] - s * d[2], 0.0, s * d[0] + c * d[2]])
    light_dir = np.array([-s * t[2], 0.0, s * t[0]]) * 0.6 + t * 0.4
    light = look_at((target + 3 * radius * (light_dir + np.array([0, 1.6, 0]))).astype('f'), target.astype('f'), WORLD_UP)
    return {'pose': look_at(eye.astype('f'), target.astype('f'), WORLD_UP), 'light': light, 'floor_y': floor_y,
            'height': height, 'focus': np.array([target[0], floor_y, target[2]]), 'radius': radius}


# ------------------------------------------------------------------------------------
# meshes
# ------------------------------------------------------------------------------------

def floor_mesh(focus, half, tile, floor_y, alpha):
    """Checkerboard fading into the background towards its edge."""
    n = int(np.ceil(half / tile))
    bg = np.array(BG)
    verts, faces, colors = [], [], []
    for i in range(-n, n):
        for j in range(-n, n):
            x0, z0 = focus[0] + j * tile, focus[2] + i * tile
            base = len(verts)
            quad = [[x0, floor_y, z0], [x0 + tile, floor_y, z0], [x0 + tile, floor_y, z0 + tile], [x0, floor_y, z0 + tile]]
            verts += quad
            faces += [[base, base + 2, base + 1], [base, base + 3, base + 2]]
            for v in quad:
                fade = np.clip((np.hypot(v[0] - focus[0], v[2] - focus[2]) - 0.45 * half) / (0.5 * half), 0, 1)
                colors.append(list(np.array(FLOOR_COLORS[(i + j) % 2]) * (1 - fade) + bg * fade) + [alpha])
    return pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, faces, vertex_colors=np.array(colors), process=False),
                                      smooth=False)


def trail_mesh(points, color, width):
    """Flat ribbon on the floor fading from the floor colour (oldest) to the person's (newest)."""
    if len(points) < 2:
        return None
    verts, colors, faces = [], [], []
    floor = np.array(FLOOR_COLORS[0])
    for k, p in enumerate(points):
        d = points[min(k + 1, len(points) - 1)] - points[max(k - 1, 0)]
        side = np.array([-d[2], 0.0, d[0]])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-9 else np.array([1.0, 0.0, 0.0])
        a = (k + 1) / len(points)
        col = list(floor * (1 - a) + color * a) + [1.0]
        verts += [p + side * width / 2, p - side * width / 2]
        colors += [col, col]
    for k in range(len(points) - 1):
        i = 2 * k
        faces += [[i, i + 1, i + 3], [i, i + 3, i + 2], [i, i + 3, i + 1], [i, i + 2, i + 3]]
    return pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(np.array(verts), faces, vertex_colors=np.array(colors), process=False), smooth=False)


def glyph_vertices(pose_gl, size):
    """Corners of the camera pyramid: apex at the optical centre, opening along its view."""
    w, h, d = 0.8 * size, 0.45 * size, size
    verts = np.array([[0, 0, 0], [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d]], dtype=float)
    return verts @ pose_gl[:3, :3].T + pose_gl[:3, 3]


def camera_glyph(pose_gl, size):
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3]]
    faces += [f[::-1] for f in faces]
    return pyrender.Mesh.from_trimesh(trimesh.Trimesh(glyph_vertices(pose_gl, size), faces, process=False),
                                      material=material(np.array(CAMERA_BODY)), smooth=False)


def mark_offscreen_camera(img, corners, view_pose):
    """If the camera pyramid is out of view, draw an arrow at the frame edge pointing to it."""
    h, w = img.shape[:2]
    to_view = np.linalg.inv(view_pose)
    f = 1.0 / np.tan(WORLD_FOV / 2)

    def project(x):
        p = to_view @ np.append(x, 1.0)
        if p[2] >= 0:
            return None
        return np.array([(f * h / w * p[0] / -p[2] + 1) / 2 * w, (1 - f * p[1] / -p[2]) / 2 * h])

    if any(q is not None and 0 <= q[0] <= w and 0 <= q[1] <= h for q in map(project, corners)):
        return img
    q = project(corners[0])
    if q is None:
        return img
    pad = EDGE_MARKER * 1.6
    c = np.array([w / 2, h / 2])
    d = q - c
    d /= np.linalg.norm(d)
    t = min((w / 2 - pad) / max(abs(d[0]), 1e-9), (h / 2 - pad) / max(abs(d[1]), 1e-9))
    tip = c + d * (t + EDGE_MARKER * 0.6)
    side = np.array([-d[1], d[0]])
    back = tip - d * EDGE_MARKER * 1.5
    notch = tip - d * EDGE_MARKER * 1.0
    head = np.array([tip, back + side * EDGE_MARKER * 0.8, notch, back - side * EDGE_MARKER * 0.8])
    img = np.ascontiguousarray(img)
    cv2.fillPoly(img, [np.round(head * 16).astype(np.int32)], tuple(int(255 * x) for x in CAMERA_BODY),
                 cv2.LINE_AA, shift=4)
    return img


def path_segments(positions, radius):
    return [trimesh.creation.cylinder(radius=radius, segment=[a, b], sections=8)
            if np.linalg.norm(b - a) > 1e-6 else None for a, b in zip(positions[:-1], positions[1:])]


def letterbox(img, width, height):
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    out = np.zeros((height, width, 3), 'u1')
    x, y = (width - nw) // 2, (height - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def downsample(img, width=OUT_W):
    return cv2.resize(img, (width, OUT_H), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------
# rendering
# ------------------------------------------------------------------------------------

def main(results: str, video: str, output: str, max_frames: int = 0, device: str = 'cuda'):
    """
    results     a results.npz
    video       the video it was made from
    output      the .mp4 to write
    max_frames  stop after this many frames (0: all)
    """
    body_models = BodyModels(torch.device(device if torch.cuda.is_available() else 'cpu'))
    tracks, faces, cam = load_scene(results, body_models)
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    frames_rgb = [frame for frame in reader]
    reader.close()

    colors = assign_colors(tracks, cam)
    mats = [material(c) for c in colors]
    undetected = material(rgb(UNDETECTED), UNDETECTED_ALPHA)
    lookup = [{int(f): k for k, f in enumerate(t['frames'])} for t in tracks]

    view = world_view(tracks, cam)
    height = view['height']
    cam_positions = cam['Twc']
    alpha = FLOOR_SEE_THROUGH if cam_positions[:, 1].min() < view['floor_y'] else 1.0
    floor = floor_mesh(view['focus'], 2.6 * view['radius'], height / 1.7, view['floor_y'], alpha)
    segments = path_segments(cam_positions, 0.015 * height)
    lift = np.array([0.0, view['floor_y'] + 0.004 * height, 0.0])
    path_mat = material(rgb(CAMERA_PATH))
    world_cam = pyrender.PerspectiveCamera(yfov=WORLD_FOV, aspectRatio=WORLD_W / OUT_H)

    # the intrinsics are in pixels of the video the results were made from
    frame_h, frame_w = frames_rgb[0].shape[:2]
    native_h = 2 * float(np.mean(np.atleast_2d(cam['center'])[:, 1]))
    focals = np.asarray(cam['focal'], float) * frame_h / native_h
    fit = min(OUT_W / frame_w, OUT_H / frame_h)

    renderer = pyrender.OffscreenRenderer(OUT_W * SUPERSAMPLE, OUT_H * SUPERSAMPLE)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, codec='libx264', quality=None, pixelformat='yuv420p',
                                macro_block_size=16, output_params=['-crf', '22', '-preset', 'slow'])
    num_frames = min(len(cam['Rcw']), len(frames_rgb), max_frames or len(frames_rgb))
    for f in tqdm(range(num_frames), desc='frames'):
        people = []
        for t, lk, mat in zip(tracks, lookup, mats):
            if f in lk:
                people.append(pyrender.Mesh.from_trimesh(trimesh.Trimesh(t['verts'][lk[f]], faces, process=False),
                                                         material=mat if t['detected'][lk[f]] else undetected))

        # world view
        scene = pyrender.Scene(bg_color=BG + [1.0], ambient_light=[0.42] * 3)
        scene.add(floor)
        for m in people:
            scene.add(m)
        for t, lk, col in zip(tracks, lookup, colors):
            idx = [lk[g] for g in range(max(0, f - TRAIL_FRAMES), f + 1) if g in lk]
            trail = trail_mesh(t['root'][idx] * np.array([1.0, 0.0, 1.0]) + lift, col, TRAIL_WIDTH * height)
            if trail is not None:
                scene.add(trail)
        segs = [s for s in segments[:f] if s is not None]
        if segs:
            scene.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(segs), material=path_mat))
        pose = camera_pose_gl(cam, f)
        scene.add(camera_glyph(pose, 0.32 * height))
        scene.add(world_cam, pose=view['pose'])
        scene.add(pyrender.DirectionalLight(intensity=2.6), pose=view['light'])
        renderer.viewport_width = WORLD_W * SUPERSAMPLE
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        world = mark_offscreen_camera(downsample(color, WORLD_W), glyph_vertices(pose, 0.32 * height), view['pose'])
        renderer.viewport_width = OUT_W * SUPERSAMPLE

        # overlay
        image = letterbox(frames_rgb[f], OUT_W, OUT_H).astype('f')
        if people:
            focal = float(focals) if focals.ndim == 0 else float(focals[f])
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35] * 3)
            for m in people:
                scene.add(m)
            scene.add(pyrender.PerspectiveCamera(yfov=2 * np.arctan(OUT_H / (2 * focal * fit)),
                                                 aspectRatio=OUT_W / OUT_H), pose=pose)
            scene.add(pyrender.DirectionalLight(intensity=2.8), pose=pose)
            color, depth = renderer.render(scene)
            mask = downsample((depth > 0).astype('f'))[..., None] * 0.92
            image = downsample(color).astype('f') * mask + image * (1 - mask)
        writer.append_data(np.hstack([image.astype('u1'), world]))

    writer.close()
    renderer.delete()
    print(f'wrote {output}')


if __name__ == '__main__':
    tyro.cli(main)
