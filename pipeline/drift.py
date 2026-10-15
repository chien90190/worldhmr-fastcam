"""
Drift correction for SMPL-X world trajectories.

Camera drift makes everyone's world trajectory slide with the camera. In a window where a
person's foot stays planted, its real displacement is ~0, so its measured displacement over
the camera's displacement is the fraction of camera motion that leaked into the people. These
ratios are pooled into one scale s, and s times the camera's displacement since frame 0 is
subtracted from people and camera alike.

Camera convention, matching post_optimization() in post_optimization.py:
    Rwc = Rcw^T,  Twc = -(Rwc @ Tcw),  Tcw = -(Rcw @ Twc)
"""
import numpy as np
import torch

from prompt_hmr.vis.traj import fit_floor_height
from pipeline import log
from pipeline.joints import FOOT_JOINT_IDS



# Why a sample was dropped, as printed. "cam_static" counts windows, the rest (window, person) pairs.
REJECT_REASONS = {
    "cam_static": "camera barely moved",
    "not_tracked": "person missing at either end",
    "cos_thr": "person motion not along the camera's",
    "depth": "person too close to the camera",
    "snr": "camera rotation dominates",
    "contact": "no foot planted through the window",
}


def sigmoid_np(x):
    x = np.asarray(x, np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def stabilize_world_drift(results, smplx, cfg, stride=1):
    """
    Remove camera drift from every person's world trajectory and from the camera, in place.
    Writes results["camera_world"]["drift"]; its "mode" is "none" when no window qualified,
    and nothing is changed then.
    """
    people = results["people"]
    cam = results["camera_world"]
    cam_pos_world = np.asarray(cam["Twc"], np.float32)     # (T,3)
    cam_rot_world = np.asarray(cam["Rwc"], np.float32)     # (T,3,3)
    num_frames = len(cam_pos_world)
    cam_delta = (cam_pos_world - cam_pos_world[0]).astype(np.float32)

    last_tracked = max(int(f) for p in people.values() for f in p["frames"])
    if last_tracked < num_frames - 1:
        log.detail(f"no person tracked on the last {num_frames - 1 - last_tracked} frames")

    C, D, W, reject, num_windows = collect_drift_samples(people, smplx, cam_delta, cam_rot_world, cfg, num_frames, stride)

    if len(C) == 0:
        log.detail("no drift correction, no window qualified: "
                   + ", ".join(f"{REJECT_REASONS[k]} {v}" for k, v in reject.items() if v))
        cam["drift"] = {
            "mode": "none",
            "s_global": 0.0,
            "cam_delta": cam_delta,
            "floor_raise": 0.0,
            "reject_samples": reject,
        }
        return results

    C = np.stack(C, axis=0).astype(np.float32)   # (K,3)
    D = np.stack(D, axis=0).astype(np.float32)   # (K,3)
    W = np.asarray(W, np.float32)                # (K,)

    # Weighted least squares of person displacement on camera displacement, pooled over axes.
    num_axis = (W[:, None] * (C * D)).sum(axis=0)
    den_axis = (W[:, None] * (C * C)).sum(axis=0)
    s_global = np.float32(float(num_axis.sum()) / (float(den_axis.sum()) + 1e-8))
    den_d = float((W * (D * D).sum(axis=1)).sum()) + 1e-8
    cos_global = float(num_axis.sum()) / (np.sqrt(float(den_axis.sum()) + 1e-8) * np.sqrt(den_d))

    estimated_drift = s_global * cam_delta       # (T,3)
    for person in people.values():
        frames = np.asarray(person["frames"], np.int32)
        trans_world = np.asarray(person["smplx_world"]["trans"], np.float32)
        trans_world[:, :3] -= estimated_drift[frames]
        person["smplx_world"]["trans"] = trans_world

    # Removing drift shifts everyone vertically too, so re-seat the floor.
    floor_raise = estimate_floor_offset(results, smplx, contact_thr=cfg.contact_thr)
    for person in people.values():
        person["smplx_world"]["trans"] = np.asarray(person["smplx_world"]["trans"], np.float32) - floor_raise

    sample_s = ((C * D).sum(axis=1) / ((C * C).sum(axis=1) + 1e-8)).astype(np.float32)
    cam["drift"] = {
        "mode": "global_scalar",
        "s_global": float(s_global),
        "cos_global": float(cos_global),
        "num_samples": int(len(W)),
        "num_windows": int(num_windows),
        "reject_samples": reject,
        "sample_s": sample_s,                                   # (K,) per-sample scale
        "s_per_frame": np.full(num_frames, s_global, np.float32),
        "cam_delta": cam_delta,                                 # (T,3)
        "floor_raise": floor_raise,                             # (3,)
        "contact_thr": cfg.contact_thr,
    }
    log.detail(f"drift scale {float(s_global):.3f} from {len(W)} samples in {num_windows} windows "
               f"(alignment with the camera, cos {cos_global:.2f})")

    # The camera moves with the people, including the floor re-seating.
    shift_cam(results, estimated_drift + floor_raise)
    return results


def collect_drift_samples(people, smplx, cam_delta, cam_rot_world, cfg, num_frames, stride):
    """
    Slide a window of cfg.win_size frames over the video and keep one sample per
    (window, person) where a foot stays planted and the camera moved enough to measure
    against. The person's displacement is measured at that planted foot joint.

    Returns the per-sample weighted camera displacements C, person displacements D,
    contact weights W, rejection counts and the number of windows with a sample.
    """
    rot_err = cfg.rot_err_deg * np.pi / 180.0
    C, D, W = [], [], []
    reject = dict.fromkeys(REJECT_REASONS, 0)
    num_windows = 0

    tracks = []
    for person in people.values():
        probs = sigmoid_np(person["smplx_cam"]["static_conf_logits"])[..., :len(FOOT_JOINT_IDS)]
        tracks.append((
            np.asarray(person["frames"], np.int32),                    # (Ti,)
            foot_joints_world(person, smplx),                          # (Ti,4,3)
            probs.astype(np.float32),                                  # (Ti,4) contact per foot joint
            np.asarray(person["smplx_cam"]["trans"], np.float32),      # (Ti,3)
        ))

    t0 = 0
    while t0 < num_frames - cfg.win_size:
        t1 = t0 + cfg.win_size

        cam_disp = cam_delta[t1] - cam_delta[t0]
        cam_trans_mag = np.linalg.norm(cam_disp)
        if cam_trans_mag < 1e-3:
            reject["cam_static"] += 1
            t0 += stride
            continue

        # Rotation over the window inflates the depth-dependent error floor below.
        R_rel = cam_rot_world[t1] @ cam_rot_world[t0].T
        theta = np.arccos(np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0))
        theta_err = rot_err + 0.1 * theta

        # Soft axis weights: axes the camera barely moved along say little about the scale.
        axis_w = np.abs(cam_disp)
        axis_w = axis_w / (axis_w.max() + 1e-8)
        sqrt_w = np.sqrt(np.clip(axis_w, 1e-3, None)).astype(np.float32)
        cam_disp = cam_disp * sqrt_w

        window_has_sample = False
        for frame, feet, contact, cam_trans in tracks:
            if not (t0 in frame and t1 in frame):
                reject["not_tracked"] += 1
                continue
            idx0 = np.where(frame == t0)[0][0]
            idx1 = np.where(frame == t1)[0][0]

            # One joint has to stay planted the whole window for its displacement to be ~0.
            j = int(contact[idx0:idx1 + 1].mean(axis=0).argmax())
            disp = (feet[idx1, j] - feet[idx0, j]) * sqrt_w

            # Drift moves people with the camera, so a usable sample is (anti)parallel to it.
            cos_sim = float(disp @ cam_disp) / (np.linalg.norm(disp) * np.linalg.norm(cam_disp) + 1e-8)
            if np.abs(cos_sim) < cfg.cos_thr:
                reject["cos_thr"] += 1
                continue

            depth = abs(float(cam_trans[idx0, 2]))
            if depth < 0.1:
                reject["depth"] += 1
                continue

            # Rotation fakes translation in proportion to depth; real motion must dominate it.
            if (theta_err * depth) * cfg.drift_snr > cam_trans_mag:
                reject["snr"] += 1
                continue

            w_seq = contact[idx0:idx1 + 1, j]
            if (w_seq > cfg.contact_thr).mean() < cfg.align_frac:
                reject["contact"] += 1
                continue

            C.append(cam_disp)
            D.append(disp)
            W.append(float(w_seq.mean()))
            window_has_sample = True

        num_windows += window_has_sample
        t0 += stride

    return C, D, W, reject, num_windows


def foot_joints_world(person, smplx, device="cuda"):
    """(Ti,4,3) world positions of FOOT_JOINT_IDS, columns aligned with static_conf_logits."""
    smplx = smplx.to(device)
    p = torch.from_numpy(np.asarray(person["smplx_world"]["pose"], np.float32)).to(device)            # (Ti,55*3)
    t = torch.from_numpy(np.asarray(person["smplx_world"]["trans"], np.float32)).to(device)           # (Ti,3)
    b = torch.from_numpy(np.asarray(person["smplx_world"]["shape"], np.float32)[:, :10]).to(device)   # (Ti,10)
    zeros3 = torch.zeros((len(p), 3), device=device)

    with torch.no_grad():
        smplx_out = smplx(
            body_pose=p[:, 3:66],
            global_orient=p[:, :3],
            betas=b,
            left_hand_pose=p[:, 75:120],
            right_hand_pose=p[:, 120:165],
            transl=t,
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=torch.zeros((len(p), 10), device=device),
            pose2rot=True,
        )
    return smplx_out.joints[:, FOOT_JOINT_IDS].cpu().numpy().astype(np.float32)


def estimate_floor_offset(results, smplx, contact_thr, device="cuda"):
    """
    Floor height under the posed meshes of people standing on both feet: a (3,) offset with
    only y set, or zeros if nobody qualifies. fit_floor_height is a RANSAC on np.random.
    """
    smplx = smplx.to(device)
    all_verts = []

    for person in results["people"].values():
        trans = np.asarray(person["smplx_world"]["trans"], np.float32)              # (Ti,3)
        pose = np.asarray(person["smplx_world"]["pose"], np.float32)                # (Ti,55*3)
        betas = np.asarray(person["smplx_world"]["shape"], np.float32)[:, :10]      # (Ti,10)

        probs = sigmoid_np(person["smplx_cam"]["static_conf_logits"])[..., :len(FOOT_JOINT_IDS)]
        both_feet = np.minimum(probs[:, 0:2].max(axis=-1), probs[:, 2:4].max(axis=-1))
        keep = np.where(both_feet > contact_thr)[0]
        if keep.size == 0:
            continue

        p = torch.from_numpy(pose[keep]).float().to(device)
        t = torch.from_numpy(trans[keep]).float().to(device)
        b = torch.from_numpy(betas[keep]).float().to(device)
        zeros3 = torch.zeros((len(keep), 3), device=device)

        with torch.no_grad():
            smplx_out = smplx(
                body_pose=p[:, 3:66],
                global_orient=p[:, :3],
                betas=b,
                left_hand_pose=p[:, 75:120],
                right_hand_pose=p[:, 120:165],
                transl=t,
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                expression=torch.zeros((len(keep), 10), device=device),
                pose2rot=True,
            )
        all_verts.append(smplx_out.vertices)

    if len(all_verts) == 0:
        return np.zeros(3, np.float32)

    all_verts = torch.cat(all_verts).detach().cpu()
    return fit_floor_height(all_verts, "ransac", "y").numpy()


def shift_cam(results, drift):
    """Subtract a (T,3) drift from the camera's world position; rotation is unchanged."""
    cam = results["camera_world"]
    Twc = np.asarray(cam["Twc"], np.float32)          # (T,3)
    Rcw = np.asarray(cam["Rcw"], np.float32)          # (T,3,3)

    Twc_new = Twc.copy()
    Twc_new[:len(drift)] -= np.asarray(drift, np.float32)

    cam["Twc"] = Twc_new
    cam["Rwc"] = np.transpose(Rcw, (0, 2, 1))
    cam["Tcw"] = -(Rcw @ Twc_new[..., None])[..., 0]
    cam["Rcw"] = Rcw
    return results
