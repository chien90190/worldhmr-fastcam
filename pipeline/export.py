"""Write a pipeline result as one npz of flat arrays with '/'-grouped keys.

P people x T frames, every array on the clip's timeline:

    body_model                 ()          "smplx": the body is under the smplx/ keys
    camera/intrinsics          (3,3)       (T,3,3) when the focal length is per frame
    camera/extrinsics          (T,4,4)     world -> camera, x_cam = E @ x_world
    track_id                   (P,)
    detected                   (P,T)       person detected on this frame; False where the pose is filled in
    bbox_xyxy                  (P,T,4)     NaN where the person has no box
    keypoints_2d               (P,T,25,3)  OpenPose BODY_25, x y confidence; NaN likewise
    contact                    (P,T,6)     joint is static, per contact_joints
    contact_joints             (6,)        SMPL-X joint name of each contact column
    smplx/betas                (P,10)
    smplx/global_orient        (P,T,3)     world, axis-angle; NaN where the person has no pose
    smplx/body_pose            (P,T,63)
    smplx/transl               (P,T,3)
    smplx_incam/global_orient  (P,T,3)     camera frame, before world optimization; NaN on gap frames
    smplx_incam/body_pose      (P,T,63)
    smplx_incam/transl         (P,T,3)

Build the mesh with SMPLX(use_pca=False, flat_hand_mean=True, num_betas=10) and zero
jaw, eye, hand and expression parameters.
"""
import numpy as np

from pipeline.joints import CONTACT_JOINTS, CONTACT_THRESHOLD



def contact_flags(static_conf_logits):
    return 1 / (1 + np.exp(-np.asarray(static_conf_logits, np.float64))) > CONTACT_THRESHOLD


def intrinsics_matrix(focal, center):
    focal = np.asarray(focal, np.float32)
    center = np.asarray(center, np.float32)
    K = np.zeros(focal.shape + (3, 3), np.float32)
    K[..., 0, 0] = focal
    K[..., 1, 1] = focal
    K[..., :2, 2] = center
    K[..., 2, 2] = 1
    return K


def extrinsics_matrix(R, t):
    E = np.zeros((len(R), 4, 4), np.float32)
    E[:, :3, :3] = R
    E[:, :3, 3] = t
    E[:, 3, 3] = 1
    return E


def spread(values, frames, num_frames, fill=np.nan):
    """Put per-frame values onto the full clip timeline, `fill` elsewhere."""
    values = np.asarray(values)
    out = np.full((num_frames,) + values.shape[1:], fill, dtype=values.dtype)
    out[frames] = values
    return out


def stack_people(people, num_frames, world_key):
    """Per-person arrays stacked to (P, T, ...). world_key names the world SMPL-X dict, or None
    when the person dict holds it directly; det_frames, if present, lists the detected frames."""
    ids = sorted(people)
    cols = {k: [] for k in ["detected", "bbox_xyxy", "keypoints_2d", "smplx/betas",
                            "smplx/global_orient", "smplx/body_pose", "smplx/transl"]}
    for tid in ids:
        p = people[tid]
        w = p[world_key] if world_key else p
        frames = np.asarray(p["frames"])
        # bboxes, keypoints_2d and detected are parallel to the frames that have a detection
        det_frames = np.asarray(p["det_frames"]) if "det_frames" in p else frames[~np.asarray(p["gaps"], bool)]
        betas = np.asarray(w["shape"], np.float32)
        assert np.allclose(betas, betas[0], atol=1e-5), (tid, "betas vary over frames")
        pose = np.asarray(w["pose"], np.float32)

        # narrower than p["detected"]: also False on gap frames, where there is no box at all
        cols["detected"].append(spread(np.asarray(p["detected"], bool), det_frames, num_frames, False))
        cols["bbox_xyxy"].append(spread(np.asarray(p["bboxes"], np.float32), det_frames, num_frames))
        cols["keypoints_2d"].append(spread(np.asarray(p["keypoints_2d"], np.float32)[:, :25], det_frames, num_frames))
        cols["smplx/betas"].append(betas[0])
        cols["smplx/global_orient"].append(spread(pose[:, :3], frames, num_frames))
        cols["smplx/body_pose"].append(spread(pose[:, 3:66], frames, num_frames))
        cols["smplx/transl"].append(spread(np.asarray(w["trans"], np.float32), frames, num_frames))
    out = {"track_id": np.array(ids, np.int32)}
    out.update({k: np.stack(v) for k, v in cols.items()})
    return out


def results_to_arrays(results):
    cam = results["camera_world"]
    num_frames = len(cam["Rcw"])
    out = {"body_model": np.array("smplx"),
           "camera/intrinsics": intrinsics_matrix(cam["img_focal"], cam["img_center"]),
           "camera/extrinsics": extrinsics_matrix(cam["Rcw"], cam["Tcw"])}
    out.update(stack_people(results["people"], num_frames, "smplx_world"))

    cols = {k: [] for k in ["contact", "smplx_incam/global_orient", "smplx_incam/body_pose", "smplx_incam/transl"]}
    for tid in sorted(results["people"]):
        p = results["people"][tid]
        c = p["smplx_cam"]
        frames = np.asarray(p["frames"])
        # pose_interpolation fills static_conf_logits over frames but leaves the camera-frame
        # pose on the frames that have a detection
        det_frames = frames[~np.asarray(p["gaps"], bool)]
        pose = np.asarray(c["pose"], np.float32)
        cols["contact"].append(spread(contact_flags(c["static_conf_logits"]), frames, num_frames, False))
        cols["smplx_incam/global_orient"].append(spread(pose[:, :3], det_frames, num_frames))
        cols["smplx_incam/body_pose"].append(spread(pose[:, 3:66], det_frames, num_frames))
        cols["smplx_incam/transl"].append(spread(np.asarray(c["trans"], np.float32), det_frames, num_frames))
    out.update({k: np.stack(v) for k, v in cols.items()})
    out["contact_joints"] = np.array([CONTACT_JOINTS[j] for j in results["contact_joint_ids"]])
    return out


def save_results_npz(results, path):
    np.savez_compressed(path, **results_to_arrays(results))
