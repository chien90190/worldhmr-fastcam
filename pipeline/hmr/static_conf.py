"""
Sidecar transfer of per-person contact confidence between pipeline runs.

Only some HPS models predict contact confidence. save_static_conf() dumps it from a donor
run and load_static_conf() replays it into a model that doesn't, so the contact-gated stages
downstream see the same evidence whichever model produced the poses.

Call both right after HPS estimation: pose_interpolation() later resamples the logits and
blends them toward 1.0 inside gaps, so the copy kept in the results is no longer raw output.
"""
import os

import joblib
import numpy as np

from pipeline import log


def save_static_conf(results, path, source=""):
    """
    Dump raw per-person contact confidence to a sidecar file, so a later run of a model that
    doesn't predict it can load it back.

    Payload:
      source             hmr_type of the run that produced the logits
      contact_joint_ids  joint id per logit column (results["contact_joint_ids"])
      num_frames         video length this run saw
      people             {pid: {"frames": (Ti,) int32,          absolute frame indices
                                "bboxes": (Ti,4) float32,       x1y1x2y2, for track re-matching
                                "static_conf_logits": (Ti,J) float32}}
    """
    people = {}
    for pid, person in results["people"].items():
        logits = person.get("smplx_cam", {}).get("static_conf_logits", None)
        if logits is None:
            log.warning(f"person {pid} has no contact confidence, not saved")
            continue

        frames = np.asarray(person["frames"], np.int32)
        logits = np.asarray(logits, np.float32)
        if len(logits) != len(frames):
            log.warning(f"person {pid}: contact confidence on {len(logits)} frames, "
                        f"track on {len(frames)}, not saved")
            continue

        entry = {"frames": frames, "static_conf_logits": logits}
        # Pre-interpolation, bboxes are 1:1 with frames; keep them so a future loader can
        # fall back to IoU matching when track ids don't line up across runs.
        bboxes = person.get("bboxes", None)
        if bboxes is not None and len(bboxes) == len(frames):
            entry["bboxes"] = np.asarray(bboxes, np.float32)
        else:
            log.warning(f"person {pid} has no box on every frame, contact confidence saved "
                        "without boxes")
        people[pid] = entry

    if len(people) == 0:
        log.warning("no person has contact confidence, nothing saved")
        return None

    num_frames = 1 + max(int(f) for e in people.values() for f in e["frames"])
    payload = {
        "source": str(source),
        "contact_joint_ids": list(results.get("contact_joint_ids", [])),
        "num_frames": num_frames,
        "people": people,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(payload, path)
    log.detail(f"contact confidence saved to {path}")
    return path


def _mean_iou_on_shared_frames(frames_a, boxes_a, frames_b, boxes_b):
    """Mean box IoU over the frames two tracks have in common. Returns (iou, n_shared)."""
    shared, ia, ib = np.intersect1d(frames_a, frames_b, return_indices=True)
    if shared.size == 0:
        return 0.0, 0

    a, b = boxes_a[ia], boxes_b[ib]
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter
    return float((inter / np.maximum(union, 1e-8)).mean()), int(shared.size)


def _require_static_conf(results, path):
    """
    Fail here rather than several stages later.

    Falling back to "the model's own logits" only works for a model that predicts them.
    A loader-fed model (hmr_type: tram) has none, so a donor that didn't arrive leaves
    static_conf_logits absent and the run dies in stabilize_world_drift_contact with a
    bare KeyError, after tracking, SLAM and HPS have already been paid for.
    """
    missing = [pid for pid, person in results["people"].items()
               if "smplx_cam" not in person
               or "static_conf_logits" not in person["smplx_cam"]]
    if missing:
        raise KeyError(
            f"no contact confidence for people {missing}: this model does not predict "
            f"static_conf_logits and the donor at {path} did not supply it")


def load_static_conf(results, path, iou_thr=0.3, default_logit=0.0):
    """
    Overwrite each person's contact confidence with a donor run's, as saved by
    save_static_conf().

    People are matched by track id first; anyone unmatched falls back to the donor track with
    the highest mean bbox IoU over shared frames (> iou_thr). Donor logits are re-indexed by
    absolute frame, so differing track spans are fine. Frames the donor never saw get
    `default_logit` (0.0 -> prob 0.5), which the contact_thr gates reject -- a missing frame
    reads as no evidence of contact, not as contact.

    A person left unmatched keeps its own model's prediction, and a missing donor file is a
    warning rather than an error, so a batch sweep doesn't die on one absent sequence.
    """
    if not os.path.exists(path):
        log.warning(f"{path} not found, no contact confidence loaded")
        _require_static_conf(results, path)
        return results

    payload = joblib.load(path)
    donor = payload.get("people", {})
    if len(donor) == 0:
        log.warning(f"{path} has no people, no contact confidence loaded")
        _require_static_conf(results, path)
        return results

    donor_joints = list(payload.get("contact_joint_ids", []))
    own_joints = list(results.get("contact_joint_ids", []))
    if donor_joints and own_joints and donor_joints != own_joints:
        log.warning(f"contact joints differ: {donor_joints} in {path}, {own_joints} here")

    # 1. exact track-id matches
    matches = {pid: pid for pid in results["people"] if pid in donor}
    unmatched_donor = set(donor.keys()) - set(matches.values())

    # 2. IoU fallback for the leftovers (track ids didn't line up across runs)
    for pid, person in results["people"].items():
        if pid in matches or len(unmatched_donor) == 0:
            continue
        boxes = person.get("bboxes", None)
        frames = np.asarray(person["frames"], np.int64)
        if boxes is None or len(boxes) != len(frames):
            continue
        boxes = np.asarray(boxes, np.float32)

        best_id, best_iou, best_n = None, iou_thr, 0
        for did in sorted(unmatched_donor, key=str):
            entry = donor[did]
            if "bboxes" not in entry:
                continue
            iou, n = _mean_iou_on_shared_frames(
                frames, boxes, np.asarray(entry["frames"], np.int64), entry["bboxes"])
            if n > 0 and iou > best_iou:
                best_id, best_iou, best_n = did, iou, n

        if best_id is not None:
            matches[pid] = best_id
            unmatched_donor.discard(best_id)
            log.detail(f"person {pid} takes contact from person {best_id} of {path} by box overlap "
                       f"(IoU {best_iou:.2f} over {best_n} frames)")

    # 3. re-index donor logits onto each person's own frames
    num_applied = 0
    for pid, person in results["people"].items():
        did = matches.get(pid, None)
        if did is None:
            log.warning(f"person {pid} matches no person in {path}, no contact confidence")
            continue

        entry = donor[did]
        donor_frames = np.asarray(entry["frames"], np.int64)
        donor_logits = np.asarray(entry["static_conf_logits"], np.float32)
        frames = np.asarray(person["frames"], np.int64)

        size = int(max(donor_frames.max(), frames.max())) + 1
        dense = np.full((size, donor_logits.shape[-1]), default_logit, np.float32)
        dense[donor_frames] = donor_logits
        seen = np.zeros(size, bool)
        seen[donor_frames] = True

        person.setdefault("smplx_cam", {})["static_conf_logits"] = dense[frames]
        num_applied += 1

        missing = int((~seen[frames]).sum())
        if missing:
            log.detail(f"person {pid}: {missing} of {len(frames)} frames have no contact confidence "
                       f"in {path}, set to probability {1 / (1 + np.exp(-default_logit)):.1f}")

    log.detail(f"contact confidence for {num_applied} of {len(results['people'])} people "
               f"from {path}")
    _require_static_conf(results, path)
    return results
