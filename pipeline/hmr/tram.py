import os
import glob
import numpy as np
import torch

from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from pipeline import log
from pipeline.smpl_conversion import SmplToSmplx


def _box_iou_rows(a, b):
    """Per-row IoU of two aligned (N,4) xyxy box arrays."""
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a + area_b - inter, 1e-8)


def _assign_tracks_per_frame(frames, boxes, donors, iou_thr):
    """
    Per-frame: which TRAM track's box best covers this person's box.

    A pipeline person is not guaranteed to correspond to one TRAM track for the whole
    video -- when two dancers cross under occlusion, the tracker may follow one before
    the crossing and the other after, so the person's box sits on a different TRAM track
    in each stretch. Assigning per frame follows the box rather than an identity, which
    is what PromptHMR-Vid does implicitly (it fits whatever body is inside the box).

    Nothing here smooths the assignment: rapid switching during a crossing is TRAM's
    prediction for those boxes, and correcting it would misrepresent the method.

    Returns an array of donor keys per frame, -1 where nothing matches well enough.
    """
    assign = np.full(len(frames), -1, np.int64)
    best = np.zeros(len(frames), np.float32)

    for k, donor in donors.items():
        shared, i_person, i_donor = np.intersect1d(
            frames, donor['frames'], return_indices=True)
        if shared.size == 0:
            continue
        iou = _box_iou_rows(boxes[i_person], donor['boxes'][i_donor]).astype(np.float32)
        better = iou > np.maximum(best[i_person], iou_thr)
        assign[i_person[better]] = k
        best[i_person[better]] = iou[better]

    return assign


def _tram_track_boxes(trk):
    """Detected frames and xyxy boxes of one TRAM track (a list of per-frame dicts)."""
    frames, boxes = [], []
    for entry in trk:
        if not entry.get('det', False):
            continue
        frames.append(int(entry['frame']))
        boxes.append(np.asarray(entry['det_box'], np.float32).reshape(-1)[:4])
    return np.asarray(frames, np.int64), np.asarray(boxes, np.float32).reshape(-1, 4)


def _hps_index_to_track_id(tracks):
    """
    Rebuild TRAM's hps_track_<k> -> track id mapping.

    estimate_humans.py sorts the tracks by len(trk) descending and enumerates them, so
    hps_track_k belongs to the k-th track of that ordering. np.argsort is not stable, so
    the caller must verify the result (hps frames must be a subset of the track's
    detected frames) rather than trust it.
    """
    tids = list(tracks.keys())
    lens = [len(trk) for trk in tracks.values()]
    rank = np.argsort(lens)[::-1]
    return [tids[r] for r in rank]


def smpl_to_smplx_pose(pred_rotmat):
    """
    SMPL (24 joints) -> the pipeline's SMPLX layout, mirroring how gvhmr_worker.py and
    phmr.py build theirs so downstream stages see an identical structure.

    Returns pose (T,165) axis-angle as [global_orient(3), body(63), zeros(99)] and
    rotmat (T,55,3,3) as the first 25 joints plus 30 identity matrices for the hands.
    SMPL joints 22/23 (its single hand joints) are dropped: SMPLX models hands with
    15 joints per side, and the pipeline runs SMPLX with flat_hand_mean.
    """
    rotmat = torch.as_tensor(np.asarray(pred_rotmat, np.float32))       # (T,24,3,3)
    num_frames = rotmat.shape[0]

    body = matrix_to_axis_angle(rotmat[:, :22].reshape(-1, 3, 3)).reshape(num_frames, 22 * 3)
    pose = torch.cat([body, torch.zeros(num_frames, 165 - 22 * 3)], dim=-1)   # (T,165)

    rotmat_25 = axis_angle_to_matrix(pose[:, :75].reshape(-1, 3)).reshape(-1, 25, 3, 3)
    hand_rotmat = torch.eye(3).repeat(num_frames, 30, 1, 1)
    rotmat_55 = torch.cat([rotmat_25, hand_rotmat], dim=1)               # (T,55,3,3)
    return pose.numpy(), rotmat_55.numpy()


class TRAM_Video():
    """
    Load precomputed TRAM results (TRAM's scripts/estimate_humans.py) into the
    pipeline's `results` instead of running an HMR model.

    TRAM does its own detection/tracking, so its track ids are unrelated to the
    pipeline's, so poses are matched by box IoU -- valid because both run on the same
    frames at the same resolution.

    Matching is per frame, not per track: when two dancers cross under occlusion the
    pipeline's tracker may follow one before the crossing and the other after, so a
    single person's box can sit on different TRAM tracks over the video. Following the
    box rather than an identity is what PromptHMR-Vid does implicitly, which keeps the
    methods comparable -- including through a crossing, where a short segment is still
    the pose TRAM produced for whatever the box was on. Nothing is smoothed or filtered
    here: jitter is a property of the prediction, not something to fix while loading.

    Frames with no confident match are left out of `frames` -- pose_interpolation()
    later refills the pose arrays over the full range and derives `gaps` from `frames`.
    bboxes and keypoints_2d are subset alongside `frames`, because post_optimization() indexes them by
    frames[~gaps] and requires the lengths to agree.

    TRAM predicts SMPL. The body pose is copied to SMPL-X; the shape is fitted and the
    translation re-referenced (pipeline/smpl_conversion.py) so the SMPL-X mesh lands on
    TRAM's SMPL mesh.

    TRAM predicts no contact confidence; static_conf_logits is left absent for the
    static_conf_dir donor path in pipeline.py to fill in.
    """

    def run(self, results, tram_seq_dir, iou_thr=0.3):
        tracks_path = os.path.join(tram_seq_dir, 'tracks.npy')
        if not os.path.exists(tracks_path):
            raise FileNotFoundError(f"TRAM tracks not found: {tracks_path}")

        tram_tracks = np.load(tracks_path, allow_pickle=True).item()
        ordered_ids = _hps_index_to_track_id(tram_tracks)

        hps_files = sorted(glob.glob(os.path.join(tram_seq_dir, 'hps', 'hps_track_*.npy')),
                           key=lambda p: int(os.path.basename(p)[len('hps_track_'):-len('.npy')]))
        if len(hps_files) == 0:
            raise FileNotFoundError(f"No TRAM hps files under {tram_seq_dir}/hps")

        # 1. donor side: hps payload + the matching track's detected boxes
        donors = {}
        for path in hps_files:
            k = int(os.path.basename(path)[len('hps_track_'):-len('.npy')])
            if k >= len(ordered_ids):
                raise ValueError(f"{path}: no track for hps index {k} "
                                 f"({len(ordered_ids)} tracks in tracks.npy)")

            hps = np.load(path, allow_pickle=True).item()
            hps_frames = np.asarray(hps['frame'], np.int64)
            tid = ordered_ids[k]
            det_frames, det_boxes = _tram_track_boxes(tram_tracks[tid])

            # verifies the reconstructed ordering; argsort ties are not stable
            if not np.isin(hps_frames, det_frames).all():
                raise ValueError(
                    f"{os.path.basename(path)} frames are not a subset of track {tid}'s "
                    "detected frames -- the hps_track -> track id ordering is wrong")

            keep = np.isin(det_frames, hps_frames)
            donors[k] = {'tid': tid, 'hps': hps, 'frames': hps_frames,
                         'boxes': det_boxes[keep]}

        to_smplx = SmplToSmplx()

        # 2. per frame, pick the TRAM track whose box covers this person's box
        for pid in list(results['people'].keys()):
            person = results['people'][pid]
            frames = np.asarray(person['frames'], np.int64)
            boxes = np.asarray(person['bboxes'], np.float32)
            if len(boxes) != len(frames):
                log.warning(f"person {pid} has no box on every frame, dropped")
                del results['people'][pid]
                continue

            assign = _assign_tracks_per_frame(frames, boxes, donors, iou_thr)
            keep = assign >= 0
            if not keep.any():
                log.warning(f"person {pid} matches no TRAM track, dropped")
                del results['people'][pid]
                continue

            # 3. gather the poses, one row per kept frame, from whichever track owns it
            kept_frames = frames[keep]
            kept_assign = assign[keep]
            num = len(kept_frames)
            pred_rotmat = np.empty((num, 24, 3, 3), np.float32)
            shape = np.empty((num, 10), np.float32)
            trans = np.empty((num, 3), np.float32)

            for k in np.unique(kept_assign):
                sel = kept_assign == k
                hps = donors[k]['hps']
                rows = np.searchsorted(donors[k]['frames'], kept_frames[sel])
                pred_rotmat[sel] = np.asarray(hps['pred_rotmat'], np.float32)[rows]
                shape[sel] = np.asarray(hps['pred_shape'], np.float32)[rows]
                trans[sel] = np.asarray(hps['pred_trans'], np.float32)[rows, 0]

            num_tracks = len(np.unique(kept_assign))
            log.detail(f"person {pid}: TRAM poses on {num} of {len(frames)} frames"
                       + (f", from {num_tracks} TRAM tracks" if num_tracks > 1 else ""))

            pose, rotmat = smpl_to_smplx_pose(pred_rotmat)
            shape, trans = to_smplx(shape, pred_rotmat[:, 0], trans)
            person['smplx_cam'] = {
                'rotmat': rotmat,       # (T,55,3,3)
                'pose': pose,           # (T,165)
                'shape': shape,         # (T,10)
                'trans': trans,         # (T,3)
            }
            person['smplx_pose'] = torch.from_numpy(rotmat)
            person['smplx_transl'] = torch.from_numpy(trans)
            person['smplx_betas'] = torch.from_numpy(shape)

            person['frames'] = kept_frames.astype(frames.dtype)
            person['bboxes'] = boxes[keep]
            # every other per-frame array has to follow `frames` too, or it silently
            # desynchronises (phmr.py indexes masks by frames, post_optimization
            # indexes bboxes/keypoints_2d by frames[~gaps])
            for key in ('keypoints_2d', 'vitpose', 'masks', 'detected', 'gaps'):
                if key in person and len(person[key]) == len(frames):
                    person[key] = np.asarray(person[key])[keep]

        if len(results['people']) == 0:
            raise ValueError(f"No pipeline person matched a TRAM track in {tram_seq_dir}")

        log.detail(f"TRAM poses for {len(results['people'])} people from {tram_seq_dir}")
        return results
