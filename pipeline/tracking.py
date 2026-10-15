import os
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt
from ultralytics import YOLO

import sam2
from sam2_video.predictor import build_sam2_video_predictor
from pipeline import log


SAM2_CONFIG = os.path.join(os.path.dirname(sam2.__file__), "configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_CHECKPOINT = "data/pretrain/sam2_ckpts/sam2.1_hiera_large.pt"
YOLO_CHECKPOINT = "data/pretrain/yolo11x-pose.pt"


def detect_segment_track(images, cfg):
    """
    Detect people with YOLO11-pose, pick the frame that shows the most of them clearly, and
    track them through the video with SAM 2.1 in both directions. A second SAM 2.1 pass
    re-prompts each track with the YOLO box it best matches every 15 frames.

    images: (T,H,W,3) RGB. Returns {track_id: {frames, bboxes, masks, detected, gaps,
    keypoints_yolo}}.
    """
    predictor = build_sam2_video_predictor('/' + os.path.abspath(SAM2_CONFIG), SAM2_CHECKPOINT)

    log.detail("detecting people with YOLO11-pose")
    det_cache = detect_people(images, cfg.nms_oks_thr, cfg.det_thresh, min_box_hw=10)
    start_frame, boxes, scores, keypoints = find_start_frame(
        images, det_cache, cfg.det_score_thresh, cfg.det_height_thresh,
        believable_min_trunc=cfg.believable_min_trunc, believable_min_kp_count=cfg.believable_min_kp_count,
        init_min_kp_count=cfg.init_min_kp_count, init_min_kp_score=cfg.init_min_kp_score,
        init_min_trunc=cfg.init_min_trunc, init_fallback=cfg.init_fallback,
    )

    def init_state():
        return predictor.init_state(video_frames=images, async_loading_frames=False, offload_video_to_cpu=True)

    def propagate(reverse, base_tracks=None):
        """One SAM 2.1 pass from start_frame; with base_tracks, re-prompt from YOLO first."""
        state = init_state()
        add_prompts(predictor, state, start_frame, boxes, keypoints)
        if base_tracks is not None:
            refine_prompts(base_tracks, det_cache, predictor, state, len(images), refine_interval=15, kpt_conf_thresh=0.3)
        tracks = collect_tracks(predictor, state, start_frame, reverse)
        del state
        torch.cuda.empty_cache()
        return tracks

    log.detail(f"tracking with SAM 2.1 from frame {start_frame}")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        log.detail("pass 1 / 4: forward")
        fwd_base = propagate(reverse=False)
        log.detail("pass 2 / 4: backward")
        bwd_base = propagate(reverse=True)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        log.detail("pass 3 / 4: forward,  re-prompted with YOLO boxes every 15 frames")
        fwd = propagate(reverse=False, base_tracks=fwd_base)
        log.detail("pass 4 / 4: backward, re-prompted with YOLO boxes every 15 frames")
        bwd = propagate(reverse=True, base_tracks=bwd_base)
        tracks = merge_tracks(fwd, bwd)

    for obj_id in list(tracks.keys()):
        track = clean_track(tracks[obj_id], len(images))
        if track is None:
            log.detail(f"track {obj_id} dropped: no valid box")
            del tracks[obj_id]
        else:
            tracks[obj_id] = track

    tracks = dict(sorted(tracks.items(), key=lambda x: len(x[1]['frames']), reverse=True))
    keep_ids = list(tracks.keys())[:cfg.num_max_people]
    tracks = {k: v for k, v in tracks.items() if k in keep_ids}

    tracks = deduplicate_tracks(tracks, iou_thresh=0.85)

    return match_yolo_to_tracks(tracks, det_cache)


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------

def detect_people(images, nms_oks_thr, det_thresh, min_box_hw):
    """YOLO11-pose on every frame: {frame: (keypoints (N,17,3), scores (N,), boxes (N,4))}."""
    model = YOLO(YOLO_CHECKPOINT)
    det_cache = {}
    for i in log.progress(range(len(images)), 'frames'):
        r = model(images[i][..., ::-1].copy(), conf=det_thresh, verbose=False)[0]
        if len(r) == 0:
            det_cache[i] = (np.empty((0, 17, 3), dtype=np.float32),
                            np.empty((0,), dtype=np.float32),
                            np.empty((0, 4), dtype=np.float32))
            continue

        box = r.boxes.xyxy.cpu().numpy()
        scr = r.boxes.conf.cpu().numpy()
        kps = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)

        kps, scr, box = filter_detections(kps, scr, box, nms_oks_thr, det_thresh, min_box_hw)
        keep = dedup_boxes(box, scr, iou_thr=0.8)
        det_cache[i] = (kps[keep], scr[keep], box[keep])
    del model
    return det_cache


def filter_detections(keypoints, scores, boxes, nms_oks_thr, det_thresh, min_box_hw):
    """OKS non-maximum suppression, then a score and a minimum box size threshold."""
    if len(scores) == 0:
        return keypoints, scores, boxes

    keep = nms_oks(keypoints.transpose(0, 2, 1), boxes, nms_oks_thr)
    keypoints, scores, boxes = keypoints[keep], scores[keep], boxes[keep]

    conf = scores > det_thresh
    keypoints, scores, boxes = keypoints[conf], scores[conf], boxes[conf]
    if len(scores) == 0:
        return keypoints, scores, boxes

    hw = np.stack([boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]], axis=1)
    ok = np.logical_and(hw[:, 0] >= min_box_hw, hw[:, 1] >= min_box_hw)
    return keypoints[ok], scores[ok], boxes[ok]


def dedup_boxes(boxes, scores, iou_thr):
    """Indices of boxes kept by greedy IoU suppression in descending score order."""
    keep = []
    for idx in np.argsort(scores)[::-1]:
        if all(box_iou(boxes[idx], boxes[j]) < iou_thr for j in keep):
            keep.append(idx)
    return np.array(keep, dtype=np.int64)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))


def nms_oks(kp_predictions, rois, thresh):
    """Non-maximum suppression on keypoint similarity. kp_predictions: (N,3,K)."""
    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(kp_predictions[i], rois[i], kp_predictions[order[1:]], rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """OKS of one detection (3,K) against N others (N,3,K), normalized by the source box area."""
    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]
    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    return np.sum(np.exp(-e), axis=1) / e.shape[1]


# --------------------------------------------------------------------------------------
# start frame
# --------------------------------------------------------------------------------------

def find_start_frame(images, det_cache, score_thresh, height_thresh, occ_w=0.2, blur_w=0.2, sep_w=0.3, det_w=0.3,
                     believable_min_trunc=0.05, believable_min_kp_count=3,
                     init_min_kp_count=5, init_min_kp_score=0.35, init_min_trunc=0.10, init_fallback=True):
    """
    Pick the frame SAM 2.1 starts from. Frames are ranked first by how many believable people
    they show (capped at the 95th percentile count over the video), then by how many of those
    pass the stricter prompt criteria, then by a quality score (overlap, confidence,
    separation, truncation, keypoints, temporal support).

    Returns (start_frame, boxes, scores, keypoints) of the people used as prompts.
    """
    def spatial_separation(boxes):
        if len(boxes) < 2:
            return 1.0
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        iu = np.triu_indices(len(centers), k=1)
        return float(np.mean(dists[iu])) if len(iu[0]) > 0 else 0.0

    def occlusion_score(boxes):
        if len(boxes) < 2:
            return 1.0
        ious = [box_iou_plain(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
        ious = np.asarray(ious, dtype=np.float32)
        overlap_penalty = 0.7 * np.max(ious) + 0.3 * np.mean(ious)
        return float(np.clip(1.0 - overlap_penalty, 0.0, 1.0))

    def border_score(boxes, img_h, img_w, margin=0.03):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        full_ok = ((x1 > img_w * margin) & (y1 > img_h * margin) &
                   (x2 < img_w * (1.0 - margin)) & (y2 < img_h * (1.0 - margin))).astype(np.float32)
        soft = np.minimum.reduce([
            np.clip(x1 / (img_w * margin + 1e-6), 0.0, 1.0),
            np.clip(y1 / (img_h * margin + 1e-6), 0.0, 1.0),
            np.clip((img_w - x2) / (img_w * margin + 1e-6), 0.0, 1.0),
            np.clip((img_h - y2) / (img_h * margin + 1e-6), 0.0, 1.0),
        ])
        return np.maximum(full_ok, soft).astype(np.float32)

    def normalize_confidence(scores):
        scores = scores.astype(np.float32)
        if np.max(scores) <= 1.0:
            return np.clip(scores, 0.0, 1.0)
        scale = max(np.percentile(scores, 95), 1e-6)
        return np.clip(scores / scale, 0.0, 1.0)

    def keypoint_score(keypoints):
        """Share of visible keypoints, weighted toward head, shoulders and hips."""
        conf = keypoints[:, :, 2]
        all_vis = conf >= 0.25
        visible_count = np.sum(all_vis, axis=1).astype(np.int32)
        all_ratio = np.mean(all_vis, axis=1).astype(np.float32)
        preferred_ids = [0, 1, 2, 3, 4, 5, 6, 11, 12]
        pref_ratio = np.mean(conf[:, preferred_ids] >= 0.35, axis=1).astype(np.float32)
        return np.clip(0.7 * pref_ratio + 0.3 * all_ratio, 0.0, 1.0), visible_count

    def dedup_indices(idxs, boxes, ref_scores, iou_thr=0.75):
        if len(idxs) <= 1:
            return list(idxs)
        keep = []
        for idx in sorted(idxs, key=lambda k: float(ref_scores[k]), reverse=True):
            if all(box_iou_plain(boxes[idx], boxes[kept]) < iou_thr for kept in keep):
                keep.append(idx)
        return keep

    frame_info_cache = {}

    def get_frame_info(i):
        if i in frame_info_cache:
            return frame_info_cache[i]

        keypoints, scores, boxes = det_cache[i]
        if len(scores) == 0:
            frame_info_cache[i] = None
            return None

        img_h, img_w = images[i].shape[:2]
        img_diag = np.sqrt(img_h ** 2 + img_w ** 2)

        scores_n = normalize_confidence(scores)
        bbox_height = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        trunc_score = border_score(boxes, img_h, img_w)
        kp_score, kp_visible_count = keypoint_score(keypoints)
        size_score = np.clip(bbox_height / max(img_h, 1) / 0.35, 0.0, 1.0)
        person_quality = 0.50 * scores_n + 0.20 * kp_score + 0.15 * size_score + 0.15 * trunc_score

        # Lenient set: the people this frame is counted as showing.
        min_height_px = img_h * float(height_thresh)
        believable_idxs = [
            k for k in range(len(scores))
            if scores_n[k] >= float(score_thresh) and bbox_height[k] >= min_height_px
            and trunc_score[k] >= believable_min_trunc and kp_visible_count[k] >= believable_min_kp_count
        ]
        believable_idxs = dedup_indices(believable_idxs, boxes, scores_n, iou_thr=0.75)
        if len(believable_idxs) == 0:
            frame_info_cache[i] = None
            return None
        believable_idxs = sorted(believable_idxs, key=lambda k: person_quality[k], reverse=True)

        # Strict set: the people used as SAM 2.1 prompts.
        init_idxs = [
            k for k in believable_idxs
            if person_quality[k] >= float(score_thresh) and bbox_height[k] >= min_height_px
            and kp_visible_count[k] >= init_min_kp_count and kp_score[k] >= init_min_kp_score
            and trunc_score[k] >= init_min_trunc
        ]
        if init_fallback and len(init_idxs) == 0:
            init_idxs = believable_idxs[:1]

        selected_boxes_count = boxes[believable_idxs]
        info = {
            "frame_idx": i,
            "count_idxs": believable_idxs,
            "best_idxs": init_idxs,
            "quality_det": float(np.mean(person_quality[init_idxs])),
            "quality_blur": float(np.mean(scores_n[init_idxs])),
            "quality_kp": float(np.mean(kp_score[init_idxs])),
            "quality_occ": occlusion_score(selected_boxes_count),
            "quality_sep": float(np.clip(spatial_separation(selected_boxes_count) / (img_diag * 0.28 + 1e-6), 0.0, 1.0)),
            "quality_trunc": float(np.mean(trunc_score[init_idxs])),
            "combined_score": person_quality,
            "keypoints": keypoints,
            "scores": scores,
            "boxes": boxes,
        }
        frame_info_cache[i] = info
        return info

    def temporal_support(frame_idx, expected_count):
        vals = []
        for j in range(max(0, frame_idx - 1), min(len(images), frame_idx + 2)):
            if j == frame_idx:
                continue
            info_j = get_frame_info(j)
            if info_j is None:
                vals.append(0.0)
                continue
            count_ratio = min(len(info_j["count_idxs"]) / max(expected_count, 1), 1.0)
            vals.append(0.5 * count_ratio + 0.25 * info_j["quality_det"] + 0.25 * info_j["quality_kp"])
        return float(np.mean(vals)) if vals else 0.5

    candidates = [info for info in (get_frame_info(i) for i in range(len(images))) if info is not None]
    if len(candidates) == 0:
        raise ValueError("No persons detected in any of the images satisfying thresholds.")

    person_counts = np.asarray([len(c["count_idxs"]) for c in candidates], dtype=np.float32)
    target_count = max(int(np.round(np.percentile(person_counts, 95))), 1)

    extra_w_trunc, extra_w_temp, extra_w_kp = 0.08, 0.08, 0.12
    total_w = occ_w + blur_w + sep_w + det_w + extra_w_trunc + extra_w_temp + extra_w_kp

    best, best_rank = None, None
    for cand in candidates:
        n_count = len(cand["count_idxs"])
        quality_score = (
            occ_w * cand["quality_occ"] +
            blur_w * cand["quality_blur"] +
            sep_w * cand["quality_sep"] +
            det_w * cand["quality_det"] +
            extra_w_trunc * cand["quality_trunc"] +
            extra_w_temp * temporal_support(cand["frame_idx"], max(n_count, target_count)) +
            extra_w_kp * cand["quality_kp"]
        ) / total_w
        rank = (min(n_count, target_count), -max(0, n_count - target_count),
                len(cand["best_idxs"]), quality_score, -cand["frame_idx"])
        if best_rank is None or rank > best_rank:
            best, best_rank = cand, rank

    idxs = best["best_idxs"]
    log.detail(f"start frame {best['frame_idx']}: {len(best['count_idxs'])} people counted, "
               f"{len(idxs)} used as SAM 2.1 prompts")
    return best["frame_idx"], best["boxes"][idxs], best["scores"][idxs], best["keypoints"][idxs]


def box_iou_plain(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


# --------------------------------------------------------------------------------------
# SAM 2.1 tracking
# --------------------------------------------------------------------------------------

def add_prompts(predictor, state, frame_idx, boxes, keypoints, kp_thres=0.5):
    """Prompt each person with their box, their keypoints as positive points, and other
    people's keypoints and box centers inside their box as negative points."""
    for idx in range(len(boxes)):
        pos_kp = keypoints[idx][keypoints[idx, :, 2] > kp_thres]
        neg_kp = []
        x1, y1, x2, y2 = boxes[idx]

        for pid in range(len(boxes)):
            if pid == idx:
                continue
            other_kp = keypoints[pid][keypoints[pid, :, 2] > kp_thres]
            if len(other_kp) == 0:
                continue
            m = (other_kp[:, 0] >= x1) & (other_kp[:, 0] <= x2) & (other_kp[:, 1] >= y1) & (other_kp[:, 1] <= y2)
            if m.any():
                neg_kp.append(other_kp[m])
            ox1, oy1, ox2, oy2 = boxes[pid]
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            if x1 <= ocx <= x2 and y1 <= ocy <= y2:
                neg_kp.append(np.array([[ocx, ocy, 1.0]]))

        points, labels = [], []
        if len(pos_kp) > 0:
            points.append(pos_kp[:, :2])
            labels.append(np.ones(len(pos_kp), dtype=np.int32))
        if len(neg_kp) > 0:
            neg_kp = np.concatenate(neg_kp, axis=0)
            points.append(neg_kp[:, :2])
            labels.append(np.zeros(len(neg_kp), dtype=np.int32))

        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=idx + 1,
            points=np.concatenate(points, axis=0) if points else None,
            labels=np.concatenate(labels, axis=0) if labels else None,
            box=boxes[idx],
        )


def collect_tracks(predictor, state, start_frame, reverse):
    tracks = {}
    for t, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=start_frame, reverse=bool(reverse)):
        boxes_t = batched_mask_to_box(mask_logits > 0.0)
        for obj_id, box, obj_msk in zip(obj_ids, boxes_t, mask_logits):
            obj_id = int(obj_id)
            if obj_id not in tracks:
                tracks[obj_id] = {'track_id': obj_id, 'bbox_format': 'x1y1x2y2', 'frames': [], 'bboxes': [], 'masks': []}
            tracks[obj_id]['frames'].append(int(t))
            tracks[obj_id]['bboxes'].append(box[0].cpu().numpy())
            tracks[obj_id]['masks'].append((obj_msk[0] > 0.0).cpu().numpy().astype(np.bool_))
    return tracks


def merge_tracks(track_fwd, track_bwd):
    """Union of the two passes per object; the forward pass wins on frames both cover."""
    merged = {}
    for obj_id in set(track_fwd.keys()) | set(track_bwd.keys()):
        obj_id = int(obj_id)
        by_frame = {}
        for tracks in (track_bwd, track_fwd):
            if obj_id in tracks:
                for t, b, m in zip(tracks[obj_id]['frames'], tracks[obj_id]['bboxes'], tracks[obj_id]['masks']):
                    by_frame[int(t)] = (b, m)
        ts = sorted(by_frame.keys())
        merged[obj_id] = {'track_id': obj_id, 'bbox_format': 'x1y1x2y2', 'frames': ts,
                          'bboxes': [by_frame[t][0] for t in ts], 'masks': [by_frame[t][1] for t in ts]}
    return merged


def refine_prompts(base_tracks, det_cache, predictor, state, num_frames, refine_interval, kpt_conf_thresh):
    """Every refine_interval frames, match YOLO people to the first pass's masks by the share
    of their keypoints inside each mask, and add each matched YOLO box and keypoints as a new
    prompt for that object."""
    sam_obj_ids = list(base_tracks.keys())
    for frame_idx in range(0, num_frames, refine_interval):
        kps, scr, boxes = det_cache[frame_idx]
        if len(boxes) == 0 or len(sam_obj_ids) == 0:
            continue

        cost = np.ones((len(sam_obj_ids), len(boxes)), dtype=np.float32)
        for i, obj_id in enumerate(sam_obj_ids):
            idx_in_track = np.where(np.array(base_tracks[obj_id]['frames']) == frame_idx)[0]
            if len(idx_in_track) == 0:
                continue
            sam_mask = base_tracks[obj_id]['masks'][idx_in_track[0]]
            height, width = sam_mask.shape

            for j in range(len(boxes)):
                valid_kps = kps[j][kps[j][:, 2] > kpt_conf_thresh]
                if len(valid_kps) == 0:
                    continue
                in_mask = 0
                for kp in valid_kps:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < width and 0 <= y < height and sam_mask[y, x]:
                        in_mask += 1
                cost[i, j] = 1.0 - in_mask / len(valid_kps)

        for row, col in zip(*linear_sum_assignment(cost)):
            if 1.0 - cost[row, col] > 0.30:
                valid_kps = kps[col][kps[col][:, 2] > kpt_conf_thresh]
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=sam_obj_ids[row],
                    box=boxes[col],
                    points=valid_kps[:, :2] if len(valid_kps) > 0 else None,
                    labels=np.ones(len(valid_kps), dtype=np.int32) if len(valid_kps) > 0 else None,
                )


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    XYXY boxes around masks, [0,0,0,0] for an empty mask. For input shape C1xC2x...xHxW, the
    output shape is C1xC2x...x4.
    """
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)

    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]


# --------------------------------------------------------------------------------------
# track clean-up
# --------------------------------------------------------------------------------------

def clean_track(track, num_frames):
    """
    Trim empty boxes at both ends, drop boxes under 10 px, fill gaps of up to 5 frames by
    linear interpolation, and replace boxes that suddenly shrink below half their running
    median area. Returns None when nothing is left.
    """
    frames = np.array(track['frames'])
    bboxes = np.array(track['bboxes'])
    masks = np.array(track['masks'])

    heights = bboxes[:, 3] - bboxes[:, 1]
    widths = bboxes[:, 2] - bboxes[:, 0]
    non_zero = np.where(np.logical_and(heights > 0, widths > 0))[0]
    if len(non_zero) > 0:
        s, e = non_zero[0], non_zero[-1] + 1
        frames, bboxes, masks, heights, widths = frames[s:e], bboxes[s:e], masks[s:e], heights[s:e], widths[s:e]

    ok = np.logical_and(heights > 10, widths > 10)
    bboxes, frames, masks = bboxes[ok], frames[ok], masks[ok]
    if len(bboxes) == 0:
        return None

    bboxes, frames, masks = interpolate_bboxes(bboxes, frames, masks, num_frames=num_frames)
    track['frames'] = frames
    track['masks'] = masks
    track['detected'] = np.sum(masks, axis=(1, 2)) > 1
    track['gaps'] = np.logical_not(np.isin(frames, range(num_frames)))

    bboxes = bboxes.copy()
    if len(bboxes) > 5:
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        bad = areas < (medfilt(areas, kernel_size=5) * 0.5)
        if np.any(bad):
            good_idxs = np.where(~bad)[0]
            if len(good_idxs) > 2:
                for c in range(4):
                    f = interp1d(good_idxs, bboxes[good_idxs, c], kind='linear', fill_value='extrapolate')
                    bboxes[bad, c] = f(np.where(bad)[0])
    track['bboxes'] = bboxes
    return track


def interpolate_bboxes(bboxes, frames, masks, max_gap=5, num_frames=None):
    """
    Fill gaps of up to max_gap frames by linear interpolation, and extend the track to the
    first or last frame of the video when it starts or ends within max_gap of it. Filled
    frames get an empty mask.
    """
    frames = np.asarray(frames, dtype=np.int32)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    first_frame, last_frame = int(frames[0]), int(frames[-1])

    left = 0 if first_frame <= max_gap else first_frame
    right = (num_frames - 1) if (num_frames - 1) - last_frame <= max_gap else last_frame
    full_frames = np.arange(left, right + 1, dtype=np.int32)

    missing = np.diff(frames) - 1
    long_gaps = np.flatnonzero(missing > max_gap)
    if long_gaps.size:
        start = frames[long_gaps] + 1
        end = frames[long_gaps + 1] - 1
        mark = np.zeros(full_frames.size + 1, dtype=np.int32)
        np.add.at(mark, start - left, 1)
        np.add.at(mark, end - left + 1, -1)
        all_frames = full_frames[~(np.cumsum(mark[:-1]) > 0)]
    else:
        all_frames = full_frames

    interp_bboxes = interp1d(frames, bboxes, axis=0, kind="linear", bounds_error=False,
                             fill_value=(bboxes[0], bboxes[-1]))(all_frames)
    all_masks = np.full((len(all_frames), masks.shape[1], masks.shape[2]), False, dtype=bool)
    all_masks[np.searchsorted(all_frames, frames)] = masks
    return interp_bboxes, all_frames, all_masks


def deduplicate_tracks(tracks, iou_thresh=0.85):
    """Of two tracks whose boxes overlap by more than iou_thresh on average over at least 3
    shared frames, keep the longer one."""
    track_ids = sorted(tracks.keys())
    to_remove = set()

    def get_bbox(track, frame_idx):
        loc = np.where(track['frames'] == frame_idx)[0]
        return track['bboxes'][loc[0]] if len(loc) > 0 else None

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    for i in range(len(track_ids)):
        id_a = track_ids[i]
        if id_a in to_remove:
            continue
        for j in range(i + 1, len(track_ids)):
            id_b = track_ids[j]
            if id_b in to_remove:
                continue
            frames_a = set(tracks[id_a]['frames'])
            frames_b = set(tracks[id_b]['frames'])
            common = sorted(list(frames_a & frames_b))
            if len(common) < 3:
                continue

            ious = []
            for f in common:
                box_a, box_b = get_bbox(tracks[id_a], f), get_bbox(tracks[id_b], f)
                if box_a is not None and box_b is not None:
                    ious.append(iou(box_a, box_b))
            if not ious:
                continue

            avg_iou = np.mean(ious)
            if avg_iou > iou_thresh:
                if len(frames_a) >= len(frames_b):
                    to_remove.add(id_b)
                    log.detail(f"track {id_b} dropped: same person as track {id_a} (box IoU {avg_iou:.2f})")
                else:
                    to_remove.add(id_a)
                    log.detail(f"track {id_a} dropped: same person as track {id_b} (box IoU {avg_iou:.2f})")
                    break

    for tid in to_remove:
        del tracks[tid]
    return tracks


def match_yolo_to_tracks(tracks, det_cache, kpt_conf_thresh=0.3):
    """Attach to each track, per frame, the YOLO keypoints whose share inside its mask is
    highest (Hungarian matching, at least 30%); zeros where nothing matches."""
    sam_obj_ids = list(tracks.keys())
    if not sam_obj_ids:
        return tracks

    num_frames = 0
    for obj_id in sam_obj_ids:
        if len(tracks[obj_id]['frames']) > 0:
            num_frames = max(num_frames, np.max(tracks[obj_id]['frames']) + 1)
        tracks[obj_id]['keypoints_yolo'] = np.zeros((len(tracks[obj_id]['frames']), 17, 3), dtype=np.float32)

    for frame_idx in range(num_frames):
        kps, scr, boxes = det_cache[frame_idx]
        if len(boxes) == 0:
            continue

        active = []
        for obj_id in sam_obj_ids:
            idx_in_track = np.where(tracks[obj_id]['frames'] == frame_idx)[0]
            if len(idx_in_track) == 1:
                active.append((obj_id, idx_in_track[0]))
        if len(active) == 0:
            continue

        cost = np.ones((len(active), len(boxes)), dtype=np.float32)
        for i, (obj_id, idx_f) in enumerate(active):
            sam_mask = tracks[obj_id]['masks'][idx_f]
            h, w = sam_mask.shape
            for j in range(len(boxes)):
                valid_kps = kps[j][kps[j][:, 2] > kpt_conf_thresh]
                if len(valid_kps) == 0:
                    continue
                xs, ys = valid_kps[:, 0].astype(int), valid_kps[:, 1].astype(int)
                in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                if not np.any(in_bounds):
                    continue
                cost[i, j] = 1.0 - np.sum(sam_mask[ys[in_bounds], xs[in_bounds]]) / len(valid_kps)

        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] < 0.7:
                obj_id, idx_f = active[r]
                tracks[obj_id]['keypoints_yolo'][idx_f] = kps[c]

    return tracks
