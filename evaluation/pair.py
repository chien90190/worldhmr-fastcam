"""
Scoring a target clip against the pseudo ground truth of its reference clip.

The two clips film the same moment from two cameras, so their people and world frames
are matched first. Both worlds are gravity-aligned, so the transform between them is a
scale, a rotation about the vertical axis and a translation. The match is solved once per
clip, in three stages that each only look at the people the previous ones left unmatched:

  1. one similarity for the whole clip, then Hungarian on pelvis distance
  2. the same in WINDOWS short windows, a person is matched by majority vote
  3. Hungarian on body pose (joint rotations), below MAX_POSE_DIST

Metrics, per clip, on the frames where both people of a pair are detected:

  WA-MPJPE   joint error after one yaw similarity fitted to every matched person at once
  W-MPJPE    the same with the similarity fitted on the first START_FRAMES frames only
  PA-MPJPE   joint error after a full similarity per person per frame
  pair dist  mean error of the pelvis distances between people
  formation  pelvis layout error on the floor after a 2D similarity, frames with 3+ people
  scale      the scale of the whole-clip similarity, prediction to reference
"""
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

PELVIS = 0
YAW_STEPS = 360
REFINE_ITERS = 3
MAX_MATCH_DIST = 1.0    # m
WINDOWS = 3
MAX_POSE_DIST = 0.20    # mean per-joint axis-angle distance, rad
START_FRAMES = 2


# ------------------------------------------------------------------------------------
# similarity fits
# ------------------------------------------------------------------------------------

def yaw_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def umeyama_yaw(X, Y):
    """Similarity (s, R, t) minimizing |X - (s R Y + t)|, R a rotation about +y."""
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    A = float((Xc[:, 0] * Yc[:, 0] + Xc[:, 2] * Yc[:, 2]).sum())
    B = float((Xc[:, 0] * Yc[:, 2] - Xc[:, 2] * Yc[:, 0]).sum())
    norm = np.hypot(A, B)
    c, s_ = (A / norm, B / norm) if norm > 1e-12 else (1.0, 0.0)
    R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]], np.float32)
    num = A * c + B * s_ + float((Xc[:, 1] * Yc[:, 1]).sum())
    den = float((Yc ** 2).sum())
    scale = num / den if den > 1e-12 else 1.0
    return scale, R, muX - scale * R @ muY


def umeyama(X, Y):
    """Unconstrained similarity (s, R, t) minimizing |X - (s R Y + t)|."""
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    U, S, Vt = np.linalg.svd(Xc.T @ Yc / len(X))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    var = float((Yc ** 2).sum()) / len(X)
    s = float(S @ [1.0, 1.0, d]) / var if var > 1e-12 else 1.0
    return s, R, muX - s * R @ muY


def apply_sim(P, s, R, t):
    return s * (P @ R.T) + t


def joint_err(A, B):
    return float(np.linalg.norm(A - B, axis=-1).mean())


def dist_matrix(pts):
    return np.linalg.norm(pts[:, None] - pts[None], axis=-1)


def sim2d_rmse(X, Y):
    """RMSE after the best 2D similarity taking Y onto X, both (K,2)."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    A = float((Xc * Yc).sum())
    B = float((Xc[:, 1] * Yc[:, 0] - Xc[:, 0] * Yc[:, 1]).sum())
    norm = np.hypot(A, B)
    c, s_ = (A / norm, B / norm) if norm > 1e-12 else (1.0, 0.0)
    den = float((Yc ** 2).sum())
    scale = norm / den if den > 1e-12 else 1.0
    D = Xc - scale * (Yc @ np.array([[c, -s_], [s_, c]]).T)
    return float(np.sqrt((D ** 2).sum(1).mean()))


# ------------------------------------------------------------------------------------
# matching
# ------------------------------------------------------------------------------------

def formation_size(P):
    """Mean pairwise pelvis distance per frame, NaN where fewer than two are present."""
    out = np.full(len(P), np.nan, np.float32)
    for t in range(len(P)):
        pts = P[t][~np.isnan(P[t]).any(-1)]
        if len(pts) > 1:
            d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            out[t] = d[~np.eye(len(pts), dtype=bool)].mean()
    return out


def centre_per_frame(P):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(P, axis=1, keepdims=True)
    return P - mu


def cost_matrix(G, Q):
    """(Kg,Kp) mean distance over the frames where both tracks are present, inf if none."""
    Kg, Kp = G.shape[1], Q.shape[1]
    cost = np.full((Kg, Kp), np.inf, np.float32)
    for i in range(Kg):
        for j in range(Kp):
            d = np.linalg.norm(G[:, i] - Q[:, j], axis=-1)
            ok = ~np.isnan(d)
            if ok.any():
                cost[i, j] = d[ok].mean()
    return cost


def solve(cost):
    """Hungarian over the finite entries -> [(gt slot, pred slot), ...]."""
    big = np.nanmax(cost[np.isfinite(cost)]) * 10 + 1 if np.isfinite(cost).any() else 1.0
    rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
    return [(r, c) for r, c in zip(rows, cols) if np.isfinite(cost[r, c])]


def stack_pairs(Gp, Pp, pairs):
    """Reference and prediction points of every pair, over the frames both are present."""
    if not pairs:
        return np.empty((0, 3)), np.empty((0, 3))
    ok = [(~np.isnan(Gp[:, r]).any(-1)) & (~np.isnan(Pp[:, c]).any(-1)) for r, c in pairs]
    X = [Gp[:, r][k] for (r, _), k in zip(pairs, ok)]
    Y = [Pp[:, c][k] for (_, c), k in zip(pairs, ok)]
    return np.concatenate(X), np.concatenate(Y)


def global_match(Gp, Pp):
    """Stage 1 on pelvis tracks (T,K,3): yaw grid on centred formations, then refinement."""
    gs, ps = formation_size(Gp), formation_size(Pp)
    both = ~np.isnan(gs) & ~np.isnan(ps) & (ps > 1e-6)
    scale = float(np.median(gs[both] / ps[both])) if both.any() else 1.0

    Gc, Pc = centre_per_frame(Gp), centre_per_frame(Pp)
    best = None
    for theta in np.linspace(0, 2 * np.pi, YAW_STEPS, endpoint=False):
        cost = cost_matrix(Gc, scale * (Pc @ yaw_matrix(theta).T))
        pairs = solve(cost)
        if not pairs:
            continue
        total = float(np.mean([cost[r, c] for r, c in pairs]))
        if best is None or total < best[0]:
            best = (total, theta, pairs)
    if best is None:
        return None

    pairs = best[2]
    s, R, t = scale, yaw_matrix(best[1]), np.zeros(3, np.float32)
    for _ in range(REFINE_ITERS):
        X, Y = stack_pairs(Gp, Pp, pairs)
        if len(X) < 3:
            break
        s, R, t = umeyama_yaw(X, Y)
        pairs = solve(cost_matrix(Gp, s * (Pp @ R.T) + t))
        if not pairs:
            return None

    cost = cost_matrix(Gp, s * (Pp @ R.T) + t)
    return [(r, c) for r, c in solve(cost) if cost[r, c] <= MAX_MATCH_DIST]


def windowed_pairs(Gp, Pp, need_gt, free_pred):
    """Stage 2: stage 1 per window; a person needs a majority of the windows they are in."""
    votes, present_in = {}, {}
    for sl in np.array_split(np.arange(len(Gp)), WINDOWS):
        here = ~np.isnan(Gp[sl]).any(-1)
        for r in need_gt:
            present_in[r] = present_in.get(r, 0) + int(here[:, r].any())
        got = global_match(Gp[sl], Pp[sl])
        if got is None:
            continue
        for r, c in got:
            if r in need_gt and c in free_pred:
                votes.setdefault(r, []).append(c)

    ranked = []
    for r, v in votes.items():
        c = max(set(v), key=v.count)
        share = v.count(c) / max(present_in[r], 1)
        if share > 0.5:
            ranked.append((share, r, c))
    out, taken = {}, set()
    for _, r, c in sorted(ranked, reverse=True):
        if c not in taken:
            out[r] = c
            taken.add(c)
    return out


def pose_pairs(Gx, Px, need_gt, free_pred):
    """Stage 3: Hungarian on body pose (T,K,21,3)."""
    sub = cost_matrix(Gx, Px)[np.ix_(need_gt, free_pred)]
    if not np.isfinite(sub).any():
        return {}
    return {need_gt[i]: free_pred[j] for i, j in solve(sub) if sub[i, j] <= MAX_POSE_DIST}


def match_clip(Gp, Pp, Gx, Px):
    """Sorted [(gt slot, pred slot), ...] from pelvis tracks (T,K,3) and body poses, or None."""
    got = global_match(Gp, Pp)
    if got is None:
        return None
    pairs = list(got)
    matched = {r for r, _ in pairs}

    def remaining():
        taken = {c for _, c in pairs}
        return ([r for r in range(Gp.shape[1]) if r not in matched],
                [c for c in range(Pp.shape[1]) if c not in taken])

    need, free = remaining()
    if need and free:
        for r, c in windowed_pairs(Gp, Pp, set(need), set(free)).items():
            pairs.append((r, c))
            matched.add(r)

    need, free = remaining()
    if need and free:
        for r, c in pose_pairs(Gx, Px, need, free).items():
            pairs.append((r, c))
            matched.add(r)

    if len(stack_pairs(Gp, Pp, pairs)[0]) < 3:
        return None
    return sorted(pairs)


# ------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------

def start_mask(ok, num_pairs):
    """The first START_FRAMES frames where at least half of the pairs are present."""
    counts = np.sum(ok, axis=0)
    need = max(2, (num_pairs + 1) // 2) if num_pairs >= 2 else 1
    idx = np.flatnonzero(counts >= need)[:START_FRAMES]
    if not len(idx):
        idx = np.flatnonzero(counts > 0)[:START_FRAMES]
    mask = np.zeros(len(counts), bool)
    mask[idx] = True
    return mask


def clip_metrics(G, P, pairs):
    """Metrics of one clip from (T,K,J,3) joints, NaN where not detected, and the pairing."""
    ok = [(~np.isnan(G[:, r]).any((1, 2))) & (~np.isnan(P[:, c]).any((1, 2))) for r, c in pairs]
    if not any(m.any() for m in ok):
        return None

    def gather(masks):
        X = np.concatenate([G[m, r].reshape(-1, 3) for (r, _), m in zip(pairs, masks)])
        Y = np.concatenate([P[m, c].reshape(-1, 3) for (_, c), m in zip(pairs, masks)])
        return X, Y

    whole = umeyama_yaw(*gather(ok))
    X0, Y0 = gather([m & start_mask(ok, len(pairs)) for m in ok])
    frozen = umeyama_yaw(X0, Y0) if len(X0) >= 3 else whole

    wa, w, pa = [], [], []
    for (r, c), m in zip(pairs, ok):
        if not m.any():
            continue
        Gj, Pj = G[m, r], P[m, c]
        wa.append(joint_err(Gj, apply_sim(Pj, *whole)))
        w.append(joint_err(Gj, apply_sim(Pj, *frozen)))
        pa.append(np.mean([joint_err(g, apply_sim(p, *umeyama(g, p))) for g, p in zip(Gj, Pj)]))

    scale = whole[0]
    Gm = G[:, [r for r, _ in pairs], PELVIS]
    Pm = P[:, [c for _, c in pairs], PELVIS] * scale
    present = (~np.isnan(Gm).any(-1)) & (~np.isnan(Pm).any(-1))
    pair_err, shape_rmse = [], []
    for t in range(len(Gm)):
        k = np.flatnonzero(present[t])
        if len(k) < 2:
            continue
        Dg, Dp = dist_matrix(Gm[t, k]), dist_matrix(Pm[t, k])
        off = ~np.eye(len(k), dtype=bool)
        pair_err.append(float(np.abs(Dg[off] - Dp[off]).mean()))
        if len(k) >= 3:
            shape_rmse.append(sim2d_rmse(Gm[t, k][:, [0, 2]], Pm[t, k][:, [0, 2]]))

    return {
        'wa_mpjpe': np.mean(wa) * 100,
        'w_mpjpe': np.mean(w) * 100,
        'pa_mpjpe': np.mean(pa) * 100,
        'pair_dist': np.mean(pair_err) * 100 if pair_err else np.nan,
        'formation': np.mean(shape_rmse) * 100 if shape_rmse else np.nan,
        'scale': scale,
    }


def score_pair(G, P, Gx, Px):
    """
    Match and score one pair from (T,K,22,3) joints and (T,K,21,3) body poses of the
    reference (G, Gx) and the prediction (P, Px), NaN where a person is not detected.
    Returns (metrics, number of matched people), or None if nothing could be matched.
    """
    pairs = match_clip(G[:, :, PELVIS], P[:, :, PELVIS], Gx, Px)
    if pairs is None:
        return None
    met = clip_metrics(G, P, pairs)
    if met is None:
        return None
    return met, len(pairs)
