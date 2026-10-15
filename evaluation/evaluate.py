"""
Score predictions on the K-pop dataset, one row per run.

    python evaluation/evaluate.py --dataset <dataset dir> --pred <predictions dir> --set main
    python evaluation/evaluate.py --dataset <dataset dir> --pred <predictions dir> --set pair

Predictions are <pred>/<clip name>/results.npz in the pipeline's output format;
evaluation/convert.py writes it from other methods' outputs.

main  every clip of main.csv: PCK against yolo11n_pose/main, jitter and foot skating, and
      Mask IoU when --masks gives person masks (format in mask_iou.py). Every metric is
      pooled over all its samples in all clips.
pair  every NN_target of pair.csv, against the pseudo ground truth NN_reference: WA-, W- and
      PA-MPJPE, pair distance, formation, scale and matched people. Means over pairs.
"""
import csv
import os
import sys
from typing import Literal, Optional

import numpy as np
import torch
import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bodies import BodyModels, load_people  # noqa: E402
from motion import track_motion  # noqa: E402
from pair import score_pair  # noqa: E402
from reprojection import clip_pck  # noqa: E402

PAIR_METRICS = [('wa_mpjpe', 'WA-MPJPE'), ('w_mpjpe', 'W-MPJPE'), ('pa_mpjpe', 'PA-MPJPE'),
                ('pair_dist', 'pair dist'), ('formation', 'formation'), ('scale', 'scale')]


def clip_names(dataset, csv_name):
    with open(os.path.join(dataset, csv_name)) as f:
        return [row['clip_name'] for row in csv.DictReader(f)]


def prediction_paths(pred, names):
    paths = {n: os.path.join(pred, n, 'results.npz') for n in names}
    missing = [n for n, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(names)} clips have no {pred}/<clip>/results.npz: '
                         f'{", ".join(missing[:10])}{" ..." if len(missing) > 10 else ""}')
    return paths


def evaluate_main(dataset, paths, masks, body_models):
    if masks is not None:
        from mask_iou import clip_mask_iou

    rows = []
    for name, path in paths.items():
        people, camera = load_people(path)
        jitter, skating = [], []
        for person in people:
            j, s = track_motion(body_models, person)
            jitter.append(j)
            skating.append(s)
        jitter, skating = np.concatenate(jitter), np.concatenate(skating)
        yolo = dict(np.load(os.path.join(dataset, 'yolo11n_pose', 'main', f'{name}.npz')))
        correct, total, matches = clip_pck(body_models, people, camera, yolo)
        row = {'clip': name, 'pck': correct / total if total else 0.0, 'pck_matches': matches,
               'jitter': jitter.mean() if len(jitter) else np.nan, 'jitter_samples': len(jitter),
               'foot_skating': skating.mean() if len(skating) else np.nan,
               'foot_skating_samples': len(skating)}
        if masks is not None:
            score_sum, mask_matches, _, _ = clip_mask_iou(
                body_models, people, camera, dict(np.load(os.path.join(masks, f'{name}.npz'))))
            row.update({'mask_iou': score_sum / mask_matches if mask_matches else 0.0,
                        'mask_matches': mask_matches})
        rows.append(row)
        print(f'  {name}  PCK {row["pck"]:.3f}  jitter {row["jitter"]:.2f}  '
              f'skating {row["foot_skating"]:.2f}'
              + (f'  mask IoU {row["mask_iou"]:.3f}' if masks is not None else ''), flush=True)

    def pooled(key, count):
        w = np.array([r[count] for r in rows], float)
        v = np.array([r[key] for r in rows], float)
        keep = w > 0
        return float(np.average(v[keep], weights=w[keep])) if keep.any() else np.nan

    table = [('clips', f'{len(rows)}')]
    if masks is not None:
        table.append(('Mask IoU', f'{pooled("mask_iou", "mask_matches"):.3f}'))
    table += [('PCK', f'{pooled("pck", "pck_matches"):.3f}'),
              ('Jitter (10m/s^3)', f'{pooled("jitter", "jitter_samples"):.2f}'),
              ('Foot skating (mm)', f'{pooled("foot_skating", "foot_skating_samples"):.2f}')]
    return rows, table


def pair_arrays(body_models, people, num_frames, joints):
    """(T,K,22,3) joints and (T,K,21,3) body poses, NaN where a person is not detected.
    joints='smplx' takes SMPL-X's own body joints, 'smpl' the SMPL joints 0-21."""
    J = np.full((num_frames, len(people), 22, 3), np.nan, np.float32)
    X = np.full((num_frames, len(people), 21, 3), np.nan, np.float32)
    for k, person in enumerate(people):
        frames = np.flatnonzero(person.detected)
        if len(frames) == 0:
            continue
        _, smpl_joints, _, body_joints = body_models.forward(person, frames)
        J[frames, k] = (body_joints if joints == 'smplx' else smpl_joints[:, :22]).cpu().numpy()
        X[frames, k] = person.body_pose[frames, :63].reshape(-1, 21, 3)
    return J, X


def evaluate_pair(dataset, paths, body_models):
    rows = []
    for name, path in paths.items():
        ref_people, ref_camera = load_people(os.path.join(dataset, 'annotations', 'pair',
                                                          name.replace('_target', '_reference') + '.npz'))
        people, camera = load_people(path)
        num_frames = len(ref_camera['extrinsics'])
        assert len(camera['extrinsics']) == num_frames, (name, 'frame count differs from the reference')
        # SMPL predictions are compared on SMPL joints, the reference's from its mesh
        joints = people[0].model
        G, Gx = pair_arrays(body_models, ref_people, num_frames, joints)
        P, Px = pair_arrays(body_models, people, num_frames, joints)
        got = score_pair(G, P, Gx, Px)
        row = {'clip': name, 'reference_people': len(ref_people), 'predicted_people': len(people)}
        if got is None:
            row['matched'] = 0
            print(f'  {name}  nothing matched', flush=True)
        else:
            met, matched = got
            row.update(met)
            row['matched'] = matched
            print(f'  {name}  ' + '  '.join(f'{label} {met[k]:.2f}' for k, label in PAIR_METRICS)
                  + f'  matched {matched}/{len(ref_people)}', flush=True)
        rows.append(row)

    scored = [r for r in rows if r['matched']]
    table = [('pairs', f'{len(scored)}')]
    for key, label in PAIR_METRICS:
        unit = '' if key == 'scale' else ' (cm)'
        table.append((label + unit, f'{np.nanmean([r[key] for r in scored]):.2f}'))
    table.append(('matched', f'{sum(r["matched"] for r in rows)} of {sum(r["reference_people"] for r in rows)}'))
    return rows, table


def main(dataset: str,
         pred: str,
         set: Literal['main', 'pair'] = 'main',
         masks: Optional[str] = None,
         clips: Optional[tuple[str, ...]] = None,
         per_clip_csv: Optional[str] = None,
         device: str = 'cuda'):
    """
    dataset       the dataset folder, with main.csv, pair.csv, annotations/ and yolo11n_pose/
    pred          predictions, <pred>/<clip name>/results.npz
    set           main or pair
    masks         main only: person masks, <masks>/<clip name>.npz, adds Mask IoU
    clips         score only these clips instead of the whole set
    per_clip_csv  also write one row per clip to this CSV
    """
    if set == 'main':
        names = clip_names(dataset, 'main.csv')
    else:
        names = [n for n in clip_names(dataset, 'pair.csv') if n.endswith('_target')]
    if clips:
        unknown = [c for c in clips if c not in names]
        if unknown:
            raise SystemExit(f'not in the {set} set: {", ".join(unknown)}')
        names = [n for n in names if n in clips]
    paths = prediction_paths(pred, names)

    body_models = BodyModels(torch.device(device if torch.cuda.is_available() else 'cpu'))
    if set == 'main':
        rows, table = evaluate_main(dataset, paths, masks, body_models)
    else:
        rows, table = evaluate_pair(dataset, paths, body_models)

    if per_clip_csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(per_clip_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    widths = [max(len(label), len(value)) for label, value in table]
    print('\n' + '  '.join(f'{label:>{w}}' for (label, _), w in zip(table, widths)))
    print('  '.join(f'{value:>{w}}' for (_, value), w in zip(table, widths)))


if __name__ == '__main__':
    tyro.cli(main)
