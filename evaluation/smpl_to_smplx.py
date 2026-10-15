"""
Convert a results.npz with an SMPL body (body_model 'smpl') to SMPL-X.

    python evaluation/smpl_to_smplx.py --input results.npz --output results_smplx.npz

Per person:
  betas        fitted so the SMPL-X rest mesh, mapped to the SMPL topology, matches the
               SMPL rest mesh (pipeline/smpl_conversion.py)
  pose         the root and the 21 body joints are copied; SMPL's two hand joints are
               dropped and SMPL-X's hands stay flat
  transl       set per frame so the posed SMPL-X mesh lands on the SMPL mesh

All other keys are copied unchanged and body_model becomes 'smplx'.
"""
import os
import sys

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pipeline.smpl_conversion import SmplToSmplx  # noqa: E402

NUM_BODY_JOINTS = 21


def convert(data, to_smplx):
    assert str(data['body_model']) == 'smpl', 'body_model is not smpl'
    out = {k: data[k] for k in data.files if not k.startswith('smpl/')}
    out['body_model'] = np.array('smplx')

    betas = data['smpl/betas']                                              # (P,10)
    orient = data['smpl/global_orient']                                     # (P,T,3), NaN where absent
    transl = data['smpl/transl'].copy()
    present = ~np.isnan(transl).any(-1)
    for k in range(len(transl)):
        p = present[k]
        R = Rotation.from_rotvec(orient[k, p]).as_matrix()
        _, transl[k, p] = to_smplx(np.repeat(betas[k][None], p.sum(), 0), R, transl[k, p])

    out['smplx/betas'] = to_smplx.fit(betas)[0].astype(np.float32)
    out['smplx/global_orient'] = orient
    out['smplx/body_pose'] = data['smpl/body_pose'][..., :NUM_BODY_JOINTS * 3]
    out['smplx/transl'] = transl.astype(np.float32)
    return out


def main(input: str, output: str):
    """
    input   a results.npz with body_model 'smpl'
    output  where the SMPL-X results.npz goes
    """
    arrays = convert(np.load(input), SmplToSmplx())
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f'{output}: {len(arrays["track_id"])} people')


if __name__ == '__main__':
    tyro.cli(main)
