import torch

from prompt_hmr.vis.traj import fit_floor_height
from prompt_hmr.utils.one_euro_filter import smooth_one_euro
from prompt_hmr.utils.rotation_conversions import (
    rotation_about_x,
    rotation_about_y,
    matrix_to_axis_angle,
)


def transform_smpl_params(root_orient, transl, R, t, smpl_t_pose_pelvis):
    '''
    Apply a rigid transform to SMPL root orientation and translation, rotating about the
    pelvis (SMPL rotates the body about the pelvis, not the origin).
    Input:
        root_orient: [B, 3, 3]
        transl: [B, 3]
        R: [B, 3, 3]
        t: [B, 3]
        smpl_t_pose_pelvis: [3]

    Output:
        root_orient: [B, 3, 3]
        transl: [B, 3]
    '''
    assert smpl_t_pose_pelvis.shape == (3,)
    smpl_t_pose_pelvis = smpl_t_pose_pelvis[None, :, None]
    transl = transl.unsqueeze(-1).float()
    t = t.unsqueeze(-1)

    transl = R @ (smpl_t_pose_pelvis + transl) + t - smpl_t_pose_pelvis
    root_orient = R @ root_orient
    transl = transl.squeeze()
    return root_orient, transl


def world_hps_estimation(cfg, results, smplx):
    """
    Lift every person from the camera frame to a gravity-aligned world: smooth the camera
    trajectory, level it with SPEC's pitch and roll of the first frame, flip to +y up, and put
    the floor fitted under all meshes at y=0. Writes results['people'][k]['smplx_world']
    (pose, shape, trans) and results['camera_world'].
    """
    pred_cam = results['camera']
    img_focal = pred_cam['img_focal']
    img_center = pred_cam['img_center']
    pred_cam_R = torch.tensor(pred_cam['pred_cam_R'])
    pred_cam_T = torch.tensor(pred_cam['pred_cam_T'])

    pred_cam_T = torch.from_numpy(smooth_one_euro(pred_cam_T.numpy(), cfg.min_cutoff, cfg.beta, cfg.d_cutoff))
    pred_cam_R = torch.from_numpy(smooth_one_euro(pred_cam_R.numpy(), cfg.min_cutoff, cfg.beta, cfg.d_cutoff, is_rot=True))

    cam_wc = torch.eye(4).repeat(len(pred_cam_R), 1, 1)
    cam_wc[:, :3, :3] = pred_cam_R
    cam_wc[:, :3, 3] = pred_cam_T

    # Gravity: rotate the camera by SPEC's pitch and roll of the first frame
    spec_calib = results['spec_calib']['first_frame']
    Rpitch = rotation_about_x(-spec_calib['pitch'])[None, :3, :3]
    Rroll = rotation_about_y(spec_calib['roll'])[None, :3, :3]
    cam_wc[:, :3, :3] = Rpitch @ Rroll @ cam_wc[:, :3, :3]
    cam_wc[:, :3, 3] = (Rpitch @ Rroll @ cam_wc[:, :3, 3][..., None]).squeeze()
    Rwc = cam_wc[:, :3, :3]
    Twc = cam_wc[:, :3, 3]

    # People: camera frame -> world
    for k, v in results['people'].items():
        pred_smpl = v['smplx_cam']
        for k_, v_ in pred_smpl.items():
            pred_smpl[k_] = torch.from_numpy(v_)

        pred_rotmat = pred_smpl['rotmat'].clone()
        pred_shape = pred_smpl['shape'].clone()
        pred_trans = pred_smpl['trans'].clone()

        frame = torch.from_numpy(v['frames'])
        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        smpl_t_pose_pelvis = smplx(
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 21*3),
            betas=mean_shape
        ).joints[0, 0]

        root_orient, pred_trans = transform_smpl_params(
            pred_rotmat[:, 0], pred_trans.squeeze(), Rwc[frame], Twc[frame], smpl_t_pose_pelvis)
        pred_rotmat[:, 0] = root_orient
        pred_trans = pred_trans.squeeze()

        pred_pose_aa = matrix_to_axis_angle(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 55*3)
        B = pred_pose_aa.shape[0]
        pred = smplx(
            global_orient=pred_pose_aa[:, :3],
            body_pose=pred_pose_aa[:, 3:66],
            left_hand_pose=pred_pose_aa[:, 75:120],
            right_hand_pose=pred_pose_aa[:, 120:],
            betas=pred_shape,
            transl=pred_trans,
            jaw_pose=torch.zeros(B, 3).to(pred_pose_aa),
            leye_pose=torch.zeros(B, 3).to(pred_pose_aa),
            reye_pose=torch.zeros(B, 3).to(pred_pose_aa),
            expression=torch.zeros(B, 10).to(pred_pose_aa),
        )

        results['people'][k]['smplx_world'] = {
            'verts': pred.vertices,
            'rotmat': pred_rotmat.clone(),
            'shape': pred_shape,
            'trans': pred_trans.clone(),
            't_pose_pelvis': smpl_t_pose_pelvis,
        }

    # Flip to +y up, and put the floor under all meshes at y = 0
    R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).float() @ torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
    all_pred_vert_w = torch.cat([v['smplx_world']['verts'] for v in results['people'].values()])
    pred_vert_gr = torch.einsum('ij,bnj->bni', R, all_pred_vert_w).detach()
    offset = fit_floor_height(pred_vert_gr, 'ransac', 'y')

    Rwc = torch.einsum('ij,bjk->bik', R, Rwc)
    Twc = torch.einsum('ij,bj->bi', R, Twc) - offset
    Rcw = Rwc.mT
    Tcw = -torch.einsum('bij,bj->bi', Rcw, Twc)

    for k, v in results['people'].items():
        world = v['smplx_world']
        pose_rotmat = world['rotmat']
        root_orient, trans = transform_smpl_params(
            root_orient=pose_rotmat[:, 0],
            transl=world['trans'],
            R=R,
            t=-offset,
            smpl_t_pose_pelvis=world['t_pose_pelvis'],
        )
        pose_rotmat[:, 0] = root_orient

        results['people'][k]['smplx_world'] = {
            'pose': matrix_to_axis_angle(pose_rotmat.reshape(-1, 3, 3)).reshape(-1, 55*3),
            'shape': world['shape'],
            'trans': trans.squeeze(),
        }

    results['camera_world'] = {
        'pred_cam_R': Rwc.numpy(),
        'pred_cam_T': Twc.numpy(),
        'Rwc': Rwc.numpy(),
        'Twc': Twc.numpy(),
        'Rcw': Rcw.numpy(),
        'Tcw': Tcw.numpy(),
        'img_focal': img_focal,
        'img_center': img_center,
    }
    return results
