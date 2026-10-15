"""Step 9: post-optimization of every person's world trajectory and the camera; the paper's
physically grounded optimization (PGO)."""
import numpy as np
import torch
import torch.nn.functional as F

from prompt_hmr.utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix
from pipeline import log
from pipeline.joints import FOOT_JOINT_IDS, SMPLX_TO_OPENPOSE_25


IGNORED_KEYPOINTS = [9, 8, 12]          # BODY_25 hips, which SMPL-X places differently


def gmof(x, sigma=150):
    """Geman-McClure error function"""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def post_optimization(cfg, results, images, smplx, opt_drift=True):
    """
    Post-optimization (PGO): refine every person's world translation, the camera's initial
    rotation and height, and (with opt_drift) a residual on the drift scale plus a floor
    raise, against 2D keypoint reprojection, smoothness, foot contact, floor and inter-person
    repulsion terms. Writes the refined translations and camera back into results.
    """
    smplx = smplx.to('cuda')
    pred_cam = results['camera_world']

    # camera_world holds intrinsics at the video's own resolution; the keypoints live in
    # `images`' (possibly resized) pixel space, so rescale a local copy.
    img_center = np.asarray(pred_cam['img_center'])
    scale = images[0].shape[0] / (float(np.mean(np.atleast_2d(img_center)[:, 1])) * 2.0)
    img_focal = float(pred_cam['img_focal'] * scale)
    img_center = img_center * scale
    K = np.eye(3)
    K[0, 0] = img_focal
    K[1, 1] = img_focal
    K[0, 2] = float(img_center[0])
    K[1, 2] = float(img_center[1])
    K_proj = torch.tensor(K).float().cuda()[None, None, None]  # [1, 1, 1, 3, 3]

    # dict of people -> (P, T, ...) arrays
    seq_len = len(images)
    num_people = len(results['people'])
    people_ids = list(results['people'].keys())

    bboxes = np.zeros((num_people, seq_len, 4))
    pred_kp_2d = np.zeros((num_people, seq_len, 25, 3))
    transl_init = np.zeros((num_people, seq_len, 3))
    mask = np.zeros((num_people, seq_len))
    smplx_pose = np.zeros((num_people, seq_len, 55*3))
    smplx_betas = np.zeros((num_people, seq_len, 10))
    rcw = np.zeros((num_people, seq_len, 3, 3))
    tcw = np.zeros((num_people, seq_len, 3))
    pred_contact = np.zeros((num_people, seq_len, 6))

    for pidx, pid in enumerate(people_ids):
        v = results['people'][pid]
        gaps = v['gaps']
        bboxes[pidx, v['frames'][~gaps]] = v['bboxes']
        pred_kp_2d[pidx, v['frames'][~gaps]] = v['keypoints_2d'][:, :25]
        transl_init[pidx, v['frames']] = v['smplx_world']['trans']
        mask[pidx, v['frames']] = 1
        smplx_pose[pidx, v['frames']] = v['smplx_world']['pose']
        smplx_betas[pidx, v['frames']] = v['smplx_world']['shape'][:, :10]
        pred_contact[pidx, v['frames']] = v['smplx_cam']['static_conf_logits']
        rcw[pidx, v['frames']] = pred_cam['Rcw'][v['frames']]
        tcw[pidx, v['frames']] = pred_cam['Tcw'][v['frames']]

    bbox_height = torch.from_numpy(bboxes[:, :, 3] - bboxes[:, :, 1]).float().cuda().clamp(min=1e-6)
    pred_kp_2d = torch.from_numpy(pred_kp_2d).float().cuda()
    pred_contact = torch.from_numpy(pred_contact).float().cuda()
    tcw = torch.from_numpy(tcw).float().cuda()
    rcw = torch.from_numpy(rcw).float().cuda()
    mask = torch.from_numpy(mask).float().cuda()
    smplx_pose = torch.from_numpy(smplx_pose).float().cuda()
    smplx_betas = torch.from_numpy(smplx_betas).float().cuda()
    transl_init = torch.from_numpy(transl_init).float().cuda()
    world_drift = torch.from_numpy(pred_cam['drift']['cam_delta']).float().cuda()  # (T,3)

    # Parameters
    transl = torch.zeros_like(transl_init, requires_grad=True)
    cam_height_offset = torch.zeros(1).float().cuda().requires_grad_(True)
    cam_init_r6d = matrix_to_rotation_6d(torch.eye(3).float().cuda()).requires_grad_(True)
    resid_scale = torch.tensor(0.0).float().cuda()
    floor_raise_y = torch.tensor(0.0).float().cuda()

    param_groups = [{"params": [transl, cam_height_offset, cam_init_r6d], "lr": cfg.postopt_lr}]
    if opt_drift:
        resid_scale.requires_grad = True
        floor_raise_y.requires_grad = True
        param_groups.append({"params": [resid_scale, floor_raise_y], "lr": cfg.postopt_lr_resid})
    optim = torch.optim.Adam(param_groups)

    joint_idxs = [i for i in range(25) if i not in IGNORED_KEYPOINTS]

    with torch.no_grad():
        B = num_people * seq_len
        smplx_out = smplx(
            body_pose=smplx_pose[:, :, 3:66].reshape(-1, 21*3),
            global_orient=smplx_pose[:, :, :3].reshape(-1, 3),
            betas=smplx_betas.reshape(-1, 10),
            left_hand_pose=smplx_pose[:, :, 75:120].reshape(-1, 15*3),
            right_hand_pose=smplx_pose[:, :, 120:165].reshape(-1, 15*3),
            transl=transl_init.reshape(-1, 3),
            jaw_pose=torch.zeros(B, 3).to(smplx_pose),
            leye_pose=torch.zeros(B, 3).to(smplx_pose),
            reye_pose=torch.zeros(B, 3).to(smplx_pose),
            expression=torch.zeros(B, 10).to(smplx_pose),
            pose2rot=True
        )
        joints = smplx_out.joints[:, SMPLX_TO_OPENPOSE_25]
        contact_joints = smplx_out.joints[:, FOOT_JOINT_IDS]
        joints = torch.cat([joints, contact_joints], dim=1).reshape(num_people, seq_len, -1, 3)

    with log.progress(range(cfg.training_iter), "iterations") as pbar:
        for i in pbar:
            optim.zero_grad()

            j_world = joints + transl[:, :, None]  # (P,T,J,3)
            d_total = resid_scale * world_drift[:seq_len]  # (T,3)
            d_total[:, 1] += floor_raise_y
            j_world = j_world - d_total[None, :, None, :]

            cam_init_rotmat = rotation_6d_to_matrix(cam_init_r6d)
            rcw_mod = cam_init_rotmat[None, None, :, :] @ rcw
            rwc_mod = rcw_mod.transpose(-1, -2)
            tcw_mod = (cam_init_rotmat[None, None, :, :] @ tcw[:, :, :, None])[..., 0]
            tcw_mod[:, :, 1] += cam_height_offset
            twc_mod = -(rwc_mod @ tcw_mod[..., None])[..., 0]  # (P,T,3)
            twc_mod = twc_mod - d_total[None, :, :]
            tcw_mod = -(rcw_mod @ twc_mod[..., None])[..., 0]

            # Reprojection, Xc = Rcw * Xw + Tcw
            j_cam = (rcw_mod[:, :, None] @ j_world[..., None])[..., 0] + tcw_mod[:, :, None]
            pj = (K_proj @ j_cam[..., None])[..., 0]
            pj = (pj / (pj[..., 2:3] + 1e-6))[..., :2]
            gt = pred_kp_2d[..., :2]

            loss_l1 = torch.abs(pj[:, :, joint_idxs] - gt[:, :, joint_idxs]) / bbox_height[:, :, None, None]
            loss_kp = gmof(pj[:, :, joint_idxs] - gt[:, :, joint_idxs]) / bbox_height[:, :, None, None] + loss_l1
            loss_kp = loss_kp * mask[:, :, None, None]
            loss_kp = ((pred_kp_2d[:, :, joint_idxs, 2] > 0.9) * loss_kp.mean(-1)).mean(-1).sum() / mask.sum()
            loss_kp = cfg.loss_kp_w * loss_kp

            # Smoothness: velocity and acceleration of the world translation
            f_transl = transl_init + transl - d_total[None, :, :]
            loss_vel = ((f_transl[:, 1:] - f_transl[:, :-1]) * cfg.fps).pow(2).mean(-1)
            loss_vel = (loss_vel * mask[:, 1:]).sum() / mask[:, 1:].sum()
            loss_acc = torch.linalg.norm((f_transl[:, 2:] + f_transl[:, :-2] - 2 * f_transl[:, 1:-1]) * cfg.fps, dim=-1)
            loss_acc = (loss_acc * mask[:, 1:-1]).sum() / mask[:, 1:-1].sum()
            loss_smooth = cfg.loss_smooth_w * loss_vel + cfg.loss_smooth_w * loss_acc

            loss_repul = cfg.loss_repel_w * inter_person_repulsion_temporal_loss(j_world, mask, margin=0.2, window=5)

            # Contact: planted joints don't move, and sit at target_foot_y
            contacts_conf = torch.sigmoid(pred_contact)[..., :len(FOOT_JOINT_IDS)]
            delta_pos = (j_world[:, 1:, 25:] - j_world[:, :-1, 25:])  # (P,T-1,Jc,3)
            loss_contact_vel = ((delta_pos.pow(2).sum(dim=-1) * contacts_conf[:, 1:]) * mask[:, 1:, None]).mean(-1).sum() / mask[:, 1:].sum()
            loss_contact_vel = cfg.loss_cont_vel_w * loss_contact_vel

            contact_high = (contacts_conf > cfg.contact_thr).float()
            valid_count = (contact_high[:, 1:] * mask[:, 1:, None]).sum() + 1e-8
            loss_stable_feet = (delta_pos.abs().sum(dim=-1) * contact_high[:, 1:] * mask[:, 1:, None]).sum() / valid_count
            loss_stable_feet = cfg.loss_stable_feet_w * loss_stable_feet

            floor_diff = torch.abs(j_world[:, :, 25:, 1] - cfg.target_foot_y)
            loss_contact_height = ((floor_diff * contacts_conf) * mask[:, :, None]).mean(-1).sum() / mask.sum()
            loss_contact_height = cfg.loss_cont_height_w * loss_contact_height

            # No joints below the floor
            below = F.relu(-j_world[:, :, :, 1]) * mask[:, :, None]
            loss_below_floor = cfg.loss_below_floor_w * (below.mean(-1).sum() / mask.sum())

            loss = loss_kp + loss_smooth + loss_below_floor + loss_repul
            loss += loss_contact_vel + loss_contact_height + loss_stable_feet

            if i % 100 == 0:
                pbar.set_postfix({"loss": loss.item(), "kp": loss_kp.item(), "smooth": loss_smooth.item(),
                                  "below": loss_below_floor.item(), "rep": loss_repul.item()})
            loss.backward()
            optim.step()

    with torch.no_grad():
        d_total = resid_scale * world_drift[:seq_len]
        d_total[:, 1] += floor_raise_y
        final_transl = transl_init + transl - d_total[None, :, :]

    resid = float(resid_scale.detach().cpu().item())
    total_scale = None
    if pred_cam['drift']['mode'] != 'none':
        total_scale = float((pred_cam['drift']['s_per_frame'][:seq_len] + resid_scale.detach().cpu().numpy()).mean())
    if total_scale is not None:
        log.detail(f"drift scale {total_scale:.3f} after post-optimization (residual {resid:+.3f})")
    log.detail(f"floor raised {100 * float(floor_raise_y.item()):.1f} cm, "
               f"camera height offset {100 * float(cam_height_offset.item()):+.1f} cm")

    for pidx, person_id in enumerate(people_ids):
        results['people'][person_id]['smplx_world']['trans'] = final_transl[pidx][mask[pidx].bool()].detach().cpu().numpy()

    d_total = d_total.detach().cpu().numpy()
    cam_init_rotmat = cam_init_rotmat.detach().cpu().numpy()
    Tcw = (cam_init_rotmat[None, :, :] @ pred_cam['Tcw'][..., None])[..., 0]
    Tcw[:, 1] += cam_height_offset.detach().cpu().numpy()
    Rcw = cam_init_rotmat[None, :, :] @ pred_cam['Rcw']
    Rwc = Rcw.transpose(0, 2, 1)
    Twc = -(Rwc @ Tcw[..., None])[..., 0] - d_total
    Tcw = -(Rcw @ Twc[..., None])[..., 0]

    pred_cam['Rcw'] = Rcw
    pred_cam['Tcw'] = Tcw
    pred_cam['Rwc'] = Rwc
    pred_cam['Twc'] = Twc
    pred_cam['drift']['postopt_resid_scale'] = resid
    pred_cam['drift']['postopt_floor_raise_y'] = float(floor_raise_y.detach().cpu().item())
    if total_scale is not None:
        pred_cam['drift']['postopt_total_scale'] = total_scale
    return results


def inter_person_repulsion_temporal_loss(joints_world, visibility, margin=0.01, window=10, alpha=0.5):
    """
    Spheres on torso, legs and arms of different people must not overlap.

    - Direct term: penalizes penetration magnitude, so overlaps are removed.
    - Temporal term: penalizes fast changes of penetration energy within a window, so the
      correction is spread over nearby frames.
    """
    device = joints_world.device
    dtype = joints_world.dtype

    joint_ids = torch.as_tensor((1, 2, 5, 8, 9, 12, 10, 13, 3, 6, 4, 7), device=device, dtype=torch.long)
    joint_radii = torch.as_tensor([0.05, 0.05, 0.05, 0.06, 0.05, 0.05, 0.05, 0.05, 0.035, 0.035, 0.03, 0.03],
                                  device=device, dtype=dtype).view(1, 1, -1)  # (1,1,S)

    num_people, num_frames, _, _ = joints_world.shape
    if num_people < 2 or num_frames < 2 or window <= 0:
        return torch.zeros((), device=device, dtype=dtype)

    centers = joints_world[:, :, joint_ids, :]                                     # (P,F,S,3)
    sphere_dist = torch.linalg.norm(
        centers[:, None, :, :, None, :] - centers[None, :, :, None, :, :], dim=-1).clamp(min=1e-8)  # (P,P,F,S,S)
    Rsum = (joint_radii[:, :, :, None] + joint_radii[:, :, None, :]) + margin       # (1,1,S,S)
    penetration = F.relu(Rsum - sphere_dist).pow(2)

    # person pairs i < j
    upper_tri = torch.triu(torch.ones(num_people, num_people, device=device, dtype=dtype), diagonal=1)
    penetration = penetration * upper_tri[:, :, None, None, None]
    valid_pair_frame = upper_tri[:, :, None].expand(num_people, num_people, num_frames)

    pair_frame_energy = penetration.sum(dim=(-1, -2))  # (P,P,F)

    active = ((pair_frame_energy > 0).to(dtype) * valid_pair_frame)
    loss_direct = (pair_frame_energy * active).sum() / active.sum().clamp(min=1.0)

    # Temporal coupling, gated by repulsion happening at the earlier frame
    repulsion_weight = pair_frame_energy / (pair_frame_energy + 1e-4)
    max_offset = min(int(window), num_frames - 1)
    total_weighted_loss = torch.zeros((), device=device, dtype=dtype)
    total_weight = torch.zeros((), device=device, dtype=dtype)
    for frame_offset in range(1, max_offset + 1):
        offset_weight = float(max_offset + 1 - frame_offset)  # triangular window
        energy_diff = pair_frame_energy[:, :, frame_offset:] - pair_frame_energy[:, :, :-frame_offset]
        valid_both = valid_pair_frame[:, :, frame_offset:] * valid_pair_frame[:, :, :-frame_offset]
        temporal_weight = repulsion_weight[:, :, :-frame_offset] * valid_both
        offset_loss = (energy_diff.pow(2) * temporal_weight).sum() / temporal_weight.sum().clamp(min=1.0)
        total_weighted_loss = total_weighted_loss + offset_weight * offset_loss
        total_weight = total_weight + offset_weight
    loss_temporal = total_weighted_loss / total_weight.clamp(min=1.0)

    return alpha * loss_direct + (1.0 - alpha) * loss_temporal
