"""Step 8: every person's world pose on every frame, filled in between and beyond detections."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp


def pose_interpolation(results):
    """
    Fill every person's world pose over the whole video: linear translation and slerped
    rotations between detections, damped extrapolation before the first and after the last.
    Contact logits are interpolated likewise and pushed toward contact (logit 1.0) more than
    15 frames from a detection. Sets `frames` to every frame and `gaps` to the filled ones.
    """
    max_frame = len(results['camera']['pred_cam_R'])
    full_range = np.arange(max_frame, dtype=np.int32)

    for person in results["people"].values():
        frames = person["frames"]
        if len(frames) < 1:
            continue
        internal_range = np.arange(frames[0], frames[-1] + 1, dtype=np.int32)
        gap_mask = ~np.isin(full_range, frames)

        def interp_and_pad(data, method='linear'):
            if len(frames) == 1:
                return np.broadcast_to(data[0], (len(full_range), *data.shape[1:])).copy()

            out = np.empty((max_frame, *data.shape[1:]), dtype=np.float32)
            if method == 'slerp':
                num_joints = data.shape[1] // 3
                pbj = data.reshape(-1, num_joints, 3)
                interp = np.empty((len(internal_range), data.shape[1]), dtype=np.float32)
                for j in range(num_joints):
                    interp[:, 3*j:3*(j+1)] = Slerp(frames, R.from_rotvec(pbj[:, j]))(internal_range).as_rotvec()
                out[:internal_range[0]] = interp[0]
                out[internal_range[-1]+1:] = interp[-1]
                out[internal_range] = interp
                return out

            interp = interp1d(frames, data, axis=0, kind=method, bounds_error=False)(internal_range)
            out[internal_range] = interp

            # extrapolate with a damped velocity at both ends
            alpha = 0.8
            window = min(10, len(frames) - 1)
            if internal_range[0] > 0:
                v_front = (data[0] - data[window]) / window
                curr_p = interp[0].copy()
                for t in range(internal_range[0] - 1, -1, -1):
                    curr_p += v_front
                    out[t] = curr_p
                    v_front *= alpha
            if internal_range[-1] < max_frame - 1:
                v_back = (data[-1] - data[-window-1]) / window
                curr_p = interp[-1].copy()
                for t in range(internal_range[-1] + 1, max_frame):
                    curr_p += v_back
                    out[t] = curr_p
                    v_back *= alpha
            return out

        person["smplx_world"]["trans"] = interp_and_pad(person["smplx_world"]["trans"])
        person["smplx_world"]["pose"] = interp_and_pad(person["smplx_world"]["pose"], method='slerp')
        person["smplx_world"]["shape"] = np.broadcast_to(person["smplx_world"]["shape"][0], (len(full_range), 10)).copy()

        logits = interp_and_pad(person["smplx_cam"]["static_conf_logits"])
        dist = np.min(np.abs(np.arange(len(full_range))[:, None] - np.where(~gap_mask)[0]), axis=1)
        weight = np.clip((dist - 15) / 10, 0, 1)[:, np.newaxis]
        person["smplx_cam"]["static_conf_logits"] = (1 - weight) * logits + weight * 1.0

        person["frames"] = full_range
        person["gaps"] = gap_mask

    return results
