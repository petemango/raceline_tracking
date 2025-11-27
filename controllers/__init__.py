import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack
from .utils import get_closest_index, get_track_errors, compute_path_tangents
from .planner import generate_speed_profile
from .pid import lower_controller, reset_lower_controller_state

CONTROL_DT = 0.1
SAFE_MAX_VEL = 100.0


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    if racetrack.raceline is None:
        return np.array([0.0, 0.0])

    # Initialize persistent state on the racetrack object
    if not hasattr(racetrack, 'speed_profile'):
        racetrack.speed_profile, racetrack.curvature_profile = generate_speed_profile(
            racetrack.raceline, parameters, SAFE_MAX_VEL)

    if not hasattr(racetrack, 'raceline_tangents'):
        racetrack.raceline_tangents = compute_path_tangents(racetrack.raceline)

    if not hasattr(racetrack, 'last_idx'):
        racetrack.last_idx = None

    path = racetrack.raceline
    profile = racetrack.speed_profile
    curvatures = racetrack.curvature_profile
    tangents = racetrack.raceline_tangents

    car_pos = state[0:2]
    car_vel = float(state[3])
    wheelbase = parameters[0]

    # 1. Tracking
    current_idx = get_closest_index(
        car_pos, path, last_idx=racetrack.last_idx, window=100)
    racetrack.last_idx = current_idx

    cte, he = get_track_errors(
        state, path, known_idx=current_idx, tangents=tangents)

    # 2. Target Velocity (Dynamic Lookahead)
    lookahead_time = 1.0
    lookahead_dist = max(car_vel * lookahead_time, 5.0)

    avg_segment_len = 5.0
    lookahead_idx_offset = int(lookahead_dist / avg_segment_len)
    lookahead_idx = (current_idx + lookahead_idx_offset) % len(path)

    desired_vel = profile[lookahead_idx]

    # 3. Error-Aware Speed Governor
    # Absolute limits: 15cm safe, 40cm danger
    cte_safe = 0.15
    cte_danger = 0.40

    he_safe = np.deg2rad(3.0)
    he_danger = np.deg2rad(10.0)

    cte_ratio = max(0.0, (abs(cte) - cte_safe) / (cte_danger - cte_safe))
    he_ratio = max(0.0, (abs(he) - he_safe) / (he_danger - he_safe))

    penalty = min(1.0, max(cte_ratio, he_ratio))

    v_max_track = min(parameters[5], SAFE_MAX_VEL)
    v_min_safe = 5.0

    if penalty <= 0.0:
        v_error_limit = v_max_track
    elif penalty >= 1.0:
        v_error_limit = v_min_safe
    else:
        v_error_limit = v_max_track - penalty * (v_max_track - v_min_safe)

    desired_vel = min(desired_vel, v_error_limit)

    # 4. Lateral Control (Feedforward + Feedback)
    # Interpolate curvature
    next_idx = (current_idx + 1) % len(path)
    prev_idx = (current_idx - 1) % len(path)
    vec_next = path[next_idx] - path[current_idx]
    vec_car = car_pos - path[current_idx]

    if np.dot(vec_car, vec_next) >= 0:
        p1 = path[current_idx]
        p2 = path[next_idx]
        idx1 = current_idx
        idx2 = next_idx
    else:
        p1 = path[prev_idx]
        p2 = path[current_idx]
        idx1 = prev_idx
        idx2 = current_idx

    seg_vec = p2 - p1
    seg_len = np.linalg.norm(seg_vec)
    t = 0.0
    if seg_len > 1e-6:
        proj = np.dot(car_pos - p1, seg_vec / seg_len)
        t = np.clip(proj / seg_len, 0.0, 1.0)

    def get_k(idx):
        p_prev = path[(idx - 1) % len(path)]
        p_curr = path[idx]
        p_next = path[(idx + 1) % len(path)]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross_z = v1[0]*v2[1] - v1[1]*v2[0]
        k = curvatures[idx]
        return k * np.sign(cross_z) if abs(cross_z) > 1e-8 else 0.0

    k_signed = (1.0 - t) * get_k(idx1) + t * get_k(idx2)
    delta_ff = np.arctan(wheelbase * k_signed)

    # Feedback (Gain Scheduling)
    ref_width = 5.0
    cte_norm_signed = cte / ref_width
    e_lat = 1.0 * cte_norm_signed + 1.5 * he

    if not hasattr(racetrack, "pid_lat_prev_error"):
        racetrack.pid_lat_prev_error = 0.0

    e_lat_prev = racetrack.pid_lat_prev_error
    de_lat = (e_lat - e_lat_prev) / CONTROL_DT
    racetrack.pid_lat_prev_error = e_lat

    base_Kp, base_Kd = -0.6, -0.1
    v_clamped = max(float(car_vel), 10.0)
    scaling = 20.0 / (10.0 + v_clamped)

    Kp_lat = base_Kp * scaling
    Kd_lat = base_Kd * scaling

    delta_corr = Kp_lat * e_lat + Kd_lat * de_lat
    desired_steer = delta_ff + delta_corr

    # Output Clamping
    max_steer = float(parameters[4])
    min_vel = float(parameters[2])
    max_vel = float(parameters[5])

    desired_steer = float(np.clip(desired_steer, -max_steer, max_steer))
    desired_vel = float(np.clip(desired_vel, min_vel, max_vel))

    return np.array([desired_steer, desired_vel])
