import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# Controller timestep (matches RaceCar.time_step)
CONTROL_DT = 0.1
SAFE_MAX_VEL = 100.0


def reset_lower_controller_state() -> None:
    """
    Reset internal PID state for the low-level controller.

    This should be called whenever a new simulation is started (new car/track)
    to avoid leaking integrals or previous errors across runs.
    """
    global _pid_state
    _pid_state = {
        "initialized": False,
        "steer_integral": 0.0,
        "steer_prev_error": 0.0,
        "vel_integral": 0.0,
        "vel_prev_error": 0.0,
    }


# Persistent PID state for low-level controller
_pid_state: dict = {}
reset_lower_controller_state()


def get_closest_index(point, path, last_idx=None, window=100):
    """
    Finds the closest point on the path.
    If last_idx is provided, searches locally around it.
    """
    n_points = len(path)

    if last_idx is None:
        # Global search
        dists = np.linalg.norm(path - point, axis=1)
        return np.argmin(dists)

    # Windowed search
    indices = []
    # Look slightly backward and mostly forward
    start = -int(window * 0.2)
    end = int(window)

    for i in range(start, end):
        indices.append((last_idx + i) % n_points)

    indices = np.array(indices)
    local_path = path[indices]

    dists = np.linalg.norm(local_path - point, axis=1)
    best_local_idx = np.argmin(dists)

    return indices[best_local_idx]


def compute_path_tangents(path):
    """
    Computes smooth tangents for the path using centered differences.
    """
    n_points = len(path)
    tangents = np.zeros_like(path)
    for i in range(n_points):
        prev_p = path[(i - 1) % n_points]
        next_p = path[(i + 1) % n_points]
        vec = next_p - prev_p
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            tangents[i] = vec / norm
        else:
            tangents[i] = np.array([1.0, 0.0])
    return tangents


def get_track_errors(car_state, path, known_idx=None, tangents=None):
    """
    Calculates the Cross Track Error (CTE) and Heading Error (HE).
    """
    car_pos = car_state[0:2]
    car_heading = car_state[4]

    # 1. Find the closest point
    if known_idx is not None:
        closest_idx = known_idx
    else:
        closest_idx = get_closest_index(car_pos, path)

    # 2. Identify the segment (Point A -> Point B)
    next_idx = (closest_idx + 1) % len(path)
    prev_idx = (closest_idx - 1) % len(path)

    vec_next = path[next_idx] - path[closest_idx]
    vec_car = car_pos - path[closest_idx]

    # Project car onto both segments to see which one is "active"
    if np.dot(vec_car, vec_next) >= 0:
        p1 = path[closest_idx]
        p2 = path[next_idx]
        idx1 = closest_idx
        idx2 = next_idx
    else:
        p1 = path[prev_idx]
        p2 = path[closest_idx]
        idx1 = prev_idx
        idx2 = closest_idx

    # 3. Calculate CTE
    segment_vec = p2 - p1
    segment_len = np.linalg.norm(segment_vec)
    if segment_len < 1e-6:
        return 0.0, 0.0

    segment_unit_vec = segment_vec / segment_len
    p1_to_car = car_pos - p1
    proj_len = np.dot(p1_to_car, segment_unit_vec)
    closest_point_on_line = p1 + segment_unit_vec * proj_len
    cte_vec = car_pos - closest_point_on_line
    cte = np.linalg.norm(cte_vec)

    # Sign of CTE
    cross_prod = segment_vec[0] * p1_to_car[1] - segment_vec[1] * p1_to_car[0]
    if cross_prod < 0:
        cte = -cte
    else:
        cte = cte

    # 4. Calculate Heading Error
    if tangents is not None:
        # Interpolate tangent for smoother HE
        t_factor = np.clip(proj_len / segment_len, 0.0, 1.0)
        t1 = tangents[idx1]
        t2 = tangents[idx2]
        t_interp = (1.0 - t_factor) * t1 + t_factor * t2
        track_heading = np.arctan2(t_interp[1], t_interp[0])
    else:
        track_heading = np.arctan2(segment_vec[1], segment_vec[0])

    heading_error = car_heading - track_heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    return cte, heading_error


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level PID controller.

    Tracks desired steering angle and desired velocity using two SISO PID
    controllers, and outputs steering rate and longitudinal acceleration.

    state:    [x, y, steer_angle, velocity, heading]
    desired:  [desired_steer_angle, desired_velocity]
    parameters:
        [wheelbase,
         min_steer, min_vel, min_heading,
         max_steer, max_vel, max_heading,
         min_steer_vel, min_accel,
         max_steer_vel, max_accel]
    """
    global _pid_state

    current_steer = float(state[2])
    current_vel = float(state[3])

    desired_steer = float(desired[0])
    desired_vel = float(desired[1])

    # Initialize PID state on first call
    if not _pid_state["initialized"]:
        _pid_state["initialized"] = True
        _pid_state["steer_integral"] = 0.0
        _pid_state["steer_prev_error"] = 0.0
        _pid_state["vel_integral"] = 0.0
        _pid_state["vel_prev_error"] = 0.0

    dt = CONTROL_DT

    # Limits
    min_steer_vel = float(parameters[7])
    max_steer_vel = float(parameters[9])
    min_accel = float(parameters[8])
    max_accel = float(parameters[10])

    # -------------------------
    # Steering PID (delta -> v_delta)
    # -------------------------
    e_steer = desired_steer - current_steer

    # Gains for steering angle tracking
    Kp_steer = 4.0
    Ki_steer = 0.5
    Kd_steer = 0.1

    # Integral candidate
    steer_integral_candidate = _pid_state["steer_integral"] + e_steer * dt
    steer_derivative = (e_steer - _pid_state["steer_prev_error"]) / dt

    steer_rate_unsat = (
        Kp_steer * e_steer
        + Ki_steer * steer_integral_candidate
        + Kd_steer * steer_derivative
    )

    steer_rate = float(np.clip(steer_rate_unsat, min_steer_vel, max_steer_vel))

    # Simple anti-windup: only accept integral update when not saturated
    if steer_rate == steer_rate_unsat:
        _pid_state["steer_integral"] = steer_integral_candidate

    _pid_state["steer_prev_error"] = e_steer

    # -------------------------
    # Velocity PID (v -> a)
    # -------------------------
    e_vel = desired_vel - current_vel

    # Gains for speed tracking
    Kp_vel = 1.5
    Ki_vel = 0.4
    Kd_vel = 0.05

    vel_integral_candidate = _pid_state["vel_integral"] + e_vel * dt
    vel_derivative = (e_vel - _pid_state["vel_prev_error"]) / dt

    accel_unsat = (
        Kp_vel * e_vel
        + Ki_vel * vel_integral_candidate
        + Kd_vel * vel_derivative
    )

    accel = float(np.clip(accel_unsat, min_accel, max_accel))

    if accel == accel_unsat:
        _pid_state["vel_integral"] = vel_integral_candidate

    _pid_state["vel_prev_error"] = e_vel

    return np.array([steer_rate, accel])


def generate_speed_profile(path, parameters):
    """
    Generates a velocity profile for the raceline.
    Returns: (speed_profile, curvature_profile)
    """
    n_points = len(path)
    speed_profile = np.zeros(n_points)
    curvature_profile = np.zeros(n_points)

    # Parameters
    max_v = min(parameters[5], SAFE_MAX_VEL)
    max_lat_accel = 20.0  # Increased to rely on steering rate limits
    max_braking = np.abs(parameters[8])  # u2_min is negative
    max_accel = parameters[10]

    # 1. Calculate Curvature and Max Cornering Speed
    # We use a simple geometric curvature: k = 1/R
    # circumcircle of 3 points (p_i-1, p_i, p_i+1)

    for i in range(n_points):
        p1 = path[(i - 1) % n_points]
        p2 = path[i]
        p3 = path[(i + 1) % n_points]

        # Vector 1->2 and 2->3
        v1 = p2 - p1
        v2 = p3 - p2

        # Area of triangle = 0.5 * |v1 x v2|
        area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])

        # Lengths
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        l3 = np.linalg.norm(p3 - p1)

        if area < 1e-6:
            k = 0.0
        else:
            k = (4 * area) / (l1 * l2 * l3)

        curvature_profile[i] = k

        if k < 1e-4:
            v_limit = max_v
        else:
            v_limit = np.sqrt(max_lat_accel / k)

        speed_profile[i] = min(v_limit, max_v)

    # 1.5. Steering Rate Limits (Heuristic)
    # v * |d(delta)/ds| <= max_steer_vel
    # delta approx L * k
    # |d(delta)/ds| approx L * |dk/ds|
    # v <= max_steer_vel / (L * |dk/ds|)
    
    wheelbase = parameters[0]
    max_steer_rate = parameters[9]
    
    for i in range(n_points - 1):
        p1 = path[i]
        p2 = path[i+1]
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            continue
            
        k1 = curvature_profile[i]
        k2 = curvature_profile[i+1]
        dk_ds = (k2 - k1) / dist
        
        if abs(dk_ds) > 1e-6:
            # Conservative limit: ignore 1/(1+(Lk)^2) factor
            v_steer_limit = max_steer_rate / (wheelbase * abs(dk_ds))
            # Allow some relaxation because steering controller can lead/lag?
            # No, physical limit is hard.
            speed_profile[i] = min(speed_profile[i], v_steer_limit)
            # Also limit the next point to be safe? 
            # The rate constraint applies to the segment.
            speed_profile[i+1] = min(speed_profile[i+1], v_steer_limit)

    # 2. Backward Pass (Braking Limits)
    # Ensure we can actually slow down to v_limit from the previous point
    # v_i^2 <= v_{i+1}^2 + 2 * a_brake * distance

    for i in range(n_points - 1, -1, -1):
        next_i = (i + 1) % n_points
        dist = np.linalg.norm(path[next_i] - path[i])

        # Allowed entry speed based on exit speed and braking
        allowed_entry_sq = speed_profile[next_i]**2 + 2 * max_braking * dist
        allowed_entry = np.sqrt(allowed_entry_sq)

        speed_profile[i] = min(speed_profile[i], allowed_entry)

    # Run backward pass twice to handle the loop closure properly
    for i in range(n_points - 1, -1, -1):
        next_i = (i + 1) % n_points
        dist = np.linalg.norm(path[next_i] - path[i])
        allowed_entry_sq = speed_profile[next_i]**2 + 2 * max_braking * dist
        speed_profile[i] = min(speed_profile[i], np.sqrt(allowed_entry_sq))

    # 3. Forward Pass (Acceleration Limits)
    # v_{i+1}^2 <= v_i^2 + 2 * a_accel * distance
    # This isn't strictly necessary for safety (the car physics handles accel limits),
    # but helps the LQR know the *realistic* speed.

    for i in range(n_points):
        prev_i = (i - 1) % n_points
        dist = np.linalg.norm(path[i] - path[prev_i])

        allowed_exit_sq = speed_profile[prev_i]**2 + 2 * max_accel * dist
        speed_profile[i] = min(speed_profile[i], np.sqrt(allowed_exit_sq))

    return speed_profile, curvature_profile


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    # state: [x, y, steer_angle, velocity, heading]

    if racetrack.raceline is None:
        return np.array([0.0, 0.0])

    # Generate speed profile if not exists
    if not hasattr(racetrack, 'speed_profile'):
        racetrack.speed_profile, racetrack.curvature_profile = generate_speed_profile(
            racetrack.raceline, parameters)

    # Generate tangents if not exists
    if not hasattr(racetrack, 'raceline_tangents'):
        racetrack.raceline_tangents = compute_path_tangents(racetrack.raceline)

    # Initialize tracking state
    if not hasattr(racetrack, 'last_idx'):
        racetrack.last_idx = None

    path = racetrack.raceline
    profile = racetrack.speed_profile
    curvatures = racetrack.curvature_profile
    tangents = racetrack.raceline_tangents

    car_pos = state[0:2]
    car_vel = state[3]
    car_steer = state[2]
    wheelbase = parameters[0]

    # 1. Find closest point (Windowed Search)
    current_idx = get_closest_index(
        car_pos, path, last_idx=racetrack.last_idx, window=100)
    racetrack.last_idx = current_idx

    # 2. Get Errors using the KNOWN index (Fixes latching onto wrong segment)
    cte, he = get_track_errors(state, path, known_idx=current_idx, tangents=tangents)

    # 3. Get Target Velocity from Profile
    # Dynamic lookahead for velocity to account for delay
    # Look ahead ~0.2s to 0.5s
    lookahead_time = 0.3
    lookahead_dist = max(car_vel * lookahead_time, 5.0) # Min 5m lookahead
    
    # Estimate index offset
    # Assuming roughly 5m per point (from analysis)
    avg_segment_len = 5.0
    lookahead_idx_offset = int(lookahead_dist / avg_segment_len)
    lookahead_idx = (current_idx + lookahead_idx_offset) % len(path)
    
    desired_vel = profile[lookahead_idx]

    # --- Error-aware speed governor (raceline-based) ---
    # STRICT ABSOLUTE LIMITS to ensure 0.5m wall safety.
    # The raceline is guaranteed to be >0.5m from the wall.
    # We must keep CTE well within this margin.
    
    # CTE limit thresholds (in meters)
    cte_safe = 0.15  # < 15cm error: Allow full speed
    cte_danger = 0.40 # > 40cm error: Crawl speed (danger of hitting 0.5m wall)
    
    # Heading error threshold (radians)
    # If we are pointing the wrong way, we will generate CTE fast.
    he_safe = np.deg2rad(3.0)
    he_danger = np.deg2rad(10.0)
    
    # Normalize error metrics to 0.0-1.0 range based on thresholds
    cte_ratio = max(0.0, (abs(cte) - cte_safe) / (cte_danger - cte_safe))
    he_ratio = max(0.0, (abs(he) - he_safe) / (he_danger - he_safe))
    
    # Combined penalty factor (0.0 = safe, 1.0 = danger)
    penalty = max(cte_ratio, he_ratio)
    penalty = min(1.0, penalty)

    # Map penalty to speed limit
    v_max_track = min(parameters[5], SAFE_MAX_VEL)
    v_min_safe = 5.0 # Maintenance speed to recover steering
    
    if penalty <= 0.0:
        v_error_limit = v_max_track
    elif penalty >= 1.0:
        v_error_limit = v_min_safe
    else:
        # Linear interpolation
        v_error_limit = v_max_track - penalty * (v_max_track - v_min_safe)

    desired_vel = min(desired_vel, v_error_limit)

    # -------------------------
    # Lateral steering target (delta_des) via PD on raceline error
    # -------------------------
    # Interpolate curvature for feedforward steering
    # Use simple interpolation at current projected position
    next_idx = (current_idx + 1) % len(path)
    prev_idx = (current_idx - 1) % len(path)
    vec_next = path[next_idx] - path[current_idx]
    vec_car = car_pos - path[current_idx]
    
    if np.dot(vec_car, vec_next) >= 0:
        p1_seg = path[current_idx]
        p2_seg = path[next_idx]
        idx1 = current_idx
        idx2 = next_idx
    else:
        p1_seg = path[prev_idx]
        p2_seg = path[current_idx]
        idx1 = prev_idx
        idx2 = current_idx

    seg_vec = p2_seg - p1_seg
    seg_len = np.linalg.norm(seg_vec)
    if seg_len > 1e-6:
        proj = np.dot(car_pos - p1_seg, seg_vec / seg_len)
        t = np.clip(proj / seg_len, 0.0, 1.0)
    else:
        t = 0.0

    def get_signed_curvature(idx):
        p_prev = path[(idx - 1) % len(path)]
        p_curr = path[idx]
        p_next = path[(idx + 1) % len(path)]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross_z = v1[0]*v2[1] - v1[1]*v2[0]
        k_mag = curvatures[idx]
        return k_mag * np.sign(cross_z) if abs(cross_z) > 1e-8 else 0.0

    k1 = get_signed_curvature(idx1)
    k2 = get_signed_curvature(idx2)
    k_signed = (1.0 - t) * k1 + t * k2

    # Feedforward steering from curvature
    delta_ff = np.arctan(wheelbase * k_signed)

    # Combine raceline CTE and heading error into a single normalized lateral
    # error. Normalize CTE by a reference width so that large offsets do not
    # immediately drive the steering into saturation.
    # Using a fixed reference width instead of dynamic track width for consistency.
    ref_width = 5.0 
    cte_norm_signed = cte / ref_width

    k_cte = 1.0
    k_he = 1.5
    e_lat = k_cte * cte_norm_signed + k_he * he

    # Simple PD on lateral error around the feedforward steering
    if not hasattr(racetrack, "pid_lat_prev_error"):
        racetrack.pid_lat_prev_error = 0.0

    e_lat_prev = racetrack.pid_lat_prev_error
    de_lat = (e_lat - e_lat_prev) / CONTROL_DT
    racetrack.pid_lat_prev_error = e_lat

    # Gains (negative: positive CTE/HE steers back toward centerline)
    # Gain scheduling: Reduce gains at high speeds to prevent oscillation
    # Base gains tuned for ~10-20 m/s
    base_Kp = -0.6
    base_Kd = -0.1
    
    v_clamped = max(float(car_vel), 10.0)
    scaling = 20.0 / (10.0 + v_clamped)
    
    Kp_lat = base_Kp * scaling
    Kd_lat = base_Kd * scaling

    delta_corr = Kp_lat * e_lat + Kd_lat * de_lat
    desired_steer = delta_ff + delta_corr

    # Clamp outputs to vehicle limits
    max_steer = float(parameters[4])
    min_vel = float(parameters[2])
    max_vel = float(parameters[5])

    desired_steer = float(np.clip(desired_steer, -max_steer, max_steer))
    desired_vel = float(np.clip(desired_vel, min_vel, max_vel))

    return np.array([desired_steer, desired_vel])
