import numpy as np
from numpy.typing import ArrayLike
import time

from racetrack import RaceTrack


# ============================================================================
# SE380 PROJECT - CONTROLLER IMPLEMENTATION
# ============================================================================
# 
# State: [x, y, delta, v, psi]
#   x, y   - position (m)
#   delta  - steering angle (rad)
#   v      - velocity (m/s)
#   psi    - heading angle (rad)
#   e, acceleration]
#   steering_rate - rate of change of steering angle (rad/s)
#   acceleration  - longitudinal acceleration (m/s^2)
#
# Parameters: [L, -max_steer, min_vel, -pi, max_steer, max_vel, pi, 
#              -max_steer_rate, -max_accel, max_steer_rate, max_accel]
# ============================================================================

# Logging for tuning
LOG_FILE = "controller_log.csv"
_log_initialized = False
_log_data = []
_start_time = None

# Global state for path tracking
_path_data = None
_path_curvature = None
_last_closest_idx = 0
_step_count = 0


# ============================================================================
# TUNABLE PARAMETERS - TARGET: 60-80s LAP, ZERO VIOLATIONS
# ============================================================================
TUNING_PARAMS = {
    # Velocity control
    'max_straight_velocity': 100.0,  # m/s - top speed on straights
    'min_velocity': 25.0,            # m/s - minimum corner speed
    'curvature_lookahead': 45,       # curvature window
    'max_lateral_accel': 18.0,       # m/s^2 (~1.84g)
    'grip_margin': 0.90,             # base grip margin
    
    # Pure pursuit
    'lookahead_base': 4.8,           # slightly longer
    'lookahead_gain': 0.56,          # scale with speed
    
    # Lower controller PID gains
    'kp_steer': 11.0,
    'kp_vel': 6.0,
}


def _precompute_curvature(path):
    """Precompute curvature at each point on the path."""
    n = len(path)
    curvature = np.zeros(n)
    
    for i in range(n):
        p0 = path[(i - 1) % n]
        p1 = path[i]
        p2 = path[(i + 1) % n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            curvature[i] = 0.0
            continue
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)
        angle_change = np.arctan2(cross, dot)
        ds = (len1 + len2) / 2.0
        curvature[i] = abs(angle_change) / ds
    
    return curvature


def _get_target_velocity(closest_idx, curvature, n_points, current_velocity=50.0):
    """
    Calculate target velocity based on physics with ADAPTIVE lookahead.
    
    Key insight: Only look far ahead if we need to brake a lot.
    If we're already slow enough, don't brake early!
    """
    max_straight_velocity = TUNING_PARAMS['max_straight_velocity']
    min_velocity = TUNING_PARAMS['min_velocity']
    base_lookahead = TUNING_PARAMS['curvature_lookahead']
    max_lateral_accel = TUNING_PARAMS['max_lateral_accel']
    base_grip_margin = TUNING_PARAMS['grip_margin']
    
    # ALWAYS use full lookahead for safety - no adaptive reduction
    # This ensures we always see corners early enough
    curvature_lookahead = base_lookahead
    
    # Find max curvature in lookahead window
    max_curv = 0.0
    for i in range(curvature_lookahead):
        idx = (closest_idx + i) % n_points
        max_curv = max(max_curv, curvature[idx])
    
    if max_curv < 1e-6:
        return max_straight_velocity
    
    # Adaptive grip margin - conservative but allows speed where safe
    if max_curv > 0.05:
        grip_margin = base_grip_margin * 0.60
    elif max_curv > 0.04:
        grip_margin = base_grip_margin * 0.68
    elif max_curv > 0.03:
        grip_margin = base_grip_margin * 0.78
    elif max_curv > 0.025:
        grip_margin = base_grip_margin * 0.86
    elif max_curv > 0.02:
        grip_margin = base_grip_margin * 0.94
    elif max_curv > 0.015:
        grip_margin = min(base_grip_margin * 1.00, 0.95)
    elif max_curv > 0.01:
        grip_margin = min(base_grip_margin * 1.04, 0.98)
    elif max_curv > 0.005:
        grip_margin = min(base_grip_margin * 1.07, 1.02)
    else:
        grip_margin = min(base_grip_margin * 1.10, 1.04)
    
    # Physics-based velocity limit
    v_physics = np.sqrt(max_lateral_accel / max_curv) * grip_margin
    return np.clip(v_physics, min_velocity, max_straight_velocity)


def _write_log():
    """Write log data to file."""
    global _log_data, _log_initialized
    
    mode = 'a' if _log_initialized else 'w'
    with open(LOG_FILE, mode) as f:
        if not _log_initialized:
            f.write("time,x,y,velocity,cross_track_error,desired_steer,desired_vel,curvature\n")
            _log_initialized = True
        
        for entry in _log_data:
            f.write(f"{entry['time']:.3f},{entry['x']:.2f},{entry['y']:.2f},"
                    f"{entry['velocity']:.2f},{entry['cross_track_error']:.3f},"
                    f"{entry['desired_steer']:.4f},{entry['desired_vel']:.2f},"
                    f"{entry['curvature']:.6f}\n")
    
    _log_data = []


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack, raceline: ArrayLike = None
) -> ArrayLike:
    """
    High-level controller using Pure Pursuit algorithm.
    
    Inputs:
        state: [x, y, delta, v, psi]
        parameters: vehicle parameters array
        racetrack: RaceTrack object with centerline
        raceline: Optional optimal racing line (x, y coordinates)
    
    Returns:
        desired: [desired_steering_angle, desired_velocity]
    """
    global _path_data, _path_curvature, _last_closest_idx
    global _log_initialized, _log_data, _start_time, _step_count
    
    # Initialize path data on first call - use raceline if available, else centerline
    if _path_data is None:
        if raceline is not None:
            _path_data = raceline[:, :2]  # Use optimal raceline
            print("Using OPTIMAL RACELINE for path tracking")
        else:
            _path_data = racetrack.centerline  # Fallback to centerline
            print("Using centerline for path tracking")
        _path_curvature = _precompute_curvature(_path_data)
    
    x, y, delta, v, psi = state
    n_points = len(_path_data)
    L = parameters[0]  # wheelbase
    max_steer = parameters[4]
    
    # Velocity-adaptive lookahead
    lookahead_base = TUNING_PARAMS['lookahead_base']
    lookahead_gain = TUNING_PARAMS['lookahead_gain']
    current_speed = max(abs(v), 1.0)
    lookahead_dist = lookahead_base + lookahead_gain * current_speed
    
    # Find closest point on path (global search)
    car_pos = np.array([x, y])
    dists = np.linalg.norm(_path_data - car_pos, axis=1)
    closest_idx = np.argmin(dists)
    
    # Prevent backwards jumps on track
    idx_diff = closest_idx - _last_closest_idx
    if idx_diff < -10 and idx_diff > -n_points + 100:
        closest_idx = _last_closest_idx
    _last_closest_idx = closest_idx
    
    # Find lookahead point by accumulating distance along path
    lookahead_idx = closest_idx
    accumulated_dist = 0.0
    
    for i in range(1, min(200, n_points // 2)):
        idx = (closest_idx + i) % n_points
        prev_idx = (closest_idx + i - 1) % n_points
        segment_dist = np.linalg.norm(_path_data[idx] - _path_data[prev_idx])
        accumulated_dist += segment_dist
        if accumulated_dist >= lookahead_dist:
            lookahead_idx = idx
            break
    
    if lookahead_idx == closest_idx:
        lookahead_idx = (closest_idx + 10) % n_points
    
    target = _path_data[lookahead_idx]
    
    # Pure Pursuit geometry
    dx = target[0] - x
    dy = target[1] - y
    angle_to_target = np.arctan2(dy, dx)
    alpha = angle_to_target - psi
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # normalize
    
    lookahead_actual = max(np.linalg.norm([dx, dy]), 0.1)
    desired_steer = np.arctan(2 * L * np.sin(alpha) / lookahead_actual)
    desired_steer = np.clip(desired_steer, -max_steer, max_steer)
    
    # Get target velocity based on curvature (pass current velocity for adaptive lookahead)
    desired_vel = _get_target_velocity(closest_idx, _path_curvature, n_points, current_speed)
    
    # Calculate cross-track error (distance from path)
    cross_track_error = dists[closest_idx]
    
    # Logging
    _step_count += 1
    
    if _start_time is None:
        _start_time = time.time()
    
    # Log every 5 steps to reduce overhead
    if _step_count % 5 == 0:
        elapsed = time.time() - _start_time
        _log_data.append({
            'time': elapsed,
            'x': x, 'y': y,
            'velocity': v,
            'cross_track_error': cross_track_error,
            'desired_steer': desired_steer,
            'desired_vel': desired_vel,
            'curvature': _path_curvature[closest_idx] if _path_curvature is not None else 0
        })
        
        # Write to file every 50 entries
        if len(_log_data) >= 50:
            _write_log()
    
    return np.array([desired_steer, desired_vel])


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Lower-level PID controller that converts desired [steer_angle, velocity]
    to control inputs [steering_rate, acceleration].
    
    Inputs:
        state: [x, y, delta, v, psi]
        desired: [desired_steering_angle, desired_velocity]
        parameters: vehicle parameters array
    
    Returns:
        control: [steering_rate, acceleration]
    """
    assert desired.shape == (2,)
    
    # Extract limits from parameters
    max_steer_rate = parameters[9]
    max_accel = parameters[10]
    
    # Current state
    current_steer = state[2]
    current_vel = state[3]
    
    # Errors
    steer_error = desired[0] - current_steer
    vel_error = desired[1] - current_vel
    
    # PID gains
    kp_steer = TUNING_PARAMS['kp_steer']
    kp_vel = TUNING_PARAMS['kp_vel']
    
    # Compute control outputs with saturation
    steering_rate = np.clip(kp_steer * steer_error, -max_steer_rate, max_steer_rate)
    acceleration = np.clip(kp_vel * vel_error, -max_accel, max_accel)
    
    return np.array([steering_rate, acceleration])
