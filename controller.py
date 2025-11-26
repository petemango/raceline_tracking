import numpy as np
from numpy.typing import ArrayLike

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
#
# Control Input: [steering_rate, acceleration]
#   steering_rate - rate of change of steering angle (rad/s)
#   acceleration  - longitudinal acceleration (m/s^2)
#
# Parameters: [L, -max_steer, min_vel, -pi, max_steer, max_vel, pi, 
#              -max_steer_rate, -max_accel, max_steer_rate, max_accel]
# ============================================================================


# Global state for path tracking
_path_data = None
_path_curvature = None
_last_closest_idx = 0


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


def _get_target_velocity(closest_idx, curvature, n_points):
    """
    Calculate target velocity based on physics.
    v_max = sqrt(a_lateral_max / curvature)
    """
    # Tunable parameters
    max_straight_velocity = 95.0   # m/s - top speed on straights
    min_velocity = 25.0            # m/s - minimum corner speed
    curvature_lookahead = 35       # points to look ahead
    max_lateral_accel = 17.0       # m/s^2 (~1.7g)
    grip_margin = 0.88             # use 88% of theoretical grip
    
    # Find max curvature ahead
    max_curv = 0.0
    for i in range(curvature_lookahead):
        idx = (closest_idx + i) % n_points
        max_curv = max(max_curv, curvature[idx])
    
    if max_curv < 1e-6:
        return max_straight_velocity
    
    # Physics-based velocity limit
    v_physics = np.sqrt(max_lateral_accel / max_curv) * grip_margin
    return np.clip(v_physics, min_velocity, max_straight_velocity)


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller using Pure Pursuit algorithm.
    
    Inputs:
        state: [x, y, delta, v, psi]
        parameters: vehicle parameters array
        racetrack: RaceTrack object with centerline
    
    Returns:
        desired: [desired_steering_angle, desired_velocity]
    """
    global _path_data, _path_curvature, _last_closest_idx
    
    # Initialize path data on first call
    if _path_data is None:
        _path_data = racetrack.centerline
        _path_curvature = _precompute_curvature(_path_data)
    
    x, y, delta, v, psi = state
    n_points = len(_path_data)
    L = parameters[0]  # wheelbase
    max_steer = parameters[4]
    
    # Velocity-adaptive lookahead
    lookahead_base = 6.0
    lookahead_gain = 0.8
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
    
    # Get target velocity based on curvature
    desired_vel = _get_target_velocity(closest_idx, _path_curvature, n_points)
    
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
    
    # PID gains (tuned for good response)
    kp_steer = 6.0
    kp_vel = 5.0
    
    # Compute control outputs with saturation
    steering_rate = np.clip(kp_steer * steer_error, -max_steer_rate, max_steer_rate)
    acceleration = np.clip(kp_vel * vel_error, -max_accel, max_accel)
    
    return np.array([steering_rate, acceleration])
