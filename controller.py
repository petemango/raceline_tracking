import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

def get_closest_index(point, path):
    dists = np.linalg.norm(path - point, axis=1)
    return np.argmin(dists)

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # state: [x, y, steer_angle, velocity, heading]
    # desired: [desired_steer_angle, desired_velocity]
    # parameters: [wheelbase, min_steer, min_vel, min_heading, max_steer, max_vel, max_heading, min_steer_vel, min_accel, max_steer_vel, max_accel]

    current_steer = state[2]
    current_vel = state[3]
    
    desired_steer = desired[0]
    desired_vel = desired[1]

    # Gains
    Kp_vel = 2.0
    Kp_steer = 5.0

    # Longitudinal Control
    accel = Kp_vel * (desired_vel - current_vel)
    
    # Lateral Control (Steering Rate)
    steer_vel = Kp_steer * (desired_steer - current_steer)

    # Limits
    min_steer_vel = parameters[7]
    max_steer_vel = parameters[9]
    min_accel = parameters[8]
    max_accel = parameters[10]

    accel = np.clip(accel, min_accel, max_accel)
    steer_vel = np.clip(steer_vel, min_steer_vel, max_steer_vel)

    return np.array([steer_vel, accel])

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    # state: [x, y, steer_angle, velocity, heading]
    
    if racetrack.raceline is None:
        # Fallback if no raceline (should not happen based on setup)
        return np.array([0.0, 0.0])

    path = racetrack.raceline
    car_pos = state[0:2]
    car_heading = state[4]
    car_vel = state[3]
    wheelbase = parameters[0]

    # 1. Find closest point
    closest_idx = get_closest_index(car_pos, path)

    # 2. Pure Pursuit Lookahead
    # Lookahead distance. Dynamic: Ld = K * v + Ld_min
    k_lookahead = 0.5
    ld_min = 5.0
    lookahead_dist = k_lookahead * car_vel + ld_min

    # Find lookahead point
    # We search forward from the closest point until we find a point distance > lookahead_dist
    # We need to handle the loop nature of the track.
    
    n_points = len(path)
    target_point = path[closest_idx] # Fallback
    
    for i in range(n_points):
        idx = (closest_idx + i) % n_points
        dist = np.linalg.norm(path[idx] - car_pos)
        if dist > lookahead_dist:
            target_point = path[idx]
            break
    
    # 3. Calculate Desired Steering
    # Vector to target
    to_target = target_point - car_pos
    # Angle of target vector
    target_angle = np.arctan2(to_target[1], to_target[0])
    
    # Alpha: Angle between vehicle heading and target vector
    alpha = target_angle - car_heading
    
    # Normalize alpha to [-pi, pi]
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    # Pure Pursuit Control Law
    # delta = arctan(2 * L * sin(alpha) / Ld)
    # We use the actual distance to the target point as Ld in the denominator for stability
    actual_lookahead_dist = np.linalg.norm(to_target)
    if actual_lookahead_dist < 0.1:
        actual_lookahead_dist = 0.1
        
    desired_steer = np.arctan2(2 * wheelbase * np.sin(alpha), actual_lookahead_dist)

    # 4. Desired Velocity
    # Adaptive velocity based on steering angle (curvature)
    MAX_LAT_ACCEL = 15.0 # m/s^2
    max_velocity = parameters[5]
    
    curvature = np.abs(np.tan(desired_steer) / wheelbase)
    
    if curvature < 1e-3:
        desired_vel = max_velocity
    else:
        desired_vel = np.sqrt(MAX_LAT_ACCEL / curvature)
        
    # Clamp velocity
    desired_vel = np.clip(desired_vel, 0.0, max_velocity)

    return np.array([desired_steer, desired_vel])