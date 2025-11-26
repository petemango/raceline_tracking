import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

def solve_lqr(A, B, Q, R):
    """
    Solves the Discrete Algebraic Riccati Equation (DARE) iteratively.
    Returns the optimal gain matrix K.
    """
    dt = 0.1 # Discretization step for the solver
    
    # Discretize (Forward Euler approximation for simplicity in gain calculation)
    # Ad = I + A * dt
    # Bd = B * dt
    eye = np.eye(A.shape[0])
    Ad = eye + A * dt
    Bd = B * dt
    
    # Iteratively solve for P
    P = Q.copy()
    max_iters = 100
    epsilon = 1e-4
    
    for _ in range(max_iters):
        P_next = Ad.T @ P @ Ad - (Ad.T @ P @ Bd) @ np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad) + Q
        if np.abs(P_next - P).max() < epsilon:
            P = P_next
            break
        P = P_next
        
    # Calculate K
    # K = (R + B^T P B)^-1 * B^T P A
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    
    return K

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

def get_track_errors(car_state, path, known_idx=None):
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
    else:
        p1 = path[prev_idx]
        p2 = path[closest_idx]
        
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
    track_heading = np.arctan2(segment_vec[1], segment_vec[0])
    heading_error = car_heading - track_heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    return cte, heading_error

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # ... (rest of lower_controller same as before, included for context if needed but we can skip if not editing)
    # For safety in 'replace', I will match the block boundaries carefully.
    # Since I cannot skip middle parts easily, I will target get_track_errors -> controller
    pass

# To minimize token usage and potential errors, I will do two replacements if needed, 
# or one big one covering get_track_errors to controller.

# Let's stick to replacing get_track_errors ... controller



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

def generate_speed_profile(path, parameters):
    """
    Generates a velocity profile for the raceline.
    """
    n_points = len(path)
    profile = np.zeros(n_points)
    
    # Parameters
    max_v = parameters[5]
    max_lat_accel = 10.0 # Reduced for safety
    max_braking = np.abs(parameters[8]) # u2_min is negative
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
            
        if k < 1e-4:
            v_limit = max_v
        else:
            v_limit = np.sqrt(max_lat_accel / k)
            
        profile[i] = min(v_limit, max_v)
        
    # 2. Backward Pass (Braking Limits)
    # Ensure we can actually slow down to v_limit from the previous point
    # v_i^2 <= v_{i+1}^2 + 2 * a_brake * distance
    
    for i in range(n_points - 1, -1, -1):
        next_i = (i + 1) % n_points
        dist = np.linalg.norm(path[next_i] - path[i])
        
        # Allowed entry speed based on exit speed and braking
        allowed_entry_sq = profile[next_i]**2 + 2 * max_braking * dist
        allowed_entry = np.sqrt(allowed_entry_sq)
        
        profile[i] = min(profile[i], allowed_entry)
        
    # Run backward pass twice to handle the loop closure properly
    for i in range(n_points - 1, -1, -1):
        next_i = (i + 1) % n_points
        dist = np.linalg.norm(path[next_i] - path[i])
        allowed_entry_sq = profile[next_i]**2 + 2 * max_braking * dist
        profile[i] = min(profile[i], np.sqrt(allowed_entry_sq))

    # 3. Forward Pass (Acceleration Limits)
    # v_{i+1}^2 <= v_i^2 + 2 * a_accel * distance
    # This isn't strictly necessary for safety (the car physics handles accel limits), 
    # but helps the LQR know the *realistic* speed.
    
    for i in range(n_points):
        prev_i = (i - 1) % n_points
        dist = np.linalg.norm(path[i] - path[prev_i])
        
        allowed_exit_sq = profile[prev_i]**2 + 2 * max_accel * dist
        profile[i] = min(profile[i], np.sqrt(allowed_exit_sq))
        
    return profile

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    # state: [x, y, steer_angle, velocity, heading]
    
    if racetrack.raceline is None:
        return np.array([0.0, 0.0])

    # Generate speed profile if not exists
    if not hasattr(racetrack, 'speed_profile'):
        racetrack.speed_profile = generate_speed_profile(racetrack.raceline, parameters)
        
    # Initialize tracking state
    if not hasattr(racetrack, 'last_idx'):
        racetrack.last_idx = None
    
    path = racetrack.raceline
    profile = racetrack.speed_profile
    
    car_pos = state[0:2]
    car_vel = state[3]
    car_steer = state[2]
    wheelbase = parameters[0]
    
    # 1. Find closest point (Windowed Search)
    current_idx = get_closest_index(car_pos, path, last_idx=racetrack.last_idx, window=100)
    racetrack.last_idx = current_idx
    
    # 2. Get Errors using the KNOWN index (Fixes latching onto wrong segment)
    cte, he = get_track_errors(state, path, known_idx=current_idx)
    
    # 3. Get Target Velocity from Profile
    # Look ahead slightly for velocity to account for delay
    lookahead_idx = (current_idx + 2) % len(path)
    desired_vel = profile[lookahead_idx]
    
    # 4. LQR Formulation
    # State x = [cte, he, delta]
    # Input u = v_delta (steering rate)
    
    # Linearization Velocity
    # Use current velocity, but prevent singularity at 0
    # Higher minimum velocity (e.g. 10.0) prevents aggressive gains at start/low speed
    v = max(car_vel, 10.0) 
    
    A = np.zeros((3, 3))
    A[0, 1] = v            # dot_e = v * theta_e
    A[1, 2] = v / wheelbase # dot_theta = (v/L) * delta
    
    B = np.zeros((3, 1))
    B[2, 0] = 1            # dot_delta = u
    
    # Tunable Weights
    # Penalize Cross Track Error heavily to avoid violations
    Q = np.diag([20.0, 0.5, 0.0]) 
    R = np.array([[5.0]])
    
    K = solve_lqr(A, B, Q, R)
    
    # Control Law: u = -K * x
    x_error = np.array([[cte], [he], [car_steer]])
    u_opt = -K @ x_error
    steer_rate = u_opt[0, 0]
    
    # 5. Interface with Lower Controller
    # The lower controller is: rate = 5.0 * (desired - current)
    # So: desired = current + rate / 5.0
    Kp_steer = 5.0
    desired_steer = car_steer + steer_rate / Kp_steer
    
    # Clamp outputs to vehicle limits
    max_steer = parameters[4]
    min_vel = parameters[2]
    max_vel = parameters[5]
    
    desired_steer = np.clip(desired_steer, -max_steer, max_steer)
    desired_vel = np.clip(desired_vel, min_vel, max_vel)
    
    return np.array([desired_steer, desired_vel])