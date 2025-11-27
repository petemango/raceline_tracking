import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack
from .utils import get_closest_index, get_track_errors, compute_path_tangents
from .planner import generate_speed_profile
from .pid import lower_controller, reset_lower_controller_state

CONTROL_TIMESTEP = 0.1
SAFE_MAX_VELOCITY = 100.0

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    if racetrack.raceline is None:
        return np.array([0.0, 0.0])

    # Initialize static data on the racetrack object if not present
    if not hasattr(racetrack, 'speed_profile'):
        racetrack.speed_profile, racetrack.curvature_profile = generate_speed_profile(
            racetrack.raceline, parameters, SAFE_MAX_VELOCITY)

    if not hasattr(racetrack, 'tangents'):
        racetrack.tangents = compute_path_tangents(racetrack.raceline)

    if not hasattr(racetrack, 'last_idx'):
        racetrack.last_idx = None

    path = racetrack.raceline
    velocity_profile = racetrack.speed_profile
    curvature_profile = racetrack.curvature_profile
    tangents = racetrack.tangents

    car_position = state[0:2]
    car_velocity = float(state[3])
    wheelbase = parameters[0]

    # 1. Tracking: Find position on path and errors
    current_index = get_closest_index(car_position, path, last_index=racetrack.last_idx, search_window=100)
    racetrack.last_idx = current_index

    cross_track_error, heading_error = get_track_errors(state, path, known_index=current_index, tangents=tangents)

    # 2. Target Velocity Control with Dynamic Lookahead
    lookahead_distance = max(car_velocity * 1.0, 5.0)
    index_offset = int(lookahead_distance / 5.0)
    target_index = (current_index + index_offset) % len(path)
    
    desired_velocity = velocity_profile[target_index]

    # 3. Error-Aware Speed Governor (Safety)
    # If errors are high, reduce speed to regain control
    cte_safe_threshold, cte_danger_threshold = 0.15, 0.40
    he_safe_threshold, he_danger_threshold = np.deg2rad(3.0), np.deg2rad(10.0)

    normalized_cte = max(0.0, (abs(cross_track_error) - cte_safe_threshold) / (cte_danger_threshold - cte_safe_threshold))
    normalized_he = max(0.0, (abs(heading_error) - he_safe_threshold) / (he_danger_threshold - he_safe_threshold))
    safety_penalty = min(1.0, max(normalized_cte, normalized_he))

    if safety_penalty > 0:
        safe_recovery_velocity = 5.0
        velocity_limit = min(parameters[5], SAFE_MAX_VELOCITY)
        governed_velocity = velocity_limit - safety_penalty * (velocity_limit - safe_recovery_velocity)
        desired_velocity = min(desired_velocity, governed_velocity)

    # 4. Lateral Control
    # Feedforward steering based on curvature
    next_index = (current_index + 1) % len(path)
    previous_index = (current_index - 1) % len(path)
    
    vector_to_next = path[next_index] - path[current_index]
    vector_to_car = car_position - path[current_index]

    if np.dot(vector_to_car, vector_to_next) >= 0:
        point_1, point_2 = path[current_index], path[next_index]
        index_1, index_2 = current_index, next_index
    else:
        point_1, point_2 = path[previous_index], path[current_index]
        index_1, index_2 = previous_index, current_index

    segment_vector = point_2 - point_1
    segment_distance = np.linalg.norm(segment_vector)
    interpolation_factor = 0.0
    if segment_distance > 1e-6:
        interpolation_factor = np.clip(np.dot(car_position - point_1, segment_vector / segment_distance) / segment_distance, 0.0, 1.0)

    def get_signed_curvature(idx):
        point_prev = path[(idx - 1) % len(path)]
        point_curr = path[idx]
        point_next = path[(idx + 1) % len(path)]
        vec_1 = point_curr - point_prev
        vec_2 = point_next - point_curr
        cross_z = vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]
        return curvature_profile[idx] * np.sign(cross_z) if abs(cross_z) > 1e-8 else 0.0

    interpolated_curvature = (1.0 - interpolation_factor) * get_signed_curvature(index_1) + interpolation_factor * get_signed_curvature(index_2)
    feedforward_steering = np.arctan(wheelbase * interpolated_curvature)

    # Feedback Control (PD) with Gain Scheduling
    if not hasattr(racetrack, "previous_lateral_error"):
        racetrack.previous_lateral_error = 0.0

    reference_width = 5.0
    normalized_lateral_error = (cross_track_error / reference_width) + 1.5 * heading_error
    derivative_lateral_error = (normalized_lateral_error - racetrack.previous_lateral_error) / CONTROL_TIMESTEP
    racetrack.previous_lateral_error = normalized_lateral_error

    # Reduce gains at high speeds to prevent oscillation
    velocity_clamped = max(car_velocity, 10.0)
    gain_scale = 20.0 / (10.0 + velocity_clamped)
    
    base_kp, base_kd = -0.6, -0.1
    feedback_steering = (base_kp * gain_scale * normalized_lateral_error) + (base_kd * gain_scale * derivative_lateral_error)
    
    desired_steering = feedforward_steering + feedback_steering

    # Clamp outputs to vehicle limits
    max_steering_angle = float(parameters[4])
    desired_steering = float(np.clip(desired_steering, -max_steering_angle, max_steering_angle))
    desired_velocity = float(np.clip(desired_velocity, float(parameters[2]), float(parameters[5])))

    return np.array([desired_steering, desired_velocity])
