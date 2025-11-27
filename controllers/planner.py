import numpy as np


def generate_speed_profile(path, parameters, safe_max_velocity):
    num_points = len(path)
    velocity_profile = np.zeros(num_points)
    curvature_profile = np.zeros(num_points)

    max_velocity = min(parameters[5], safe_max_velocity)
    max_lateral_acceleration = 20.0
    max_braking = np.abs(parameters[8])
    max_acceleration = parameters[10]

    # Calculate curvature-limited speed
    for i in range(num_points):
        point_prev = path[(i - 1) % num_points]
        point_curr = path[i]
        point_next = path[(i + 1) % num_points]

        vector_1 = point_curr - point_prev
        vector_2 = point_next - point_curr

        # Area of triangle formed by the three points
        triangle_area = 0.5 * np.abs(
            vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]
        )

        length_1 = np.linalg.norm(vector_1)
        length_2 = np.linalg.norm(vector_2)
        length_3 = np.linalg.norm(point_next - point_prev)

        # Curvature k = 1/R = 4*Area / (a*b*c)
        curvature = (
            (4 * triangle_area) / (length_1 * length_2 * length_3)
            if triangle_area > 1e-6
            else 0.0
        )
        curvature_profile[i] = curvature

        velocity_limit = (
            np.sqrt(max_lateral_acceleration / curvature)
            if curvature > 1e-4
            else max_velocity
        )
        velocity_profile[i] = min(velocity_limit, max_velocity)

    # Apply steering rate limits
    wheelbase = parameters[0]
    max_steering_rate = parameters[9]

    for i in range(num_points - 1):
        segment_distance = np.linalg.norm(path[i + 1] - path[i])
        if segment_distance < 1e-6:
            continue

        change_in_curvature = (
            abs(curvature_profile[i + 1] - curvature_profile[i]) / segment_distance
        )

        if change_in_curvature > 1e-6:
            steering_velocity_limit = max_steering_rate / (
                wheelbase * change_in_curvature
            )
            velocity_profile[i] = min(velocity_profile[i], steering_velocity_limit)
            velocity_profile[i + 1] = min(
                velocity_profile[i + 1], steering_velocity_limit
            )

    # Backward pass (braking constraints)
    # Ensure we can brake in time for upcoming lower speed limits
    for _ in range(2):
        for i in range(num_points - 1, -1, -1):
            next_index = (i + 1) % num_points
            segment_distance = np.linalg.norm(path[next_index] - path[i])

            # v_current^2 <= v_next^2 + 2 * a_brake * d
            allowed_velocity = np.sqrt(
                velocity_profile[next_index] ** 2 + 2 * max_braking * segment_distance
            )
            velocity_profile[i] = min(velocity_profile[i], allowed_velocity)

    # Forward pass (acceleration constraints)
    # Ensure the car can actually reach the target speed given acceleration limits
    for i in range(num_points):
        previous_index = (i - 1) % num_points
        segment_distance = np.linalg.norm(path[i] - path[previous_index])

        # v_current^2 <= v_prev^2 + 2 * a_accel * d
        allowed_velocity = np.sqrt(
            velocity_profile[previous_index] ** 2
            + 2 * max_acceleration * segment_distance
        )
        velocity_profile[i] = min(velocity_profile[i], allowed_velocity)

    return velocity_profile, curvature_profile
