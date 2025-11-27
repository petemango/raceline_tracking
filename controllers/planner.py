import numpy as np


def generate_speed_profile(path, parameters, safe_max_vel):
    n_points = len(path)
    speed_profile = np.zeros(n_points)
    curvature_profile = np.zeros(n_points)

    max_v = min(parameters[5], safe_max_vel)
    max_lat_accel = 20.0
    max_braking = np.abs(parameters[8])
    max_accel = parameters[10]

    # 1. Curvature and Max Cornering Speed
    for i in range(n_points):
        p1 = path[(i - 1) % n_points]
        p2 = path[i]
        p3 = path[(i + 1) % n_points]

        v1 = p2 - p1
        v2 = p3 - p2
        area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])

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

    # 2. Steering Rate Limits
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
            v_steer_limit = max_steer_rate / (wheelbase * abs(dk_ds))
            speed_profile[i] = min(speed_profile[i], v_steer_limit)
            speed_profile[i+1] = min(speed_profile[i+1], v_steer_limit)

    # 3. Backward Pass (Braking)
    for i in range(n_points - 1, -1, -1):
        next_i = (i + 1) % n_points
        dist = np.linalg.norm(path[next_i] - path[i])
        allowed_entry_sq = speed_profile[next_i]**2 + 2 * max_braking * dist
        speed_profile[i] = min(speed_profile[i], np.sqrt(allowed_entry_sq))

    # Run backward pass twice for loop closure
    for i in range(n_points - 1, -1, -1):
        next_i = (i + 1) % n_points
        dist = np.linalg.norm(path[next_i] - path[i])
        allowed_entry_sq = speed_profile[next_i]**2 + 2 * max_braking * dist
        speed_profile[i] = min(speed_profile[i], np.sqrt(allowed_entry_sq))

    # 4. Forward Pass (Accel)
    for i in range(n_points):
        prev_i = (i - 1) % n_points
        dist = np.linalg.norm(path[i] - path[prev_i])
        allowed_exit_sq = speed_profile[prev_i]**2 + 2 * max_accel * dist
        speed_profile[i] = min(speed_profile[i], np.sqrt(allowed_exit_sq))

    return speed_profile, curvature_profile
