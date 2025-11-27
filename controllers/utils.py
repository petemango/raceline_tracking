import numpy as np


def get_closest_index(point, path, last_idx=None, window=100):
    n_points = len(path)

    if last_idx is None:
        dists = np.linalg.norm(path - point, axis=1)
        return np.argmin(dists)

    indices = []
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
    car_pos = car_state[0:2]
    car_heading = car_state[4]

    if known_idx is not None:
        closest_idx = known_idx
    else:
        closest_idx = get_closest_index(car_pos, path)

    next_idx = (closest_idx + 1) % len(path)
    prev_idx = (closest_idx - 1) % len(path)

    vec_next = path[next_idx] - path[closest_idx]
    vec_car = car_pos - path[closest_idx]

    # Determine active segment
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

    segment_vec = p2 - p1
    segment_len = np.linalg.norm(segment_vec)

    if segment_len < 1e-6:
        return 0.0, 0.0

    segment_unit_vec = segment_vec / segment_len
    p1_to_car = car_pos - p1
    proj_len = np.dot(p1_to_car, segment_unit_vec)
    closest_point_on_line = p1 + segment_unit_vec * proj_len

    # Cross Track Error
    cte_vec = car_pos - closest_point_on_line
    cte = np.linalg.norm(cte_vec)

    cross_prod = segment_vec[0] * p1_to_car[1] - segment_vec[1] * p1_to_car[0]
    if cross_prod < 0:
        cte = -cte

    # Heading Error
    if tangents is not None:
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
