import numpy as np

def get_closest_index(point, path, last_index=None, search_window=100):
    num_points = len(path)

    if last_index is None:
        distances = np.linalg.norm(path - point, axis=1)
        return np.argmin(distances)

    indices = []
    start_offset = -int(search_window * 0.2)
    end_offset = int(search_window)

    for i in range(start_offset, end_offset):
        indices.append((last_index + i) % num_points)

    indices = np.array(indices)
    local_path = path[indices]
    
    distances = np.linalg.norm(local_path - point, axis=1)
    best_local_index = np.argmin(distances)
    
    return indices[best_local_index]

def compute_path_tangents(path):
    num_points = len(path)
    tangents = np.zeros_like(path)
    for i in range(num_points):
        previous_point = path[(i - 1) % num_points]
        next_point = path[(i + 1) % num_points]
        
        tangent_vector = next_point - previous_point
        vector_length = np.linalg.norm(tangent_vector)
        
        if vector_length > 1e-6:
            tangents[i] = tangent_vector / vector_length
        else:
            tangents[i] = np.array([1.0, 0.0])
    return tangents

def get_track_errors(car_state, path, known_index=None, tangents=None):
    car_position = car_state[0:2]
    car_heading = car_state[4]
    num_points = len(path)

    current_index = known_index if known_index is not None else get_closest_index(car_position, path)
    
    next_index = (current_index + 1) % num_points
    previous_index = (current_index - 1) % num_points

    vector_to_next = path[next_index] - path[current_index]
    vector_to_car = car_position - path[current_index]

    if np.dot(vector_to_car, vector_to_next) >= 0:
        point_1 = path[current_index]
        point_2 = path[next_index]
        index_1 = current_index
        index_2 = next_index
    else:
        point_1 = path[previous_index]
        point_2 = path[current_index]
        index_1 = previous_index
        index_2 = current_index

    segment_vector = point_2 - point_1
    segment_length = np.linalg.norm(segment_vector)

    if segment_length < 1e-6:
        return 0.0, 0.0

    segment_unit_vector = segment_vector / segment_length
    vector_p1_to_car = car_position - point_1
    projection_length = np.dot(vector_p1_to_car, segment_unit_vector)
    closest_point_on_line = point_1 + segment_unit_vector * projection_length
    
    cross_track_error = np.linalg.norm(car_position - closest_point_on_line)
    
    # Check which side of the line we are on
    cross_product_z = segment_vector[0] * vector_p1_to_car[1] - segment_vector[1] * vector_p1_to_car[0]
    if cross_product_z < 0:
        cross_track_error = -cross_track_error

    if tangents is not None:
        interpolation_factor = np.clip(projection_length / segment_length, 0.0, 1.0)
        interpolated_tangent = (1.0 - interpolation_factor) * tangents[index_1] + interpolation_factor * tangents[index_2]
        target_heading = np.arctan2(interpolated_tangent[1], interpolated_tangent[0])
    else:
        target_heading = np.arctan2(segment_vector[1], segment_vector[0])

    heading_error = car_heading - target_heading
    # Normalize angle to -pi to pi
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    return cross_track_error, heading_error
