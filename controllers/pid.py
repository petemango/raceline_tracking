import numpy as np
from numpy.typing import ArrayLike

_pid_state = {}


def reset_lower_controller_state() -> None:
    global _pid_state
    _pid_state = {
        "initialized": False,
        "integral_steering_error": 0.0,
        "previous_steering_error": 0.0,
        "integral_velocity_error": 0.0,
        "previous_velocity_error": 0.0,
    }


reset_lower_controller_state()


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    global _pid_state
    delta_time = 0.1

    current_steering_angle = float(state[2])
    current_velocity = float(state[3])
    desired_steering_angle = float(desired[0])
    desired_velocity = float(desired[1])

    if not _pid_state["initialized"]:
        reset_lower_controller_state()
        _pid_state["initialized"] = True

    # Limits
    min_steering_rate = float(parameters[7])
    max_steering_rate = float(parameters[9])
    min_acceleration = float(parameters[8])
    max_acceleration = float(parameters[10])

    # Steering PID
    error_steering = desired_steering_angle - current_steering_angle
    kp_steering, ki_steering, kd_steering = 4.0, 0.5, 0.1

    # Integration and Derivative
    new_integral_steering = (
        _pid_state["integral_steering_error"] + error_steering * delta_time
    )
    derivative_steering = (
        error_steering - _pid_state["previous_steering_error"]
    ) / delta_time

    raw_steering_rate = (
        kp_steering * error_steering
        + ki_steering * new_integral_steering
        + kd_steering * derivative_steering
    )
    commanded_steering_rate = float(
        np.clip(raw_steering_rate, min_steering_rate, max_steering_rate)
    )

    # Anti-windup: only update integral if not saturated
    if commanded_steering_rate == raw_steering_rate:
        _pid_state["integral_steering_error"] = new_integral_steering
    _pid_state["previous_steering_error"] = error_steering

    # Velocity PID
    error_velocity = desired_velocity - current_velocity
    kp_velocity, ki_velocity, kd_velocity = 1.5, 0.4, 0.05

    new_integral_velocity = (
        _pid_state["integral_velocity_error"] + error_velocity * delta_time
    )
    derivative_velocity = (
        error_velocity - _pid_state["previous_velocity_error"]
    ) / delta_time

    raw_acceleration = (
        kp_velocity * error_velocity
        + ki_velocity * new_integral_velocity
        + kd_velocity * derivative_velocity
    )
    commanded_acceleration = float(
        np.clip(raw_acceleration, min_acceleration, max_acceleration)
    )

    # Anti-windup
    if commanded_acceleration == raw_acceleration:
        _pid_state["integral_velocity_error"] = new_integral_velocity
    _pid_state["previous_velocity_error"] = error_velocity

    return np.array([commanded_steering_rate, commanded_acceleration])
