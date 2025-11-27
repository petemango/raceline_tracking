import numpy as np
from numpy.typing import ArrayLike

# Global state for PID
_pid_state: dict = {}


def reset_lower_controller_state() -> None:
    global _pid_state
    _pid_state = {
        "initialized": False,
        "steer_integral": 0.0,
        "steer_prev_error": 0.0,
        "vel_integral": 0.0,
        "vel_prev_error": 0.0,
    }


# Initialize on import
reset_lower_controller_state()


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    global _pid_state

    dt = 0.1

    current_steer = float(state[2])
    current_vel = float(state[3])

    desired_steer = float(desired[0])
    desired_vel = float(desired[1])

    if not _pid_state["initialized"]:
        _pid_state["initialized"] = True
        _pid_state["steer_integral"] = 0.0
        _pid_state["steer_prev_error"] = 0.0
        _pid_state["vel_integral"] = 0.0
        _pid_state["vel_prev_error"] = 0.0

    # Limits
    min_steer_vel = float(parameters[7])
    max_steer_vel = float(parameters[9])
    min_accel = float(parameters[8])
    max_accel = float(parameters[10])

    # Steering PID
    e_steer = desired_steer - current_steer
    Kp_steer, Ki_steer, Kd_steer = 4.0, 0.5, 0.1

    steer_integral_candidate = _pid_state["steer_integral"] + e_steer * dt
    steer_derivative = (e_steer - _pid_state["steer_prev_error"]) / dt

    steer_rate_unsat = (
        Kp_steer * e_steer
        + Ki_steer * steer_integral_candidate
        + Kd_steer * steer_derivative
    )

    steer_rate = float(np.clip(steer_rate_unsat, min_steer_vel, max_steer_vel))

    if steer_rate == steer_rate_unsat:
        _pid_state["steer_integral"] = steer_integral_candidate

    _pid_state["steer_prev_error"] = e_steer

    # Velocity PID
    e_vel = desired_vel - current_vel
    Kp_vel, Ki_vel, Kd_vel = 1.5, 0.4, 0.05

    vel_integral_candidate = _pid_state["vel_integral"] + e_vel * dt
    vel_derivative = (e_vel - _pid_state["vel_prev_error"]) / dt

    accel_unsat = (
        Kp_vel * e_vel
        + Ki_vel * vel_integral_candidate
        + Kd_vel * vel_derivative
    )

    accel = float(np.clip(accel_unsat, min_accel, max_accel))

    if accel == accel_unsat:
        _pid_state["vel_integral"] = vel_integral_candidate

    _pid_state["vel_prev_error"] = e_vel

    return np.array([steer_rate, accel])
