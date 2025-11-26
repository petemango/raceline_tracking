import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack


class PIDController:
    """
    Discrete-Time PID Controller with Anti-Windup.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, ts: float, min_out: float, max_out: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ts = ts
        self.min_out = min_out
        self.max_out = max_out
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error: float) -> float:
        p_term = self.kp * error
        delta_integral = (self.ki * self.ts / 2.0) * (error + self.prev_error)
        candidate_integral = self.integral + delta_integral
        d_term = self.kd * (error - self.prev_error) / self.ts
        raw_output = p_term + candidate_integral + d_term
        output = np.clip(raw_output, self.min_out, self.max_out)
        
        if raw_output != output:
            if np.sign(error) != np.sign(raw_output):
                self.integral = candidate_integral
        else:
            self.integral = candidate_integral
            
        self.prev_error = error
        return output


class VehicleController:
    """
    Hierarchical Controller:
    1. Outer Loop (Path Tracking): Pure Pursuit -> Desired Steering Angle, Velocity
    2. Inner Loop (Dynamics): PID Controllers -> Steering Rate, Acceleration
    """
    
    def __init__(self, parameters: ArrayLike, raceline_data: ArrayLike = None):
        self.parameters = parameters
        
        self.L = parameters[0]
        self.max_steer_angle = parameters[4]
        self.max_velocity = parameters[5]
        self.max_steer_rate = parameters[9]
        self.max_accel = parameters[10]
        self.min_accel = parameters[8]
        
        self.ts = 0.1

        self.path = raceline_data
        self.last_closest_idx = 0
        
        self.path_curvature = None
        if self.path is not None:
            self._precompute_curvature()
        
        # Tunable Control Parameters
        self.lookahead_base = 5.0
        self.lookahead_gain = 0.7
        
        # Physics-based velocity limits
        self.max_straight_velocity = 90.0
        self.min_velocity = 20.0
        self.curvature_lookahead = 45
        
        # Lateral grip limit
        self.max_lateral_accel = 14.0
        self.grip_margin = 0.82
        
        # PID Tunings
        kp_steer = 5.0
        ki_steer = 0.5
        kd_steer = 0.2
        
        kp_vel = 3.0
        ki_vel = 0.5
        kd_vel = 0.1
        
        self.steer_pid = PIDController(
            kp_steer, ki_steer, kd_steer, self.ts, 
            -self.max_steer_rate, self.max_steer_rate
        )
        
        self.velocity_pid = PIDController(
            kp_vel, ki_vel, kd_vel, self.ts,
            self.min_accel, self.max_accel
        )

    def set_path(self, path: ArrayLike):
        self.path = path
        if self.path is not None:
            self._precompute_curvature()
    
    def _precompute_curvature(self):
        n = len(self.path)
        self.path_curvature = np.zeros(n)
        
        for i in range(n):
            p0 = self.path[(i - 1) % n]
            p1 = self.path[i]
            p2 = self.path[(i + 1) % n]
            
            v1 = p1 - p0
            v2 = p2 - p1
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 < 1e-6 or len2 < 1e-6:
                self.path_curvature[i] = 0.0
                continue
            
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = np.dot(v1, v2)
            angle_change = np.arctan2(cross, dot)
            
            ds = (len1 + len2) / 2.0
            self.path_curvature[i] = abs(angle_change) / ds
    
    def _get_target_velocity(self, closest_idx):
        if self.path_curvature is None:
            return self.min_velocity
        
        n = len(self.path)
        
        max_curvature = 0.0
        for i in range(self.curvature_lookahead):
            idx = (closest_idx + i) % n
            max_curvature = max(max_curvature, self.path_curvature[idx])
        
        if max_curvature < 1e-6:
            return self.max_straight_velocity
        
        v_physics = np.sqrt(self.max_lateral_accel / max_curvature) * self.grip_margin
        target_vel = np.clip(v_physics, self.min_velocity, self.max_straight_velocity)
        
        return target_vel

    def compute_control(self, state: ArrayLike) -> ArrayLike:
        desired_steer, desired_vel = self._pure_pursuit(state)
        
        current_steer = state[2]
        current_vel = state[3]
        
        steer_error = desired_steer - current_steer
        vel_error = desired_vel - current_vel
        
        steering_rate = self.steer_pid.update(steer_error)
        acceleration = self.velocity_pid.update(vel_error)
        
        return np.array([steering_rate, acceleration])

    def _pure_pursuit(self, state):
        if self.path is None:
            return 0.0, 0.0
            
        x, y, delta, v, psi = state
        n_points = len(self.path)
        
        current_speed = max(abs(v), 1.0)
        lookahead_dist = self.lookahead_base + self.lookahead_gain * current_speed
        
        car_pos = np.array([x, y])
        dists = np.linalg.norm(self.path - car_pos, axis=1)
        closest_idx = np.argmin(dists)
        
        idx_diff = closest_idx - self.last_closest_idx
        if idx_diff < -10 and idx_diff > -n_points + 100:
            closest_idx = self.last_closest_idx
        
        self.last_closest_idx = closest_idx
        
        lookahead_idx = closest_idx
        accumulated_dist = 0.0
        
        for i in range(1, min(200, n_points // 2)):
            idx = (closest_idx + i) % n_points
            prev_idx = (closest_idx + i - 1) % n_points
            
            segment_dist = np.linalg.norm(self.path[idx] - self.path[prev_idx])
            accumulated_dist += segment_dist
            
            if accumulated_dist >= lookahead_dist:
                lookahead_idx = idx
                break
        
        if lookahead_idx == closest_idx:
            lookahead_idx = (closest_idx + 10) % n_points
        
        target = self.path[lookahead_idx]
        
        dx = target[0] - x
        dy = target[1] - y
        angle_to_target = np.arctan2(dy, dx)
        alpha = angle_to_target - psi
        
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        lookahead_actual = max(np.linalg.norm([dx, dy]), 0.1)
        desired_steer = np.arctan(2 * self.L * np.sin(alpha) / lookahead_actual)
        
        desired_steer = np.clip(desired_steer, -self.max_steer_angle, self.max_steer_angle)
        
        target_vel = self._get_target_velocity(closest_idx)
        
        return desired_steer, target_vel


# ============ Function interface for main branch compatibility ============

_controller_instance = None

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Lower-level controller that converts desired [steer_angle, velocity] to 
    control inputs [steering_rate, acceleration] using PID.
    """
    assert(desired.shape == (2,))
    
    # Simple proportional control for the lower level
    max_steer_rate = parameters[9]
    max_accel = parameters[10]
    
    current_steer = state[2]
    current_vel = state[3]
    
    steer_error = desired[0] - current_steer
    vel_error = desired[1] - current_vel
    
    # P control with gain
    steering_rate = np.clip(5.0 * steer_error, -max_steer_rate, max_steer_rate)
    acceleration = np.clip(3.0 * vel_error, -max_accel, max_accel)
    
    return np.array([steering_rate, acceleration])


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller that computes desired [steer_angle, velocity].
    Uses VehicleController internally for path following.
    """
    global _controller_instance
    
    # Initialize controller on first call
    if _controller_instance is None:
        _controller_instance = VehicleController(parameters, racetrack.centerline)
    
    # Get desired steering and velocity from pure pursuit
    desired_steer, desired_vel = _controller_instance._pure_pursuit(state)
    
    return np.array([desired_steer, desired_vel])
