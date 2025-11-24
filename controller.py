import numpy as np
from numpy.typing import ArrayLike
from racetrack import RaceTrack

class PIDController:
    """
    Discrete-Time PID Controller with Anti-Windup and Derivative Filtering.
    
    Implements:
    u[k] = P[k] + I[k] + D[k]
    
    P[k] = Kp * e[k]
    I[k] = I[k-1] + Ki * Ts * (e[k] + e[k-1]) / 2  (Tustin / Trapezoidal)
    D[k] = Kd * (e[k] - e[k-1]) / Ts               (Backward Difference)
    
    Anti-Windup: Clamps the integrator component if output saturates.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, ts: float, min_out: float, max_out: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ts = ts
        self.min_out = min_out
        self.max_out = max_out
        
        # State
        self.prev_error = 0.0
        self.integral = 0.0
        self.reset()

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error: float) -> float:
        # Proportional
        p_term = self.kp * error
        
        # Integral (Tustin approximation)
        # I[k] = I[k-1] + (Ki * Ts / 2) * (e[k] + e[k-1])
        # We compute a candidate integral first
        delta_integral = (self.ki * self.ts / 2.0) * (error + self.prev_error)
        candidate_integral = self.integral + delta_integral
        
        # Derivative (Backward Euler)
        d_term = self.kd * (error - self.prev_error) / self.ts
        
        # Compute raw output
        raw_output = p_term + candidate_integral + d_term
        
        # Output Saturation & Anti-Windup
        # If we are saturating, we do not update the integral term to accumulate more error (Clamping)
        output = np.clip(raw_output, self.min_out, self.max_out)
        
        # Simple Anti-Windup: Only update integral if we are not saturated 
        # or if the error attempts to bring us back from saturation.
        # Check if saturation occurred
        if raw_output != output:
             # Sign mismatch or magnitude issue. 
             # If driving further into saturation, don't integrate.
             # If driving out of saturation, do integrate.
             if np.sign(error) == np.sign(raw_output):
                 # Driving further into saturation -> clamp integral (keep previous)
                 pass 
             else:
                 # Driving out -> allow integration
                 self.integral = candidate_integral
        else:
            self.integral = candidate_integral
            
        self.prev_error = error
        
        return output

# --------------------------------------------------------------------------
# Vehicle Controller (Outer Loop + Inner Loop)
# --------------------------------------------------------------------------
class VehicleController:
    """
    Hierarchical Controller:
    1. Outer Loop (Path Tracking): Pure Pursuit -> Desired Steering Angle, Velocity
    2. Inner Loop (Dynamics): PID Controllers -> Steering Rate, Acceleration
    """
    
    def __init__(self, parameters: ArrayLike, raceline_data: ArrayLike = None):
        self.parameters = parameters
        
        # Extract Vehicle limits for PID clamping
        # parameters: [L, -max_steer, min_vel, -pi, max_steer, max_vel, pi, -max_steer_rate, -max_accel, max_steer_rate, max_accel]
        self.L = parameters[0]
        self.max_steer_angle = parameters[4]
        self.max_velocity = parameters[5]
        self.max_steer_rate = parameters[9]
        self.max_accel = parameters[10]
        self.min_accel = parameters[8] # Usually negative max accel
        
        self.ts = 0.1 # 100ms sample time (from RaceCar time_step)

        # Path to track (Raceline or Centerline)
        self.path = raceline_data
        
        # Tunable Control Parameters
        self.lookahead_dist = 8.0  # Adjusted for better tracking
        self.target_velocity = 25.0 # Slightly faster target
        
        # PID Tunings
        # Steering Rate PID
        # Fast response needed.
        kp_steer = 8.0
        ki_steer = 2.0
        kd_steer = 0.1
        
        # Velocity PID
        # Smooth acceleration
        kp_vel = 2.0
        ki_vel = 0.5
        kd_vel = 0.0
        
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

    def compute_control(self, state: ArrayLike) -> ArrayLike:
        """
        Computes control inputs [steering_rate, acceleration]
        state: [x, y, delta, v, psi]
        """
        # 1. Outer Loop: Pure Pursuit
        desired_steer, desired_vel = self._pure_pursuit(state)
        
        # 2. Inner Loop: PID Control
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
        
        # Find closest point on path
        # Optim: In a real scenario we would search locally, but here global search is okay for sim size
        dists = np.linalg.norm(self.path - np.array([x, y]), axis=1)
        closest_idx = np.argmin(dists)
        
        # Find lookahead point
        n_points = len(self.path)
        lookahead_idx = closest_idx
        
        for i in range(n_points):
            idx = (closest_idx + i) % n_points
            dist = np.linalg.norm(self.path[idx] - np.array([x, y]))
            if dist > self.lookahead_dist:
                lookahead_idx = idx
                break
        
        target = self.path[lookahead_idx]
        
        # Calculate Alpha
        dx = target[0] - x
        dy = target[1] - y
        angle_to_target = np.arctan2(dy, dx)
        alpha = angle_to_target - psi
        
        # Normalize alpha to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # Pure Pursuit Geometry
        lookahead_actual = np.linalg.norm([dx, dy])
        desired_steer = np.arctan(2 * self.L * np.sin(alpha) / lookahead_actual)
        
        desired_steer = np.clip(desired_steer, -self.max_steer_angle, self.max_steer_angle)
        
        return desired_steer, self.target_velocity
