import numpy as np

class PID():
    def __init__(self, kp: float, ki: float, kd: float,
                out_min: float = -float('inf'),
                out_max: float =  float('inf'),
                windup_limit: float = 500.0):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.out_min = out_min
        self.out_max = out_max

        self.windup_limit = windup_limit   # anti-windup clamp on integral

        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:

        if dt <= 1e-6: # if below 1 microsecond
            return 0.0

        self._integral += error * dt        
        self._integral = max(-self.windup_limit, min(self.windup_limit, self._integral)) # clamp integral

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        u_unsat = self.kp * error + self.ki * self._integral + self.kd * derivative
        u_sat = max(self.out_min, min(self.out_max, u_unsat))

        # Conditional integration anti-windup
        # - 1 : if not sat (normal case)
        # - 2 : (if sat MAX ,`less` output) and (error went other direction)
        # - 3 : (if sat MIN ,`more` output) and (error went other direction)  
        if (u_unsat == u_sat) or \
            (u_sat == self.out_max and error < 0) or \
            (u_sat == self.out_min and error > 0):
            
            self._integral += error * dt
            self._integral = max(-self.windup_limit, min(self.windup_limit, self._integral))

        return u_sat
    