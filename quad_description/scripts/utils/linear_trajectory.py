import time

class LinearTrajectory:
    """
    Generates a linear (ramp) trajectory for a single axis.

    The setpoint rises (or falls) at a constant velocity from
    z_start → z_end over `duration` seconds, then holds z_end.

    Usage
    -----
        traj = LinearTrajectory(z_start=0.0, z_end=1.5, duration=5.0)
        traj.start()                   # call once when ready to fly

        # inside your control loop:
        ref_z = traj.get_setpoint()    # smooth ramp value
    """

    def __init__(self,
                 z_start: float = 0.0,
                 z_end:   float = 1.0,
                 duration: float = 5.0):
        """
        Parameters
        ----------
        z_start  : initial altitude  [m]
        z_end    : target altitude   [m]
        duration : time to complete ramp [s]
        """
        self.z_start  = float(z_start)
        self.z_end    = float(z_end)
        self.duration = float(duration)

        self._t0      = None   # wall-clock start time
        self._running = False

    # ------------------------------------------------------------------
    def start(self):
        """Begin the trajectory from now."""
        self._t0      = time.time()
        self._running = True

    def reset(self, z_start=None, z_end=None, duration=None):
        """Update parameters and restart the trajectory."""
        if z_start  is not None: self.z_start  = float(z_start)
        if z_end    is not None: self.z_end    = float(z_end)
        if duration is not None: self.duration = float(duration)
        self.start()

    # ------------------------------------------------------------------
    def get_setpoint(self) -> float:
        """
        Returns the current altitude setpoint [m].

        Returns z_start if start() has not been called yet,
        and holds z_end once the ramp is complete.
        """
        if not self._running:
            return self.z_start

        elapsed = time.time() - self._t0

        # clamp to [0, duration]
        t = max(0.0, min(elapsed, self.duration))

        # linear interpolation
        alpha = t / self.duration if self.duration > 1e-9 else 1.0
        return self.z_start + alpha * (self.z_end - self.z_start)

    @property
    def is_done(self) -> bool:
        """True once the ramp has fully reached z_end."""
        if not self._running:
            return False
        return (time.time() - self._t0) >= self.duration


# -----------------------------------------------------------------------
# ROS 2 integration helper  (drop-in replacement for ref_altitute)
# -----------------------------------------------------------------------
# In your RPYControllerNode.__init__  add:
#
#   from utils.linear_trajectory import LinearTrajectory
#   self.z_traj = LinearTrajectory(z_start=0.0, z_end=1.0, duration=5.0)
#   self.z_traj.start()
#
# In _control_loop replace:
#
#   alt_err = self.ref_altitute - self.altitute
#
# with:
#
#   self.ref_altitute = self.z_traj.get_setpoint()
#   alt_err = self.ref_altitute - self.altitute
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Quick standalone test  (run:  python linear_trajectory.py)
# -----------------------------------------------------------------------
if __name__ == '__main__':
    import math

    traj = LinearTrajectory(z_start=0.0, z_end=1.5, duration=5.0)
    traj.start()

    print(f"{'Time(s)':>8}  {'Setpoint(m)':>12}  {'Done':>6}")
    print("-" * 35)

    dt = 0.25
    t  = 0.0
    while t <= 6.0:
        sp = traj.get_setpoint()
        print(f"{t:8.2f}  {sp:12.4f}  {str(traj.is_done):>6}")
        time.sleep(dt)
        t += dt
