#!/usr/bin/env python3
"""
trajectory_generator.py
========================
ROS 2 node that publishes a 3-D reference trajectory as PoseStamped + Path.

─────────────────────────────────────────────────────────────────────────────
TRAJECTORY SELECTION
─────────────────────────────────────────────────────────────────────────────
Change the single class variable below to switch trajectories:

    self.mode = "yaw"        ← stationary hover + slow yaw sweep (tuning yaw)
    self.mode = "helix"      ← rising helix then circle
    self.mode = "circle"     ← horizontal circle at fixed altitude
    self.mode = "figure8"    ← lemniscate figure-of-8
    self.mode = "line"       ← sinusoidal back-and-forth on x-axis
    self.mode = "hover"      ← stationary hover at [0, 0, cruise_altitude]

─────────────────────────────────────────────────────────────────────────────
YAW TUNING TRAJECTORY  (mode = "yaw")
─────────────────────────────────────────────────────────────────────────────
Purpose: isolate and tune the yaw channel.
  - Drone holds [x=0, y=0, z=cruise_altitude]  (position locked)
  - Yaw sweeps a triangle wave between -yaw_amp and +yaw_amp with period T
  - Linear ramp avoids step commands that saturate tau_yaw
  - All position/velocity references stay zero → no coupling with roll/pitch

Triangle-wave yaw profile:
    At t=0:            yaw=0, yawdot=+yaw_rate
    At t=T/4:          yaw=+yaw_amp, yawdot=-yaw_rate  (peak)
    At t=3T/4:         yaw=-yaw_amp, yawdot=+yaw_rate  (trough)
    At t=T:            yaw=0  (repeat)

Tuning sequence recommended:
    1.  Start with R_DIAG[3] (tau_yaw cost) in MPC and Q_DIAG[5] (psi cost)
    2.  Increase Q_DIAG[5] until tracking is tight without oscillation
    3.  Increase R_DIAG[3] to damp oscillation if it appears
    4.  Check Q_DIAG[11] (psidot cost) — raising it reduces overshoot

Parameters
----------
  cruise_altitude   float   2.0   m     hover / cruise altitude
  period            float  15.0   s     trajectory period (also yaw sweep period)
  radius            float   1.5   m     circle / helix / figure8 / line radius
  rise_time         float   4.0   s     altitude ramp duration
  yaw_amplitude     float   1.57  rad   half-range of yaw sweep (default π/2)
  publish_rate      float  20.0   Hz

Publications
------------
  /reference_pose  (geometry_msgs/PoseStamped)
  /reference_path  (nav_msgs/Path)   — preview for RViz
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class TrajectoryGenerator(Node):

    # ══════════════════════════════════════════════════════════════════════════
    # ► CHANGE THIS LINE TO SWITCH TRAJECTORY ◄
    # Options: "yaw" | "helix" | "circle" | "figure8" | "line" | "hover"
    # ══════════════════════════════════════════════════════════════════════════
    mode = "helix"

    def __init__(self):
        super().__init__('trajectory_generator')

        # ROS parameters
        self.declare_parameter('cruise_altitude',  2.0)
        self.declare_parameter('period',          15.0)
        self.declare_parameter('radius',           1.5)
        self.declare_parameter('rise_time',        4.0)
        self.declare_parameter('yaw_amplitude',    math.pi / 2.0)  # rad
        self.declare_parameter('publish_rate',    20.0)
        self.declare_parameter('preview_steps',   60)

        self.z_cruise      = self.get_parameter('cruise_altitude').value
        self.T             = self.get_parameter('period').value
        self.R             = self.get_parameter('radius').value
        self.t_rise        = self.get_parameter('rise_time').value
        self.yaw_amp       = self.get_parameter('yaw_amplitude').value
        rate               = self.get_parameter('publish_rate').value
        self.preview_steps = self.get_parameter('preview_steps').value
        self.dt_preview    = 1.0 / rate

        self.pub_pose = self.create_publisher(PoseStamped, '/reference_pose', 10)
        self.pub_path = self.create_publisher(Path,        '/reference_path',  1)

        self.t0 = None
        self.create_timer(1.0 / rate, self._cb)

        self.get_logger().info(
            f'TrajectoryGenerator | mode="{self.mode}"  '
            f'z={self.z_cruise}m  T={self.T}s  R={self.R}m  '
            f'yaw_amp={math.degrees(self.yaw_amp):.0f}deg  rate={rate}Hz')

    # ─────────────────────────────────────────────────────────────────────────

    def _cb(self):
        stamp = self.get_clock().now()
        t_sec = stamp.nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t_sec
        t = t_sec - self.t0

        x, y, z, yaw = self._sample(t)
        self.pub_pose.publish(self._make_pose(x, y, z, yaw, stamp.to_msg()))

        path_msg = Path()
        path_msg.header.stamp    = stamp.to_msg()
        path_msg.header.frame_id = 'odom'
        for k in range(self.preview_steps):
            px, py, pz, pyaw = self._sample(t + k * self.dt_preview)
            path_msg.poses.append(self._make_pose(px, py, pz, pyaw, stamp.to_msg()))
        self.pub_path.publish(path_msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Trajectory sampler — dispatches on self.mode
    # ─────────────────────────────────────────────────────────────────────────

    def _sample(self, t):
        """Return (x, y, z, yaw) at time t for the active mode."""
        if self.mode == "yaw":
            return self._traj_yaw(t)
        elif self.mode in ("helix", "circle"):
            return self._traj_helix(t)
        elif self.mode == "figure8":
            return self._traj_figure8(t)
        elif self.mode == "line":
            return self._traj_line(t)
        elif self.mode == "hover":
            return 0.0, 0.0, self.z_cruise, 0.0
        else:
            self.get_logger().warn(
                f'Unknown mode "{self.mode}" — defaulting to hover', once=True)
            return 0.0, 0.0, self.z_cruise, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # ── YAW TUNING ────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    def _traj_yaw(self, t):
        """
        Stationary position at (0, 0, z_cruise) with a triangle-wave yaw sweep.

        Triangle wave between −yaw_amp and +yaw_amp with period T.
        Linear ramp (constant yawdot) avoids impulsive torque demands.

        Altitude ramp: z linearly increases to z_cruise over t_rise,
        then yaw sweep begins — so the drone reaches altitude first.
        """
        # Altitude ramp (same as other trajectories)
        alpha = min(1.0, t / self.t_rise) if self.t_rise > 0.0 else 1.0
        z = self.z_cruise * alpha

        # Yaw sweep only starts after altitude ramp completes
        if alpha < 1.0:
            return 0.0, 0.0, z, 0.0

        # Time since altitude was reached
        t_yaw = t - self.t_rise

        # Triangle wave with period T
        # Normalise into [0, 1) within one period
        phase = (t_yaw % self.T) / self.T      # 0 → 1

        # Triangle:  0→1 in first half,  1→0 in second half,  scaled to ±yaw_amp
        if phase < 0.5:
            yaw = self.yaw_amp * (4.0 * phase - 1.0)        # -amp → +amp
        else:
            yaw = self.yaw_amp * (3.0 - 4.0 * phase)        # +amp → -amp

        return 0.0, 0.0, z, yaw

    # ─────────────────────────────────────────────────────────────────────────
    # ── HELIX / CIRCLE ────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    def _traj_helix(self, t):
        alpha = min(1.0, t / self.t_rise) if self.t_rise > 0.0 else 1.0
        z     = self.z_cruise * alpha
        omega = 2.0 * math.pi / self.T
        tc    = t * alpha          # blend horizontal motion with altitude ramp
        x   =  self.R * math.cos(omega * tc)
        y   =  self.R * math.sin(omega * tc)
        yaw =  omega * tc
        return x, y, z, yaw

    # ─────────────────────────────────────────────────────────────────────────
    # ── FIGURE-8 ──────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    def _traj_figure8(self, t):
        alpha = min(1.0, t / self.t_rise) if self.t_rise > 0.0 else 1.0
        z     = self.z_cruise * alpha
        omega = 2.0 * math.pi / self.T
        tc    = t * alpha
        a     = self.R
        st    = math.sin(omega * tc)
        ct    = math.cos(omega * tc)
        x     = a * st
        y     = a * st * ct
        vx    = a * omega * ct
        vy    = a * omega * (ct * ct - st * st)
        yaw   = math.atan2(vy, vx) if (vx != 0.0 or vy != 0.0) else 0.0
        return x, y, z, yaw

    # ─────────────────────────────────────────────────────────────────────────
    # ── LINE ──────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    def _traj_line(self, t):
        alpha = min(1.0, t / self.t_rise) if self.t_rise > 0.0 else 1.0
        z     = self.z_cruise * alpha
        omega = 2.0 * math.pi / self.T
        tc    = t * alpha
        x     = self.R * math.sin(omega * tc)
        vx    = self.R * omega * math.cos(omega * tc)
        yaw   = 0.0 if vx >= 0.0 else math.pi
        return x, 0.0, z, yaw

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_pose(x, y, z, yaw, stamp):
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        return msg


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()