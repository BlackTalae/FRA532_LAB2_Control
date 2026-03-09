#!/usr/bin/env python3
"""
trajectory_sender.py
====================
Publishes waypoint trajectories as:
  /target_pose      (geometry_msgs/PoseStamped)  — existing single-pose output
  /reference_path   (nav_msgs/Path)              — MPC-compatible lookahead path

The /reference_path is what the MPC subscribes to.  It contains N_HORIZON + 1
poses spaced MPC_DT seconds apart, built by looking ahead in the waypoint list
from the current position.

Compatibility with MPC
----------------------
  MPC.DT       = 0.05 s  (20 Hz)
  PUBLISH_HZ   = 100 Hz  → dt = 0.01 s per waypoint index
  LOOKAHEAD_STEP = round(MPC_DT / (1/PUBLISH_HZ)) = round(0.05 / 0.01) = 5

  Path pose k = waypoint at index (current + k * LOOKAHEAD_STEP)
  MPC finite-differences pose k+1 - pose k and divides by MPC_DT to get velocity.

  N_HORIZON = 21 poses  (MPC needs N+1=21 to finite-difference N=20 velocities)

Robustness support (added, structure unchanged)
-----------------------------------------------
  Publishes /trajectory_lap_complete (std_msgs/Bool, data=True) each time
  the waypoint index wraps around to 0 (i.e. one full loop completes).
  The trajectory_visualizer node subscribes to this to capture per-lap data.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool                    # ← ADDED: lap-complete signal

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def add_yaw_alignment(waypoints, is_round_trip=False):
    if not waypoints or len(waypoints) < 2:
        return waypoints

    aligned_waypoints = []
    forward_len = len(waypoints) // 2 + 1 if is_round_trip else len(waypoints)

    for i in range(len(waypoints)):
        wp_curr = waypoints[i]

        if is_round_trip and i >= forward_len:
            target_yaw = aligned_waypoints[-1][3] if aligned_waypoints else 0.0
            aligned_waypoints.append((wp_curr[0], wp_curr[1], wp_curr[2], target_yaw, wp_curr[4], wp_curr[5]))
            continue

        if i + 1 < forward_len:
            wp_next = waypoints[i + 1]
            dx = wp_next[0] - wp_curr[0]
            dy = wp_next[1] - wp_curr[1]
        else:
            wp_prev = waypoints[i - 1]
            dx = wp_curr[0] - wp_prev[0]
            dy = wp_curr[1] - wp_prev[1]

        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            target_yaw = math.atan2(dy, dx)
        else:
            target_yaw = aligned_waypoints[-1][3] if aligned_waypoints else 0.0

        aligned_waypoints.append((wp_curr[0], wp_curr[1], wp_curr[2], target_yaw, wp_curr[4], wp_curr[5]))

    return aligned_waypoints


def make_round_trip(waypoints):
    if not waypoints:
        return []
    return waypoints + waypoints[-2::-1]


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Library
# ──────────────────────────────────────────────────────────────────────────────

def get_hover_trajectory(z=1.0, dwell=2.0):
    return [(0.0, 0.0, z, 0.0, 0.0, dwell)]

def get_vertical_trajectory(z_min=1.0, z_max=3.0, speed=1.0, dwell=2.0, hz=100.0):
    dist = abs(z_max - z_min)
    total_duration = dist / speed if speed > 0 else 1.0
    num_points = max(1, int(total_duration * hz))
    waypoints = [(0.0, 0.0, z_min, 0.0, hz, dwell)]
    for i in range(1, num_points + 1):
        z = z_min + (i / num_points) * (z_max - z_min)
        waypoints.append((0.0, 0.0, z, 0.0, hz, dwell if i == num_points else 0.0))
    return waypoints

def get_plane_trajectory(size=1.0, x_offset=0.0, z_start=1.0, speed=1.0, dwell=2.0, hz=100.0):
    duration = size / speed if speed > 0 else 1.0
    num_points = max(1, int(duration * hz))
    dt = duration / num_points
    corners = [
        (x_offset,        0.0, z_start),
        (x_offset + size, 0.0, z_start),
        (x_offset + size, 0.0, z_start + size),
        (x_offset,        0.0, z_start),
    ]
    waypoints = [(corners[0][0], corners[0][1], corners[0][2], 0.0, dt, dwell)]
    for seg_idx in range(len(corners) - 1):
        p_start, p_end = corners[seg_idx], corners[seg_idx + 1]
        for i in range(1, num_points + 1):
            alpha = i / num_points
            x = p_start[0] + alpha * (p_end[0] - p_start[0])
            y = p_start[1] + alpha * (p_end[1] - p_start[1])
            z = p_start[2] + alpha * (p_end[2] - p_start[2])
            waypoints.append((x, y, z, 0.0, dt, dwell if i == num_points else 0.0))
    return waypoints

def get_sine_wave_trajectory(amplitude=1.0, freq=0.5, num_waves=1.0, speed=1.0, z_base=1.5, dwell=3.0, hz=100.0):
    total_duration = num_waves / freq if freq > 0 else 1.0
    num_points = max(1, int(total_duration * hz))
    waypoints = [(0.0, 0.0, z_base, 0.0, hz, dwell)]
    for i in range(1, num_points + 1):
        t = (i / num_points) * total_duration
        x = speed * t
        z = z_base + amplitude * math.sin(2.0 * math.pi * freq * t)
        waypoints.append((x, 0.0, z, 0.0, hz, dwell if i == num_points else 0.0))
    return waypoints

def get_circle_trajectory(radius=1.0, y=0.0, z_base=3.0, speed=1.0, dwell=0.0, hz=100.0):
    circumference = 2.0 * math.pi * radius
    total_duration = circumference / speed if speed > 0 else 2.0 * math.pi
    num_points = max(1, int(total_duration * hz))
    waypoints = [(0.0, y, z_base, 0.0, hz, dwell)]
    for i in range(1, num_points + 1):
        angle = 2.0 * math.pi * (i / num_points)
        x = radius * math.cos(angle) - radius
        z = z_base + radius * math.sin(angle)
        waypoints.append((x, y, z, 0.0, hz, dwell if i == num_points else 0.0))
    return waypoints

def get_linear_trajectory(start_pos, target_pos, speed=1.0, dwell=2.0, hz=100.0):
    sx, sy, sz = start_pos
    tx, ty, tz = target_pos
    dist = math.sqrt((tx-sx)**2 + (ty-sy)**2 + (tz-sz)**2)
    total_duration = dist / speed if speed > 0 else 1.0
    num_points = max(1, int(total_duration * hz))
    waypoints = [(sx, sy, sz, 0.0, hz, dwell)]
    for i in range(1, num_points + 1):
        alpha = i / num_points
        waypoints.append((sx + alpha*(tx-sx), sy + alpha*(ty-sy), sz + alpha*(tz-sz),
                          0.0, hz, dwell if i == num_points else 0.0))
    return waypoints

def get_helix_trajectory(radius=1.0, height=2.0, turns=2, speed=1.0, z_base=1.0, dwell=0.0, hz=100.0):
    circumference = 2.0 * math.pi * radius
    h_per_turn = height / turns
    total_distance = turns * math.sqrt(circumference**2 + h_per_turn**2)
    total_duration = total_distance / speed if speed > 0 else 5.0
    num_points = max(1, int(total_duration * hz))
    waypoints = [(0.0, 0.0, z_base, 0.0, hz, dwell)]
    for i in range(1, num_points + 1):
        progress = i / num_points
        angle = progress * turns * 2.0 * math.pi
        x = radius * math.sin(angle)
        y = radius - radius * math.cos(angle)
        z = z_base + progress * height
        waypoints.append((x, y, z, 0.0, hz, dwell if i == num_points else 0.0))
    return waypoints

def get_spiral_trajectory(radius_start=2.0, height=2.0, turns=3, speed=1.0, z_base=1.0, dwell=0.0, hz=100.0):
    avg_radius = radius_start / 2.0
    approx_circumference = 2.0 * math.pi * avg_radius
    h_per_turn = height / turns
    approx_distance = turns * math.sqrt(approx_circumference**2 + h_per_turn**2)
    total_duration = approx_distance / speed if speed > 0 else 5.0
    num_points = max(1, int(total_duration * hz))
    waypoints = [(0.0, 0.0, z_base, 0.0, hz, dwell)]
    for i in range(1, num_points + 1):
        progress = i / num_points
        radius = radius_start * (1.0 - progress)
        angle = progress * turns * 2.0 * math.pi
        x = radius * math.sin(angle)
        y = radius_start - radius * math.cos(angle)
        z = z_base + progress * height
        waypoints.append((x, y, z, 0.0, hz, dwell if i == num_points else 0.0))
    return waypoints

def get_figure_eight_2d_trajectory(width=2.0, height=2.0, speed=1.0, z_base=3.0, dwell=0.0, hz=100.0):
    approx_perimeter = math.pi * (width + height)
    total_duration = approx_perimeter / speed if speed > 0 else 5.0
    num_points = max(1, int(total_duration * hz))
    waypoints = [(0.0, 0.0, z_base, 0.0, hz, dwell)]
    for i in range(1, num_points + 1):
        t = 2.0 * math.pi * (i / num_points)
        x = width * math.sin(t)
        z = z_base + (height / 2.0) * math.sin(2.0 * t)
        waypoints.append((x, 0.0, z, 0.0, hz, dwell if i == num_points else 0.0))
    return waypoints

def get_figure_eight_3d(width=2.0, height=1.0, depth=1.0, speed=1.0, points=100, dwell=0.0):
    waypoints = []
    x_curr, y_curr, z_curr = 0.0, 0.0, 1.5
    for i in range(1, points + 1):
        t = (2.0 * math.pi * i) / points
        x_next = width * math.sin(t)
        y_next = depth * math.sin(2*t) / 2.0
        z_next = 1.5 + height * math.cos(t)
        seg_dist = math.sqrt((x_next-x_curr)**2 + (y_next-y_curr)**2 + (z_next-z_curr)**2)
        dt = seg_dist / speed if speed > 0 else 0.1
        waypoints.append((x_next, y_next, z_next, 0.0, dt, 0.0))
        x_curr, y_curr, z_curr = x_next, y_next, z_next
    if dwell > 0:
        wp = waypoints[-1]
        waypoints[-1] = (wp[0], wp[1], wp[2], wp[3], wp[4], dwell)
    return waypoints


# ──────────────────────────────────────────────────────────────────────────────
# Node Class
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryNode(Node):
    PUBLISH_HZ         = 100.0   # Hz  — waypoint tick rate
    POSITION_THRESHOLD = 0.05    # m   — proximity-based advancement (unused here)

    # MPC compatibility settings
    MPC_DT     = 0.05            # s   — must match MPC.DT
    N_HORIZON  = 21              # poses in /reference_path  (N+1 = 20+1)

    def __init__(self, waypoints):
        super().__init__('trajectory_node')

        if not waypoints:
            raise ValueError('Waypoints list cannot be empty')

        self._waypoints  = waypoints
        self._wp_idx     = 0
        self._start_time = self.get_clock().now().nanoseconds * 1e-9
        self._dwell_start: float | None = None
        self._start_pose = self._waypoints[0][:4]

        # Current drone pose (from odom)
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0

        # How many waypoint indices correspond to one MPC step
        # PUBLISH_HZ=100 → dt=0.01s, MPC_DT=0.05s → step=5
        self._lookahead_step = max(1, round(self.MPC_DT * self.PUBLISH_HZ))

        # ROS interfaces
        self._pub_pose = self.create_publisher(PoseStamped, '/target_pose',    10)
        self._pub_path = self.create_publisher(Path,        '/reference_path',  1)
        self._pub_lap  = self.create_publisher(Bool,        '/trajectory_lap_complete', 10)  # ← ADDED
        self._sub      = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        dt = 1.0 / self.PUBLISH_HZ
        self.create_timer(dt, self._timer_cb)

        self.get_logger().info(
            f'TrajectoryNode ready | {len(self._waypoints)} waypoints | '
            f'lookahead_step={self._lookahead_step} '
            f'(MPC_DT={self.MPC_DT}s, {self.N_HORIZON} path poses)')

    # ─────────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        self._pos_z = msg.pose.pose.position.z

    # ─────────────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        wp_data    = self._waypoints[self._wp_idx]
        tx, ty, tz, tyaw = wp_data[:4]
        dwell_time = wp_data[5] if len(wp_data) > 5 else 0.0

        # ── Publish current target pose (unchanged from original) ─────────────
        self._pub_pose.publish(self._make_pose_stamped(tx, ty, tz, tyaw))

        # ── Publish MPC-compatible lookahead path ─────────────────────────────
        self._pub_path.publish(self._build_path())

        # ── Waypoint advancement logic (unchanged from original) ──────────────
        if dwell_time <= 0:
            self._advance_waypoint(tx, ty, tz, tyaw, now)
        else:
            if self._dwell_start is None:
                self._dwell_start = now
                self.get_logger().info(
                    f'Reached WP {self._wp_idx + 1}. Dwelling {dwell_time:.1f}s...')
            elif (now - self._dwell_start) >= dwell_time:
                self._advance_waypoint(tx, ty, tz, tyaw, now)

    # ─────────────────────────────────────────────────────────────────────────

    def _build_path(self) -> Path:
        """
        Build nav_msgs/Path with N_HORIZON poses for the MPC.

        Each pose is spaced LOOKAHEAD_STEP waypoint-indices ahead of the
        previous one, so consecutive poses are exactly MPC_DT seconds apart.

        When the lookahead reaches the end of a non-looping trajectory it
        clamps to the last waypoint, giving the MPC a zero-velocity tail
        (i.e. "hover at final waypoint").

        For looping trajectories the index wraps modulo len(waypoints).
        """
        stamp    = self.get_clock().now().to_msg()
        n_wp     = len(self._waypoints)
        step     = self._lookahead_step
        path_msg = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = 'odom'

        for k in range(self.N_HORIZON):
            # Wrap index so the path loops continuously with the trajectory
            idx    = (self._wp_idx + k * step) % n_wp
            wp     = self._waypoints[idx]
            x, y, z, yaw = wp[0], wp[1], wp[2], wp[3]
            path_msg.poses.append(self._make_pose_stamped(x, y, z, yaw, stamp))

        return path_msg

    # ─────────────────────────────────────────────────────────────────────────

    def _advance_waypoint(self, tx, ty, tz, tyaw, now):
        # ── ADDED: detect lap completion when index wraps to 0 ────────────────
        next_idx = (self._wp_idx + 1) % len(self._waypoints)
        if next_idx == 0:
            lap_msg = Bool()
            lap_msg.data = True
            self._pub_lap.publish(lap_msg)
            self.get_logger().info(
                'Trajectory lap complete — /trajectory_lap_complete published.')
        # ── END ADDED ──────────────────────────────────────────────────────────

        self._wp_idx      = next_idx          # was: (self._wp_idx + 1) % len(...)
        self._start_time  = now
        self._start_pose  = (tx, ty, tz, tyaw)
        self._dwell_start = None
        self._log_current_segment()

    def _log_current_segment(self):
        wp = self._waypoints[self._wp_idx]
        self.get_logger().info(
            f'WP {self._wp_idx + 1}/{len(self._waypoints)}: '
            f'({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}) yaw={wp[3]:.3f}rad')

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_pose_stamped(x, y, z, yaw, stamp=None) -> PoseStamped:
        """Build a PoseStamped from (x, y, z, yaw)."""
        from rclpy.clock import Clock
        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        if stamp is not None:
            msg.header.stamp = stamp
        msg.pose.position.x  = float(x)
        msg.pose.position.y  = float(y)
        msg.pose.position.z  = float(z)
        half_yaw = yaw * 0.5
        msg.pose.orientation.w = math.cos(half_yaw)
        msg.pose.orientation.z = math.sin(half_yaw)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        return msg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    # ── SELECT TRAJECTORY ────────────────────────────────────────────────────
    MODE       = 'HELIX'
    SPEED      = 0.75
    ROUND_TRIP = True
    ALIGN_YAW  = False
    # ─────────────────────────────────────────────────────────────────────────

    trajectories = {
        'HOVER':     get_hover_trajectory(z=2.0, dwell=3.0),
        'VERTICAL':  get_vertical_trajectory(z_min=1.0, z_max=4.0, speed=SPEED, dwell=2.0),
        'CIRCLE':    get_circle_trajectory(radius=1.5, y=0.0, speed=SPEED, dwell=0.0),
        'XZ_SQUARE': get_plane_trajectory(size=2.0, speed=SPEED, dwell=1.0),
        'SINE':      get_sine_wave_trajectory(amplitude=0.5, freq=0.1, num_waves=1, speed=SPEED, z_base=2.0),
        'LIN_3D':    get_linear_trajectory((0.0, 0.0, 2.0), (3.0, 3.0, 3.0), speed=SPEED, dwell=2.0),
        'HELIX':     get_helix_trajectory(radius=1.5, height=2.0, turns=2, speed=SPEED, z_base=2.0, dwell=2.0),
        'SPIRAL':    get_spiral_trajectory(radius_start=2.0, height=2.0, turns=3, speed=SPEED, z_base=2.0, dwell=2.0),
        'FIG_8':     get_figure_eight_3d(width=3.0, height=1.0, depth=1.0, speed=SPEED),
        'FIG_8_2D':  get_figure_eight_2d_trajectory(width=3.0, height=3.0, speed=SPEED, z_base=2.0),
    }

    if MODE not in trajectories:
        print(f"Error: Mode '{MODE}' not found. Options: {list(trajectories.keys())}")
        rclpy.shutdown()
        return

    waypoints = trajectories[MODE]

    if ROUND_TRIP:
        waypoints = make_round_trip(waypoints)
    if ALIGN_YAW:
        waypoints = add_yaw_alignment(waypoints, is_round_trip=ROUND_TRIP)

    node = TrajectoryNode(waypoints)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()