#!/usr/bin/env python3
"""
trajectory_sender.py
====================
Publishes waypoint trajectories as:
  /target_pose      (geometry_msgs/PoseStamped)
  /reference_path   (nav_msgs/Path)
  /trajectory_started (std_msgs/Bool)
  /trajectory_lap_complete (std_msgs/Bool)

Design choices
--------------
- Trajectory is represented as dense sampled points at PUBLISH_HZ.
- No "start dwell" is inserted. The trajectory starts immediately.
- Optional terminal dwell is encoded by repeating the final waypoint samples.
- Near the end of the path, /reference_path is CLAMPED to the final waypoint
  instead of wrapping early. This avoids preview discontinuity and makes the
  terminal hold smooth.
- Actual looping happens only when the current index wraps from the last sample
  back to 0, at which point /trajectory_lap_complete is published.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def add_yaw_alignment(points, is_round_trip=False):
    """
    Recompute yaw so that each sample faces the direction of motion in x-y.
    Input/output points are 4-tuples: (x, y, z, yaw).
    """
    if not points or len(points) < 2:
        return list(points)

    aligned = []
    n = len(points)

    for i in range(n):
        x, y, z, _ = points[i]

        if i < n - 1:
            xn, yn, _, _ = points[i + 1]
            dx = xn - x
            dy = yn - y
        else:
            xp, yp, _, _ = points[i - 1]
            dx = x - xp
            dy = y - yp

        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            yaw = math.atan2(dy, dx)
        else:
            yaw = aligned[-1][3] if aligned else 0.0

        aligned.append((x, y, z, yaw))

    return aligned


def make_round_trip(points):
    """
    Example: [A, B, C] -> [A, B, C, B, A]
    """
    if not points:
        return []
    if len(points) == 1:
        return list(points)
    return list(points) + list(points[-2::-1])


def append_hold_segment(points, hold_s, hz):
    """
    Add repeated final-point samples so terminal dwell becomes part of trajectory.
    """
    if not points or hold_s <= 0.0:
        return list(points)

    hold_steps = max(1, int(round(hold_s * hz)))
    last = points[-1]
    return list(points) + [last] * hold_steps


def _num_samples_from_duration(duration_s, hz):
    return max(2, int(round(duration_s * hz)) + 1)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory library
# Each function returns list[(x, y, z, yaw)]
# Dense samples only. No dwell logic inside.
# ──────────────────────────────────────────────────────────────────────────────

def get_hover_trajectory(z=2.0, hold_s=2.0, hz=100.0):
    n = max(2, int(round(hold_s * hz)))
    return [(0.0, 0.0, z, 0.0)] * n


def get_vertical_trajectory(z_min=1.0, z_max=3.0, speed=1.0, hz=100.0):
    dist = abs(z_max - z_min)
    duration = dist / max(speed, 1e-6)
    n = _num_samples_from_duration(duration, hz)

    pts = []
    for i in range(n):
        a = i / (n - 1)
        z = z_min + a * (z_max - z_min)
        pts.append((0.0, 0.0, z, 0.0))
    return pts


def get_plane_trajectory(size=1.0, x_offset=0.0, z_start=1.0, speed=1.0, hz=100.0):
    corners = [
        (x_offset,        0.0, z_start),
        (x_offset + size, 0.0, z_start),
        (x_offset + size, 0.0, z_start + size),
        (x_offset,        0.0, z_start),
    ]

    pts = []
    for seg_idx in range(len(corners) - 1):
        p0 = corners[seg_idx]
        p1 = corners[seg_idx + 1]
        dist = math.dist(p0, p1)
        duration = dist / max(speed, 1e-6)
        n = _num_samples_from_duration(duration, hz)

        for i in range(n):
            if seg_idx > 0 and i == 0:
                continue
            a = i / (n - 1)
            x = p0[0] + a * (p1[0] - p0[0])
            y = p0[1] + a * (p1[1] - p0[1])
            z = p0[2] + a * (p1[2] - p0[2])
            pts.append((x, y, z, 0.0))
    return pts


def get_sine_wave_trajectory(amplitude=1.0, freq=0.5, num_waves=1.0,
                             speed=1.0, z_base=1.5, hz=100.0):
    total_duration = num_waves / max(freq, 1e-6)
    n = _num_samples_from_duration(total_duration, hz)

    pts = []
    for i in range(n):
        t = (i / (n - 1)) * total_duration
        x = speed * t
        z = z_base + amplitude * math.sin(2.0 * math.pi * freq * t)
        pts.append((x, 0.0, z, 0.0))
    return pts


def get_circle_trajectory(radius=1.0, y=0.0, z_base=3.0, speed=1.0, hz=100.0):
    circumference = 2.0 * math.pi * radius
    duration = circumference / max(speed, 1e-6)
    n = _num_samples_from_duration(duration, hz)

    pts = []
    for i in range(n):
        angle = 2.0 * math.pi * (i / (n - 1))
        x = radius * math.cos(angle) - radius
        z = z_base + radius * math.sin(angle)
        pts.append((x, y, z, 0.0))
    return pts


def get_linear_trajectory(start_pos, target_pos, speed=1.0, hz=100.0):
    sx, sy, sz = start_pos
    tx, ty, tz = target_pos
    dist = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2 + (tz - sz) ** 2)
    duration = dist / max(speed, 1e-6)
    n = _num_samples_from_duration(duration, hz)

    pts = []
    for i in range(n):
        a = i / (n - 1)
        x = sx + a * (tx - sx)
        y = sy + a * (ty - sy)
        z = sz + a * (tz - sz)
        pts.append((x, y, z, 0.0))
    return pts


def get_helix_trajectory(radius=1.0, height=2.0, turns=2,
                         speed=1.0, z_base=1.0, hz=100.0):
    circumference = 2.0 * math.pi * radius
    h_per_turn = height / max(turns, 1e-6)
    total_distance = turns * math.sqrt(circumference**2 + h_per_turn**2)
    duration = total_distance / max(speed, 1e-6)
    n = _num_samples_from_duration(duration, hz)

    pts = []
    for i in range(n):
        p = i / (n - 1)
        angle = p * turns * 2.0 * math.pi
        x = radius * math.sin(angle)
        y = radius - radius * math.cos(angle)
        z = z_base + p * height
        pts.append((x, y, z, 0.0))
    return pts


def get_spiral_trajectory(radius_start=2.0, height=2.0, turns=3,
                          speed=1.0, z_base=1.0, hz=100.0):
    avg_radius = radius_start / 2.0
    approx_circumference = 2.0 * math.pi * avg_radius
    h_per_turn = height / max(turns, 1e-6)
    approx_distance = turns * math.sqrt(approx_circumference**2 + h_per_turn**2)
    duration = approx_distance / max(speed, 1e-6)
    n = _num_samples_from_duration(duration, hz)

    pts = []
    for i in range(n):
        p = i / (n - 1)
        radius = radius_start * (1.0 - p)
        angle = p * turns * 2.0 * math.pi
        x = radius * math.sin(angle)
        y = radius_start - radius * math.cos(angle)
        z = z_base + p * height
        pts.append((x, y, z, 0.0))
    return pts


def get_figure_eight_2d_trajectory(width=2.0, height=2.0, speed=1.0,
                                   z_base=3.0, hz=100.0):
    approx_perimeter = math.pi * (width + height)
    duration = approx_perimeter / max(speed, 1e-6)
    n = _num_samples_from_duration(duration, hz)

    pts = []
    for i in range(n):
        t = 2.0 * math.pi * (i / (n - 1))
        x = width * math.sin(t)
        z = z_base + 0.5 * height * math.sin(2.0 * t)
        pts.append((x, 0.0, z, 0.0))
    return pts


def get_figure_eight_3d(width=2.0, height=1.0, depth=1.0,
                        speed=1.0, hz=100.0):
    approx_duration = 8.0 / max(speed, 1e-6)
    n = _num_samples_from_duration(approx_duration, hz)

    pts = []
    for i in range(n):
        t = 2.0 * math.pi * (i / (n - 1))
        x = width * math.sin(t)
        y = 0.5 * depth * math.sin(2.0 * t)
        z = 1.5 + height * math.cos(t)
        pts.append((x, y, z, 0.0))
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryNode(Node):
    PUBLISH_HZ = 100.0
    MPC_DT = 0.01          # match controller DT
    N_HORIZON = 21         # N+1 for controller with N=20

    def __init__(self, waypoints):
        super().__init__('trajectory_node')

        if not waypoints:
            raise ValueError('Waypoints list cannot be empty')

        self._waypoints = list(waypoints)
        self._wp_idx = 0

        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0

        self._lookahead_step = max(1, round(self.MPC_DT * self.PUBLISH_HZ))

        self._pub_pose  = self.create_publisher(PoseStamped, '/target_pose', 10)
        self._pub_path  = self.create_publisher(Path, '/reference_path', 1)
        self._pub_lap   = self.create_publisher(Bool, '/trajectory_lap_complete', 10)
        self._pub_start = self.create_publisher(Bool, '/trajectory_started', 10)
        self._sub_odom  = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self._active = False
        self._started_sent = False

        self._start_srv = self.create_service(
            Trigger,
            '/start_trajectory',
            self._handle_start_trajectory
        )

        dt = 1.0 / self.PUBLISH_HZ
        self.create_timer(dt, self._timer_cb)

        self.get_logger().info(
            f'TrajectoryNode ready | {len(self._waypoints)} samples | '
            f'PUBLISH_HZ={self.PUBLISH_HZ} Hz | '
            f'lookahead_step={self._lookahead_step} | '
            f'MPC_DT={self.MPC_DT}s'
        )

    def _odom_cb(self, msg: Odometry):
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        self._pos_z = msg.pose.pose.position.z

    def _timer_cb(self):
        if not self._active:
            return
        if not self._started_sent:
            self._pub_start.publish(Bool(data=True))
            self._started_sent = True
            self.get_logger().info('Published /trajectory_started')

        x, y, z, yaw = self._waypoints[self._wp_idx]

        self._pub_pose.publish(self._make_pose_stamped(x, y, z, yaw))
        self._pub_path.publish(self._build_path())

        self._advance_waypoint()

    def _build_path(self) -> Path:
        """
        Build a lookahead path for MPC.

        Important:
        - We CLAMP to the final waypoint instead of wrapping early.
        - This makes terminal dwell smooth and prevents preview discontinuity.
        """
        stamp = self.get_clock().now().to_msg()
        n_wp = len(self._waypoints)

        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = 'odom'

        for k in range(self.N_HORIZON):
            idx = min(self._wp_idx + k * self._lookahead_step, n_wp - 1)
            x, y, z, yaw = self._waypoints[idx]
            path_msg.poses.append(self._make_pose_stamped(x, y, z, yaw, stamp))

        return path_msg

    def _advance_waypoint(self):
        if self._wp_idx >= len(self._waypoints) - 1:
            self._wp_idx = 0
            self._pub_lap.publish(Bool(data=True))
            self.get_logger().info('Trajectory lap complete — /trajectory_lap_complete published.')
        else:
            self._wp_idx += 1

    @staticmethod
    def _make_pose_stamped(x, y, z, yaw, stamp=None) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        if stamp is not None:
            msg.header.stamp = stamp

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        half_yaw = 0.5 * yaw
        msg.pose.orientation.w = math.cos(half_yaw)
        msg.pose.orientation.z = math.sin(half_yaw)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        return msg
    
    def _handle_start_trajectory(self, request, response):
        self._active = True
        self._wp_idx = 0
        self._started_sent = False

        response.success = True
        response.message = 'Trajectory started.'
        self.get_logger().info('Start trajectory service called.')
        return response

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    MODE = 'HELIX'
    SPEED = 0.75
    ROUND_TRIP = True
    ALIGN_YAW = False
    END_DWELL = 3.0   # hold only at the very end, not at the start

    trajectories = {
        'HOVER':    get_hover_trajectory(z=2.0, hold_s=3.0, hz=TrajectoryNode.PUBLISH_HZ),
        'VERTICAL': get_vertical_trajectory(z_min=1.0, z_max=4.0, speed=SPEED, hz=TrajectoryNode.PUBLISH_HZ),
        'CIRCLE':   get_circle_trajectory(radius=1.5, y=0.0, z_base=2.0, speed=SPEED, hz=TrajectoryNode.PUBLISH_HZ),
        'XZ_SQUARE': get_plane_trajectory(size=2.0, speed=SPEED, hz=TrajectoryNode.PUBLISH_HZ),
        'SINE':     get_sine_wave_trajectory(amplitude=0.5, freq=0.1, num_waves=1, speed=SPEED, z_base=2.0, hz=TrajectoryNode.PUBLISH_HZ),
        'LIN_3D':   get_linear_trajectory((0.0, 0.0, 2.0), (3.0, 3.0, 3.0), speed=SPEED, hz=TrajectoryNode.PUBLISH_HZ),
        'HELIX':    get_helix_trajectory(radius=1.5, height=2.0, turns=2, speed=SPEED, z_base=2.0, hz=TrajectoryNode.PUBLISH_HZ),
        'SPIRAL':   get_spiral_trajectory(radius_start=2.0, height=2.0, turns=3, speed=SPEED, z_base=2.0, hz=TrajectoryNode.PUBLISH_HZ),
        'FIG_8':    get_figure_eight_3d(width=3.0, height=1.0, depth=1.0, speed=SPEED, hz=TrajectoryNode.PUBLISH_HZ),
        'FIG_8_2D': get_figure_eight_2d_trajectory(width=3.0, height=3.0, speed=SPEED, z_base=2.0, hz=TrajectoryNode.PUBLISH_HZ),
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

    if END_DWELL > 0.0 and MODE != 'HOVER':
        waypoints = append_hold_segment(waypoints, END_DWELL, TrajectoryNode.PUBLISH_HZ)

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