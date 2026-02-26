#!/usr/bin/env python3
"""
Trajectory Sender Node
======================
Publishes a sequence of 3D waypoints (x, y, z, yaw) as a
geometry_msgs/PoseStamped message on the /target_pose topic.

The active controller (LQR, MPC, PID …) subscribes to /target_pose and
updates its setpoint accordingly.

Waypoint advancement:
  The node subscribes to /odom and advances to the next waypoint once the
  drone is within `POSITION_THRESHOLD` metres of the current one AND has
  held that position for `DWELL_TIME` seconds.

Edit the WAYPOINTS list to define your trajectory.
Each entry is (x [m], y [m], z [m], yaw [rad]).

Usage:
  ros2 run quad_description send_trajectory.py

Topics:
  Publishes:   /target_pose  (geometry_msgs/msg/PoseStamped)
  Subscribes:  /odom         (nav_msgs/msg/Odometry)
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint list  –  edit to define your trajectory
# Each entry: (x [m], y [m], z [m], yaw [rad])
# ──────────────────────────────────────────────────────────────────────────────
WAYPOINTS = [
    (0.0, 0.0, 1.0, 0.0),   # take-off / hover
    (2.0, 0.0, 1.0, 0.0),   # fly +X
    (2.0, 2.0, 1.0, 0.0),   # fly +Y
    (0.0, 2.0, 1.5, 0.0),   # fly −X, gain altitude
    (0.0, 0.0, 1.5, 0.0),   # return home (higher)
    (0.0, 0.0, 0.3, 0.0),   # descend
]


class TrajectoryNode(Node):

    # ── Tuning ─────────────────────────────────────────────────────────────
    POSITION_THRESHOLD = 0.15   # m  – distance to waypoint before advancing
    DWELL_TIME         = 1.5    # s  – seconds to hold before advancing
    PUBLISH_HZ         = 20.0   # Hz – how often we republish the target

    def __init__(self):
        super().__init__('trajectory_node')

        self._waypoints  = WAYPOINTS
        self._wp_idx     = 0
        self._dwell_start: float | None = None

        # Current drone pose
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0

        # ── ROS interfaces ────────────────────────────────────────────────
        self._pub  = self.create_publisher(PoseStamped, '/target_pose', 10)
        self._sub  = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)

        dt = 1.0 / self.PUBLISH_HZ
        self.create_timer(dt, self._timer_cb)

        self.get_logger().info(
            f'Trajectory node started.  {len(self._waypoints)} waypoints.')
        self._log_current_wp()

    # ── Odometry callback ───────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        self._pos_z = msg.pose.pose.position.z

    # ── Periodic publisher ──────────────────────────────────────────────────
    def _timer_cb(self):
        wp = self._waypoints[self._wp_idx]
        tx, ty, tz, tyaw = wp

        # ── Check if close enough to current waypoint ───────────────────────
        dist = math.sqrt(
            (self._pos_x - tx) ** 2 +
            (self._pos_y - ty) ** 2 +
            (self._pos_z - tz) ** 2
        )

        now = self.get_clock().now().nanoseconds * 1e-9

        if dist < self.POSITION_THRESHOLD:
            if self._dwell_start is None:
                self._dwell_start = now
            elif (now - self._dwell_start) >= self.DWELL_TIME:
                # Advance to next waypoint
                self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)
                self._dwell_start = None
                self._log_current_wp()
                return   # publish next cycle
        else:
            self._dwell_start = None   # left the proximity zone; reset dwell

        # ── Build and publish PoseStamped ───────────────────────────────────
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(tx)
        msg.pose.position.y = float(ty)
        msg.pose.position.z = float(tz)

        # Convert yaw → quaternion (roll=0, pitch=0, yaw=tyaw)
        half_yaw = tyaw * 0.5
        msg.pose.orientation.w = math.cos(half_yaw)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(half_yaw)

        self._pub.publish(msg)

    # ── Logging helper ──────────────────────────────────────────────────────
    def _log_current_wp(self):
        idx = self._wp_idx
        wp  = self._waypoints[idx]
        self.get_logger().info(
            f'Waypoint {idx + 1}/{len(self._waypoints)}  →  '
            f'x={wp[0]:.2f}  y={wp[1]:.2f}  z={wp[2]:.2f}  '
            f'yaw={math.degrees(wp[3]):.1f}°')


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
