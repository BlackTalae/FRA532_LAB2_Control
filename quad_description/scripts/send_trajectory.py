#!/usr/bin/env python3
"""
Trajectory Sender Node (Modular Version)
========================================
Publishes a sequence of 3D waypoints (x, y, z, yaw) as a
geometry_msgs/PoseStamped message on the /target_pose topic.

Features:
- Modular trajectory patterns.
- Smooth timed interpolation between waypoints.
- Proximity-based checkpoint advancement.
- Adjustable dwell time at each waypoint.
- Automatic looping.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def make_round_trip(waypoints):
    """
    Takes a list of waypoints and returns a new list that 
    includes the reverse path back to the start.
    """
    if not waypoints:
        return []
    # Reverse path (excluding the very last point to avoid double-dwelling at the turn)
    # We take waypoints from N-1 down to 0
    reverse_path = waypoints[-2::-1]
    return waypoints + reverse_path

# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Library
# ──────────────────────────────────────────────────────────────────────────────

def get_hover_trajectory(z=1.0, dwell=2.0):
    """Simple hover at a fixed height (velocity 0)."""
    # Start immediately at height z. Duration 0 to jump or move instantly.
    return [
        (0.0, 0.0, z, 0.0, 0.0, dwell)
    ]

def get_vertical_trajectory(z_min=1.0, z_max=3.0, speed=1.0, dwell=2.0):
    """Moving up and down with target speed."""
    dist = abs(z_max - z_min)
    duration = dist / speed if speed > 0 else 1.0
    return [
        (0.0, 0.0, z_min, 0.0, duration, dwell),
        (0.0, 0.0, z_max, 0.0, duration, dwell),
    ]

def get_xz_square_trajectory(size=1.0, x_offset=0.0, z_start=1.0, speed=1.0, dwell=2.0):
    """A 2D square path in the X-Z plane with target speed."""
    duration = size / speed if speed > 0 else 1.0
    return [
        (x_offset,        0.0, z_start,        0.0, duration, dwell),
        (x_offset + size, 0.0, z_start,        0.0, duration, dwell),
        (x_offset + size, 0.0, z_start + size, 0.0, duration, dwell),
        (x_offset,        0.0, z_start + size, 0.0, duration, dwell),
        (x_offset,        0.0, z_start,        0.0, duration, dwell),
    ]

def get_sine_wave_trajectory(amplitude=1.0, freq=0.5, num_waves=1.0, speed=1.0, z_base=1.5, points_per_wave=50):
    """
    Constant speed sine wave in the X-Z plane.
    Instead of v_x, we use path speed (arc length).
    """
    period = 1.0 / freq
    total_points = int(num_waves * points_per_wave)
    # We estimate v_x by assuming sine path is roughly longer than linear path.
    # To get precise constant path speed, we calculate segment by segment.
    
    # Pre-calculate points to find segment lengths 
    # (Simplified: assume constant horizontal dx for now, but adjust dt)
    # For high fidelity, we'd need to solve for dt at each step, but 
    # for a smooth sine, constant dx is usually acceptable if speed is high.
    # Actually, let's do constant dx and set dt based on segment length.
    
    total_x = num_waves * 4.0 # Dummy width, let's assume 1 cycle = distance 4m or handle it differently
    # Let's use wavelength = period * speed_approx
    wavelength = speed / freq # Rough wavelength
    
    waypoints = []
    x_curr = 0.0
    z_curr = z_base
    
    for i in range(1, total_points + 1):
        x_next = (i / total_points) * (num_waves * wavelength)
        z_next = z_base + amplitude * math.sin(2.0 * math.pi * freq * (x_next / wavelength * period))
        
        seg_dist = math.sqrt((x_next - x_curr)**2 + (z_next - z_curr)**2)
        dt = seg_dist / speed if speed > 0 else 0.1
        
        waypoints.append((x_next, 0.0, z_next, 0.0, dt, 0.0))
        x_curr, z_curr = x_next, z_next
        
    return waypoints

def get_circle_trajectory(radius=1.0, y=0.0, points=100, speed=1.0, dwell=0.0):
    """
    A circle in the X-Z plane with constant speed.
    - speed: speed along the circumference [m/s]
    """
    circumference = 2.0 * math.pi * radius
    total_duration = circumference / speed if speed > 0 else 2.0 * math.pi
    dt = total_duration / points
    
    waypoints = []
    for i in range(1, points + 1):
        angle = 2.0 * math.pi * i / points
        x = radius * math.cos(angle) - radius # Start at (0,0) offset
        z = radius * math.sin(angle)
        # yaw could be tangent to circle, but let's keep 0 for simplicity or face center
        waypoints.append((x, y, z, 0.0, dt, dwell))
        
    return waypoints

def get_3d_linear_trajectory(target_pos, speed=1.0, dwell=2.0):
    """
    Straight line from (0,0,0) to target_pos=(x,y,z).
    """
    tx, ty, tz = target_pos
    dist = math.sqrt(tx**2 + ty**2 + tz**2)
    duration = dist / speed if speed > 0 else 1.0
    return [
        (tx, ty, tz, 0.0, duration, dwell)
    ]

def get_helix_trajectory(radius=1.0, height=2.0, turns=2, speed=1.0, points_per_turn=50, dwell=0.0):
    """
    Climbing circular path in 3D.
    """
    total_points = turns * points_per_turn
    # Helix arc length L = turns * sqrt((2pi*R)^2 + (H/turns)^2)
    circumference = 2.0 * math.pi * radius
    h_per_turn = height / turns
    segment_length = math.sqrt(circumference**2 + h_per_turn**2) / points_per_turn
    dt = segment_length / speed if speed > 0 else 0.1
    
    waypoints = []
    for i in range(1, total_points + 1):
        angle = (2.0 * math.pi * i) / points_per_turn
        x = radius * math.cos(angle) - radius
        y = radius * math.sin(angle)
        z = (i / total_points) * height + 1.0 # Start at Z=1.0
        waypoints.append((x, y, z, 0.0, dt, 0.0))
    
    # Final point gets the dwell time
    if dwell > 0:
        waypoints[-1] = (waypoints[-1][0], waypoints[-1][1], waypoints[-1][2], waypoints[-1][3], waypoints[-1][4], dwell)
        
    return waypoints

def get_spiral_trajectory(radius_start=2.0, height=2.0, turns=3, speed=1.0, points_per_turn=50, dwell=0.0):
    """
    Decreasing radius helix (cone spiral) toward center.
    """
    total_points = turns * points_per_turn
    waypoints = []
    x_curr, y_curr, z_curr = 0.0, 0.0, 1.0 # Initial relative assumed point
    
    for i in range(1, total_points + 1):
        ratio = 1.0 - (i / total_points) # From 1.0 to 0.0
        radius = radius_start * ratio
        angle = (2.0 * math.pi * i) / points_per_turn
        
        # Start at (0,0) offset initially
        x_next = radius * math.cos(angle) - radius_start
        y_next = radius * math.sin(angle)
        z_next = (i / total_points) * height + 1.0
        
        seg_dist = math.sqrt((x_next - x_curr)**2 + (y_next - y_curr)**2 + (z_next - z_curr)**2)
        dt = seg_dist / speed if speed > 0 else 0.1
        
        waypoints.append((x_next, y_next, z_next, 0.0, dt, 0.0))
        x_curr, y_curr, z_curr = x_next, y_next, z_next

    if dwell > 0:
        waypoints[-1] = (waypoints[-1][0], waypoints[-1][1], waypoints[-1][2], waypoints[-1][3], waypoints[-1][4], dwell)
        
    return waypoints

def get_figure_eight_3d(width=2.0, height=1.0, depth=1.0, speed=1.0, points=100, dwell=0.0):
    """
    3D Lissajous figure (Butterfly/Figure-8).
    """
    waypoints = []
    x_curr, y_curr, z_curr = 0.0, 0.0, 1.5
    for i in range(1, points + 1):
        t = (2.0 * math.pi * i) / points
        x_next = width * math.sin(t)
        y_next = depth * math.sin(2*t) / 2.0
        z_next = 1.5 + height * math.cos(t)
        
        seg_dist = math.sqrt((x_next - x_curr)**2 + (y_next - y_curr)**2 + (z_next - z_curr)**2)
        dt = seg_dist / speed if speed > 0 else 0.1
        
        waypoints.append((x_next, y_next, z_next, 0.0, dt, 0.0))
        x_curr, y_curr, z_curr = x_next, y_next, z_next

    if dwell > 0:
        waypoints[-1] = (waypoints[-1][0], waypoints[-1][1], waypoints[-1][2], waypoints[-1][3], waypoints[-1][4], dwell)
        
    return waypoints

# ──────────────────────────────────────────────────────────────────────────────
# Node Class
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryNode(Node):
    PUBLISH_HZ         = 100.0   # Hz
    POSITION_THRESHOLD = 0.05    # m 

    def __init__(self, waypoints):
        super().__init__('trajectory_node')

        if not waypoints:
            self.get_logger().error("No waypoints provided!")
            raise ValueError("Waypoints list cannot be empty")

        self._waypoints  = waypoints
        self._wp_idx     = 0
        self._start_time = self.get_clock().now().nanoseconds * 1e-9
        self._dwell_start: float | None = None
        
        # Track start of current segment for interpolation
        self._start_pose = self._waypoints[0][:4] 

        # Current drone pose (from odom)
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0

        # ROS interfaces
        self._pub  = self.create_publisher(PoseStamped, '/target_pose', 10)
        self._sub  = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        dt = 1.0 / self.PUBLISH_HZ
        self.create_timer(dt, self._timer_cb)

        self.get_logger().info(f'Trajectory Node initialized with {len(self._waypoints)} waypoints.')
        self._log_current_segment()

    def _odom_cb(self, msg: Odometry):
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        self._pos_z = msg.pose.pose.position.z

    def _timer_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        
        wp_data = self._waypoints[self._wp_idx]
        # Support both 5-tuple and 6-tuple for backward compatibility if needed, 
        # but library functions now return 6-tuple.
        tx, ty, tz, tyaw, duration = wp_data[:5]
        dwell_time = wp_data[5] if len(wp_data) > 5 else 0.0

        elapsed = now - self._start_time
        alpha = (elapsed / duration) if duration > 0 else 1.0
        alpha_clamped = min(max(alpha, 0.0), 1.0)

        # Interpolate setpoint
        curr_x = (1 - alpha_clamped) * self._start_pose[0] + alpha_clamped * tx
        curr_y = (1 - alpha_clamped) * self._start_pose[1] + alpha_clamped * ty
        curr_z = (1 - alpha_clamped) * self._start_pose[2] + alpha_clamped * tz
        curr_yaw = (1 - alpha_clamped) * self._start_pose[3] + alpha_clamped * tyaw

        # Proximity check
        dist = math.sqrt((self._pos_x - tx)**2 + (self._pos_y - ty)**2 + (self._pos_z - tz)**2)

        # Segment advancement logic
        if alpha >= 1.0 and dist < self.POSITION_THRESHOLD:
            if dwell_time <= 0:
                # Advance immediately for continuous movement
                self._advance_waypoint(tx, ty, tz, tyaw, now)
            else:
                # Handle dwell time
                if self._dwell_start is None:
                    self._dwell_start = now
                    self.get_logger().info(f'Reached WP {self._wp_idx + 1}. Dwelling for {dwell_time}s...')
                elif (now - self._dwell_start) >= dwell_time:
                    self._advance_waypoint(tx, ty, tz, tyaw, now)
        else:
            # Not yet at the waypoint or still interpolating
            pass

        # Publish
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(curr_x)
        msg.pose.position.y = float(curr_y)
        msg.pose.position.z = float(curr_z)
        
        half_yaw = curr_yaw * 0.5
        msg.pose.orientation.w = math.cos(half_yaw)
        msg.pose.orientation.z = math.sin(half_yaw)
        self._pub.publish(msg)

    def _advance_waypoint(self, tx, ty, tz, tyaw, now):
        self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)
        self._start_time = now
        self._start_pose = (tx, ty, tz, tyaw)
        self._dwell_start = None
        self._log_current_segment()

    def _log_current_segment(self):
        wp = self._waypoints[self._wp_idx]
        self.get_logger().info(f'WP {self._wp_idx + 1}/{len(self._waypoints)}: target=({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})')

# ──────────────────────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    # ──── EASY SELECTION ──────────────────────────────────────────────────────
    # Choose your trajectory mode: 'HOVER', 'VERTICAL', 'CIRCLE', 'XZ_SQUARE', 'SINE'
    # ──────────────────────────────────────────────────────────────────────────
    MODE = 'HELIX'
    SPEED = 5.0      # m/s
    ROUND_TRIP = True # Forward and then back along the same path
    
    trajectories = {
        'HOVER':     get_hover_trajectory(z=1.0, dwell=3.0),
        'VERTICAL':  get_vertical_trajectory(z_min=1.0, z_max=4.0, speed=SPEED, dwell=1.0),
        'CIRCLE':    get_circle_trajectory(radius=1.5, y=0.0, speed=SPEED, points=100),
        'XZ_SQUARE': get_xz_square_trajectory(size=2.0, speed=SPEED, dwell=1.0),
        'SINE':      get_sine_wave_trajectory(amplitude=1.0, freq=2.0, num_waves=1, speed=SPEED, z_base=2.0),
        'LIN_3D':    get_3d_linear_trajectory(target_pos=(3.0, 3.0, 4.0), speed=SPEED),
        'HELIX':     get_helix_trajectory(radius=1.5, height=3.0, turns=2, speed=SPEED),
        'SPIRAL':    get_spiral_trajectory(radius_start=2.0, height=3.0, turns=4, speed=SPEED),
        'FIG_8':     get_figure_eight_3d(width=3.0, height=1.5, depth=2.0, speed=SPEED),
        'CUSTOM':    [(0.0, 0.0, 1.0, 0.0, 1.0, 1.0)]
    }

    try:
        if MODE not in trajectories:
            print(f"Error: Mode '{MODE}' not found.")
            return

        waypoints = trajectories[MODE]
        
        if ROUND_TRIP:
            waypoints = make_round_trip(waypoints)

        node = TrajectoryNode(waypoints)
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
