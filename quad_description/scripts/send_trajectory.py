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

def add_yaw_alignment(waypoints, is_round_trip=False):
    """
    Calculates the yaw angle based on the direction of movement in the X-Y plane
    and updates the waypoints with this aligned yaw.
    If is_round_trip is True, the drone keeps the yaw from the end of the 
    forward path and flies backwards.
    """
    if not waypoints or len(waypoints) < 2:
        return waypoints
        
    aligned_waypoints = []
    
    # If round trip, the second half is the reverse path
    forward_len = len(waypoints) // 2 + 1 if is_round_trip else len(waypoints)
    
    for i in range(len(waypoints)):
        wp_curr = waypoints[i]
        x_curr, y_curr = wp_curr[0], wp_curr[1]
        
        # If in the backward path of a round trip, just freeze the yaw
        if is_round_trip and i >= forward_len:
            target_yaw = aligned_waypoints[-1][3] if aligned_waypoints else 0.0
            aligned_waypoints.append((wp_curr[0], wp_curr[1], wp_curr[2], target_yaw, wp_curr[4], wp_curr[5]))
            continue
            
        # Find next point to determine direction (or use previous if at the end)
        if i + 1 < forward_len:
            wp_next = waypoints[i + 1]
            dx = wp_next[0] - x_curr
            dy = wp_next[1] - y_curr
        else:
            wp_prev = waypoints[i - 1]
            dx = x_curr - wp_prev[0]
            dy = y_curr - wp_prev[1]
            
        # Calculate yaw if there is a movement in XY plane, otherwise keep 0.0 or previous
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            target_yaw = math.atan2(dy, dx)
        else:
            target_yaw = aligned_waypoints[-1][3] if aligned_waypoints else 0.0
            
        aligned_waypoints.append((wp_curr[0], wp_curr[1], wp_curr[2], target_yaw, wp_curr[4], wp_curr[5]))
        
    return aligned_waypoints

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

def get_vertical_trajectory(z_min=1.0, z_max=3.0, speed=1.0, dwell=2.0, hz=100.0):
    """Moving up and down with target speed, generating points based on desired Hz."""
    dist = abs(z_max - z_min)
    total_duration = dist / speed if speed > 0 else 1.0
    num_points = max(1, int(total_duration * hz))
    # dt = total_duration / num_points
    
    waypoints = []
    # Start point with initial dwell
    waypoints.append((0.0, 0.0, z_min, 0.0, hz, dwell))
    
    # Intermediate points between z_min and z_max
    for i in range(1, num_points + 1):
        z = z_min + (i / num_points) * (z_max - z_min)
        point_dwell = dwell if i == num_points else 0.0
        waypoints.append((0.0, 0.0, z, 0.0, hz, point_dwell))
        
    return waypoints

def get_plane_trajectory(size=1.0, x_offset=0.0, z_start=1.0, speed=1.0, dwell=2.0, hz=100.0):
    """A 2D square path in the X-Z plane with target speed, generating points based on desired Hz."""
    duration = size / speed if speed > 0 else 1.0
    num_points = max(1, int(duration * hz))
    dt = duration / num_points
    
    corners = [
        (x_offset,        0.0, z_start),
        (x_offset + size, 0.0, z_start),
        (x_offset + size, 0.0, z_start + size),
        # (x_offset,        0.0, z_start + size),
        (x_offset,        0.0, z_start),
    ]
    
    waypoints = []
    # Start corner with initial dwell
    waypoints.append((corners[0][0], corners[0][1], corners[0][2], 0.0, dt, dwell))
    
    # Generate points for each of the 4 edges
    for seg_idx in range(len(corners)-1):
        p_start = corners[seg_idx]
        p_end = corners[seg_idx + 1]
        
        for i in range(1, num_points + 1):
            alpha = i / num_points
            x = p_start[0] + alpha * (p_end[0] - p_start[0])
            y = p_start[1] + alpha * (p_end[1] - p_start[1])
            z = p_start[2] + alpha * (p_end[2] - p_start[2])
            
            # Apply dwell at corners
            point_dwell = dwell if i == num_points else 0.0
            waypoints.append((x, y, z, 0.0, dt, point_dwell))
            
    return waypoints

def get_sine_wave_trajectory(amplitude=1.0, freq=0.5, num_waves=1.0, speed=1.0, z_base=1.5, dwell=3.0, hz=100.0):
    """
    Sine wave in the X-Z plane, generating points based on desired Hz.
    'speed' here defines the horizontal velocity (v_x).
    """
    total_duration = num_waves / freq if freq > 0 else 1.0
    num_points = max(1, int(total_duration * hz))
    
    waypoints = []
    # Start point with initial dwell
    waypoints.append((0.0, 0.0, z_base, 0.0, hz, dwell))
    
    for i in range(1, num_points + 1):
        t = (i / num_points) * total_duration
        x = speed * t
        z = z_base + amplitude * math.sin(2.0 * math.pi * freq * t)
        
        point_dwell = dwell if i == num_points else 0.0
        waypoints.append((x, 0.0, z, 0.0, hz, point_dwell))
        
    return waypoints

def get_circle_trajectory(radius=1.0, y=0.0, z_base=3.0, speed=1.0, dwell=0.0, hz=100.0):
    """
    A circle in the X-Z plane with constant speed, generating points based on desired Hz.
    - speed: speed along the circumference [m/s]
    """
    circumference = 2.0 * math.pi * radius
    total_duration = circumference / speed if speed > 0 else 2.0 * math.pi
    num_points = max(1, int(total_duration * hz))
    
    waypoints = []
    # Start point with initial dwell
    waypoints.append((0.0, y, z_base, 0.0, hz, dwell))
    
    for i in range(1, num_points + 1):
        angle = 2.0 * math.pi * (i / num_points)
        x = radius * math.cos(angle) - radius # Start at (0,0) offset
        z = z_base + radius * math.sin(angle)
        # yaw could be tangent to circle, but let's keep 0 for simplicity or face center
        point_dwell = dwell if i == num_points else 0.0
        waypoints.append((x, y, z, 0.0, hz, point_dwell))
        
    return waypoints

def get_linear_trajectory(start_pos, target_pos, speed=1.0, dwell=2.0, hz=100.0):
    """
    Straight line from start_pos to target_pos=(x,y,z), generating points based on desired Hz.
    """
    sx, sy, sz = start_pos
    tx, ty, tz = target_pos
    dist = math.sqrt((tx - sx)**2 + (ty - sy)**2 + (tz - sz)**2)
    total_duration = dist / speed if speed > 0 else 1.0
    num_points = max(1, int(total_duration * hz))
    
    waypoints = []
    # Start point with initial dwell
    waypoints.append((sx, sy, sz, 0.0, hz, dwell))
    
    for i in range(1, num_points + 1):
        alpha = i / num_points
        x = sx + alpha * (tx - sx)
        y = sy + alpha * (ty - sy)
        z = sz + alpha * (tz - sz)
        
        point_dwell = dwell if i == num_points else 0.0
        waypoints.append((x, y, z, 0.0, hz, point_dwell))
        
    return waypoints

def get_helix_trajectory(radius=1.0, height=2.0, turns=2, speed=1.0, z_base=1.0, dwell=0.0, hz=100.0):
    """
    Climbing circular path in 3D, generating points based on desired Hz.
    """
    # Helix arc length L = turns * sqrt((2pi*R)^2 + (H/turns)^2)
    circumference = 2.0 * math.pi * radius
    h_per_turn = height / turns
    total_distance = turns * math.sqrt(circumference**2 + h_per_turn**2)
    total_duration = total_distance / speed if speed > 0 else 5.0
    num_points = max(1, int(total_duration * hz))
    
    waypoints = []
    # Start point
    waypoints.append((0.0, 0.0, z_base, 0.0, hz, dwell))
    
    for i in range(1, num_points + 1):
        progress = i / num_points
        angle = progress * turns * 2.0 * math.pi
        
        # x = radius * math.cos(angle) - radius
        # y = radius * math.sin(angle)
        x = radius * math.sin(angle)
        y = radius - radius * math.cos(angle)
        z = z_base + progress * height
        
        point_dwell = dwell if i == num_points else 0.0
        waypoints.append((x, y, z, 0.0, hz, point_dwell))
        
    return waypoints

def get_spiral_trajectory(radius_start=2.0, height=2.0, turns=3, speed=1.0, z_base=1.0, dwell=0.0, hz=100.0):
    """
    Decreasing radius helix (cone spiral) toward center, generating points based on desired Hz.
    """
    # Approximate spiral arc length using average radius
    avg_radius = radius_start / 2.0
    approx_circumference = 2.0 * math.pi * avg_radius
    h_per_turn = height / turns
    approx_distance = turns * math.sqrt(approx_circumference**2 + h_per_turn**2)
    total_duration = approx_distance / speed if speed > 0 else 5.0
    num_points = max(1, int(total_duration * hz))
    
    waypoints = []
    # Start point
    waypoints.append((0.0, 0.0, z_base, 0.0, hz, dwell))
    
    for i in range(1, num_points + 1):
        progress = i / num_points
        ratio = 1.0 - progress # From 1.0 to 0.0
        radius = radius_start * ratio
        angle = progress * turns * 2.0 * math.pi
        
        # Start at (0,0) offset initially
        # x = radius * math.cos(angle) - radius_start
        # y = radius * math.sin(angle)
        x = radius * math.sin(angle)
        y = radius_start - radius * math.cos(angle)
        z = z_base + progress * height
        
        point_dwell = dwell if i == num_points else 0.0
        waypoints.append((x, y, z, 0.0, hz, point_dwell))
        
    return waypoints

def get_figure_eight_2d_trajectory(width=2.0, height=2.0, speed=1.0, z_base=3.0, dwell=0.0, hz=100.0):
    """
    2D Lissajous figure (Figure-8) in the X-Z plane.
    """
    approx_perimeter = math.pi * (width + height)
    total_duration = approx_perimeter / speed if speed > 0 else 5.0
    num_points = max(1, int(total_duration * hz))
    
    waypoints = []
    # Start point
    waypoints.append((0.0, 0.0, z_base, 0.0, hz, dwell))
    
    for i in range(1, num_points + 1):
        t = 2.0 * math.pi * (i / num_points)
        x_next = width * math.sin(t)
        # Shift Z up from base using sin(2t)
        z_next = z_base + (height / 2.0) * math.sin(2.0 * t)
        
        point_dwell = dwell if i == num_points else 0.0
        waypoints.append((x_next, 0.0, z_next, 0.0, hz, point_dwell))
        
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
        tx, ty, tz, tyaw = wp_data[:4]
        dwell_time = wp_data[5] if len(wp_data) > 5 else 0.0

        # Publish exactly the waypoint target for this timestep
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(tx)
        msg.pose.position.y = float(ty)
        msg.pose.position.z = float(tz)
        
        half_yaw = tyaw * 0.5
        msg.pose.orientation.w = math.cos(half_yaw)
        msg.pose.orientation.z = math.sin(half_yaw)
        self._pub.publish(msg)

        # Advance logic every timestep
        if dwell_time <= 0:
            self._advance_waypoint(tx, ty, tz, tyaw, now)
        else:
            # Handle dwell time if specified for this waypoint
            if self._dwell_start is None:
                self._dwell_start = now
                self.get_logger().info(f'Reached WP {self._wp_idx + 1}. Dwelling for {dwell_time}s...')
            elif (now - self._dwell_start) >= dwell_time:
                self._advance_waypoint(tx, ty, tz, tyaw, now)

    def _advance_waypoint(self, tx, ty, tz, tyaw, now):
        self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)
        self._start_time = now
        self._start_pose = (tx, ty, tz, tyaw)
        self._dwell_start = None
        self._log_current_segment()

    def _log_current_segment(self):
        wp = self._waypoints[self._wp_idx]
        self.get_logger().info(f'WP {self._wp_idx + 1}/{len(self._waypoints)}: target=({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}) yaw={wp[3]:.3f}rad')

# ──────────────────────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    # ──── EASY SELECTION ──────────────────────────────────────────────────────
    # Choose your trajectory mode: 'HOVER', 'VERTICAL', 'CIRCLE', 'XZ_SQUARE', 'SINE'
    # ──────────────────────────────────────────────────────────────────────────
    MODE = 'FIG_8_2D'
    SPEED = 0.75      # m/s
    ROUND_TRIP = False # Forward and then back along the same path
    ALIGN_YAW = False  # Check and adjust yaw toward the direction of travel
    
    trajectories = {
        'HOVER':     get_hover_trajectory(z=3.0, dwell=3.0),
        'VERTICAL':  get_vertical_trajectory(z_min=1.0, z_max=4.0, speed=SPEED, dwell=2.0),
        'CIRCLE':    get_circle_trajectory(radius=1.5, y=0.0, speed=SPEED, dwell=0.0, hz=100.0),
        'XZ_SQUARE': get_plane_trajectory(size=2.0, speed=SPEED, dwell=1.0),
        'SINE':      get_sine_wave_trajectory(amplitude=0.5, freq=0.1, num_waves=1, speed=SPEED, z_base=1.0),
        'LIN_3D':    get_linear_trajectory(start_pos=(0.0, 0.0, 3.0), target_pos=(3.0, 3.0, 4.0), speed=SPEED, dwell=2.0, hz=100.0),
        'HELIX':     get_helix_trajectory(radius=1.5, height=3.0, turns=2, speed=SPEED, z_base=3.0, dwell=2.0, hz=100.0),
        'SPIRAL':    get_spiral_trajectory(radius_start=2.0, height=3.0, turns=4, speed=SPEED, z_base=3.0, dwell=2.0, hz=100.0),
        'FIG_8':     get_figure_eight_3d(width=3.0, height=1.0, depth=1.0, speed=SPEED),
        'FIG_8_2D':  get_figure_eight_2d_trajectory(width=3.0, height=3.0, speed=SPEED, z_base=3.0, hz=100.0),
    }

    try:
        if MODE not in trajectories:
            print(f"Error: Mode '{MODE}' not found.")
            return

        waypoints = trajectories[MODE]
        
        if ROUND_TRIP:
            waypoints = make_round_trip(waypoints)
            
        if ALIGN_YAW:
            waypoints = add_yaw_alignment(waypoints, is_round_trip=ROUND_TRIP)

        node = TrajectoryNode(waypoints)
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
