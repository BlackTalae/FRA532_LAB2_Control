#!/usr/bin/env python3
"""
Quadrotor PID Hover Controller
================================
Controls a quadrotor to hold a fixed position (x, y, z) using PID controllers.
Subscribes to: /odom  (nav_msgs/msg/Odometry)
Publishes to:  /motor_commands  (actuator_msgs/msg/Actuators)

Physical parameters (from robot_params.xacro):
  mass    = 1.5 kg
  k_F     = 8.54858e-06  [N / (rad/s)^2]
  k_M     = 0.06
  omega_max = 1500 rad/s

Hover motor speed:
  4 * k_F * omega_h^2 = m * g
  omega_h = sqrt(m*g / (4*k_F)) ≈ 656 rad/s

Motor layout (top view, X-config):
        front (+x)
          2   0
      left     right
          1   3
        rear (-x)

  Motor 0: front-right (+x, -y)  CCW
  Motor 1: rear-left   (-x, +y)  CCW
  Motor 2: front-left  (+x, +y)  CW
  Motor 3: rear-right  (-x, -y)  CW

Mixing (in omega space, working with motor speed directly):
  w0 = base + dz + pitch - roll + yaw   (front-right, CCW)
  w1 = base + dz - pitch + roll + yaw   (rear-left,  CCW)
  w2 = base + dz + pitch + roll - yaw   (front-left,  CW)
  w3 = base + dz - pitch - roll - yaw   (rear-right,  CW)

  where:
    dz    = Z-PID output      [rad/s]  — altitude control
    pitch = X-PID output      [rad/s]  — front/rear differential → pitches drone
    roll  = Y-PID output      [rad/s]  — left/right differential  → rolls drone
    yaw   = Yaw-PID output    [rad/s]  — CW vs CCW differential

Tune order: Kp_z first (hover), then Kp_x/y, then Kd terms, finally Ki.
"""

import math
import threading
from collections import deque

import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg not available
import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators


# ---------------------------------------------
# PID controller Class
# ---------------------------------------------
class PID:
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

        if dt <= 1e-6:
            return 0.0

        self._integral += error * dt
        self._integral = max(-self.windup_limit, min(self.windup_limit, self._integral)) # clamp integral

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        return max(self.out_min, min(self.out_max, output)) # clamp output


# ---------------------------------------------
#  ROS 2 Node
# ---------------------------------------------
class QuadrotorPIDNode(Node):

    # --- Physical constants -------------------
    MASS       = 1.5          # kg
    GRAVITY    = 9.81         # m/s²
    K_F        = 8.54858e-06  # N / (rad/s)²
    OMEGA_MAX  = 1500.0       # rad/s

    # --- Target pose (setpoint) -- edit here to change hover goal ---
    TARGET_X   = 0.0   # m
    TARGET_Y   = 0.0   # m
    TARGET_Z   = 1.0   # m
    TARGET_YAW = 0.0   # rad

    # --- History length for the live plot ---
    HISTORY_LEN = 600   # samples at 100 Hz → 6 s window

    def __init__(self):
        super().__init__('quadrotor_pid_node')

        # Hover motor speed (rad/s)
        self.omega_hover = math.sqrt(self.MASS * self.GRAVITY / (4.0 * self.K_F))
        self.get_logger().info(f'Hover motor speed: {self.omega_hover:.1f} rad/s')

        # --- PID controllers -------------------
        self.pid_z = PID(kp=50.0, ki=2.0, kd=10.0,
                         out_min=-400.0, out_max=400.0,
                         windup_limit=300.0)

        self.pid_x = PID(kp=30.0, ki=2.0, kd=40.0,
                         out_min=-150.0, out_max=150.0,
                         windup_limit=100.0)

        self.pid_y = PID(kp=30.0, ki=2.0, kd=40.0,
                         out_min=-150.0, out_max=150.0,
                         windup_limit=100.0)

        self.pid_yaw = PID(kp=50.0, ki=0.0, kd=30.0,
                           out_min=-100.0, out_max=100.0)

        # --- State -------------------------------
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self._odom_received = False
        self._last_time: float | None = None
        self._t0: float | None = None

        # --- History deques for plotting ----------
        n = self.HISTORY_LEN
        self.t_hist   = deque(maxlen=n)
        self.x_hist   = deque(maxlen=n);  self.tx_hist = deque(maxlen=n)
        self.y_hist   = deque(maxlen=n);  self.ty_hist = deque(maxlen=n)
        self.z_hist   = deque(maxlen=n);  self.tz_hist = deque(maxlen=n)

        # --- ROS interfaces ----------------------
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.cmd_pub  = self.create_publisher(
            Actuators, '/motor_commands', 10)

        # Control loop at 100 Hz
        self.create_timer(0.01, self._control_loop)

        # Live plot in a background thread
        self._plot_lock = threading.Lock()
        threading.Thread(target=self._plot_loop, daemon=True).start()

        self.get_logger().info(
            f'PID node ready. Target → x={self.TARGET_X}, '
            f'y={self.TARGET_Y}, z={self.TARGET_Z} m')


# --- Entry point -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorPIDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()   # keep final plot open after Ctrl+C


if __name__ == '__main__':
    main()
