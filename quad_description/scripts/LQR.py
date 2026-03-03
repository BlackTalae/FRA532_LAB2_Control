#!/usr/bin/env python3
"""
Quadrotor LQR Hover Controller
================================
Controls a quadrotor to hold a fixed position (x, y, z) using a Linear
Quadratic Regulator (LQR).

Subscribes to: /odom  (nav_msgs/msg/Odometry)
Publishes to:  /motor_commands  (actuator_msgs/msg/Actuators)

Physical parameters (from robot_params.xacro):
  mass        = 1.5 kg
  k_F         = 8.54858e-06  [N / (rad/s)^2]
  k_M         = 0.06         [N·m / (rad/s)^2]
  Ixx         = 0.0347563    [kg·m²]
  Iyy         = 0.07         [kg·m²]
  Izz         = 0.0977       [kg·m²]
  omega_max   = 1500         rad/s
  arm_length  = ~0.22 m  (from joint origins in URDF)

Linearisation point: hovering at rest (x,y,z) = target, roll=pitch=yaw=0,
all velocities = 0.

State vector  x  (12x1):
  [x, y, z, φ(roll), θ(pitch), ψ(yaw),
   ẋ, ẏ, ż, φ̇, θ̇, ψ̇]

Input vector  u  (4x1):
  [F_total, τ_roll, τ_pitch, τ_yaw]   (thrust + three torques)

The LQR gain matrix K (4x12) maps the state error to inputs:
  u = u_hover - K · (x - x_ref)

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

Motor mixing (force/torque → omega²):
  Given k_F, k_M, arm length L (moment arm), the allocation matrix Γ:

      ┌  k_F    k_F    k_F    k_F  ┐  ┌ w0² ┐   ┌ F_total ┐
      │  k_F·L -k_F·L  k_F·L -k_F·L│  │ w1² │ = │ τ_roll  │
      │  k_F·L  k_F·L -k_F·L -k_F·L│  │ w2² │   │ τ_pitch │
      └ -k_M    k_M   -k_M    k_M  ┘  └ w3² ┘   └ τ_yaw   ┘

  (Exact signs follow from motor positions and rotation directions.)
  We solve: [w0², w1², w2², w3²] = Γ⁻¹ · [F, τ_r, τ_p, τ_y]
  then wi = sqrt(wi²), clamped to [0, omega_max].
"""

import math
import threading
from collections import deque

import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg not available
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped


# ──────────────────────────────────────────────────────────────────────────────
# Helper: LQR gain solver
# ──────────────────────────────────────────────────────────────────────────────

def lqr(A: np.ndarray, B: np.ndarray,
        Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve the continuous-time LQR problem.

    Minimises:  J = ∫ (x'Qx + u'Ru) dt

    Returns the gain matrix K such that u = -K x is optimal.
    Internally solves the Continuous Algebraic Riccati Equation (CARE):
        A'P + PA − PBR⁻¹B'P + Q = 0
    then  K = R⁻¹ B' P.
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


# ──────────────────────────────────────────────────────────────────────────────
# LQR controller wrapper (stateless – the node owns the state)
# ──────────────────────────────────────────────────────────────────────────────

class LQR:
    """
    Thin wrapper around a pre-computed LQR gain matrix.

    Usage:
        controller = LQR(K)
        u_correction = controller.compute(state_error)   # shape (4,)
    """

    def __init__(self, K: np.ndarray):
        """
        Parameters
        ----------
        K : ndarray, shape (4, 12)
            Full-state feedback gain matrix (computed offline via lqr()).
        """
        self.K = K  # (4 × 12)

    def compute(self, error: np.ndarray) -> np.ndarray:
        """
        Return the control *correction* for the given state error.

        Parameters
        ----------
        error : ndarray, shape (12,)
            State error  x_ref - x_current  (positive = below target).

        Returns
        -------
        u_corr : ndarray, shape (4,)
            [dF, dτ_roll, dτ_pitch, dτ_yaw] to add to the hover trim.
        """
        return self.K @ error   # K (4×12) @ error (12,) → (4,)


# ──────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ──────────────────────────────────────────────────────────────────────────────

class QuadrotorLQRNode(Node):

    # ── Physical constants ─────────────────────────────────────────────────
    MASS       = 1.5            # kg
    GRAVITY    = 9.81           # m/s²
    K_F        = 8.54858e-06    # N / (rad/s)²
    K_M        = 0.06           # N·m / (rad/s)²   (torque coefficient ratio k_M/k_F ≈ 0.016 m, but xacro gives 0.06 directly)
    OMEGA_MAX  = 1500.0         # rad/s

    # Moments of inertia (from URDF)
    IXX = 0.0347563   # kg·m²
    IYY = 0.07        # kg·m²
    IZZ = 0.0977      # kg·m²

    # Arm length: average of rotor joint offsets in x and y
    # joint origins: (0.13, ±0.22) and (-0.13, ±0.20) → use 0.22 m
    L = 0.22   # m  (distance from CoM to rotor along body x/y axis)

    # ── Default target pose (overridden live by /target_pose topic) ───────────
    DEFAULT_TARGET_X   = 0.0   # m
    DEFAULT_TARGET_Y   = 0.0   # m
    DEFAULT_TARGET_Z   = 1.0   # m
    DEFAULT_TARGET_YAW = 0.0   # rad

    # ── Plot history ───────────────────────────────────────────────────────
    HISTORY_LEN = 600   # samples @ 100 Hz → 6 s window

    def __init__(self):
        super().__init__('quadrotor_lqr_node')

        # ── Hover trim ──────────────────────────────────────────────────────
        self.F_hover     = self.MASS * self.GRAVITY          # total thrust at hover
        self.omega_hover = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.get_logger().info(f'Hover ω: {self.omega_hover:.1f} rad/s  '
                               f'F_hover: {self.F_hover:.2f} N')

        # ── Motor allocation matrix Γ (4×4)  [w0², w1², w2², w3²] ─────────
        #
        #  Rows: [F_total, τ_roll, τ_pitch, τ_yaw]
        #  Sign conventions (X-config, body frame):
        #    τ_roll  = k_F·L·(w2²+w3²−w0²−w1²)  … NOT used; see below
        #  We follow the same sign pattern as the PID mixer:
        #    w0 (front-right, CCW): +F  −roll  +pitch  −yaw
        #    w1 (rear-left ,  CCW): +F  +roll  −pitch  −yaw
        #    w2 (front-left,   CW): +F  +roll  +pitch  +yaw
        #    w3 (rear-right,   CW): +F  −roll  −pitch  +yaw
        kF, kM, L = self.K_F, self.K_M, self.L
        self.Gamma = np.array([
            [ kF,       kF,       kF,       kF      ],   # Total thrust
            [-kF * L,   kF * 0.2,   kF * L,  -kF * 0.2  ],   # τ_roll  (body x)
            [-kF * L,   kF * 0.2,  -kF * L,   kF * 0.2  ],   # τ_pitch (body y)
            [-kM,      -kM,       kM,       kM      ],   # τ_yaw   (body z)
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)
        # self.get_logger().info(f'Gamma_inv: {self.Gamma_inv.tolist()}')

        # ── Build linearised model A, B ─────────────────────────────────────
        A, B = self._build_linear_model()

        # ── LQR weight matrices Q and R ─────────────────────────────────────
        #   State: [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]
        Q = np.diag([
            120.0,  120.0,  900.0,   # position  x, y, z     — penalise position error
            10.0,   10.0,   500.0,    # attitude  φ, θ, ψ
            10.0,   10.0,   50.0,    # velocity  ẋ, ẏ, ż
            1.0,   1.0,   10.0,    # ang-rate  φ̇, θ̇, ψ̇
        ])
        #   Input: [F_total, τ_roll, τ_pitch, τ_yaw]
        R = np.diag([
            10.0,   # F  (large  → cheap to vary thrust)
            1.0,    # τ_roll
            1.0,    # τ_pitch
            0.1,    # τ_yaw
        ])

        K = lqr(A, B, Q, R)
        self.lqr_ctrl = LQR(K)
        df = pd.DataFrame(K, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
        df.to_csv('my_data.csv', index=False)
        self.get_logger().info(f'LQR gain K computed {K.tolist()}')

        # ── Dynamic target (updated by /target_pose) ────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # ── State ────────────────────────────────────────────────────────────
        self.pos_x = 0.0;  self.pos_y = 0.0;  self.pos_z = 0.0
        self.vel_x = 0.0;  self.vel_y = 0.0;  self.vel_z = 0.0
        self.roll  = 0.0;  self.pitch = 0.0;  self.yaw   = 0.0
        self.ang_vx = 0.0; self.ang_vy = 0.0; self.ang_vz = 0.0

        self._odom_received = False
        self._last_time: float | None = None
        self._t0:        float | None = None

        # ── History deques for live plot ─────────────────────────────────────
        n = self.HISTORY_LEN
        self.t_hist  = deque(maxlen=n)
        self.x_hist  = deque(maxlen=n);  self.tx_hist = deque(maxlen=n)
        self.y_hist  = deque(maxlen=n);  self.ty_hist = deque(maxlen=n)
        self.z_hist  = deque(maxlen=n);  self.tz_hist = deque(maxlen=n)

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.odom_sub   = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.target_sub = self.create_subscription(
            PoseStamped, '/target_pose', self._target_cb, 10)
        self.cmd_pub    = self.create_publisher(
            Actuators, '/motor_commands', 10)

        # Control loop at 100 Hz
        self.create_timer(0.01, self._control_loop)

        self.get_logger().info(
            f'LQR node ready. Listening on /target_pose. '
            f'Default target → x={self.TARGET_X}, y={self.TARGET_Y}, z={self.TARGET_Z} m')

    # ── Linearised quadrotor model ──────────────────────────────────────────
    def _build_linear_model(self):
        """
        Return (A, B) for the linearised quadrotor dynamics around hover.

        State  x  = [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]  (12)
        Input  u  = [F_total, τ_roll, τ_pitch, τ_yaw]           (4)

        Hover equilibrium: φ=θ=ψ=0, ẋ=ẏ=ż=φ̇=θ̇=ψ̇=0, F0=mg.

        Linearised translational accelerations (body≈world at hover):
            ẍ =  g·θ      (pitch forward → + x)
            ÿ = -g·φ      (roll right   → - y)
            z̈ = F/m − g

        Linearised rotational accelerations:
            φ̈ = τ_roll  / Ixx
            θ̈ = τ_pitch / Iyy
            ψ̈ = τ_yaw   / Izz
        """
        m   = self.MASS
        g   = self.GRAVITY
        Ixx = self.IXX
        Iyy = self.IYY
        Izz = self.IZZ

        # State indices:  0  1  2  3  4  5  6   7   8   9   10  11
        #                 x  y  z  φ  θ  ψ  ẋ   ẏ   ż   φ̇   θ̇   ψ̇
        n, p = 12, 4
        A = np.zeros((n, n))
        B = np.zeros((n, p))

        # Kinematics: ṗos = vel
        A[0, 6]  = 1.0   # ẋ → dx/dt
        A[1, 7]  = 1.0   # ẏ → dy/dt
        A[2, 8]  = 1.0   # ż → dz/dt
        A[3, 9]  = 1.0   # φ̇ → dφ/dt
        A[4, 10] = 1.0   # θ̇ → dθ/dt
        A[5, 11] = 1.0   # ψ̇ → dψ/dt

        # Translational dynamics (linearised gravity coupling)
        A[6, 4]  =  g        # ẍ ← g·θ
        A[7, 3]  = -g        # ÿ ← −g·φ
        # z: controlled by thrust

        # Input coupling — translational
        B[8, 0]  = 1.0 / m   # z̈ ← F/m

        # Input coupling — rotational
        B[9,  1] = 1.0 / Ixx  # φ̈ ← τ_roll / Ixx
        B[10, 2] = 1.0 / Iyy  # θ̈ ← τ_pitch / Iyy
        B[11, 3] = 1.0 / Izz  # ψ̈ ← τ_yaw / Izz

        return A, B

    # ── Target pose callback ────────────────────────────────────────────────
    def _target_cb(self, msg: PoseStamped):
        """Update the setpoint from the trajectory / goal publisher."""
        self.TARGET_X = msg.pose.position.x
        self.TARGET_Y = msg.pose.position.y
        self.TARGET_Z = msg.pose.position.z

        # Extract yaw from quaternion
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.TARGET_YAW = math.atan2(siny, cosy)

    # ── Odometry callback ───────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        q = msg.pose.pose.orientation

        self.pos_x, self.pos_y, self.pos_z = p.x, p.y, p.z
        self.vel_x, self.vel_y, self.vel_z = v.x, v.y, v.z
        self.ang_vx, self.ang_vy, self.ang_vz = w.x, w.y, w.z

        # Quaternion → Euler (ZYX)
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x ** 2 + q.y ** 2)
        self.roll = math.atan2(sinr, cosr)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        self.pitch = math.asin(sinp)

        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.yaw = math.atan2(siny, cosy)

        self._odom_received = True

        # self.get_logger().info(f'Odom: x={self.pos_x:.2f}, y={self.pos_y:.2f}, z={self.pos_z:.2f}')

    # ── Main control loop ───────────────────────────────────────────────────
    def _control_loop(self):
        if not self._odom_received:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self._last_time is None:
            self._last_time = now
            self._t0 = now
            return

        dt = now - self._last_time
        self._last_time = now
        if dt <= 1e-6:
            return

        # ── State error: x_ref − x_current ──────────────────────────────────
        #  [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]
        yaw_err = self.TARGET_YAW - self.yaw
        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi   # wrap

        error = np.array([
            self.TARGET_X   - self.pos_x,
            self.TARGET_Y   - self.pos_y,
            self.TARGET_Z   - self.pos_z,
            0.0             - self.roll,     # desired roll = 0
            0.0             - self.pitch,    # desired pitch = 0
            yaw_err,
            0.0             - self.vel_x,
            0.0             - self.vel_y,
            0.0             - self.vel_z,
            0.0             - self.ang_vx,
            0.0             - self.ang_vy,
            0.0             - self.ang_vz,
        ])

        # ── LQR correction ───────────────────────────────────────────────────
        u_corr = self.lqr_ctrl.compute(error)   # [dF, dτ_r, dτ_p, dτ_y]

        F_total  = self.F_hover + u_corr[0]
        tau_roll  = u_corr[1]
        tau_pitch = u_corr[2]
        tau_yaw   = u_corr[3]

        # ── Clamp total thrust ───────────────────────────────────────────────
        F_max = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        F_total = float(np.clip(F_total, 0.0, F_max))
        # self.get_logger().info(f'u_corr: {u_corr[0]:.2f} N')
        # self.get_logger().info(f'F_total: {F_total:.2f} N')
        # self.get_logger().info(f'F_max: {F_max:.2f} N')

        # ── Motor mixing: solve for ωi² ──────────────────────────────────────
        wrench = np.array([F_total, tau_roll, tau_pitch, tau_yaw])
        # wrench = np.array([self.F_hover, 0, 0, 0])
        omega_sq = self.Gamma_inv @ wrench          # [w0², w1², w2², w3²]
        omega_sq = np.clip(omega_sq, 0.0, self.OMEGA_MAX ** 2)
        # self.get_logger().info(f'omega_sq: {omega_sq.tolist()} rad²/s²')

        omega = np.sqrt(omega_sq)

        # ── Publish ─────────────────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)

# ── Entry point ─────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorLQRNode()
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
