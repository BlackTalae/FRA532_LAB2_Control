#!/usr/bin/env python3
"""
Quadrotor Integral LQR (LQI) Controller
========================================
Extends the standard LQR with an **integrator** on position and yaw errors
to eliminate steady-state offset from disturbances or model mismatch.

Architecture
------------
The control law is:

    u = u_hover + K · error_12 + Ki · integ_4

where:
  - K   (4×12)  : exact same gain as LQR.py (solves 12-state CARE)
  - Ki  (4×4)   : separate integral gain matrix, physics-motivated
  - error_12    : full 12-state error vector  (ref − current)
  - integ_4     : integral of [ex, ey, ez, eψ] over time

This two-gain structure is cleaner and safer than augmenting the state for
CARE: K is guaranteed to match the working LQR baseline (stable hover), and
Ki adds a slow correction layer that pulls the drone to the reference over
several seconds without disturbing the proportional action.

Ki sign convention (from linearised dynamics at hover):
  ẍ = g·θ  →  ∫ex > 0 → want +θ → +τ_pitch  (Ki[τ_p, ∫ex] > 0)
  ÿ = -g·φ →  ∫ey > 0 → want -φ → -τ_roll   (Ki[τ_r, ∫ey] < 0)
  z̈ = F/m  →  ∫ez > 0 → want +F              (Ki[F,   ∫ez] > 0)
  ψ̈ = τy/Iz→  ∫eψ > 0 → want +τ_yaw         (Ki[τ_y, ∫eψ] > 0)

Subscribes to: /odom         (nav_msgs/Odometry)
               /target_pose  (geometry_msgs/PoseStamped)
Publishes to:  /motor_commands (actuator_msgs/Actuators)
"""

import math
import numpy as np
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def quat2rpy(qx, qy, qz, qw):
    """Quaternion → (roll, pitch, yaw) in radians."""
    sr = 2.0 * (qw * qx + qy * qz)
    cr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sr, cr)
    sp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    sy = 2.0 * (qw * qz + qx * qy)
    cy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(sy, cy)
    return roll, pitch, yaw


def wrap_angle(a):
    """Wrap angle to (−π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# LQR gain solver
# ─────────────────────────────────────────────────────────────────────────────

def lqr(A: np.ndarray, B: np.ndarray,
        Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve continuous-time LQR:  A'P + PA − PBR⁻¹B'P + Q = 0  →  K = R⁻¹B'P.
    Returns K (m×n) such that u = -K·x is optimal.
    """
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


# ─────────────────────────────────────────────────────────────────────────────
# LQI controller
# ─────────────────────────────────────────────────────────────────────────────

class LQI:
    """
    Integral LQR controller with explicit proportional K and integral Ki.

    Usage
    -----
        ctrl = LQI(K, Ki)
        u_corr = ctrl.compute(error_12, dt)

    Call ctrl.reset() when the target changes significantly.
    """

    # Anti-windup limits for [∫ex, ∫ey, ∫ez, ∫eψ]  (m·s, m·s, m·s, rad·s)
    # ∫ey is larger: 4 m/s wind in -Y needs proportionally more integral
    INTEG_LIM = np.array([5.0, 10.0, 3.0, 0.8])

    def __init__(self, K: np.ndarray, Ki: np.ndarray):
        """
        Parameters
        ----------
        K  : (4, 12)  proportional-derivative gain from 12-state CARE
        Ki : (4,  4)  integral gain matrix mapping [∫ex,∫ey,∫ez,∫eψ] → u
        """
        self.K    = K                  # 4 × 12
        self.Ki   = Ki                 # 4 × 4
        self.integ = np.zeros(4)       # [∫ex, ∫ey, ∫ez, ∫eψ]

    def reset(self):
        """Zero the integrator (call when the target changes significantly)."""
        self.integ[:] = 0.0

    def compute(self, error: np.ndarray, dt: float) -> np.ndarray:
        """
        Return the control correction u_corr (shape 4).

        Parameters
        ----------
        error : (12,)  state error  x_ref - x_current
        dt    : float  time step [s]
        """
        # Integrate position and yaw errors
        self.integ += np.array([
            error[0],   # ∫(x_ref − x)
            error[1],   # ∫(y_ref − y)
            error[2],   # ∫(z_ref − z)
            error[5],   # ∫(ψ_ref − ψ)
        ]) * dt

        # Anti-windup
        self.integ = np.clip(self.integ, -self.INTEG_LIM, self.INTEG_LIM)

        # Proportional + integral correction
        return self.K @ error + self.Ki @ self.integ


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────

class QuadrotorLQINode(Node):

    # ── Physical constants ─────────────────────────────────────────────────────
    GRAVITY   = 9.81
    K_F       = 8.54858e-06
    K_M       = 0.06
    OMEGA_MAX = 1500.0

    MASS = 1.525
    IXX  = 0.0356547
    IYY  = 0.0705152
    IZZ  = 0.0990924

    L = 0.22

    # ── Default target pose ────────────────────────────────────────────────────
    DEFAULT_TARGET_X   = 0.0
    DEFAULT_TARGET_Y   = 0.0
    DEFAULT_TARGET_Z   = 1.0
    DEFAULT_TARGET_YAW = 0.0

    def __init__(self):
        super().__init__('quadrotor_lqi_node')

        # ── Hover trim ────────────────────────────────────────────────────────
        self.F_hover = self.MASS * self.GRAVITY
        self.u_hover = np.array([self.F_hover, 0.0, 0.0, 0.0])
        omega_hover  = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.get_logger().info(
            f'Hover ω: {omega_hover:.1f} rad/s  F_hover: {self.F_hover:.2f} N')

        # ── Motor allocation matrix Γ (4×4) ──────────────────────────────────
        kF, kM, LL = self.K_F, self.K_M, self.L
        self.Gamma = np.array([
            [ kF,       kF,       kF,       kF      ],
            [-kF * LL,  kF * 0.2, kF * LL, -kF * 0.2],
            [-kF * LL,  kF * 0.2,-kF * LL,  kF * 0.2],
            [-kM,      -kM,       kM,        kM     ],
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        # ── Build 12-state linearised model (identical to LQR.py) ────────────
        A, B = self._build_linear_model()

        # ── Q and R — tuned for Helix Tracking (0.75 m/s) ───────────────────
        # Q(x/y) ↑ 200 : faster position correction (reduced lag)
        # Q(vx/vy) ↑ 40 : more damping for stable high-speed tracking
        # Q(z)   900   : baseline altitude hold
        # Q(yaw) 2000  : firm yaw hold (less than the 5000 wind-stiffness)
        # R(F) ↓ 0.8   : allow more aggressive thrust for vertical segments
        Q = np.diag([
            200.0, 200.0, 900.0,    # position  x↑, y↑, z
            10.0,  10.0,  2000.0,   # attitude  φ, θ, ψ
            40.0,  40.0,  50.0,     # velocity  ẋ↑, ẏ↑, ż
            1.0,   1.0,   10.0,     # ang-rate  φ̇, θ̇, ψ̇
        ])
        R = np.diag([
            0.8,    # F_total (lowered to 0.8)
            1.0,    # τ_roll
            1.0,    # τ_pitch
            0.001,  # τ_yaw
        ])
        K = lqr(A, B, Q, R)

        # ── Integral gain Ki (4×4) ─────────────────────────────────────────────
        # ki_r moderate (not too large, oscillation risk):
        #   wind pushes -Y → integ[1] goes negative → -ki_r * neg = positive τ_r
        #   → positive roll → thrust Y-component pushes back → wind balanced
        ki_F   = 0.15    # thrust integral per m·s of z-error
        ki_r   = 0.12    # roll-torque integral
        ki_p   = 0.08    # pitch-torque integral
        ki_yaw = 0.02    # yaw-torque integral (4× raised: fight wind yaw drift)
        Ki = np.array([
            [0.0,   0.0,    ki_F,   0.0    ],   # dF ← ∫ez
            [0.0,  -ki_r,   0.0,    0.0    ],   # dτ_roll ← -∫ey
            [ki_p,  0.0,    0.0,    0.0    ],   # dτ_pitch ← +∫ex
            [0.0,   0.0,    0.0,    ki_yaw ],   # dτ_yaw ← ∫eψ
        ])

        self.lqi_ctrl = LQI(K, Ki)
        self.get_logger().info(
            f'LQI ready | K shape {K.shape} | Ki:\n{Ki}')

        # ── Target setpoint ───────────────────────────────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # ── Drone state ───────────────────────────────────────────────────────
        self.state       = np.zeros(12)
        self.state_ready = False
        self._last_time  = None

        # ── ROS interfaces ─────────────────────────────────────────────────────
        self.create_subscription(Odometry,    '/odom',        self._cb_odom,   10)
        self.create_subscription(PoseStamped, '/target_pose', self._cb_target,  10)
        self.cmd_pub = self.create_publisher(Actuators, '/motor_commands', 10)

        self.create_timer(0.01, self._cb_control)

        self.get_logger().info(
            f'LQI node ready | default target → '
            f'({self.TARGET_X}, {self.TARGET_Y}, {self.TARGET_Z}) m')

    # ─────────────────────────────────────────────────────────────────────────
    # Linearised quadrotor model (identical to LQR.py)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_linear_model(self):
        """
        Return (A, B) for linearised hover dynamics.

        State  x = [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]  (12)
        Input  u = [F, τ_roll, τ_pitch, τ_yaw]                 (4)
        """
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ

        n, p = 12, 4
        A = np.zeros((n, n))
        B = np.zeros((n, p))

        A[0, 6] = A[1, 7] = A[2, 8] = 1.0
        A[3, 9] = A[4, 10] = A[5, 11] = 1.0

        A[6, 4] =  g    # ẍ ← g·θ
        A[7, 3] = -g    # ÿ ← −g·φ

        B[8, 0]  = 1.0 / m
        B[9,  1] = 1.0 / Ixx
        B[10, 2] = 1.0 / Iyy
        B[11, 3] = 1.0 / Izz

        return A, B

    # ─────────────────────────────────────────────────────────────────────────
    # Odometry callback
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        lv  = msg.twist.twist.linear
        av  = msg.twist.twist.angular
        roll, pitch, yaw = quat2rpy(q.x, q.y, q.z, q.w)
        self.state = np.array([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw,
            lv.x, lv.y, lv.z,
            av.x, av.y, av.z,
        ])
        self.state_ready = True

    # ─────────────────────────────────────────────────────────────────────────
    # Target pose callback
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_target(self, msg: PoseStamped):
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        new_z = msg.pose.position.z

        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        new_yaw = math.atan2(siny, cosy)

        dist = math.sqrt((new_x - self.TARGET_X)**2 +
                         (new_y - self.TARGET_Y)**2 +
                         (new_z - self.TARGET_Z)**2)
        if dist > 0.3 or abs(wrap_angle(new_yaw - self.TARGET_YAW)) > 0.175:
            self.lqi_ctrl.reset()
            self.get_logger().info(
                f'Target changed ({dist:.2f} m) → integrators reset')

        self.TARGET_X   = new_x
        self.TARGET_Y   = new_y
        self.TARGET_Z   = new_z
        self.TARGET_YAW = new_yaw

    # ─────────────────────────────────────────────────────────────────────────
    # Control loop (100 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_control(self):
        if not self.state_ready:
            return

        now_ns  = self.get_clock().now().nanoseconds
        now_sec = now_ns * 1e-9

        # dt (clamped)
        if self._last_time is None:
            dt = 0.01
        else:
            dt = float(np.clip((now_ns - self._last_time) * 1e-9, 0.001, 0.1))
        self._last_time = now_ns

        # ── State error in BODY FRAME ──────────────────────────────────────────
        # Rotate world-frame x,y errors into body frame. This is essential
        # for helix tracking where yaw angle changes constantly.
        yaw     = self.state[5]
        cos_psi = math.cos(yaw)
        sin_psi = math.sin(yaw)
        ex_world = self.TARGET_X - self.state[0]
        ey_world = self.TARGET_Y - self.state[1]

        # Rotate world → body (2D rotation by -yaw)
        ex_body  =  cos_psi * ex_world + sin_psi * ey_world
        ey_body  = -sin_psi * ex_world + cos_psi * ey_world

        # Same for velocities: rotate world velocities into body frame
        vx_world = self.state[6]
        vy_world = self.state[7]
        vx_body  =  cos_psi * vx_world + sin_psi * vy_world
        vy_body  = -sin_psi * vx_world + cos_psi * vy_world

        yaw_err = wrap_angle(self.TARGET_YAW - self.state[5])
        error = np.array([
            ex_body,
            ey_body,
            self.TARGET_Z - self.state[2],
            0.0           - self.state[3],
            0.0           - self.state[4],
            yaw_err,
            0.0 - vx_body,
            0.0 - vy_body,
            0.0           - self.state[8],
            0.0           - self.state[9],
            0.0           - self.state[10],
            0.0           - self.state[11],
        ])

        # ── LQI correction ────────────────────────────────────────────────────
        u_corr = self.lqi_ctrl.compute(error, dt)
        u_opt  = self.u_hover + u_corr

        # Clamp thrust
        F_max    = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        u_opt[0] = float(np.clip(u_opt[0], 0.0, F_max))

        # ── Motor mixing ──────────────────────────────────────────────────────
        omega_sq = self.Gamma_inv @ u_opt
        omega_sq = np.clip(omega_sq, 0.0, self.OMEGA_MAX ** 2)
        omega    = np.sqrt(omega_sq)

        # ── Publish ───────────────────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)

        # ── Diagnostics (~2 Hz) ───────────────────────────────────────────────
        if int(now_sec * 2) % 4 == 0:
            pos   = self.state[:3]
            rpy   = np.degrees(self.state[3:6])
            ref   = np.array([self.TARGET_X, self.TARGET_Y, self.TARGET_Z])
            err   = float(np.linalg.norm(pos - ref))
            du    = u_opt - self.u_hover
            integ = self.lqi_ctrl.integ
            self.get_logger().info(
                f'pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] '
                f'ref=[{ref[0]:+.2f},{ref[1]:+.2f},{ref[2]:+.2f}] '
                f'err={err:.3f}m | '
                f'rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]deg | '
                f'integ=[{integ[0]:+.3f},{integ[1]:+.3f},'
                f'{integ[2]:+.3f},{integ[3]:+.3f}] | '
                f'dF={du[0]:+.2f}N | '
                f'omega={np.round(omega).astype(int)} rad/s'
            )


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorLQINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
