#!/usr/bin/env python3
"""
Quadrotor LQR Controller
========================
Controls a quadrotor to hold a fixed position (x, y, z) using a Linear
Quadratic Regulator (LQR).

Subscribes to: /odom         (nav_msgs/Odometry)
               /target_pose  (geometry_msgs/PoseStamped)
Publishes to:  /motor_commands (actuator_msgs/Actuators)

Physical parameters (from robot_params.xacro):
  mass        = 1.525 kg
  k_F         = 8.54858e-06  [N / (rad/s)^2]
  k_M         = 0.06         [dimensionless torque ratio]
  Ixx         = 0.0356547    [kg·m²]
  Iyy         = 0.0705152    [kg·m²]
  Izz         = 0.0990924    [kg·m²]
  omega_max   = 1500         rad/s
  arm_length  = 0.22 m

State vector  x  (12×1):
  [x, y, z, φ(roll), θ(pitch), ψ(yaw), ẋ, ẏ, ż, φ̇, θ̇, ψ̇]

Input vector  u  (4×1):
  [F_total, τ_roll, τ_pitch, τ_yaw]

The LQR gain matrix K (4×12) maps state error to inputs:
  u = u_hover + K · (x_ref - x)

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
  Given k_F, k_M, arm length L the allocation matrix Γ maps
  [F, τ_r, τ_p, τ_y] → [w0², w1², w2², w3²] via Γ⁻¹.
"""

import math
import numpy as np
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions  (same interface as MPC.py)
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
    Solve the continuous-time LQR problem.

    Minimises:  J = ∫ (x'Qx + u'Ru) dt

    Returns the gain matrix K such that u = -Kx is optimal.
    Internally solves the Continuous Algebraic Riccati Equation (CARE):
        A'P + PA − PBR⁻¹B'P + Q = 0
    then  K = R⁻¹ B' P.
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


# ─────────────────────────────────────────────────────────────────────────────
# LQR controller wrapper (stateless – the node owns the state)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────

class QuadrotorLQRNode(Node):

    # ── Physical constants ────────────────────────────────────────────────────
    GRAVITY   = 9.81            # m/s²
    K_F       = 8.54858e-06     # N / (rad/s)²
    K_M       = 0.06            # N·m / (rad/s)²
    OMEGA_MAX = 1500.0          # rad/s

    MASS = 1.525                # kg
    IXX  = 0.0356547            # kg·m²
    IYY  = 0.0705152            # kg·m²
    IZZ  = 0.0990924            # kg·m²

    U_MIN = np.array([0.0, -5.0, -5.0, -3.0])
    U_MAX = np.array([4.0 * MASS * GRAVITY, 5.0, 5.0, 3.0])

    # Arm length (distance from CoM to rotor along body x/y axis)
    L = 0.22                    # m

    # ── Default target pose (overridden live by /target_pose) ─────────────────
    DEFAULT_TARGET_X   = 0.0   # m
    DEFAULT_TARGET_Y   = 0.0   # m
    DEFAULT_TARGET_Z   = 2.0   # m
    DEFAULT_TARGET_YAW = 0.0   # rad
    START_DELAY = 8.0
    START_REQ_PERIOD = 0.25

    def __init__(self):
        super().__init__('quadrotor_lqr_node')

        # ── Hover trim ────────────────────────────────────────────────────────
        self.F_hover  = self.MASS * self.GRAVITY
        self.u_hover  = np.array([self.F_hover, 0.0, 0.0, 0.0])
        omega_hover   = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.get_logger().info(
            f'Hover ω: {omega_hover:.1f} rad/s  F_hover: {self.F_hover:.2f} N')

        # ── Motor allocation matrix Γ (4×4)  [w0², w1², w2², w3²] ───────────
        #  Sign conventions (X-config from URDF):
        #    Motor 0 (FR): +0.13x, -0.22y  CCW (+)
        #    Motor 1 (RL): -0.13x, +0.20y  CCW (+)
        #    Motor 2 (FL): +0.13x, +0.22y  CW  (-)
        #    Motor 3 (RR): -0.13x, -0.20y  CW  (-)
        #  τ_roll  = Σ (y_i * F_i)
        #  τ_pitch = Σ (-x_i * F_i)
        #  τ_yaw   = Σ (-k_M * F_i) for CCW, (+k_M * F_i) for CW
        kF, kM = self.K_F, self.K_M
        self.Gamma = np.array([
            [ kF,        kF,        kF,        kF       ],   # Thrust
            [-kF * 0.22, kF * 0.20, kF * 0.22, -kF * 0.20],  # τ_roll  (lever y)
            # [-kF * 0.22, kF * 0.20,-kF * 0.22, kF * 0.20],  # τ_pitch (lever -x)
            [-kF * 0.13, kF * 0.13, -kF * 0.13, kF * 0.13],  # τ_pitch (lever -x)
            [-kF * kM,  -kF * kM,   kF * kM,   kF * kM  ],   # τ_yaw   (kM ratio)
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        # ── Linearised model A, B ─────────────────────────────────────────────
        A, B = self._build_linear_model()

        # Ken
        Q = np.diag([
            400.0,  400.0,  80.0,   # position  x, y↑, z↓
            60.0,   60.0,   15.0,  # attitude  φ, θ, ψ
            45.0,   40.0,   30.0,    # velocity  ẋ, ẏ↑, ż
            5.0,    5.0,    4.0,    # ang-rate  φ̇, θ̇, ψ̇
        ])
        
        #   Input: [F_total, τ_roll, τ_pitch, τ_yaw]
        R = np.diag([
            0.1,    # F
            3.0,    # τ_roll  (Smoothed)
            3.0,    # τ_pitch (Smoothed)
            5.0,    # τ_yaw   (Smoothed)
        ])

        K = lqr(A, B, Q, R)
        self.lqr_ctrl = LQR(K)
        self.get_logger().info(f'LQR gain K computed (shape {K.shape})')

        # ── Target setpoint (updated by /target_pose) ─────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        self.TARGET_X_   = self.DEFAULT_TARGET_X
        self.TARGET_Y_   = self.DEFAULT_TARGET_Y
        self.TARGET_Z_   = self.DEFAULT_TARGET_Z

        # ── Drone state  [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇] ─────────────
        #    Same layout as MPC.py: self.state (12,), self.state_ready flag
        self.state       = np.zeros(12)
        self.state_ready = False

        # ── ROS interfaces ─────────────────────────────────────────────────────
        self.create_subscription(Odometry,    '/odom',        self._cb_odom,   10)
        self.create_subscription(PoseStamped, '/target_pose', self._cb_target,  10)
        self.cmd_pub = self.create_publisher(Actuators, '/motor_commands', 10)
        self.pub_motors = self.create_publisher(Actuators, '/motor_commands', 10)
        self.start_cli = self.create_client(Trigger, '/start_trajectory')
        self.node_start_time = self.get_clock().now().nanoseconds * 1e-9

        self.control_pub = self.create_publisher(Float64MultiArray, '/control_u', 10)
        self.min_range_pub = self.create_publisher(Float64MultiArray, '/range_min', 10)
        self.max_range_pub = self.create_publisher(Float64MultiArray, '/range_max', 10)
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)

        self.robot_path = Path()
        self.robot_path.header.frame_id = 'world'   # หรือ 'odom' ให้ตรงกับ frame ของ /odom
        self.max_path_len = 10000

        # Control loop at 100 Hz
        self.create_timer(0.01, self._cb_control)
        self.start_req_timer = self.create_timer(self.START_REQ_PERIOD, self._cb_start_request_timer)
        self.start_requested = False

        self.get_logger().info(
            f'LQR node ready | listening on /target_pose | '
            f'default target → ({self.TARGET_X}, {self.TARGET_Y}, {self.TARGET_Z}) m')

    # ─────────────────────────────────────────────────────────────────────────
    # Linearised quadrotor model
    # ─────────────────────────────────────────────────────────────────────────

    def _build_linear_model(self):
        """
        Return (A, B) for the linearised quadrotor dynamics around hover.

        State  x  = [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]  (12)
        Input  u  = [F_total, τ_roll, τ_pitch, τ_yaw]           (4)

        Hover equilibrium: φ=θ=ψ=0, ẋ=ẏ=ż=φ̇=θ̇=ψ̇=0, F0=mg.

        Linearised translational accelerations (body≈world at hover):
            ẍ =  g·θ
            ÿ = -g·φ
            z̈ = F/m − g

        Linearised rotational accelerations:
            φ̈ = τ_roll  / Ixx
            θ̈ = τ_pitch / Iyy
            ψ̈ = τ_yaw   / Izz
        """
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ

        # State indices:  0  1  2  3  4  5  6   7   8   9   10  11
        #                 x  y  z  φ  θ  ψ  ẋ   ẏ   ż   φ̇   θ̇   ψ̇
        n, p = 12, 4
        A = np.zeros((n, n))
        B = np.zeros((n, p))

        # Kinematics: ṗos = vel
        A[0, 6] = A[1, 7] = A[2, 8] = 1.0   # ẋ, ẏ, ż → dx/dt
        A[3, 9] = A[4, 10] = A[5, 11] = 1.0  # φ̇, θ̇, ψ̇ → dφ/dt

        # Translational dynamics (linearised gravity coupling)
        A[6, 4] =  g    # ẍ ← g·θ
        A[7, 3] = -g    # ÿ ← −g·φ

        # Input coupling — translational
        B[8, 0] = 1.0 / m    # z̈ ← F/m

        # Input coupling — rotational
        B[9,  1] = 1.0 / Ixx  # φ̈ ← τ_roll / Ixx
        B[10, 2] = 1.0 / Iyy  # θ̈ ← τ_pitch / Iyy
        B[11, 3] = 1.0 / Izz  # ψ̈ ← τ_yaw / Izz

        return A, B

    # ─────────────────────────────────────────────────────────────────────────
    # Odometry callback  (mirrors MPC._cb_odom)
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        """Pack sensor data into self.state — same layout as MPC.py."""
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
        self._publish_robot_path(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Target pose callback
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_target(self, msg: PoseStamped):
        """Update setpoint from /target_pose."""
        self.TARGET_X = msg.pose.position.x
        self.TARGET_Y = msg.pose.position.y
        self.TARGET_Z = msg.pose.position.z

        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.TARGET_YAW = math.atan2(siny, cosy)

    # ─────────────────────────────────────────────────────────────────────────
    # Control loop
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_control(self):
        self.min_range_pub.publish(Float64MultiArray(data=[float(v) for v in self.U_MIN]))
        self.max_range_pub.publish(Float64MultiArray(data=[float(v) for v in self.U_MAX]))

        if not self.state_ready:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # ── State error in BODY FRAME ──────────────────────────────────────────
        # Rotate world-frame x,y errors into body-heading frame.
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

        v_x_target = (self.TARGET_X - self.TARGET_X_)/ 0.01
        v_y_target = (self.TARGET_Y - self.TARGET_Y_)/ 0.01
        v_z_target = (self.TARGET_Z - self.TARGET_Z_)/ 0.01

        yaw_err = wrap_angle(self.TARGET_YAW - self.state[5])
        error = np.array([
            ex_body,
            ey_body,
            self.TARGET_Z - self.state[2],
            0.0           - self.state[3],
            0.0           - self.state[4],
            yaw_err,
            v_x_target - vx_body,
            v_y_target - vy_body,
            v_z_target    - self.state[8],
            0.0           - self.state[9],
            0.0           - self.state[10],
            0.0           - self.state[11],
        ])

        self.TARGET_X_ = self.TARGET_X
        self.TARGET_Y_ = self.TARGET_Y
        self.TARGET_Z_ = self.TARGET_Z

        # ── LQR correction ────────────────────────────────────────────────────
        u_corr    = self.lqr_ctrl.compute(error)   # [dF, dτ_r, dτ_p, dτ_y]
        u_opt     = self.u_hover + u_corr

        # Clamp total thrust
        F_max     = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        u_opt[0]  = float(np.clip(u_opt[0], 0.0, F_max))

        # ── Motor mixing: Γ⁻¹ · [F, τ_r, τ_p, τ_y] → ω ──────────────────────
        omega_sq  = self.Gamma_inv @ u_opt
        omega_sq  = np.clip(omega_sq, 0.0, self.OMEGA_MAX ** 2)
        omega     = np.sqrt(omega_sq)

        # ── Publish ───────────────────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)

        self.control_pub.publish(Float64MultiArray(data=[float(u) for u in u_opt]))

        # ── Diagnostics (~2 Hz) ───────────────────────────────────────────────
        if int(now_sec * 2) % 4 == 0:
            pos  = self.state[:3]
            rpy  = np.degrees(self.state[3:6])
            ref  = np.array([self.TARGET_X, self.TARGET_Y, self.TARGET_Z])
            err  = float(np.linalg.norm(pos - ref))
            du   = u_opt - self.u_hover
            # self.get_logger().info(
            #     f'pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] '
            #     f'ref=[{ref[0]:+.2f},{ref[1]:+.2f},{ref[2]:+.2f}] '
            #     f'err={err:.3f}m | '
            #     f'rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]deg | '
            #     f'dF={du[0]:+.2f}N | '
            #     f'omega={np.round(omega).astype(int)} rad/s'
            # )

    def _cb_start_request_timer(self):
        """
        Separate timer for requesting /start_trajectory.
        This avoids blocking the high-rate MPC control loop.
        """

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        startup_elapsed = now_sec - self.node_start_time
        if self.start_requested:
            return
        req = Trigger.Request()
        self.start_future = self.start_cli.call_async(req)
        self.start_future.add_done_callback(self._on_start_response)
        self.start_requested = True
        self.get_logger().info('Requested /start_trajectory service')

    def _on_start_response(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.traj_enabled = True
                self.get_logger().info(f'Trajectory start confirmed: {resp.message}')
            else:
                self.start_requested = False
                self.get_logger().warn(f'Trajectory start rejected: {resp.message}')
        except Exception as e:
            self.start_requested = False
            self.get_logger().error(f'Start trajectory service failed: {e}')

    def _publish_robot_path(self, msg: Odometry):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.robot_path.header.stamp = msg.header.stamp
        self.robot_path.header.frame_id = msg.header.frame_id
        self.robot_path.poses.append(pose)

        # กัน path ยาวไม่สิ้นสุด
        if len(self.robot_path.poses) > self.max_path_len:
            self.robot_path.poses.pop(0)

        self.path_pub.publish(self.robot_path)
# ─────────────────────────────────────────────────────────────────────────────

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


if __name__ == '__main__':
    main()