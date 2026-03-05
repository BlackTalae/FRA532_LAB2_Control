import math
import threading
from collections import deque

import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg not available
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped
from utils.MPCken2 import *
from utils.mpc_utils import *
from std_msgs.msg import Float64MultiArray

class MPCNode(Node):
    # ── Physical constants ─────────────────────────────────────────────────
    MASS       = 1.5            # kg
    GRAVITY    = 9.81           # m/s²
    K_F        = 8.54858e-06    # N / (rad/s)²
    K_M        = 0.06           # N·m / (rad/s)²
    OMEGA_MAX  = 900.0          # rad/s
    OMEGA_MIN  = 600.0
 
    # Moments of inertia (from URDF)
    IXX = 0.0347563   # kg·m²
    IYY = 0.07        # kg·m²
    IZZ = 0.0977      # kg·m²

    # Arm length
    L = 0.22   # m

    # ── Control loop sample time ───────────────────────────────────────────
    DT = 0.01   # s  (100 Hz)

    # ── Default target pose (overridden live by /target_pose topic) ───────────
    DEFAULT_TARGET_X   = 0.0   # m
    DEFAULT_TARGET_Y   = 0.0   # m
    DEFAULT_TARGET_Z   = 1.0   # m
    DEFAULT_TARGET_YAW = 0.0   # rad

    # ── Plot history ───────────────────────────────────────────────────────
    HISTORY_LEN = 600   # samples @ 100 Hz → 6 s window

    def __init__(self):
        super().__init__('quadrotor_mpc_node')

        # ── ROS interfaces ─────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        # self.target_sub = self.create_subscription(
        #     PoseStamped, '/target_pose', self._target_cb, 10)
        self.cmd_pub = self.create_publisher(
            Actuators, '/motor_commands', 10)
        self.u_pub = self.create_publisher(
            Float64MultiArray, '/mpc_u', 10
        )

        # ── Hover trim ──────────────────────────────────────────────────────
        self.F_hover     = self.MASS * self.GRAVITY
        self.omega_hover = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.F_max       = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        self.F_min       = 4.0 * self.K_F * self.OMEGA_MIN ** 2
        self.get_logger().info(f'Hover ω: {self.omega_hover:.1f} rad/s  '
                               f'F_hover: {self.F_hover:.2f} N  '
                               f'F_max: {self.F_max:.2f} N')

        # ── Hover trim input vector (shape (m,) = (4,)) ───────────────────────
        u_hover = np.array([self.F_hover, 0.0, 0.0, 0.0])

        # ── Motor allocation matrix Γ (4×4)  [w0², w1², w2², w3²] ─────────
        kF, kM, L = self.K_F, self.K_M, self.L
        self.Gamma = np.array([
            [ kF,         kF,        kF,        kF       ],   # Total thrust
            [-kF * L,     kF * 0.2,  kF * L,   -kF * 0.2],   # τ_roll  (body x)
            [-kF * L,     kF * 0.2, -kF * L,    kF * 0.2],   # τ_pitch (body y)
            [-kM,        -kM,        kM,         kM      ],   # τ_yaw   (body z)
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        # ── Input bounds (absolute, shape (m,)) ───────────────────────────────
        tau_max = kF * L * self.OMEGA_MAX ** 2   # rough single-motor torque limit
        tau_min = kF * L * self.OMEGA_MIN ** 2
        # u_min = np.array([0.0,      -tau_max, -tau_max, -tau_max])
        # u_min = np.array([0.0,      tau_min, tau_min, tau_min])
        # u_max = np.array([self.F_max, tau_max,  tau_max,  tau_max])
        u_min = np.array([0.0,      -0.3, -0.3, -0.3])
        # u_min = np.array([0.0,      tau_min, tau_min, tau_min])
        u_max = np.array([self.F_max, 0.3,  0.3,  0.3])
        print("Tau Max" , tau_max)
        # ── Build & discretise linearised model ───────────────────────────────
        A_c, B_c = self._build_linear_model()
        Ad, Bd   = c2d(A_c, B_c, self.DT)   # FIX: MPC needs discrete matrices

        # ── Weight matrices Q and R ───────────────────────────────────────────
        #   State: [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]
        Q = np.diag([
            1.0,   1.0,   0.0,   # x y z
            200.0,   200.0,  0.0,   # roll pitch yaw
            1.0,   1.0,  50.0,   # vx vy vz
            100.0,  100.0,  1.0,    # wx wy wz
        ])
        #   Input: [F_total, τ_roll, τ_pitch, τ_yaw]
        R = np.diag([
            1.0,   # thrust
            1.0,   # tau_roll
            1.0,   # tau_pitch
            1.0,    # tau_yaw
        ])

        Qf = 2*Q
        # ── Create MPC controller ─────────────────────────────────────────────
        self.mpc_controller = MPC_Opti(
            Ad=Ad,
            Bd=Bd,
            Q=Q,
            R=R,
            Qf=Qf,
            N=5,#10,
            Nc=5,#2,
            u_min=u_min,
            u_max=u_max,
            state_idx=[3,4],
            state_lower_bound=np.array([-0.15, -0.15]),
            state_upper_bound=np.array([ 0.15,  0.15]),
            constrain_initial_state=False,
            constrain_terminal_state=True,
            solver="ipopt",
        )
        self.mpc_controller.build()

        # ── Targets ───────────────────────────────────────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # ── State ─────────────────────────────────────────────────────────────
        self.pos_x = 0.0;  self.pos_y = 0.0;  self.pos_z = 0.0
        self.vel_x = 0.0;  self.vel_y = 0.0;  self.vel_z = 0.0
        self.roll  = 0.0;  self.pitch = 0.0;  self.yaw   = 0.0
        self.ang_vx = 0.0; self.ang_vy = 0.0; self.ang_vz = 0.0

        self._odom_received = False
        self._last_time: float | None = None
        self._t0:        float | None = None

        # ── Control timer ─────────────────────────────────────────────────────
        # FIX: _control_loop was never scheduled; add a periodic timer
        self.create_timer(self.DT, self._control_loop)



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

        # ── Current state vector x0 ────────────────────────────────────────
        x0 = np.array([
            self.pos_x,
            self.pos_y,
            self.pos_z,
            self.roll,
            self.pitch,
            self.yaw,
            self.vel_x,
            self.vel_y,
            self.vel_z,
            self.ang_vx,
            self.ang_vy,
            self.ang_vz,
        ], dtype=float)

        # ── Build references ────────────────────────────────────────────────
        xref, uref = self._build_mpc_references()

        # ── MPC solve ───────────────────────────────────────────────────────
        try:
            u0, X_opt, U_opt, info = self.mpc_controller.solve(
                x0_val=x0,
                xref_val=xref,
                uref_val=uref,
                allow_debug_fallback=False,
                return_info=True,
            )
        except RuntimeError as e:
            self.get_logger().warn(f"MPC solve failed: {e}")
            return

        # u0 = [delta_F, tau_roll, tau_pitch, tau_yaw]
        delta_F   = float(u0[0])
        tau_roll  = float(u0[1])
        tau_pitch = float(u0[2])
        tau_yaw   = float(u0[3])

        # Convert delta thrust -> total thrust
        F_total = float(np.clip(self.F_hover + delta_F, self.F_min, self.F_max))

        # ── Motor mixing ────────────────────────────────────────────────────
        wrench = np.array([F_total, tau_roll, tau_pitch, tau_yaw], dtype=float)
        omega_sq = self.Gamma_inv @ wrench
        omega_sq = np.clip(omega_sq, self.OMEGA_MIN ** 2, self.OMEGA_MAX ** 2)
        omega = np.sqrt(omega_sq)

        # ── Publish ────────────────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)

        print(u0)
        u_msg = Float64MultiArray()
        u_msg.data = [float(v) for v in u0]
        self.u_pub.publish(u_msg)


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
        print(self.roll , self.pitch , self.yaw)
        self._odom_received = True

        # self.get_logger().info(f'Odom: z={self.pos_z:.2f}')

    # ── Linearised quadrotor model ──────────────────────────────────────────
    def _build_linear_model(self):
        """
        Return (A, B) for the linearised quadrotor dynamics around hover.

        State  x  = [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]  (12)
        Input  u  = [F_total, τ_roll, τ_pitch, τ_yaw]           (4)
        """
        m   = self.MASS
        g   = self.GRAVITY
        Ixx = self.IXX
        Iyy = self.IYY
        Izz = self.IZZ

        n, p = 12, 4
        A = np.zeros((n, n))
        B = np.zeros((n, p))

        # Kinematics: ṗos = vel
        A[0, 6]  = 1.0
        A[1, 7]  = 1.0
        A[2, 8]  = 1.0
        A[3, 9]  = 1.0
        A[4, 10] = 1.0
        A[5, 11] = 1.0

        # Translational dynamics (linearised gravity coupling)
        A[6, 4]  =  g    # ẍ ← g·θ
        A[7, 3]  = -g    # ÿ ← −g·φ

        # Input coupling — translational
        B[8, 0]  = 1.0 / m      # z̈ ← F/m

        # Input coupling — rotational
        B[9,  1] = 1.0 / Ixx   # φ̈ ← τ_roll / Ixx
        B[10, 2] = 1.0 / Iyy   # θ̈ ← τ_pitch / Iyy
        B[11, 3] = 1.0 / Izz   # ψ̈ ← τ_yaw / Izz

        return A, B
    
    def _build_mpc_references(self):
        """
        Build xref: (n, N+1)
        Build uref: (m, N)

        State order:
        [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]

        Input order:
        [delta_F, tau_roll, tau_pitch, tau_yaw]
        """
        x_ref_single = np.array([
            self.TARGET_X,
            self.TARGET_Y,
            self.TARGET_Z,
            0.0,                # roll
            0.0,                # pitch
            self.TARGET_YAW,    # yaw
            0.0,                # vx
            0.0,                # vy
            0.0,                # vz
            0.0,                # wx
            0.0,                # wy
            0.0,                # wz
        ], dtype=float)

        xref = np.tile(
            x_ref_single.reshape(-1, 1),
            (1, self.mpc_controller.N + 1)
        )

        # hover equilibrium in delta-input coordinates = zero
        uref = np.zeros((self.mpc_controller.m, self.mpc_controller.N), dtype=float)

        return xref, uref

def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()