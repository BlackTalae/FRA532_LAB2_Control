#!/usr/bin/env python3
"""
Quadrotor MPC Hover Controller
================================
Controls a quadrotor to hold / track a position setpoint using a
**Model Predictive Controller** (MPC).

Drop-in replacement for the LQR node:
  Subscribes:  /odom           (nav_msgs/msg/Odometry)
               /target_pose    (geometry_msgs/msg/PoseStamped)
  Publishes:   /motor_commands (actuator_msgs/msg/Actuators)

MPC formulation
---------------
Prediction model:   discrete-time linearisation of the 6-DOF quadrotor
                    around hover (same A, B as the LQR variant).

Horizon:            N = 20 steps  (dt = 0.01 s  →  0.20 s look-ahead)

Cost:
    J = Σ_{k=0}^{N-1} [ e_k' Q e_k  +  Δu_k' R Δu_k ]
        + e_N' P e_N          (P = DARE terminal cost)

Where:
    e_k = x_k - x_ref        state error
    Δu_k = u_k - u_hover     deviation from trim

Constraints (applied every step):
    F_total   ∈ [ 0,        F_max   ]
    τ_roll    ∈ [ -τ_max,   τ_max   ]
    τ_pitch   ∈ [ -τ_max,   τ_max   ]
    τ_yaw     ∈ [ -τ_yaw_m, τ_yaw_m ]
    roll      ∈ [ -φ_max,   φ_max   ]   (soft safety)
    pitch     ∈ [ -θ_max,   θ_max   ]

Solver:   CasADi + IPOPT   (pip install casadi)
          Falls back to scipy SLSQP if CasADi is absent
          (slower, no warm-starting, but no extra C++ deps).

Physical parameters identical to the LQR version (robot_params.xacro).

State vector x (12×1):
  [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]

Input vector u (4×1):
  [F_total, τ_roll, τ_pitch, τ_yaw]

Motor layout / mixing: identical to LQR version (X-config).
"""

from __future__ import annotations
import math
import threading
import warnings
from collections import deque

import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are

# ── Optional CasADi import ──────────────────────────────────────────────────
try:
    import casadi as ca
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False
    warnings.warn(
        "CasADi not found – falling back to scipy SLSQP solver. "
        "Install CasADi for better performance:  pip install casadi",
        stacklevel=2,
    )

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped
from utils.mpc_controller import *

# ═══════════════════════════════════════════════════════════════════════════════
#  ROS 2 Node
# ═══════════════════════════════════════════════════════════════════════════════

class QuadrotorMPCNode(Node):

    # ── Physical constants (identical to LQR version) ──────────────────────
    MASS       = 1.5
    GRAVITY    = 9.81
    K_F        = 8.54858e-06
    K_M        = 0.06
    OMEGA_MAX  = 1500.0

    IXX = 0.0347563
    IYY = 0.07
    IZZ = 0.0977

    L = 0.22   # arm length [m]

    # ── MPC parameters ─────────────────────────────────────────────────────
    DT         = 0.01   # control period [s]
    HORIZON    = 20     # prediction steps (= 0.20 s look-ahead)

    # Maximum attitude deviations (state constraints, rad)
    PHI_MAX   = math.radians(30)   # roll
    THETA_MAX = math.radians(30)   # pitch

    # ── Default target ──────────────────────────────────────────────────────
    DEFAULT_TARGET_X   = 0.0
    DEFAULT_TARGET_Y   = 0.0
    DEFAULT_TARGET_Z   = 1.0
    DEFAULT_TARGET_YAW = 0.0

    HISTORY_LEN = 600

    def __init__(self):
        super().__init__('quadrotor_mpc_node')

        # ── Hover trim ──────────────────────────────────────────────────────
        self.F_hover     = self.MASS * self.GRAVITY
        self.omega_hover = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.get_logger().info(
            f'Hover ω: {self.omega_hover:.1f} rad/s  F_hover: {self.F_hover:.2f} N')

        # ── Motor allocation matrix ─────────────────────────────────────────
        kF, kM, L = self.K_F, self.K_M, self.L
        self.Gamma = np.array([
            [ kF,         kF,       kF,        kF      ],
            [-kF * L,   kF * 0.2,   kF * L,  -kF * 0.2 ],
            [-kF * L,   kF * 0.2,  -kF * L,   kF * 0.2 ],
            [-kM,        -kM,        kM,        kM      ],
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        # ── Continuous-time linearised model ───────────────────────────────
        Ac, Bc = self._build_continuous_model()

        # ── Discrete-time model for MPC ─────────────────────────────────────
        Ad, Bd = c2d(Ac, Bc, self.DT)
        self.get_logger().info(
            f'Discrete model: spectral radius Ad = '
            f'{max(abs(np.linalg.eigvals(Ad))):.4f}')

        # ── LQR / MPC weight matrices ───────────────────────────────────────
        #   State: [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]
        Q = np.diag([
            120.0, 10.0, 900.0,   # position x, y, z
             10.0, 10.0, 100.0,   # attitude φ, θ, ψ
             10.0,  4.0,  50.0,   # velocity ẋ, ẏ, ż
              1.0,  1.0,   1.0,   # angular rate φ̇, θ̇, ψ̇
        ])
        #   Input deviations: [ΔF, Δτ_roll, Δτ_pitch, Δτ_yaw]
        R = np.diag([10.0, 1.0, 1.0, 1.0])

        # ── Terminal cost: infinite-horizon DARE solution ───────────────────
        try:
            P = dare_terminal_cost(Ad, Bd, Q, R)
            self.get_logger().info('Terminal cost P computed via DARE.')
        except Exception as e:
            self.get_logger().warn(f'DARE failed ({e}); using Q as terminal cost.')
            P = Q

        # ── Input bounds (deviation from hover trim) ────────────────────────
        F_max  = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        tau_max   = 2.0   # N·m  (roll / pitch torque limit)
        tau_yaw_m = 0.5   # N·m  (yaw torque limit)

        u_min = np.array([0.0    - self.F_hover, -tau_max,    -tau_max,    -tau_yaw_m])
        u_max = np.array([F_max  - self.F_hover,  tau_max,     tau_max,     tau_yaw_m])

        # ── State bounds (applied as constraints inside the MPC) ─────────────
        #  Very permissive on position / velocity; tight on attitude
        BIG = 1e6
        x_lb = np.array([-BIG, -BIG, 0.05,
                          -self.PHI_MAX, -self.THETA_MAX, -BIG,
                          -BIG, -BIG, -BIG,
                          -BIG, -BIG, -BIG])
        x_ub = np.array([ BIG,  BIG,  BIG,
                           self.PHI_MAX,  self.THETA_MAX,  BIG,
                           BIG,  BIG,  BIG,
                           BIG,  BIG,  BIG])

        # ── Build the MPC solver ─────────────────────────────────────────────
        solver_cls = MPCSolverCasADi if _CASADI_AVAILABLE else MPCSolverScipy
        self.get_logger().info(
            f'Using {"CasADi/IPOPT" if _CASADI_AVAILABLE else "scipy/SLSQP"} MPC solver  '
            f'(N={self.HORIZON}, dt={self.DT} s)')

        self.mpc = solver_cls(
            Ad=Ad, Bd=Bd,
            Q=Q,   R=R,   P=P,
            N=self.HORIZON,
            u_min=u_min, u_max=u_max,
            x_lb=x_lb,   x_ub=x_ub,
        )

        # ── Dynamic target ───────────────────────────────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # ── State ────────────────────────────────────────────────────────────
        self.pos_x = 0.0; self.pos_y = 0.0; self.pos_z = 0.0
        self.vel_x = 0.0; self.vel_y = 0.0; self.vel_z = 0.0
        self.roll  = 0.0; self.pitch = 0.0; self.yaw   = 0.0
        self.ang_vx = 0.0; self.ang_vy = 0.0; self.ang_vz = 0.0

        self._odom_received = False
        self._last_time: float | None = None
        self._t0:        float | None = None

        # ── History for plotting ─────────────────────────────────────────────
        n = self.HISTORY_LEN
        self.t_hist  = deque(maxlen=n)
        self.x_hist  = deque(maxlen=n); self.tx_hist = deque(maxlen=n)
        self.y_hist  = deque(maxlen=n); self.ty_hist = deque(maxlen=n)
        self.z_hist  = deque(maxlen=n); self.tz_hist = deque(maxlen=n)

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.odom_sub   = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.target_sub = self.create_subscription(
            PoseStamped, '/target_pose', self._target_cb, 10)
        self.cmd_pub    = self.create_publisher(
            Actuators, '/motor_commands', 10)

        self.create_timer(self.DT, self._control_loop)

        self.get_logger().info(
            f'MPC node ready. Default target → '
            f'x={self.TARGET_X}, y={self.TARGET_Y}, z={self.TARGET_Z} m')

    # ── Continuous-time linearised model ────────────────────────────────────
    def _build_continuous_model(self):
        """
        Identical A, B to the LQR version.

        State  x = [x,y,z, φ,θ,ψ, ẋ,ẏ,ż, φ̇,θ̇,ψ̇]   (12)
        Input  u = [F, τ_r, τ_p, τ_y]                  (4)
        """
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ

        A = np.zeros((12, 12))
        B = np.zeros((12, 4))

        # Kinematics
        A[0, 6] = A[1, 7] = A[2, 8] = 1.0
        A[3, 9] = A[4, 10] = A[5, 11] = 1.0

        # Linearised gravity coupling
        A[6, 4] =  g   # ẍ ← g·θ
        A[7, 3] = -g   # ÿ ← -g·φ

        # Thrust → vertical acceleration
        B[8, 0] = 1.0 / m

        # Torques → angular accelerations
        B[9,  1] = 1.0 / Ixx
        B[10, 2] = 1.0 / Iyy
        B[11, 3] = 1.0 / Izz

        return A, B

    # ── Target pose callback ─────────────────────────────────────────────────
    def _target_cb(self, msg: PoseStamped):
        self.TARGET_X = msg.pose.position.x
        self.TARGET_Y = msg.pose.position.y
        self.TARGET_Z = msg.pose.position.z

        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.TARGET_YAW = math.atan2(siny, cosy)

    # ── Odometry callback ────────────────────────────────────────────────────
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

        sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        self.pitch = math.asin(sinp)

        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.yaw = math.atan2(siny, cosy)

        self._odom_received = True
        self.get_logger().info(f'Odom: z={self.pos_z:.2f}')

    # ── Main control loop ────────────────────────────────────────────────────
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
        yaw_err = (self.TARGET_YAW - self.yaw + math.pi) % (2 * math.pi) - math.pi

        error = np.array([
            self.TARGET_X - self.pos_x,
            self.TARGET_Y - self.pos_y,
            self.TARGET_Z - self.pos_z,
            0.0 - self.roll,
            0.0 - self.pitch,
            yaw_err,
            0.0 - self.vel_x,
            0.0 - self.vel_y,
            0.0 - self.vel_z,
            0.0 - self.ang_vx,
            0.0 - self.ang_vy,
            0.0 - self.ang_vz,
        ])

        # ── MPC solve ────────────────────────────────────────────────────────
        # Returns the first optimal input *deviation* Δu_0 = u_0 − u_hover
        try:
            du_opt = self.mpc.solve(error)   # shape (4,)
        except Exception as exc:
            self.get_logger().error(f'MPC solve failed: {exc}')
            du_opt = np.zeros(4)

        # ── Reconstruct absolute control input ──────────────────────────────
        #   u = u_hover + Δu_0
        F_total   = self.F_hover + du_opt[0]
        tau_roll  = du_opt[1]
        tau_pitch = du_opt[2]
        tau_yaw   = du_opt[3]

        # Safety clamp on total thrust
        F_max = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        F_total = float(np.clip(F_total, 0.0, F_max))

        # ── Motor mixing: solve for ωi² ──────────────────────────────────────
        wrench   = np.array([F_total, tau_roll, tau_pitch, tau_yaw])
        omega_sq = self.Gamma_inv @ wrench
        omega_sq = np.clip(omega_sq, 0.0, self.OMEGA_MAX ** 2)
        omega    = np.sqrt(omega_sq)

        # ── Publish ──────────────────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)

        # ── Log state for debugging / plotting ──────────────────────────────
        self.get_logger().debug(
            f't={now - self._t0:.2f}s  '
            f'pos=({self.pos_x:.2f},{self.pos_y:.2f},{self.pos_z:.2f})  '
            f'err_z={error[2]:.3f}  du=[{du_opt[0]:.2f},{du_opt[1]:.3f},'
            f'{du_opt[2]:.3f},{du_opt[3]:.3f}]')


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
