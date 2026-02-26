#!/usr/bin/env python3
"""
Quadrotor Constrained MPC Hover Controller
===========================================
Controls a quadrotor to hold a fixed position (x, y, z) using Model
Predictive Control (MPC) with a finite receding horizon and hard constraints.

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

State vector  x  (12×1):
  [x, y, z, φ(roll), θ(pitch), ψ(yaw),
   ẋ, ẏ, ż, φ̇, θ̇, ψ̇]

Input vector  u  (4×1):
  [ΔF, Δτ_roll, Δτ_pitch, Δτ_yaw]   (deviation from hover trim)
  Equilibrium input u_eq = 0.

Constrained MPC QP (dense / condensed form):
  ─────────────────────────────────────────────────────────────────
  min   ½ U' H U  +  q' U
  s.t.
    (1) Input box:         u_min  ≤  u_k  ≤  u_max     ∀ k ∈ [0, N-1]
    (2) Attitude limits:  |φ_k|, |θ_k|  ≤  φ_max       ∀ k ∈ [1, N]
    (3) Altitude floor:   z_k  ≥  z_floor               ∀ k ∈ [1, N]
  ─────────────────────────────────────────────────────────────────
  where x_{k+1} = Ad x_k + Bd u_k,  x_0 = current state error,
        H = Φ'Q̄Φ + R̄,  q = Φ'Q̄Ψ x_0,
        Φ, Ψ are condensed prediction matrices.

  Solved with scipy.optimize.minimize (SLSQP) — no extra dependencies.

Terminal cost P: solution of Discrete Algebraic Riccati Equation (DARE).

Motor layout (top view, X-config):
        front (+x)
          2   0
      left     right
          1   3
        rear (-x)

  Motor 0: front-right  CCW    Motor 2: front-left   CW
  Motor 1: rear-left    CCW    Motor 3: rear-right   CW

Motor mixing:  wrench = Γ · [w0², w1², w2², w3²]
  → solved as  [wi²] = Γ⁻¹ · [F, τ_r, τ_p, τ_y]
  → wi = sqrt(wi²), clamped to [0, omega_max].
"""

import math
import threading
from collections import deque

import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg is unavailable
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import minimize, Bounds, LinearConstraint

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped


# ──────────────────────────────────────────────────────────────────────────────
# Helper: discrete-time ZOH model
# ──────────────────────────────────────────────────────────────────────────────

def c2d_zoh(A: np.ndarray, B: np.ndarray, dt: float):
    """Zero-Order Hold discretisation using matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    eM = expm(M * dt)
    return eM[:n, :n], eM[:n, n:]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: terminal cost (DARE)
# ──────────────────────────────────────────────────────────────────────────────

def dare_terminal(Ad, Bd, Q, R):
    """Solve DARE for the infinite-horizon LQR terminal cost matrix P."""
    return solve_discrete_are(Ad, Bd, Q, R)


# ──────────────────────────────────────────────────────────────────────────────
# MPC controller (constrained, dense / condensed QP)
# ──────────────────────────────────────────────────────────────────────────────

class MPC:
    """
    Constrained finite-horizon MPC using a condensed QP.

    Decision variable:  U = [u_0; u_1; …; u_{N-1}]  shape (m·N,)

    Cost:
        J(U) = ½ U' H U  +  q(x0)' U  +  const
        H    = Φ' Q̄ Φ + R̄        (pre-computed, positive-definite)
        q(x0)= Φ' Q̄ Ψ x0         (updated every step)

    Constraints (all expressed as linear in U):
        ① Input box:      lb_u ≤ U ≤ ub_u         → scipy Bounds
        ② Attitude limits |φ_k|, |θ_k| ≤ φ_max   → LinearConstraint
        ③ Altitude floor  z_k ≥ z_floor            → LinearConstraint

    Only u_0 (first action) is applied (receding-horizon principle).

    Parameters
    ----------
    Ad, Bd      : discrete-time state-space matrices
    Q, R, P     : stage / terminal cost matrices
    N           : prediction horizon (steps)
    u_min/max   : input box bounds (shape m, applied to every step)
    phi_max     : max absolute roll and pitch [rad]  (attitude constraint)
    z_floor_err : lower bound on z-error (= z_floor − z_target);
                  set to -inf to disable altitude constraint
    x0_ref      : reference state (usually zeros); constraints use it to
                  compute the actual state from the error signal
    target_z    : target altitude for converting error → absolute altitude
    """

    def __init__(self,
                 Ad: np.ndarray, Bd: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, P: np.ndarray,
                 N: int,
                 u_min: np.ndarray, u_max: np.ndarray,
                 phi_max: float = 0.5,    # rad  (~28°)
                 theta_max: float = 0.5,  # rad  (~28°)
                 z_floor: float = 0.1):   # absolute altitude floor [m]
        self.Ad, self.Bd = Ad, Bd
        self.Q, self.R, self.P = Q, R, P
        self.N = N
        self.u_min, self.u_max = u_min, u_max
        self.phi_max   = phi_max
        self.theta_max = theta_max
        self.z_floor   = z_floor

        n, m = Ad.shape[0], Bd.shape[1]
        self.n, self.m = n, m

        # ── Pre-compute condensed prediction matrices ─────────────────────────
        self.Psi, self.Phi = self._build_prediction_matrices(Ad, Bd, N)

        # ── Pre-compute cost matrices ─────────────────────────────────────────
        self._H, self._Q_bar = self._build_cost(self.Phi, Q, R, P, N, n, m)
        # Hessian is constant — pre-factorise for warm gradient computation
        self._PhiT_Qbar = self.Phi.T @ self._Q_bar   # (m*N) × (n*(N+1))
        self._PsiT_Qbar = self.Psi.T @ self._Q_bar   # n     × (n*(N+1))

        # ── Scipy Bounds (input box, repeated N times) ────────────────────────
        lb_u = np.tile(u_min, N)
        ub_u = np.tile(u_max, N)
        self._bounds = Bounds(lb=lb_u, ub=ub_u)

        # ── Pre-compute attitude and altitude constraint matrices ──────────────
        # State prediction: X = Psi x0 + Phi U   (shape n*(N+1))
        # For step k (k=1..N): x_k = Psi[k*n:(k+1)*n] x0 + Phi[k*n:(k+1)*n] U
        #
        # Attitude: φ (index 3), θ (index 4)  in x_k
        # Altitude: z (index 2)               in x_k
        #
        # Constraint on φ error: |φ_ref - φ_k| ≤ φ_max → expressed on x_k:
        #   -φ_max ≤ x_k[3] ≤ φ_max   (error = x_ref - x = target - current)
        # But the error vector already is e_k = x_ref - x_k, so e_k[3] = -φ_k
        # when x_ref[3]=0 → φ_k = -e_k[3]. We constrain the error e_k[3] = -φ_k.
        # Actually simpler: just constrain U so that the predicted error in
        # roll/pitch stays bounded, and compute the absolute state separately.
        # We store Phi_att and Phi_z so we can rebuild the constraint with the
        # current x0 every solve step.

        # Row selectors for each predicted step (k=1..N)
        self._C_roll    = self._extract_state_rows(n, N, 3)   # roll error rows
        self._C_pitch   = self._extract_state_rows(n, N, 4)   # pitch error rows
        self._C_z       = self._extract_state_rows(n, N, 2)   # z error rows

        self._prev_U = np.zeros(m * N)   # warm-start cache

    @staticmethod
    def _extract_state_rows(n, N, state_idx):
        """
        Return the rows of the (N×n*(N+1)) matrix that pick out state_idx
        for prediction steps k=1..N.
        """
        rows = np.zeros((N, n * (N + 1)))
        for k in range(1, N + 1):
            rows[k - 1, k * n + state_idx] = 1.0
        return rows   # shape (N, n*(N+1))

    @staticmethod
    def _build_prediction_matrices(Ad, Bd, N):
        """Build Ψ (n*(N+1)×n) and Φ (n*(N+1)×m*N)."""
        n, m = Ad.shape[0], Bd.shape[1]
        Psi = np.zeros((n * (N + 1), n))
        Phi = np.zeros((n * (N + 1), m * N))
        Ak  = np.eye(n)
        for k in range(N + 1):
            Psi[k * n:(k + 1) * n, :] = Ak
            for j in range(k):
                Phi[k * n:(k + 1) * n, j * m:(j + 1) * m] = (
                    np.linalg.matrix_power(Ad, k - 1 - j) @ Bd
                )
            Ak = Ak @ Ad
        return Psi, Phi

    @staticmethod
    def _build_cost(Phi, Q, R, P, N, n, m):
        """Build Hessian H = Φ'Q̄Φ + R̄ and Q̄."""
        Q_bar = np.zeros((n * (N + 1), n * (N + 1)))
        for k in range(N):
            Q_bar[k * n:(k + 1) * n, k * n:(k + 1) * n] = Q
        Q_bar[N * n:(N + 1) * n, N * n:(N + 1) * n] = P
        R_bar = np.kron(np.eye(N), R)
        H = Phi.T @ Q_bar @ Phi + R_bar
        return H, Q_bar

    def compute(self, x0: np.ndarray,
                current_z: float,
                current_roll: float,
                current_pitch: float) -> np.ndarray:
        """
        Solve the constrained MPC QP for current state error x0 and return
        the first optimal control action (deviation from hover trim).

        Parameters
        ----------
        x0           : state *error* x_ref - x_current  (shape n)
        current_z    : current absolute altitude [m]  (for altitude constraint)
        current_roll : current roll angle  [rad]      (for attitude constraint)
        current_pitch: current pitch angle [rad]      (for attitude constraint)

        Returns
        -------
        u0 : first optimal control deviation  [ΔF, Δτ_r, Δτ_p, Δτ_y]
        """
        n, m, N = self.n, self.m, self.N

        # ── Linear cost gradient:  q = Φ'Q̄Ψ x0 ──────────────────────────────
        q = self._PhiT_Qbar @ self.Psi @ x0   # (m*N,)

        # ── Attitude constraints (linear in U) ────────────────────────────────
        # Predicted roll error at step k: C_roll[k] @ (Psi x0 + Phi U) in [−φ,φ]
        # → C_roll @ Phi  U  ≤  φ_max  − C_roll @ Psi x0
        # → -C_roll @ Phi U  ≤  φ_max  + C_roll @ Psi x0
        Psi_x0 = self.Psi @ x0   # (n*(N+1),)

        A_roll  = self._C_roll  @ self.Phi   # (N, m*N)
        A_pitch = self._C_pitch @ self.Phi
        A_z     = self._C_z     @ self.Phi

        b_roll_Psi  = self._C_roll  @ Psi_x0   # (N,)
        b_pitch_Psi = self._C_pitch @ Psi_x0
        b_z_Psi     = self._C_z     @ Psi_x0

        # Roll:  -φ_max ≤ e_roll_k ≤ φ_max
        #   lower: -φ_max - b_Psi ≤ A_roll U
        #   upper:  φ_max - b_Psi ≥ A_roll U
        lb_roll = -self.phi_max   * np.ones(N) - b_roll_Psi
        ub_roll =  self.phi_max   * np.ones(N) - b_roll_Psi

        lb_pitch = -self.theta_max * np.ones(N) - b_pitch_Psi
        ub_pitch =  self.theta_max * np.ones(N) - b_pitch_Psi

        # Altitude floor: absolute z = z_target - e_z  ≥ z_floor
        #    z_target - (e_z_Psi + A_z U) ≥ z_floor
        #    -A_z U ≤ z_target - z_floor - e_z_Psi
        # error e_z = z_ref - z_current = x0[2] (entry 2 of state error)
        # We want the *absolute* predicted altitude, which is:
        #   z_k = z_target - e_z_k   (since e_z = z_ref - z)
        # z_k = z_target - (b_z_Psi[k] + A_z[k] U)  ≥ z_floor
        #   A_z U ≤ z_target - z_floor - b_z_Psi
        # z_target is fixed (TARGET_Z), so we store it via the sign of x0[2]
        z_target = current_z + x0[2]   # reconstruct from error
        ub_z_abs = (z_target - self.z_floor) * np.ones(N) - b_z_Psi
        # lower bound: no upper altitude limit (−∞)
        lb_z_abs = -np.inf * np.ones(N)

        # Stack attitude + altitude constraints
        A_ineq = np.vstack([A_roll, A_pitch, A_z])
        lb_ineq = np.concatenate([lb_roll, lb_pitch, lb_z_abs])
        ub_ineq = np.concatenate([ub_roll, ub_pitch, ub_z_abs])

        lin_con = LinearConstraint(A_ineq, lb_ineq, ub_ineq)

        # ── Cost functions for SLSQP ──────────────────────────────────────────
        H = self._H

        def cost(U):
            return 0.5 * U @ H @ U + q @ U

        def cost_grad(U):
            return H @ U + q

        # ── Solve ─────────────────────────────────────────────────────────────
        res = minimize(
            cost,
            self._prev_U,          # warm start
            jac=cost_grad,
            method='SLSQP',
            bounds=self._bounds,
            constraints=lin_con,
            options={'ftol': 1e-6, 'maxiter': 100, 'disp': False},
        )

        U_opt = res.x if res.success else self._prev_U
        self._prev_U = U_opt.copy()

        return U_opt[:m]   # first action


# ──────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ──────────────────────────────────────────────────────────────────────────────

class QuadrotorMPCNode(Node):

    # ── Physical constants ─────────────────────────────────────────────────
    MASS       = 1.5            # kg
    GRAVITY    = 9.81           # m/s²
    K_F        = 8.54858e-06    # N / (rad/s)²
    K_M        = 0.06           # N·m / (rad/s)²
    OMEGA_MAX  = 1500.0         # rad/s

    # Moments of inertia (from URDF)
    IXX = 0.0347563   # kg·m²
    IYY = 0.07        # kg·m²
    IZZ = 0.0977      # kg·m²

    # Arm length
    L = 0.22   # m

    # ── Default target pose (overridden live by /target_pose topic) ───────────
    DEFAULT_TARGET_X   = 0.0   # m
    DEFAULT_TARGET_Y   = 0.0   # m
    DEFAULT_TARGET_Z   = 1.0   # m
    DEFAULT_TARGET_YAW = 0.0   # rad

    # ── MPC settings ────────────────────────────────────────────────────────
    MPC_N  = 15    # prediction / control horizon
    MPC_DT = 0.01  # sample time [s]

    # ── Constraint parameters ────────────────────────────────────────────────
    PHI_MAX   = 0.4    # max |roll|  [rad]  (~23°)
    THETA_MAX = 0.4    # max |pitch| [rad]  (~23°)
    Z_FLOOR   = 0.05   # absolute altitude floor [m]

    # ── Plot history ───────────────────────────────────────────────────────
    HISTORY_LEN = 600

    def __init__(self):
        super().__init__('quadrotor_mpc_node')

        # ── Hover trim ──────────────────────────────────────────────────────
        self.F_hover     = self.MASS * self.GRAVITY
        self.omega_hover = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.get_logger().info(
            f'Hover ω: {self.omega_hover:.1f} rad/s  F_hover: {self.F_hover:.2f} N')

        # ── Motor allocation matrix Γ (4×4) ────────────────────────────────
        kF, kM, L = self.K_F, self.K_M, self.L
        self.Gamma = np.array([
            [ kF,         kF,         kF,        kF       ],
            [-kF * L,     kF * 0.2,   kF * L,   -kF * 0.2],
            [-kF * L,     kF * 0.2,  -kF * L,    kF * 0.2],
            [-kM,        -kM,         kM,         kM      ],
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        # ── Continuous-time linear model ────────────────────────────────────
        A_c, B_c = self._build_linear_model()

        # ── ZOH discretisation ──────────────────────────────────────────────
        Ad, Bd = c2d_zoh(A_c, B_c, self.MPC_DT)

        # ── Cost matrices ───────────────────────────────────────────────────
        Q = np.diag([
            120.0,  10.0,  900.0,   # position  x, y, z
             10.0,  10.0,  100.0,   # attitude  φ, θ, ψ
             10.0,   4.0,   50.0,   # velocity  ẋ, ẏ, ż
              1.0,   1.0,    1.0,   # ang-rate  φ̇, θ̇, ψ̇
        ])
        R = np.diag([10.0, 1.0, 1.0, 1.0])   # [ΔF, Δτ_r, Δτ_p, Δτ_y]
        P_terminal = dare_terminal(Ad, Bd, Q, R)

        # ── Input bounds (deviation from hover) ─────────────────────────────
        F_max = 4.0 * kF * self.OMEGA_MAX ** 2
        u_min = np.array([-self.F_hover, -5.0, -5.0, -1.0])
        u_max = np.array([ F_max - self.F_hover, 5.0, 5.0, 1.0])

        # ── Instantiate constrained MPC ─────────────────────────────────────
        self.get_logger().info('Building MPC prediction matrices …')
        self.mpc = MPC(
            Ad, Bd, Q, R, P_terminal,
            N        = self.MPC_N,
            u_min    = u_min,
            u_max    = u_max,
            phi_max  = self.PHI_MAX,
            theta_max= self.THETA_MAX,
            z_floor  = self.Z_FLOOR,
        )
        self.get_logger().info(
            f'Constrained MPC ready.  N={self.MPC_N}  dt={self.MPC_DT}s  '
            f'φ_max=±{math.degrees(self.PHI_MAX):.0f}°  '
            f'z_floor={self.Z_FLOOR}m')

        # ── State variables ─────────────────────────────────────────────────
        self.pos_x = 0.0;  self.pos_y = 0.0;  self.pos_z = 0.0
        self.vel_x = 0.0;  self.vel_y = 0.0;  self.vel_z = 0.0
        self.roll  = 0.0;  self.pitch = 0.0;  self.yaw   = 0.0
        self.ang_vx = 0.0; self.ang_vy = 0.0; self.ang_vz = 0.0

        self._odom_received  = False
        self._last_time: float | None = None
        self._t0:        float | None = None

        # ── Dynamic target (updated by /target_pose) ────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.odom_sub   = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.target_sub = self.create_subscription(
            PoseStamped, '/target_pose', self._target_cb, 10)
        self.cmd_pub    = self.create_publisher(
            Actuators, '/motor_commands', 10)

        self.create_timer(self.MPC_DT, self._control_loop)

        self.get_logger().info(
            f'MPC node ready. Target → x={self.TARGET_X}, '
            f'y={self.TARGET_Y}, z={self.TARGET_Z} m')

    # ── Linearised model ─────────────────────────────────────────────────────
    def _build_linear_model(self):
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ
        n, p = 12, 4
        A = np.zeros((n, n))
        B = np.zeros((n, p))
        # Kinematics
        for i in range(6):
            A[i, i + 6] = 1.0
        # Gravity coupling
        A[6, 4] =  g
        A[7, 3] = -g
        # Input coupling
        B[8, 0]  = 1.0 / m
        B[9, 1]  = 1.0 / Ixx
        B[10, 2] = 1.0 / Iyy
        B[11, 3] = 1.0 / Izz
        return A, B

    # ── Target pose callback ─────────────────────────────────────────────────
    def _target_cb(self, msg: PoseStamped):
        """Update the MPC setpoint from the trajectory / goal publisher."""
        self.TARGET_X = msg.pose.position.x
        self.TARGET_Y = msg.pose.position.y
        self.TARGET_Z = msg.pose.position.z

        # Extract yaw from quaternion
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

        # ── Build state error ────────────────────────────────────────────────
        yaw_err = self.TARGET_YAW - self.yaw
        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi

        x_err = np.array([
            self.TARGET_X - self.pos_x,
            self.TARGET_Y - self.pos_y,
            self.TARGET_Z - self.pos_z,
            0.0           - self.roll,
            0.0           - self.pitch,
            yaw_err,
            0.0           - self.vel_x,
            0.0           - self.vel_y,
            0.0           - self.vel_z,
            0.0           - self.ang_vx,
            0.0           - self.ang_vy,
            0.0           - self.ang_vz,
        ])

        # ── Solve constrained MPC ────────────────────────────────────────────
        u0 = self.mpc.compute(x_err, self.pos_z, self.roll, self.pitch)

        F_total   = self.F_hover + u0[0]
        tau_roll  = u0[1]
        tau_pitch = u0[2]
        tau_yaw   = u0[3]

        # ── Clamp total thrust ───────────────────────────────────────────────
        F_max   = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        F_total = float(np.clip(F_total, 0.0, F_max))

        # ── Motor mixing ─────────────────────────────────────────────────────
        wrench   = np.array([F_total, tau_roll, tau_pitch, tau_yaw])
        omega_sq = self.Gamma_inv @ wrench
        omega_sq = np.clip(omega_sq, 0.0, self.OMEGA_MAX ** 2)
        omega    = np.sqrt(omega_sq)

        # ── Publish ──────────────────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)


# ── Entry point ─────────────────────────────────────────────────────────────

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
