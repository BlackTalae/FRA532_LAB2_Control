#!/usr/bin/env python3
"""
Quadrotor iterative LQR (iLQR) Controller
==========================================
Controls a quadrotor to hold a fixed position using an iterative Linear
Quadratic Regulator (iLQR) over a finite receding horizon.

iLQR Algorithm
--------------
Given a nominal trajectory (x̄, ū), iLQR solves the finite-horizon optimal
control problem by iterating two passes until the cost converges:

  Backward pass  – compute time-varying feedback/feedforward gains
                   (k_t, K_t) by solving the discrete Riccati recursion
                   backward from the terminal cost along the nominal traj.

  Forward pass   – rollout the *nonlinear* dynamics with the gains to
                   produce a new (potentially better) trajectory, using a
                   line-search on step size α ∈ (0, 1] to guarantee
                   cost decrease.

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
"""

import math
import numpy as np

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
# iLQR Solver
# ─────────────────────────────────────────────────────────────────────────────

class ILQRSolver:
    """
    Iterative LQR for a general discrete-time system f(x, u).

    The solver produces time-varying affine feedback policies:
        δu_t = k_t + K_t · δx_t
    where δx_t = x_t − x̄_t  is the deviation from the nominal trajectory.

    Parameters
    ----------
    f        : callable (x, u) → x_next       nonlinear dynamics
    cost     : callable (x, u, x_ref) → float  running cost
    cost_terminal : callable (x, x_ref) → float terminal cost
    n        : int   state dimension
    m        : int   input dimension
    N        : int   horizon length (steps)
    Q        : (n,n) running state cost
    R        : (m,m) running input cost
    Q_f      : (n,n) terminal state cost
    max_iter : int   maximum iLQR iterations
    tol      : float convergence threshold on cost improvement
    """

    def __init__(self, f, n, m, N, Q, R, Q_f,
                 max_iter=20, tol=1e-4, reg_init=1e-3):
        self.f        = f
        self.n        = n
        self.m        = m
        self.N        = N
        self.Q        = Q
        self.R        = R
        self.Q_f      = Q_f
        self.max_iter = max_iter
        self.tol      = tol
        self.reg      = reg_init   # regularisation (Marquardt)

    # ── Numerical Jacobians ────────────────────────────────────────────────

    def _jacobians(self, x, u, eps=1e-5):
        """
        Finite-difference Jacobians of f(x, u) w.r.t. x and u.
        Returns A (n×n), B (n×m).
        """
        n, m = self.n, self.m
        x_nom = self.f(x, u)
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            A[:, i] = (self.f(x + dx, u) - x_nom) / eps
        for j in range(m):
            du = np.zeros(m); du[j] = eps
            B[:, j] = (self.f(x, u + du) - x_nom) / eps
        return A, B

    # ── Running cost quadratic expansion ──────────────────────────────────

    def _cost_quadratic(self, x, u, x_ref):
        """
        Quadratic expansion of the running cost l(x, u) around (x, u).

        l(x, u) = (x−x_ref)' Q (x−x_ref) + u' R u
        """
        ex = x - x_ref
        lx  = 2.0 * self.Q @ ex           # ∂l/∂x
        lu  = 2.0 * self.R @ u             # ∂l/∂u
        lxx = 2.0 * self.Q                 # ∂²l/∂x²
        luu = 2.0 * self.R                 # ∂²l/∂u²
        lxu = np.zeros((self.n, self.m))   # ∂²l/∂x∂u (zero for separable cost)
        return lx, lu, lxx, luu, lxu

    # ── Terminal cost quadratic expansion ─────────────────────────────────

    def _terminal_quadratic(self, x, x_ref):
        """
        Quadratic expansion of the terminal cost phi(x).

        phi(x) = (x−x_ref)' Q_f (x−x_ref)
        """
        ex  = x - x_ref
        Vx  = 2.0 * self.Q_f @ ex
        Vxx = 2.0 * self.Q_f
        return Vx, Vxx

    # ── Backward pass ──────────────────────────────────────────────────────

    def backward_pass(self, X, U, x_ref):
        """
        Compute the iLQR backward pass given a nominal trajectory (X, U).

        X : (N+1, n)  nominal state trajectory
        U : (N,   m)  nominal input trajectory
        x_ref : (n,)  reference / goal state

        Returns
        -------
        ks : list of N arrays (m,)     feedforward gains
        Ks : list of N arrays (m, n)   feedback   gains
        dV : (2,) expected cost improvement [dV1, dV2]
        """
        n, m, N = self.n, self.m, self.N

        ks = [None] * N
        Ks = [None] * N

        # Initialise from terminal cost
        Vx, Vxx = self._terminal_quadratic(X[N], x_ref)

        dV = np.zeros(2)   # dV[0] = Σ k'l_u,  dV[1] = Σ k'Q_uu k / 2

        for t in reversed(range(N)):
            x_t, u_t = X[t], U[t]

            # Linearise dynamics
            A_t, B_t = self._jacobians(x_t, u_t)

            # Cost quadratics at step t
            lx_t, lu_t, lxx_t, luu_t, lxu_t = self._cost_quadratic(x_t, u_t, x_ref)

            # Q-function matrices
            Qx  = lx_t  + A_t.T @ Vx
            Qu  = lu_t  + B_t.T @ Vx
            Qxx = lxx_t + A_t.T @ Vxx @ A_t
            Quu = luu_t + B_t.T @ Vxx @ B_t
            Qux = lxu_t.T + B_t.T @ Vxx @ A_t

            # Regularise Quu for numerical stability (Levenberg-Marquardt)
            Quu_reg = Quu + self.reg * np.eye(m)

            try:
                Quu_inv = np.linalg.inv(Quu_reg)
            except np.linalg.LinAlgError:
                # If still singular, increase regularisation
                self.reg = min(self.reg * 10, 1e6)
                Quu_inv = np.linalg.inv(Quu + self.reg * np.eye(m))

            # Optimal gains
            k_t = -Quu_inv @ Qu
            K_t = -Quu_inv @ Qux

            ks[t] = k_t
            Ks[t] = K_t

            # Expected cost improvement
            dV[0] += float(k_t @ Qu)
            dV[1] += 0.5 * float(k_t @ Quu @ k_t)

            # Riccati recursion
            Vx  = Qx  + K_t.T @ Quu @ k_t + K_t.T @ Qu  + Qux.T @ k_t
            Vxx = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
            Vxx = 0.5 * (Vxx + Vxx.T)   # symmetrise

        return ks, Ks, dV

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward_pass(self, X, U, ks, Ks, alpha=1.0):
        """
        Rollout the nonlinear dynamics with the computed gains and step size α.

        X_new[t+1] = f(X_new[t], U_new[t])
        U_new[t]   = U[t] + α·k[t] + K[t]·(X_new[t] − X[t])

        Returns X_new (N+1, n), U_new (N, m).
        """
        N, n, m = self.N, self.n, self.m
        X_new = np.zeros((N + 1, n))
        U_new = np.zeros((N, m))
        X_new[0] = X[0].copy()

        for t in range(N):
            dx = X_new[t] - X[t]
            U_new[t] = U[t] + alpha * ks[t] + Ks[t] @ dx
            X_new[t + 1] = self.f(X_new[t], U_new[t])

        return X_new, U_new

    # ── Total cost ────────────────────────────────────────────────────────

    def total_cost(self, X, U, x_ref):
        """Compute the total trajectory cost."""
        cost = 0.0
        for t in range(self.N):
            ex = X[t] - x_ref
            cost += float(ex @ self.Q @ ex + U[t] @ self.R @ U[t])
        ex_f = X[self.N] - x_ref
        cost += float(ex_f @ self.Q_f @ ex_f)
        return cost

    # ── Main solve ────────────────────────────────────────────────────────

    def solve(self, x0, X_init, U_init, x_ref):
        """
        Run iLQR from initial state x0.

        Parameters
        ----------
        x0     : (n,)       current state
        X_init : (N+1, n)   initial nominal state trajectory
        U_init : (N,   m)   initial nominal input trajectory
        x_ref  : (n,)       reference / goal state

        Returns
        -------
        X_opt : (N+1, n)   optimised state trajectory
        U_opt : (N,   m)   optimised input trajectory
        """
        X = X_init.copy()
        U = U_init.copy()
        X[0] = x0

        old_cost = self.total_cost(X, U, x_ref)

        for i in range(self.max_iter):
            ks, Ks, dV = self.backward_pass(X, U, x_ref)

            # Line search: try α = 1, 0.5, 0.25, ...
            success = False
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                X_new, U_new = self.forward_pass(X, U, ks, Ks, alpha)
                new_cost = self.total_cost(X_new, U_new, x_ref)
                if new_cost < old_cost:
                    success = True
                    break

            if not success:
                # Increase regularisation and retry
                self.reg = min(self.reg * 10.0, 1e6)
                continue
            else:
                # Accept step; decrease regularisation
                self.reg = max(self.reg / 5.0, 1e-6)

            improvement = old_cost - new_cost
            X, U = X_new, U_new
            old_cost = new_cost

            if improvement < self.tol:
                break

        return X, U


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────

class QuadrotorILQRNode(Node):

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

    # ── Default target ─────────────────────────────────────────────────────────
    DEFAULT_TARGET_X   = 0.0
    DEFAULT_TARGET_Y   = 0.0
    DEFAULT_TARGET_Z   = 1.0
    DEFAULT_TARGET_YAW = 0.0

    # ── iLQR horizon & cost ────────────────────────────────────────────────────
    HORIZON   = 20       # number of steps in the finite horizon
    DT        = 0.01     # seconds per step (must match control timer period)
    MAX_ITER  = 10       # maximum iLQR iterations per control cycle

    def __init__(self):
        super().__init__('quadrotor_ilqr_node')

        # ── Hover trim ────────────────────────────────────────────────────────
        self.F_hover = self.MASS * self.GRAVITY
        self.u_hover = np.array([self.F_hover, 0.0, 0.0, 0.0])
        omega_hover  = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.get_logger().info(
            f'Hover ω: {omega_hover:.1f} rad/s  F_hover: {self.F_hover:.2f} N')

        # ── Motor allocation matrix Γ (4×4) ───────────────────────────────────
        #  Maps [F, τ_roll, τ_pitch, τ_yaw] → [ω0², ω1², ω2², ω3²]
        kF, kM, LL = self.K_F, self.K_M, self.L
        self.Gamma = np.array([
            [ kF,       kF,       kF,       kF      ],   # Total thrust
            [-kF * LL,  kF * 0.2, kF * LL, -kF * 0.2],  # τ_roll
            [-kF * LL,  kF * 0.2,-kF * LL,  kF * 0.2],  # τ_pitch
            [-kM,      -kM,       kM,        kM     ],   # τ_yaw
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        # ── Input bounds ──────────────────────────────────────────────────────
        self.F_max   = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        self.tau_max = 2.0 * self.K_F * self.L * self.OMEGA_MAX ** 2
        self.u_min   = np.array([0.0,        -self.tau_max, -self.tau_max, -self.tau_max])
        self.u_max   = np.array([self.F_max,  self.tau_max,  self.tau_max,  self.tau_max])

        # ── iLQR cost matrices ────────────────────────────────────────────────
        # State: [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]
        Q = np.diag([
            120.0, 120.0, 900.0,    # position  x, y, z
            10.0,  10.0,  1500.0,   # attitude  φ, θ, ψ
            10.0,  10.0,  50.0,     # velocity  ẋ, ẏ, ż
            1.0,   1.0,   10.0,     # ang-rate  φ̇, θ̇, ψ̇
        ])
        # Terminal cost (heavier to pull state to goal at end of horizon)
        Q_f = 5.0 * Q
        # Input cost
        R = np.diag([1.0, 1.0, 1.0, 0.001])

        # ── Nonlinear quadrotor dynamics (discrete, Euler integration) ────────
        def f(x, u):
            return self._dynamics(x, u)

        # ── Build iLQR solver ─────────────────────────────────────────────────
        self.solver = ILQRSolver(
            f        = f,
            n        = 12,
            m        = 4,
            N        = self.HORIZON,
            Q        = Q,
            R        = R,
            Q_f      = Q_f,
            max_iter = self.MAX_ITER,
            tol      = 1e-4,
        )

        # ── Warm-start trajectory ─────────────────────────────────────────────
        #    Initialise with hover trim repeated for all steps
        self._X_warm = None   # (N+1, 12)  – set on first odom
        self._U_warm = np.tile(self.u_hover, (self.HORIZON, 1))  # (N, 4)

        # ── Target setpoint ───────────────────────────────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # ── Drone state ───────────────────────────────────────────────────────
        self.state       = np.zeros(12)
        self.state_ready = False

        # ── ROS interfaces ─────────────────────────────────────────────────────
        self.create_subscription(Odometry,    '/odom',        self._cb_odom,   10)
        self.create_subscription(PoseStamped, '/target_pose', self._cb_target,  10)
        self.cmd_pub = self.create_publisher(Actuators, '/motor_commands', 10)

        self.create_timer(self.DT, self._cb_control)

        self.get_logger().info(
            f'iLQR node ready | horizon={self.HORIZON} steps | '
            f'dt={self.DT*1e3:.0f} ms | '
            f'default target → ({self.TARGET_X}, {self.TARGET_Y}, {self.TARGET_Z}) m')

    # ─────────────────────────────────────────────────────────────────────────
    # Nonlinear quadrotor dynamics (Euler integration over DT)
    # ─────────────────────────────────────────────────────────────────────────

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Discrete-time quadrotor dynamics using Euler integration.

        State   x  = [px, py, pz, φ, θ, ψ, vx, vy, vz, φ̇, θ̇, ψ̇]
        Input   u  = [F, τ_φ, τ_θ, τ_ψ]

        The translational equations use the full body-to-world rotation
        (small-angle approximation for efficiency near hover):

            ẍ_world =  (F/m)(sθ cψ + sφ sψ)   – for nonlinear sim
            at hover:  ẍ ≈ g θ,  ÿ ≈ −g φ  (linearised coupling kept)

        For the iLQR we use the nonlinear rigid-body model so the
        Jacobians (computed numerically) correctly capture attitude coupling.
        """
        dt = self.DT
        m  = self.MASS
        g  = self.GRAVITY

        px, py, pz = x[0], x[1], x[2]
        phi, tht, psi = x[3], x[4], x[5]
        vx, vy, vz = x[6], x[7], x[8]
        dphi, dtht, dpsi = x[9], x[10], x[11]

        F, tau_phi, tau_tht, tau_psi = u[0], u[1], u[2], u[3]

        # ── Translational accelerations (full rotation matrix, ZYX Euler) ─────
        cphi = math.cos(phi);  sphi = math.sin(phi)
        ctht = math.cos(tht);  stht = math.sin(tht)
        cpsi = math.cos(psi);  spsi = math.sin(psi)

        ax = (F / m) * (stht * cpsi + sphi * spsi)
        ay = (F / m) * (stht * spsi - sphi * cpsi)
        az = (F / m) * (cphi * ctht) - g

        # ── Rotational accelerations (simplified – diagonal inertia) ──────────
        ddphi = tau_phi  / self.IXX
        ddtht = tau_tht  / self.IYY
        ddpsi = tau_psi  / self.IZZ

        # ── Euler integration ─────────────────────────────────────────────────
        x_next = np.array([
            px   + vx   * dt,
            py   + vy   * dt,
            pz   + vz   * dt,
            phi  + dphi * dt,
            tht  + dtht * dt,
            psi  + dpsi * dt,
            vx   + ax   * dt,
            vy   + ay   * dt,
            vz   + az   * dt,
            dphi + ddphi * dt,
            dtht + ddtht * dt,
            dpsi + ddpsi * dt,
        ])

        return x_next

    # ─────────────────────────────────────────────────────────────────────────
    # Odometry callback
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        """Pack sensor data into self.state."""
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
        """Update setpoint from /target_pose."""
        self.TARGET_X = msg.pose.position.x
        self.TARGET_Y = msg.pose.position.y
        self.TARGET_Z = msg.pose.position.z

        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.TARGET_YAW = math.atan2(siny, cosy)

        # Reset warm-start when goal changes significantly
        self._X_warm = None

    # ─────────────────────────────────────────────────────────────────────────
    # Control loop (runs at 1/DT Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_control(self):
        if not self.state_ready:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # ── Reference state: desired position / yaw, zero velocities ─────────
        x_ref = np.array([
            self.TARGET_X,
            self.TARGET_Y,
            self.TARGET_Z,
            0.0,                  # desired roll  = 0
            0.0,                  # desired pitch = 0
            self.TARGET_YAW,
            0.0, 0.0, 0.0,        # desired velocities = 0
            0.0, 0.0, 0.0,        # desired ang-rates  = 0
        ])
        x_ref[5] = wrap_angle(x_ref[5])

        # ── Warm-start trajectory ─────────────────────────────────────────────
        if self._X_warm is None:
            # Initialise by rolling out from current state with hover trim
            self._X_warm = np.zeros((self.HORIZON + 1, 12))
            self._X_warm[0] = self.state.copy()
            for t in range(self.HORIZON):
                self._X_warm[t + 1] = self._dynamics(
                    self._X_warm[t], self._U_warm[t])
        else:
            # Shift warm-start by one step (receding horizon)
            self._X_warm[:-1] = self._X_warm[1:]
            self._X_warm[-1]  = self._X_warm[-2].copy()
            self._U_warm[:-1] = self._U_warm[1:]
            # Keep the last u; the next iLQR solve will update it

        # Override the initial state with the fresh measurement
        self._X_warm[0] = self.state.copy()

        # ── Run iLQR ──────────────────────────────────────────────────────────
        X_opt, U_opt = self.solver.solve(
            x0     = self.state,
            X_init = self._X_warm,
            U_init = self._U_warm,
            x_ref  = x_ref,
        )

        # Update warm-start for next cycle
        self._X_warm = X_opt
        self._U_warm = U_opt

        # ── Apply first control input (receding horizon) ──────────────────────
        u_opt = U_opt[0].copy()
        u_opt = np.clip(u_opt, self.u_min, self.u_max)

        # ── Motor mixing: Γ⁻¹ · [F, τ_r, τ_p, τ_y] → ω ──────────────────────
        omega_sq = self.Gamma_inv @ u_opt
        omega_sq = np.clip(omega_sq, 0.0, self.OMEGA_MAX ** 2)
        omega    = np.sqrt(omega_sq)

        # ── Publish ───────────────────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)

        # ── Diagnostics (~2 Hz) ───────────────────────────────────────────────
        if int(now_sec * 2) % 4 == 0:
            pos = self.state[:3]
            rpy = np.degrees(self.state[3:6])
            ref = np.array([self.TARGET_X, self.TARGET_Y, self.TARGET_Z])
            err = float(np.linalg.norm(pos - ref))
            du  = u_opt - self.u_hover
            self.get_logger().info(
                f'pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] '
                f'ref=[{ref[0]:+.2f},{ref[1]:+.2f},{ref[2]:+.2f}] '
                f'err={err:.3f}m | '
                f'rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]deg | '
                f'dF={du[0]:+.2f}N | '
                f'omega={np.round(omega).astype(int)} rad/s'
            )


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorILQRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
