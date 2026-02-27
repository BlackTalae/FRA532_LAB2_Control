#!/usr/bin/env python3
"""
mpc_controller.py
=================
Pure-Python MPC controller for a quadrotor.
No ROS dependencies – can be unit-tested or used in simulation independently.

Public API
----------
    from mpc_controller import QuadrotorMPC

    mpc = QuadrotorMPC()          # build model, discretise, compile solver
    du  = mpc.solve(error)        # error = x_ref - x_current  (12,)
    # du = [ΔF, Δτ_roll, Δτ_pitch, Δτ_yaw]  (deviations from hover trim)

MPC formulation
---------------
Prediction model:
    Discrete-time LTI (ZOH) derived from the linearised 6-DOF quadrotor
    around hover.

Horizon:
    N = 20 steps, dt = 0.01 s  →  0.20 s look-ahead.

Cost:
    J = Σ_{k=0}^{N-1} [e_k' Q e_k  +  Δu_k' R Δu_k]
        + e_N' P e_N        (P = DARE terminal cost)

    e_k   = x_k - x_ref
    Δu_k  = u_k - u_hover  (deviation from trim)

Constraints (enforced inside the optimiser every step):
    F_total  ∈ [0,      F_max  ]
    τ_roll   ∈ [-τ_max,  τ_max ]
    τ_pitch  ∈ [-τ_max,  τ_max ]
    τ_yaw    ∈ [-τ_ym,   τ_ym  ]
    roll     ∈ [-φ_max,  φ_max ]   (±30 deg)
    pitch    ∈ [-θ_max,  θ_max ]   (±30 deg)

Solver:
    CasADi + IPOPT  (pip install casadi)  – preferred, warm-started.
    Falls back to scipy SLSQP if CasADi is absent.

State vector x (12×1):
    [x, y, z, φ, θ, ψ, ẋ, ẏ, ż, φ̇, θ̇, ψ̇]

Input vector u (4×1):
    [F_total, τ_roll, τ_pitch, τ_yaw]
"""

from __future__ import annotations

import math
import warnings
from utils.mpc_utils import *
import numpy as np

# ── Optional CasADi ──────────────────────────────────────────────────────────
try:
    import casadi as ca
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False
    warnings.warn(
        "CasADi not found – falling back to scipy SLSQP solver. "
        "Install for better performance:  pip install casadi",
        stacklevel=2,
    )




# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level solver: CasADi / IPOPT
# ═══════════════════════════════════════════════════════════════════════════════

class _MPCSolverCasADi:
    """
    Condensed MPC QP solved with CasADi + IPOPT.

    The decision variable is:
        Z = [Δu_0, …, Δu_{N-1}]  ∈ ℝ^{N·m}

    The problem is parameterised by the current state error e0, so the
    solver object is compiled once and queried at each control step.
    """

    def __init__(
        self,
        Phi:   np.ndarray,
        Theta: np.ndarray,
        Q_bar: np.ndarray,
        R_bar: np.ndarray,
        n: int, m: int, N: int,
        u_min: np.ndarray,
        u_max: np.ndarray,
        x_lb:  np.ndarray | None,
        x_ub:  np.ndarray | None,
    ):
        self.m, self.N = m, N
        H = Theta.T @ Q_bar @ Theta + R_bar
        H = (H + H.T) / 2.0

        Z  = ca.MX.sym('Z',  N * m)
        e0 = ca.MX.sym('e0', n)

        f_mat = 2.0 * Theta.T @ Q_bar @ Phi
        cost  = 0.5 * ca.mtimes([Z.T, H, Z]) + ca.mtimes(f_mat, e0).T @ Z

        lbz = np.tile(u_min, N)
        ubz = np.tile(u_max, N)

        g_list, lbg, ubg = [], [], []
        if x_lb is not None and x_ub is not None:
            X_pred = Phi @ e0 + ca.mtimes(Theta, Z)
            g_list.append(X_pred)
            lbg = np.tile(x_lb, N)
            ubg = np.tile(x_ub, N)

        g = ca.vertcat(*g_list) if g_list else ca.MX(0, 1)

        nlp  = {'x': Z, 'p': e0, 'f': cost, 'g': g}
        opts = {
            'ipopt.print_level':           0,
            'ipopt.max_iter':              200,
            'ipopt.tol':                   1e-6,
            'ipopt.warm_start_init_point': 'yes',
            'print_time':                  False,
        }
        self._solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._lbz, self._ubz = lbz, ubz
        self._lbg = np.array(lbg)
        self._ubg = np.array(ubg)
        self._prev_sol: np.ndarray | None = None

    def solve(self, error: np.ndarray) -> np.ndarray:
        m, N = self.m, self.N
        z0   = self._prev_sol if self._prev_sol is not None else np.zeros(N * m)

        res   = self._solver(
            x0=z0, p=error,
            lbx=self._lbz, ubx=self._ubz,
            lbg=self._lbg, ubg=self._ubg,
        )
        Z_opt = np.array(res['x']).flatten()
        self._prev_sol = np.concatenate([Z_opt[m:], Z_opt[-m:]])   # shift warm-start
        return Z_opt[:m]


# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level solver: scipy / SLSQP
# ═══════════════════════════════════════════════════════════════════════════════

class _MPCSolverScipy:
    """
    Condensed MPC QP solved with scipy.optimize.minimize (SLSQP).
    Slower than CasADi; no C++ dependencies.
    """

    def __init__(
        self,
        Phi:   np.ndarray,
        Theta: np.ndarray,
        Q_bar: np.ndarray,
        R_bar: np.ndarray,
        n: int, m: int, N: int,
        u_min: np.ndarray,
        u_max: np.ndarray,
        x_lb:  np.ndarray | None,
        x_ub:  np.ndarray | None,
    ):
        self.m, self.N = m, N
        self._Phi   = Phi
        self._Theta = Theta
        self._x_lb, self._x_ub = x_lb, x_ub

        H = Theta.T @ Q_bar @ Theta + R_bar
        self._H     = (H + H.T) / 2.0
        self._f_mat = 2.0 * Theta.T @ Q_bar @ Phi
        self._bounds = list(zip(np.tile(u_min, N), np.tile(u_max, N)))
        self._prev_sol: np.ndarray | None = None

    def solve(self, error: np.ndarray) -> np.ndarray:
        from scipy.optimize import minimize

        m, N = self.m, self.N
        H    = self._H
        f    = self._f_mat @ error
        z0   = self._prev_sol if self._prev_sol is not None else np.zeros(N * m)

        constraints = []
        if self._x_lb is not None and self._x_ub is not None:
            A   = self._Theta
            rhs = self._Phi @ error
            lb  = np.tile(self._x_lb, N) - rhs
            ub  = np.tile(self._x_ub, N) - rhs
            constraints += [
                {'type': 'ineq', 'fun': lambda Z: A @ Z - lb, 'jac': lambda _: A},
                {'type': 'ineq', 'fun': lambda Z: ub - A @ Z, 'jac': lambda _: -A},
            ]

        res   = minimize(
            lambda Z: 0.5 * Z @ H @ Z + f @ Z,
            z0,
            jac=lambda Z: H @ Z + f,
            method='SLSQP',
            bounds=self._bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8},
        )
        Z_opt = res.x
        self._prev_sol = np.concatenate([Z_opt[m:], Z_opt[-m:]])
        return Z_opt[:m]


# ═══════════════════════════════════════════════════════════════════════════════
#  Public interface: QuadrotorMPC
# ═══════════════════════════════════════════════════════════════════════════════

class QuadrotorMPC:
    """
    MPC controller for a quadrotor.

    Parameters (all optional – match robot_params.xacro defaults)
    ----------
    mass        : float   vehicle mass [kg]
    gravity     : float   gravitational acceleration [m/s²]
    k_F         : float   motor thrust coefficient  [N / (rad/s)²]
    k_M         : float   motor torque coefficient  [N·m / (rad/s)²]
    omega_max   : float   max motor speed [rad/s]
    Ixx, Iyy, Izz : float moments of inertia [kg·m²]
    arm_length  : float   CoM to rotor distance [m]
    dt          : float   control period [s]
    horizon     : int     MPC prediction horizon (steps)
    Q           : ndarray (12,12) state-error weight matrix
    R           : ndarray (4,4)  input-deviation weight matrix
    phi_max     : float   roll  constraint [rad]   default 30°
    theta_max   : float   pitch constraint [rad]   default 30°
    tau_max     : float   roll/pitch torque limit [N·m]  default 2.0
    tau_yaw_max : float   yaw torque limit        [N·m]  default 0.5
    use_casadi  : bool    force CasADi (True) or scipy (False); None = auto

    Usage
    -----
        mpc = QuadrotorMPC()
        du  = mpc.solve(error)   # error shape (12,)
        # du[0] = ΔF,  du[1] = Δτ_r,  du[2] = Δτ_p,  du[3] = Δτ_y
    """

    # ── Physical defaults ────────────────────────────────────────────────────
    _DEFAULT = dict(
        mass        = 1.5,
        gravity     = 9.81,
        k_F         = 8.54858e-06,
        k_M         = 0.06,
        omega_max   = 1500.0,
        Ixx         = 0.0347563,
        Iyy         = 0.07,
        Izz         = 0.0977,
        arm_length  = 0.22,
        dt          = 0.01,
        horizon     = 20,
        phi_max     = math.radians(30),
        theta_max   = math.radians(30),
        tau_max     = 2.0,
        tau_yaw_max = 0.5,
        use_casadi  = None,   # auto
    )

    def __init__(self, **kwargs):
        cfg = {**self._DEFAULT, **kwargs}

        # ── Store physical parameters ────────────────────────────────────────
        self.mass       = cfg['mass']
        self.gravity    = cfg['gravity']
        self.k_F        = cfg['k_F']
        self.k_M        = cfg['k_M']
        self.omega_max  = cfg['omega_max']
        self.Ixx        = cfg['Ixx']
        self.Iyy        = cfg['Iyy']
        self.Izz        = cfg['Izz']
        self.L          = cfg['arm_length']
        self.dt         = cfg['dt']
        self.N          = cfg['horizon']

        # ── Hover trim ───────────────────────────────────────────────────────
        self.F_hover     = self.mass * self.gravity
        self.omega_hover = math.sqrt(self.F_hover / (4.0 * self.k_F))

        # ── Cost matrices (user-provided or default) ─────────────────────────
        if 'Q' in kwargs:
            Q = np.asarray(kwargs['Q'], dtype=float)
        else:
            Q = np.diag([
                120.0, 10.0, 900.0,   # position  x, y, z
                 10.0, 10.0, 100.0,   # attitude  φ, θ, ψ
                 10.0,  4.0,  50.0,   # velocity  ẋ, ẏ, ż
                  1.0,  1.0,   1.0,   # ang-rate  φ̇, θ̇, ψ̇
            ])

        if 'R' in kwargs:
            R = np.asarray(kwargs['R'], dtype=float)
        else:
            R = np.diag([10.0, 1.0, 1.0, 1.0])

        # ── Discrete-time model ──────────────────────────────────────────────
        Ac, Bc = self._build_continuous_model()
        Ad, Bd = c2d(Ac, Bc, self.dt)

        # ── Terminal cost via DARE ───────────────────────────────────────────
        try:
            P = solve_discrete_are(Ad, Bd, Q, R)
        except Exception:
            warnings.warn("DARE failed; using Q as terminal cost.", stacklevel=2)
            P = Q.copy()

        # ── Block-diagonal cost matrices over horizon ────────────────────────
        n, m = 12, 4
        Q_bar = np.block([
            [np.kron(np.eye(self.N - 1), Q), np.zeros(((self.N - 1) * n, n))],
            [np.zeros((n, (self.N - 1) * n)), P],
        ])
        R_bar = np.kron(np.eye(self.N), R)

        # ── Condensed prediction ─────────────────────────────────────────────
        Phi, Theta = build_condensed(Ad, Bd, self.N)

        # ── Input bounds (as deviations from hover trim) ─────────────────────
        F_max = 4.0 * self.k_F * self.omega_max ** 2
        u_min = np.array([0.0   - self.F_hover,
                          -cfg['tau_max'],
                          -cfg['tau_max'],
                          -cfg['tau_yaw_max']])
        u_max = np.array([F_max - self.F_hover,
                           cfg['tau_max'],
                           cfg['tau_max'],
                           cfg['tau_yaw_max']])

        # ── State bounds ─────────────────────────────────────────────────────
        BIG = 1e6
        x_lb = np.array([-BIG, -BIG,  0.05,
                          -cfg['phi_max'], -cfg['theta_max'], -BIG,
                          -BIG, -BIG, -BIG,
                          -BIG, -BIG, -BIG])
        x_ub = np.array([ BIG,  BIG,  BIG,
                           cfg['phi_max'],  cfg['theta_max'],  BIG,
                           BIG,  BIG,  BIG,
                           BIG,  BIG,  BIG])

        # ── Instantiate low-level solver ─────────────────────────────────────
        use_casadi = cfg['use_casadi']
        if use_casadi is None:
            use_casadi = _CASADI_AVAILABLE

        SolverCls = _MPCSolverCasADi if use_casadi else _MPCSolverScipy
        self._solver = SolverCls(
            Phi=Phi, Theta=Theta,
            Q_bar=Q_bar, R_bar=R_bar,
            n=n, m=m, N=self.N,
            u_min=u_min, u_max=u_max,
            x_lb=x_lb,   x_ub=x_ub,
        )

        self.backend = 'CasADi/IPOPT' if use_casadi else 'scipy/SLSQP'

    # ── Public method ────────────────────────────────────────────────────────

    def solve(self, error: np.ndarray) -> np.ndarray:
        """
        Compute the optimal first input deviation.

        Parameters
        ----------
        error : ndarray, shape (12,)
            State error  x_ref − x_current.
            Order: [ex, ey, ez, eφ, eθ, eψ, eẋ, eẏ, eż, eφ̇, eθ̇, eψ̇]

        Returns
        -------
        du : ndarray, shape (4,)
            Optimal deviation  [ΔF, Δτ_roll, Δτ_pitch, Δτ_yaw].
            Add to (F_hover, 0, 0, 0) to get the absolute wrench.
        """
        return self._solver.solve(error)

    # ── Internal: continuous-time linearised model ────────────────────────────

    def _build_continuous_model(self):
        """
        Return (A, B) for the hover-linearised 6-DOF quadrotor.

        Hover equilibrium: φ=θ=ψ=0, all velocities=0, F0=mg.

        Linearised translational accelerations (body ≈ world at hover):
            ẍ =  g·θ
            ÿ = −g·φ
            z̈ =  F/m − g

        Linearised rotational accelerations:
            φ̈ = τ_roll  / Ixx
            θ̈ = τ_pitch / Iyy
            ψ̈ = τ_yaw   / Izz
        """
        m, g = self.mass, self.gravity

        A = np.zeros((12, 12))
        B = np.zeros((12, 4))

        # Kinematics: ṗos = vel, Euler rates ≈ body rates (at hover)
        for i in range(6):
            A[i, i + 6] = 1.0

        # Linearised gravity coupling
        A[6, 4] =  g   # ẍ ← g·θ
        A[7, 3] = -g   # ÿ ← −g·φ

        # Thrust → vertical acceleration
        B[8, 0] = 1.0 / m

        # Torques → angular accelerations
        B[9,  1] = 1.0 / self.Ixx
        B[10, 2] = 1.0 / self.Iyy
        B[11, 3] = 1.0 / self.Izz

        return A, B
