#!/usr/bin/env python3
"""
mpc_controller.py
=================
Linear Model Predictive Controller for a quadrotor — built with CasADi.

Formulation  (multiple-shooting QP via CasADi Opti)
----------------------------------------------------
The controller receives a **full reference trajectory** of length N+1:

    x_ref_traj : (n, N+1)   — one reference state per prediction step

At step k the tracked deviation is:
    δx_k  =  x_k  −  x_ref_traj[:, k]

This lets the node feed a time-varying setpoint sequence (e.g. a waypoint
spline, minimum-snap trajectory, or simple step) instead of a single
constant hover point.  If the caller only has one target point, it simply
tiles that column N+1 times — see MPCController.make_hover_trajectory().

Decision variables
------------------
    X  (n, N+1)  state trajectory  x_0 … x_N          (absolute)
    U  (m, N)    input trajectory  u_0 … u_{N-1}       (absolute)

Multiple-shooting dynamics (absolute states):
    x_{k+1}  =  Ad · x_k  +  Bd · (u_k − u_hover)  +  x_hover_offset
              =  Ad · x_k  +  Bd · δu_k

where  δu_k = u_k − u_hover  (deviation from trim).

Stage cost:
    ℓ_k  =  (x_k − x_ref_k)' Q (x_k − x_ref_k)
           + (u_k − u_hover)'  R (u_k − u_hover)

Terminal cost:
    ℓ_N  =  (x_N − x_ref_N)' Qf (x_N − x_ref_N)

Input constraints:
    u_min ≤ u_k ≤ u_max     ∀ k

Parameters (updated each solve, no recompilation)
-------------------------------------------------
    p_x0       (n,)      current absolute state  x_0
    p_xref     (n, N+1)  full reference trajectory  x_ref_traj

Solver backends
---------------
    'ipopt'   — interior-point NLP  (always bundled with CasADi)
    'qrqp'    — CasADi built-in active-set QP  (fast, no extra install)
    'osqp'    — OSQP via CasADi plugin  (pip install osqp; best for large N)

Warm-starting
-------------
Previous optimal (X*, U*) is shifted one step and used as the next
initial guess, reducing solver iterations significantly.
"""

from __future__ import annotations

from typing import Optional

import casadi as ca
import numpy as np
from scipy.linalg import expm


# ─────────────────────────────────────────────────────────────────────────────
class MPCController:
    """
    Parametric linear MPC with full trajectory tracking.

    The QP is compiled once in __init__.  At every control step, solve()
    updates only the Opti parameter values (x0 and the N+1 reference
    states), warm-starts from the previous solution, and returns the first
    optimal absolute input u_0.

    Parameters
    ----------
    Ad      : (n, n)   Discrete-time A matrix (ZOH).
    Bd      : (n, m)   Discrete-time B matrix (ZOH).  Maps δu → δx.
    Q       : (n, n)   Stage state-error cost (PSD).
    R       : (m, m)   Stage input-deviation cost (PD).
    Qf      : (n, n)   Terminal state-error cost (PSD).
    N       : int      Prediction horizon (number of steps).
    u_hover : (m,)     Trim / hover absolute input  [F_hover, 0, 0, 0].
    u_min   : (m,)     Absolute lower bounds on u.
    u_max   : (m,)     Absolute upper bounds on u.
    solver  : str      CasADi solver: 'ipopt' | 'qrqp' | 'osqp'.
    """

    def __init__(
        self,
        Ad:      np.ndarray,
        Bd:      np.ndarray,
        Q:       np.ndarray,
        R:       np.ndarray,
        Qf:      np.ndarray,
        N:       int,
        u_hover: np.ndarray,
        u_min:   np.ndarray,
        u_max:   np.ndarray,
        solver:  str = "ipopt",
    ) -> None:
        self.n       = Ad.shape[0]    # state dim  (12)
        self.m       = Bd.shape[1]    # input dim  (4)
        self.N       = N
        self.u_hover = u_hover.copy()   # (m,)  — trim input
        self.u_min   = u_min.copy()     # (m,)  — absolute bounds
        self.u_max   = u_max.copy()     # (m,)

        # ── CasADi constant matrices ──────────────────────────────────────
        self._Ad      = ca.DM(Ad)
        self._Bd      = ca.DM(Bd)
        self._Q       = ca.DM(Q)
        self._R       = ca.DM(R)
        self._Qf      = ca.DM(Qf)
        self._uhover  = ca.DM(u_hover)

        # ── Warm-start buffers ────────────────────────────────────────────
        self._X_prev: Optional[np.ndarray] = None   # (n, N+1)
        self._U_prev: Optional[np.ndarray] = None   # (m, N)

        # ── Build the parametric Opti problem (compiled once) ─────────────
        (
            self._opti,
            self._X,       # (n, N+1)  absolute state trajectory
            self._U,       # (m, N)    absolute input trajectory
            self._p_x0,    # (n,)      current state parameter
            self._p_xref,  # (n, N+1)  reference trajectory parameter
        ) = self._build_problem(solver)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def solve(
        self,
        x0:        np.ndarray,
        xref_traj: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the MPC QP and return the first optimal absolute input u_0.

        Parameters
        ----------
        x0        : (n,)      Current absolute state x.
        xref_traj : (n, N+1)  Full reference trajectory.
                              Column k is the desired state at prediction
                              step k  (k = 0 … N).
                              Use make_hover_trajectory() to convert a
                              single target point into this format.

        Returns
        -------
        u_opt : (m,)  First optimal absolute input u_0.
                      Caller passes it directly to the motor mixer.
        """
        assert xref_traj.shape == (self.n, self.N + 1), (
            f"xref_traj must be ({self.n}, {self.N + 1}), "
            f"got {xref_traj.shape}"
        )

        # ── 1. Update Opti parameter values ──────────────────────────────
        self._opti.set_value(self._p_x0,   x0)
        self._opti.set_value(self._p_xref, xref_traj)

        # ── 2. Set initial guess (warm-start or cold-start) ───────────────
        if self._X_prev is None:
            X_guess, U_guess = self._cold_start(x0, xref_traj)
        else:
            X_guess, U_guess = self._shift_warm_start(x0, xref_traj)

        self._opti.set_initial(self._X, X_guess)
        self._opti.set_initial(self._U, U_guess)

        # ── 3. Solve ──────────────────────────────────────────────────────
        try:
            sol = self._opti.solve()
            X_sol = np.array(sol.value(self._X))   # (n, N+1)
            U_sol = np.array(sol.value(self._U))   # (m, N)
        except RuntimeError:
            # Solver failed — fall back to shifted warm-start or hover trim
            if self._U_prev is not None:
                return np.clip(
                    self._U_prev[:, 0], self.u_min, self.u_max
                )
            return self.u_hover.copy()

        # ── 4. Cache for next call ────────────────────────────────────────
        self._X_prev = X_sol
        self._U_prev = U_sol

        # Return first optimal absolute input u_0  (m,)
        return U_sol[:, 0].copy()

    def make_hover_trajectory(self, x_ref: np.ndarray) -> np.ndarray:
        """
        Convenience: tile a single reference state into a full trajectory.

        Equivalent to tracking a constant setpoint over the entire horizon.

        Parameters
        ----------
        x_ref : (n,)  Single reference state (absolute).

        Returns
        -------
        xref_traj : (n, N+1)  Same column repeated N+1 times.
        """
        return np.tile(x_ref[:, np.newaxis], (1, self.N + 1))

    def reset(self) -> None:
        """Discard warm-start buffers (call after large setpoint jumps)."""
        self._X_prev = None
        self._U_prev = None

    # ─────────────────────────────────────────────────────────────────────
    # Problem construction — executed once in __init__
    # ─────────────────────────────────────────────────────────────────────

    def _build_problem(
        self, solver: str
    ) -> tuple[ca.Opti, ca.MX, ca.MX, ca.MX, ca.MX]:
        """
        Declare the parametric CasADi Opti (multiple-shooting QP).

        State and input trajectories are in ABSOLUTE coordinates.
        Reference tracking errors are formed inside the cost:
            e_k  = X[:, k] − p_xref[:, k]

        Dynamics use the deviation input  δu = u − u_hover:
            X[:, k+1] = Ad · X[:, k] + Bd · (U[:, k] − u_hover)

        This formulation keeps the absolute state constraint
            X[:, 0] == p_x0
        clean (no subtraction of a varying reference).
        """
        n, m, N = self.n, self.m, self.N
        opti = ca.Opti()

        # ── Decision variables (absolute) ─────────────────────────────────
        X = opti.variable(n, N + 1)   # absolute state trajectory
        U = opti.variable(m, N)       # absolute input trajectory

        # ── Parameters ───────────────────────────────────────────────────
        p_x0   = opti.parameter(n)         # current state  (n,)
        p_xref = opti.parameter(n, N + 1)  # reference traj (n, N+1)

        # ── Cost function ─────────────────────────────────────────────────
        cost = ca.MX(0)

        # Stage costs  k = 0 … N−1
        for k in range(N):
            e_x  = X[:, k]     - p_xref[:, k]     # state error
            e_u  = U[:, k]     - self._uhover      # input deviation
            cost = cost + ca.bilin(self._Q,  e_x, e_x)
            cost = cost + ca.bilin(self._R,  e_u, e_u)

        # Terminal cost  k = N
        e_xN = X[:, N] - p_xref[:, N]
        cost = cost + ca.bilin(self._Qf, e_xN, e_xN)

        opti.minimize(cost)

        # ── Initial state constraint ──────────────────────────────────────
        opti.subject_to(X[:, 0] == p_x0)

        # ── Dynamics constraints (multiple-shooting) ──────────────────────
        # x_{k+1} = Ad·x_k + Bd·(u_k − u_hover)
        for k in range(N):
            du_k   = U[:, k] - self._uhover
            x_next = self._Ad @ X[:, k] + self._Bd @ du_k
            opti.subject_to(X[:, k + 1] == x_next)

        # ── Input bounds (absolute) ───────────────────────────────────────
        u_lb = ca.DM(self.u_min)
        u_ub = ca.DM(self.u_max)
        for k in range(N):
            opti.subject_to(opti.bounded(u_lb, U[:, k], u_ub))

        # ── Solver ───────────────────────────────────────────────────────
        opti.solver(solver, _solver_options(solver))

        return opti, X, U, p_x0, p_xref

    # ─────────────────────────────────────────────────────────────────────
    # Warm-start helpers
    # ─────────────────────────────────────────────────────────────────────

    def _cold_start(
        self,
        x0:        np.ndarray,
        xref_traj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initial guess: propagate with hover trim input.

        State: X[:,0] = x0,  X[:,k+1] = Ad·X[:,k]  (δu = 0)
        Input: U[:,k] = u_hover  ∀ k
        """
        Ad = np.array(self._Ad)
        n, m, N = self.n, self.m, self.N

        X_guess = np.zeros((n, N + 1))
        X_guess[:, 0] = x0
        for k in range(N):
            X_guess[:, k + 1] = Ad @ X_guess[:, k]  # hover input → δu=0

        U_guess = np.tile(self.u_hover[:, np.newaxis], (1, N))  # (m, N)
        return X_guess, U_guess

    def _shift_warm_start(
        self,
        x0:        np.ndarray,
        xref_traj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Shift the previous optimal trajectory by one step.

        State shift
        -----------
        X_guess[:, k]   = X_prev[:, k+1]      k = 0 … N−1
        X_guess[:, N]   = Ad · X_prev[:, N]   (propagate terminal)
        X_guess[:, 0]   = x0                  (pin to current state)

        Input shift
        -----------
        U_guess[:, k]   = U_prev[:, k+1]      k = 0 … N−2
        U_guess[:, N-1] = U_prev[:, N-1]      (repeat last input)
        """
        Ad = np.array(self._Ad)
        n, m, N = self.n, self.m, self.N

        X_guess = np.zeros((n, N + 1))
        X_guess[:, :N]  = self._X_prev[:, 1:]           # shift
        X_guess[:, N]   = Ad @ self._X_prev[:, N]       # propagate terminal
        X_guess[:, 0]   = x0                             # pin initial

        U_guess = np.zeros((m, N))
        U_guess[:, :N - 1] = self._U_prev[:, 1:]        # shift
        U_guess[:, N - 1]  = self._U_prev[:, N - 1]     # repeat last

        return X_guess, U_guess


# ─────────────────────────────────────────────────────────────────────────────
# Solver option presets
# ─────────────────────────────────────────────────────────────────────────────

def _solver_options(solver: str) -> dict:
    """
    Quiet, convergence-tuned options for each supported backend.

    ipopt  — Robust NLP interior-point (bundled with CasADi).
    qrqp   — CasADi built-in active-set QP.  Zero extra dependencies.
    osqp   — Operator-splitting QP.  pip install osqp.  Best for large N.
    """
    if solver == "ipopt":
        return {
            "ipopt.print_level":                0,
            "ipopt.sb":                         "yes",
            "ipopt.max_iter":                   300,
            "ipopt.tol":                        1e-6,
            "ipopt.acceptable_tol":             1e-4,
            "ipopt.warm_start_init_point":      "yes",
            "ipopt.warm_start_bound_push":      1e-9,
            "ipopt.warm_start_mult_bound_push": 1e-9,
            "print_time":                       False,
        }
    elif solver == "qrqp":
        return {
            "print_iter":      False,
            "print_header":    False,
            "max_iter":        1000,
            "constr_viol_tol": 1e-7,
        }
    elif solver == "osqp":
        return {
            "osqp.verbose":       False,
            "osqp.max_iter":      4000,
            "osqp.eps_abs":       1e-5,
            "osqp.eps_rel":       1e-5,
            "osqp.warm_starting": True,
            "osqp.adaptive_rho":  True,
        }
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Zero-Order-Hold (ZOH) discretisation
# ─────────────────────────────────────────────────────────────────────────────

def discretise_zoh(
    Ac: np.ndarray,
    Bc: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ZOH discretisation via matrix exponential (augmented-matrix method).

        [ Ad  Bd ]   =  expm( [ Ac  Bc ] * dt )
        [  0   I ]            [  0   0 ]

    Parameters
    ----------
    Ac : (n, n)  Continuous-time A.
    Bc : (n, m)  Continuous-time B.
    dt : float   Sample period [s].

    Returns
    -------
    Ad : (n, n)   Discrete A.
    Bd : (n, m)   Discrete B  (maps δu → δx).
    """
    n, m = Ac.shape[0], Bc.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac * dt
    M[:n, n:] = Bc * dt
    eM = expm(M)
    return eM[:n, :n], eM[:n, n:]