#!/usr/bin/env python3
from __future__ import annotations

import casadi as ca
import numpy as np


class MPC_Opti:
    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        N: int,
        u_min: np.ndarray,
        u_max: np.ndarray,
        Nc: int | None = None,
        state_idx: list[int] | None = None,
        state_lower_bound: np.ndarray | None = None,
        state_upper_bound: np.ndarray | None = None,
        constrain_initial_state: bool = False,
        constrain_terminal_state: bool = True,
        solver: str = "ipopt",
    ) -> None:
        # ---------- basic dimensions ----------
        Ad = np.asarray(Ad, dtype=float)
        Bd = np.asarray(Bd, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        Qf = np.asarray(Qf, dtype=float)

        if Ad.ndim != 2 or Ad.shape[0] != Ad.shape[1]:
            raise ValueError("Ad must be square with shape (n, n)")
        if Bd.ndim != 2 or Bd.shape[0] != Ad.shape[0]:
            raise ValueError("Bd must have shape (n, m)")
        if Q.shape != Ad.shape:
            raise ValueError("Q must have shape (n, n)")
        if Qf.shape != Ad.shape:
            raise ValueError("Qf must have shape (n, n)")
        if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] != Bd.shape[1]:
            raise ValueError("R must have shape (m, m)")

        self.n = Ad.shape[0]
        self.m = Bd.shape[1]
        self.N = int(N)
        self.Nc = self.N if Nc is None else int(np.clip(Nc, 1, self.N))

        self.u_min = np.broadcast_to(u_min, (self.m,)).astype(float).copy()
        self.u_max = np.broadcast_to(u_max, (self.m,)).astype(float).copy()
        if np.any(self.u_min > self.u_max):
            raise ValueError("u_min must be <= u_max elementwise")

        # ---------- constants ----------
        self._Ad = ca.DM(Ad)
        self._Bd = ca.DM(Bd)
        self._Q = ca.DM(Q)
        self._R = ca.DM(R)
        self._Qf = ca.DM(Qf)

        # ---------- state constraints ----------
        self.state_idx = [] if state_idx is None else list(state_idx)
        self.constrain_initial_state = bool(constrain_initial_state)
        self.constrain_terminal_state = bool(constrain_terminal_state)

        if len(self.state_idx) > 0:
            if state_lower_bound is None or state_upper_bound is None:
                raise ValueError(
                    "When state_idx is provided, state_lower_bound and "
                    "state_upper_bound must also be provided."
                )

            self.state_lower_bound = np.asarray(state_lower_bound, dtype=float).reshape(-1)
            self.state_upper_bound = np.asarray(state_upper_bound, dtype=float).reshape(-1)

            if len(self.state_lower_bound) != len(self.state_idx):
                raise ValueError("state_lower_bound must have same length as state_idx")
            if len(self.state_upper_bound) != len(self.state_idx):
                raise ValueError("state_upper_bound must have same length as state_idx")
            if np.any(self.state_lower_bound > self.state_upper_bound):
                raise ValueError("state_lower_bound must be <= state_upper_bound elementwise")

            for idx in self.state_idx:
                if not (0 <= idx < self.n):
                    raise ValueError(f"state index {idx} out of range for state dimension n={self.n}")
        else:
            self.state_lower_bound = np.array([], dtype=float)
            self.state_upper_bound = np.array([], dtype=float)

        self.solver_name = solver

        # allocated after build()
        self.opti: ca.Opti | None = None
        self.X = None
        self.Uc = None
        self.x0 = None
        self.xref = None
        self.uref = None

        self._Xw: np.ndarray | None = None
        self._Uw: np.ndarray | None = None
        self._built = False

    def _check_solve_shapes(
        self,
        x0_val: np.ndarray,
        xref_val: np.ndarray,
        uref_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0_arr = np.asarray(x0_val, dtype=float).reshape(-1)
        xref_arr = np.asarray(xref_val, dtype=float)
        uref_arr = np.asarray(uref_val, dtype=float)

        if x0_arr.shape != (self.n,):
            raise ValueError(f"x0_val must have shape ({self.n},), got {x0_arr.shape}")
        if xref_arr.shape != (self.n, self.N + 1):
            raise ValueError(
                f"xref_val must have shape ({self.n}, {self.N + 1}), got {xref_arr.shape}"
            )
        if uref_arr.shape != (self.m, self.N):
            raise ValueError(
                f"uref_val must have shape ({self.m}, {self.N}), got {uref_arr.shape}"
            )

        return x0_arr, xref_arr, uref_arr

    def build(self) -> None:
        n, m, N, Nc = self.n, self.m, self.N, self.Nc

        opti = ca.Opti()

        # ---------- decision variables ----------
        X = opti.variable(n, N + 1)   # predicted states
        Uc = opti.variable(m, Nc)     # free control inputs over control horizon

        # ---------- parameters ----------
        x0 = opti.parameter(n, 1)
        xref = opti.parameter(n, N + 1)
        uref = opti.parameter(m, N)

        # ---------- helpers ----------
        def add_selected_state_bounds(x_vec: ca.MX) -> None:
            """Apply bounds only to selected state components."""
            if len(self.state_idx) == 0:
                return
            for j, idx in enumerate(self.state_idx):
                opti.subject_to(
                    opti.bounded(
                        self.state_lower_bound[j],
                        x_vec[idx],
                        self.state_upper_bound[j],
                    )
                )

        # ---------- initial condition ----------
        opti.subject_to(X[:, 0] == x0)

        # optionally constrain current state too
        if self.constrain_initial_state:
            add_selected_state_bounds(X[:, 0])

        # ---------- input bounds on free control variables ----------
        for k in range(Nc):
            opti.subject_to(opti.bounded(self.u_min, Uc[:, k], self.u_max))

        # ---------- cost + dynamics + future state constraints ----------
        cost = 0
        for k in range(N):
            xk = X[:, k]
            uk = Uc[:, k] if k < Nc else Uc[:, Nc - 1]

            # dynamics
            x_next = self._Ad @ xk + self._Bd @ uk
            opti.subject_to(X[:, k + 1] == x_next)

            # constrain predicted future states X[:, 1], ..., X[:, N]
            if (k < N - 1) or self.constrain_terminal_state:
                add_selected_state_bounds(X[:, k + 1])

            # stage cost
            e = xk - xref[:, k]
            du = uk - uref[:, k]
            cost += ca.mtimes([e.T, self._Q, e]) + ca.mtimes([du.T, self._R, du])

        # terminal cost
        eN = X[:, N] - xref[:, N]
        cost += ca.mtimes([eN.T, self._Qf, eN])

        opti.minimize(cost)

        # ---------- solver ----------
        if self.solver_name == "ipopt":
            p_opts = {"expand": True, "print_time": 0}
            s_opts = {"print_level": 0}
            opti.solver("ipopt", p_opts, s_opts)
        else:
            # generic fallback for other solvers
            opti.solver(self.solver_name, {"expand": True})

        # ---------- store ----------
        self.opti = opti
        self.X = X
        self.Uc = Uc
        self.x0 = x0
        self.xref = xref
        self.uref = uref

        # ---------- warm start buffers ----------
        self._Xw = np.zeros((n, N + 1), dtype=float)
        self._Uw = np.zeros((m, Nc), dtype=float)

        self._built = True

    def solve(
        self,
        x0_val: np.ndarray,
        xref_val: np.ndarray,
        uref_val: np.ndarray,
        *,
        allow_debug_fallback: bool = False,
        return_info: bool = False,
    ):
        """
        Solve the MPC problem.

        Parameters
        ----------
        x0_val : (n,)
            Current state.
        xref_val : (n, N+1)
            State reference trajectory.
        uref_val : (m, N)
            Input reference trajectory.
        allow_debug_fallback : bool, optional
            If True and solver fails, return Opti debug values instead of raising.
        return_info : bool, optional
            If True, return an extra info dict.

        Returns
        -------
        u0 : (m,)
            First optimal control input.
        X_opt : (n, N+1)
            Predicted optimal state trajectory.
        U_opt : (m, N)
            Predicted optimal input trajectory expanded from control horizon.

        If return_info=True, also returns:
        info : dict
            Solver success/failure information.
        """
        if not self._built or self.opti is None:
            raise RuntimeError("Call build() before solve().")

        x0_arr, xref_arr, uref_arr = self._check_solve_shapes(x0_val, xref_val, uref_val)

        # set parameters
        self.opti.set_value(self.x0, x0_arr.reshape(self.n, 1))
        self.opti.set_value(self.xref, xref_arr)
        self.opti.set_value(self.uref, uref_arr)

        # warm start
        self.opti.set_initial(self.X, self._Xw)
        self.opti.set_initial(self.Uc, self._Uw)

        info = {
            "success": False,
            "solver": self.solver_name,
            "status": None,
            "cost": None,
            "used_debug_fallback": False,
        }

        try:
            sol = self.opti.solve()

            X_opt = np.array(sol.value(self.X), dtype=float)
            Uc_opt = np.array(sol.value(self.Uc), dtype=float).reshape(self.m, self.Nc)

            info["success"] = True
            try:
                stats = self.opti.stats()
                info["status"] = stats.get("return_status", None)
            except Exception:
                info["status"] = "success"
            try:
                info["cost"] = float(sol.value(self.opti.f))
            except Exception:
                info["cost"] = None

        except RuntimeError as exc:
            # solver failed
            status = None
            try:
                stats = self.opti.stats()
                status = stats.get("return_status", None)
            except Exception:
                pass

            info["status"] = status if status is not None else str(exc)

            if not allow_debug_fallback:
                raise RuntimeError(
                    f"MPC solve failed. Solver status: {info['status']}. "
                    f"Try checking feasibility, bounds, references, or use "
                    f"allow_debug_fallback=True for debugging."
                ) from exc

            # Use debug values as fallback
            try:
                X_opt = np.array(self.opti.debug.value(self.X), dtype=float)
                Uc_opt = np.array(self.opti.debug.value(self.Uc), dtype=float).reshape(self.m, self.Nc)
            except Exception as dbg_exc:
                raise RuntimeError(
                    f"MPC solve failed and debug fallback was unavailable. "
                    f"Solver status: {info['status']}"
                ) from dbg_exc

            info["used_debug_fallback"] = True

        # expand control horizon to full prediction horizon
        U_opt = np.hstack(
            [Uc_opt, np.tile(Uc_opt[:, [-1]], (1, self.N - self.Nc))]
        )

        # update warm start only if we got numerically valid arrays
        self._Xw = np.hstack([X_opt[:, 1:], X_opt[:, [-1]]])
        self._Uw = np.hstack([Uc_opt[:, 1:], Uc_opt[:, [-1]]])

        u0 = U_opt[:, 0]

        if return_info:
            return u0, X_opt, U_opt, info
        return u0, X_opt, U_opt