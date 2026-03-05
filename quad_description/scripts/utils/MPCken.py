#!/usr/bin/env python3
from __future__ import annotations

import casadi as ca
import numpy as np


class MPC:
    def __init__(
        self,
        Ad:      np.ndarray,   # (n, n)  discrete-time A matrix
        Bd:      np.ndarray,   # (n, m)  discrete-time B matrix
        Q:       np.ndarray,   # (n, n)  stage state cost
        R:       np.ndarray,   # (m, m)  stage input cost
        Qf:      np.ndarray,   # (n, n)  terminal state cost (use DARE solution)
        N:       int,          # prediction horizon (steps)
        u_hover: np.ndarray,   # (m,)    hover / trim input
        u_min:   np.ndarray,   # (m,)    absolute lower input bound
        u_max:   np.ndarray,   # (m,)    absolute upper input bound
        Nc:      int | None = None,  # control horizon (steps); None → Nc = N
        state_idx: list[int] | None = None,   # <- เพิ่ม
        state_lower_bound: np.ndarray | None = None,   # <- เพิ่ม
        state_upper_bound: np.ndarray | None = None,   # <- เพิ่ม
        solver:  str = "ipopt",
    ) -> None:

        self.n  = Ad.shape[0]
        self.m  = Bd.shape[1]
        self.N  = N
        # Nc ≤ N.  After Nc steps the input is held constant at u_{Nc-1}.
        # Fewer decision variables → faster solve with only small loss of optimality.
        self.Nc = N if Nc is None else int(np.clip(Nc, 1, N))

        self.u_min   = np.broadcast_to(u_min,   (self.m,)).copy()
        self.u_max   = np.broadcast_to(u_max,   (self.m,)).copy()

        # Constant Variable
        self._Ad     = ca.DM(Ad)
        self._Bd     = ca.DM(Bd)
        self._Q      = ca.DM(Q)
        self._R      = ca.DM(R)
        self._Qf     = ca.DM(Qf)

        # state constraints
        self.state_idx = [] if state_idx is None else list(state_idx)
        if len(self.state_idx) > 0:
            self.state_lower_bound = np.asarray(state_lower_bound, dtype=float).reshape(-1)
            self.state_upper_bound = np.asarray(state_upper_bound, dtype=float).reshape(-1)

            if len(self.state_lower_bound) != len(self.state_idx) or len(self.state_upper_bound) != len(self.state_idx):
                raise ValueError("state_idx, state_lb, state_ub must have same length")
        else:
            self.state_lower_bound = np.array([], dtype=float)
            self.state_upper_bound = np.array([], dtype=float)

    def build(self) -> None:
        """Build the NLP once.  Call before solve()."""
        n, m, N, Nc = self.n, self.m, self.N, self.Nc

        # ── Decision variables ────────────────────────────────────────────────
        X  = ca.SX.sym("X",  n, N + 1)   # predicted states  (n × N+1)
        Uc = ca.SX.sym("Uc", m, Nc)      # free inputs only for k = 0…Nc-1

        # ── Parameters ────────────────────────────────────────────────────────
        x0   = ca.SX.sym("x0",   n)
        xref = ca.SX.sym("xref", n, N + 1)
        uref = ca.SX.sym("uref", m, N)   # input reference over horizon
        
        cost = 0
        g = []
        lbg = []
        ubg = []

        g.append(X[:, 0] - x0)   # initial condition
        lbg.extend([0.0] * n)
        ubg.extend([0.0] * n)

        for k in range(N):
            xk = X[:, k]

            # Input at step k:  free if k < Nc, else held at u_{Nc-1}
            uk = Uc[:, k] if k < Nc else Uc[:, Nc - 1]

            # dynamics equality
            g.append(X[:, k + 1] - (self._Ad @ xk + self._Bd @ uk))
            lbg.extend([0.0] * n)
            ubg.extend([0.0] * n)

            # state inequality for selected states only
            if len(self.state_idx) > 0:
                xk_sel = ca.vertcat(*[xk[i] for i in self.state_idx])
                g.append(xk_sel)
                lbg.extend(self.state_lower_bound.tolist())
                ubg.extend(self.state_upper_bound.tolist())

            # Stage cost
            e  = xk - xref[:, k]
            du = uk - uref[:, k]
            cost += ca.mtimes([e.T, self._Q, e]) + ca.mtimes([du.T, self._R, du])

        # Terminal cost
        eN = X[:, N] - xref[:, N]
        cost += ca.mtimes([eN.T, self._Qf, eN])

        # ── Vectorise ─────────────────────────────────────────────────────────
        w = ca.vertcat(ca.reshape(X,  -1, 1),
                       ca.reshape(Uc, -1, 1)) # Optimize this
        p = ca.vertcat(
            x0,
            ca.reshape(xref, -1, 1),
            ca.reshape(uref, -1, 1)
        ) # Given parameter not optimize

        nlp  = {"x": w, "f": cost, "g": ca.vertcat(*g), "p": p}
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)

        # ── Sizes ─────────────────────────────────────────────────────────────
        self._nX  = n * (N + 1)
        self._nUc = m * Nc          # ← Nc columns, not N
        self._nW  = self._nX + self._nUc
        self._gd  = int(ca.vertcat(*g).size()[0])

        self._lbg = np.zeros(self._gd)
        self._ubg = np.zeros(self._gd)

        lbw = -np.inf * np.ones(self._nW)
        ubw =  np.inf * np.ones(self._nW)
        for k in range(Nc):
            idx = self._nX + k * m
            lbw[idx:idx + m] = self.u_min
            ubw[idx:idx + m] = self.u_max
        self._lbw = lbw
        self._ubw = ubw

        self._w0 = np.zeros(self._nW)

    def solve(self, x0_val: np.ndarray, xref_val: np.ndarray, uref_val: np.ndarray):
        """
        Parameters
        ----------
        x0_val   : (n,)       current absolute state
        xref_val : (n, N+1)   reference trajectory over full prediction horizon

        Returns
        -------
        u0    : (m,)        first optimal input
        X_opt : (n, N+1)    predicted state trajectory
        U_opt : (m, N)      full input trajectory (control horizon expanded)
        """
        pval = np.concatenate([
            x0_val.flatten(),
            xref_val.flatten(order='F'),
            uref_val.flatten(order='F')
        ])

        sol   = self.solver(
            x0  = self._w0,
            lbx = self._lbw,
            ubx = self._ubw,
            lbg = self._lbg,
            ubg = self._ubg,
            p   = pval,
        )

        w_opt = np.array(sol["x"]).flatten()

        X_opt  = w_opt[:self._nX ].reshape((self.n, self.N + 1), order='F')
        Uc_opt = w_opt[self._nX:  ].reshape((self.m, self.Nc),   order='F')

        # Expand control horizon back to N columns for warm-start & return
        # Steps Nc…N-1 hold the last free input u_{Nc-1}
        U_opt = np.hstack([Uc_opt,
                           np.tile(Uc_opt[:, [-1]], (1, self.N - self.Nc))])

        # Shifted warm start
        X_shifted  = np.hstack([X_opt[:, 1:],   X_opt[:, [-1]]])
        Uc_shifted = np.hstack([Uc_opt[:, 1:],  Uc_opt[:, [-1]]])
        self._w0   = np.concatenate([X_shifted.flatten(order='F'),
                                     Uc_shifted.flatten(order='F')])

        return U_opt[:, 0], X_opt, U_opt