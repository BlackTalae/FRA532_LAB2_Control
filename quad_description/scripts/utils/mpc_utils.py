import numpy as np
from scipy.signal import cont2discrete            # ← needed for c2d

def build_xref_from_traj(traj, N, n):
    """
    traj : np.ndarray shape (n, T)   — state_dim × timesteps
    N    : prediction horizon
    n    : state dimension

    Returns
    -------
    xref : np.ndarray shape (n, N+1)
        Columns 0…min(T,N+1)-1 come from traj.
        If T < N+1 the last column is repeated to fill the remainder.
        If T > N+1 the trajectory is truncated.
    """
    traj = np.asarray(traj)
    if traj.shape[0] != n:
        raise ValueError(f"Expected traj.shape[0]=={n}, got {traj.shape[0]}")

    T    = traj.shape[1]
    xref = np.zeros((n, N + 1))
    L    = min(T, N + 1)
    xref[:, :L] = traj[:, :L]
    if T < N + 1:
        xref[:, L:] = traj[:, -1].reshape(n, 1)   # hold final
    return xref

# ── Continuous-to-discrete conversion ────────────────────────────────────
def c2d(A: np.ndarray, B: np.ndarray, dt: float):
    """Zero-order hold discretisation using scipy."""
    # FIX: MPC expects discrete (Ad, Bd); build them here from continuous A, B
    sys_d = cont2discrete((A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1]))),
                            dt, method='zoh')
    Ad, Bd = sys_d[0], sys_d[1]
    return Ad, Bd
