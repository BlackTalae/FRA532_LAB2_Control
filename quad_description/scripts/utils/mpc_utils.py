from scipy.linalg import expm, solve_discrete_are

# ═══════════════════════════════════════════════════════════════════════════════
#  Discrete-time LTI system helper
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np

def c2d(A: np.ndarray, B: np.ndarray, dt: float):
    """
    Zero-order-hold discretisation of a continuous LTI system.

    Returns (Ad, Bd) such that  x_{k+1} = Ad x_k + Bd u_k.
    Uses the matrix-exponential method via the Van Loan algorithm:
        M = expm( [[-A, B B'] ; [0, A']] * dt )   (Scipy approximation)

    For simplicity we use the 1st-order Euler + Cayley approximation:
        Ad ≈ (I + A·dt/2)·(I - A·dt/2)⁻¹   (Tustin / bilinear)
    For our drone (dt = 0.01 s) this gives negligible discretisation error.
    """
    from scipy.linalg import expm
    n = A.shape[0]
    M = np.zeros((n + B.shape[1], n + B.shape[1]))
    # Build augmented matrix for ZOH
    em_upper = np.hstack([A, B])
    em_lower = np.zeros((B.shape[1], n + B.shape[1]))
    em = np.block([[em_upper], [em_lower]])
    Ms = expm(em * dt)
    Ad = Ms[:n, :n]
    Bd = Ms[:n, n:]
    return Ad, Bd


def dare_terminal_cost(Ad, Bd, Q, R):
    """Solve the DARE to get the infinite-horizon terminal cost matrix P."""
    P = solve_discrete_are(Ad, Bd, Q, R)
    return P

# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: continuous → discrete (ZOH)
# ═══════════════════════════════════════════════════════════════════════════════

def c2d(A: np.ndarray, B: np.ndarray, dt: float):
    """
    Zero-order-hold discretisation via matrix exponential.

    Returns (Ad, Bd) such that  x_{k+1} = Ad x_k + Bd u_k.
    """
    n, m = A.shape[0], B.shape[1]
    em = np.zeros((n + m, n + m))
    em[:n, :n] = A
    em[:n, n:] = B
    Ms = expm(em * dt)
    return Ms[:n, :n], Ms[:n, n:]


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: condensed prediction matrices
# ═══════════════════════════════════════════════════════════════════════════════

def build_condensed(Ad: np.ndarray, Bd: np.ndarray, N: int):
    """
    Build Φ (free-response) and Θ (forced-response) matrices such that:

        X = Φ · e0 + Θ · Z

    where  X = [x_1', …, x_N']'  and  Z = [Δu_0', …, Δu_{N-1}']'.

    Returns
    -------
    Phi   : ndarray, shape (N·n, n)
    Theta : ndarray, shape (N·n, N·m)
    """
    n, m = Ad.shape[0], Bd.shape[1]
    Phi   = np.zeros((N * n, n))
    Theta = np.zeros((N * n, N * m))

    Ak = np.eye(n)
    for i in range(N):
        Ak = Ad @ Ak
        Phi[i * n:(i + 1) * n, :] = Ak
        for j in range(i + 1):
            Theta[i * n:(i + 1) * n, j * m:(j + 1) * m] = (
                np.linalg.matrix_power(Ad, i - j) @ Bd
            )
    return Phi, Theta
