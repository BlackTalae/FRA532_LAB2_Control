#!/usr/bin/env python3
"""
mpc_controller.py
=================
ROS 2 MPC node for quadrotor 3-D trajectory tracking in Ignition Gazebo.

State   x = [x, y, z, phi, theta, psi, xdot, ydot, zdot, phidot, thetadot, psidot]  n=12
Control u = [F_total, tau_roll, tau_pitch, tau_yaw]                                   p=4

External trajectory feed
------------------------
Subscribes to /reference_path (nav_msgs/Path) published by trajectory_generator.py.
Each pose in the path = one future reference point for the MPC horizon.

Phase logic
-----------
  IDLE       : waiting for first odometry
  HOVER      : climb to HOVER_ALT and hold indefinitely
               → transitions to TRAJECTORY only when a fresh /reference_path arrives
               → never times out into a trajectory on its own
  TRAJECTORY : tracking the external /reference_path
               → reverts to HOVER if the path goes silent > EXTERNAL_TRAJ_TIMEOUT s
               → reverts to HOVER if the path node is killed

"No trajectory? Just hover." — the drone will stay at HOVER_ALT forever until
trajectory_generator.py publishes a path, then track it, then hover again if lost.

Solver
------
Hessian H is normalised by trace(H)/(p*N) to fix L-BFGS-B ill-conditioning.
"""

import math
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from actuator_msgs.msg import Actuators
from std_msgs.msg import Float64MultiArray

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat2rpy(qx, qy, qz, qw):
    sr = 2.0 * (qw * qx + qy * qz)
    cr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sr, cr)
    sp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    sy = 2.0 * (qw * qz + qx * qy)
    cy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(sy, cy)
    return roll, pitch, yaw


def quat2yaw(qz, qw):
    return 2.0 * math.atan2(qz, qw)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Flight phase
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    IDLE       = auto()
    HOVER      = auto()
    TRAJECTORY = auto()


# ─────────────────────────────────────────────────────────────────────────────
# MPC Controller Node
# ─────────────────────────────────────────────────────────────────────────────

class MPCController(Node):

    # ── Physical parameters ───────────────────────────────────────────────────
    MASS      = 1.5
    GRAVITY   = 9.81
    IXX       = 0.0347563
    IYY       = 0.07
    IZZ       = 0.0977
    K_F       = 8.54858e-06
    K_M       = 0.06
    OMEGA_MAX = 1500.0

    ROTOR_POS = np.array([[ 0.13, -0.22],
                           [-0.13,  0.20],
                           [ 0.13,  0.22],
                           [-0.13, -0.20]])
    ROTOR_DIR = np.array([-1.0, -1.0, +1.0, +1.0])

    # ── MPC hyper-parameters ──────────────────────────────────────────────────
    DT = 0.05
    N  = 20

    # Q_DIAG = np.array([60., 60., 30.,  # x ,y ,z
    #                    30., 30.,  5.,  # roll , pitch , yaw
    #                     20.,  20.,  10.,
    #                     25.,  25.,  8.0])

    Q_DIAG = np.array([100., 100., 50.,  # x ,y ,z
                       30., 30.,  2.,  # roll , pitch , yaw
                        20.,  20.,  10.,
                        25.,  25.,  8.0])
    QN_SCALE = 3.0
    R_DIAG = np.array([0.1, 2.0, 2.0, 1.5])

    U_MIN = np.array([0.0,   -8.0, -8.0, -3.0])
    U_MAX = np.array([4.0 * MASS * GRAVITY, 8.0, 8.0, 3.0])

    # ── Hover settings ────────────────────────────────────────────────────────
    HOVER_ALT         = 2.0    # m    target altitude
    HOVER_STABLE_TIME = 0.0 # 3.0    # s    must be stable this long before accepting trajectory
    HOVER_POS_TOL     = 0.5 # 0.10   # m    position error threshold for "stable"
    HOVER_VEL_TOL     = 0.25 # 0.05   # m/s  velocity threshold for "stable"

    # ── External trajectory timeout ───────────────────────────────────────────
    # If no /reference_path message arrives within this window, revert to HOVER.
    EXTERNAL_TRAJ_TIMEOUT = 0.5   # s

    def __init__(self):
        super().__init__('quad_mpc_controller')
        self.get_logger().info('Initialising quadrotor MPC controller...')

        m, g = self.MASS, self.GRAVITY
        self.u_hover = np.array([m * g, 0.0, 0.0, 0.0])

        self.Q  = np.diag(self.Q_DIAG)
        self.QN = self.QN_SCALE * self.Q
        self.R  = np.diag(self.R_DIAG)

        self.Ad, self.Bd = self._build_model()
        self.Sx, self.Su = self._build_prediction_matrices()
        self._build_qp_matrices()

        # Motor allocation
        px, py = self.ROTOR_POS[:, 0], self.ROTOR_POS[:, 1]
        M = np.array([np.ones(4), py, -px, self.ROTOR_DIR * self.K_M])
        self.M_inv = np.linalg.pinv(M)

        # Flight-phase state machine
        self.phase         = Phase.IDLE
        self.hover_origin  = np.zeros(3)   # [x, y, yaw] locked at takeoff
        self.hover_entry_t = None
        self.stable_since  = None          # time when hover stability was first met
        self.traj_start_t  = None

        # External trajectory
        self.ext_path        = None    # latest nav_msgs/Path received
        self.ext_path_stamp  = None    # float seconds (ROS clock) when it arrived
        self._prev_phase     = None    # for transition logging

        # Solver warm-start
        self.state       = np.zeros(12)
        self.state_ready = False
        self.dU_warm     = np.zeros(4 * self.N)

        # ROS 2 I/O
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._cb_odom, 10)
        self.sub_path = self.create_subscription(
            Path, '/reference_path', self._cb_path, 1)
        self.pub_motors = self.create_publisher(
            Actuators, '/motor_commands', 10)
        self.timer = self.create_timer(self.DT, self._cb_control)
        self.control_pub = self.create_publisher(
            Float64MultiArray, '/control_u', 10)

        self.get_logger().info(
            f'MPC ready | DT={self.DT}s  N={self.N}  '
            f'horizon={self.N*self.DT:.1f}s  '
            f'hover_alt={self.HOVER_ALT}m  '
            f'hover_thrust={m*g:.2f}N  '
            f'QP_scale={self.qp_scale:.1f}')
        self.get_logger().info(
            'Behaviour: HOVER until /reference_path received, '
            'revert to HOVER if path lost.')

    # ─────────────────────────────────────────────────────────────────────────
    # External path callback
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_path(self, msg: Path):
        self.ext_path       = msg
        self.ext_path_stamp = self.get_clock().now().nanoseconds * 1e-9

    # ─────────────────────────────────────────────────────────────────────────
    # Discrete model (ZOH)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_model(self):
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ
        n, p, dt = 12, 4, self.DT

        A = np.zeros((n, n)); B = np.zeros((n, p))
        A[0,6]=A[1,7]=A[2,8]=1.0
        A[3,9]=A[4,10]=A[5,11]=1.0
        A[6,4]=g; A[7,3]=-g
        B[8,0]=1./m
        B[9,1]=1./Ixx; B[10,2]=1./Iyy; B[11,3]=1./Izz

        AB = np.zeros((n+p, n+p))
        AB[:n,:n]=A*dt; AB[:n,n:]=B*dt
        eAB = expm(AB)
        return eAB[:n,:n], eAB[:n,n:]

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction matrices  X = Sx*x0 + Su*dU
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prediction_matrices(self):
        n, p, N = 12, 4, self.N
        Ad, Bd = self.Ad, self.Bd

        Ad_pow = [np.eye(n)]
        for _ in range(N):
            Ad_pow.append(Ad @ Ad_pow[-1])

        Sx = np.zeros((n*N, n))
        Su = np.zeros((n*N, p*N))
        for k in range(N):
            Sx[k*n:(k+1)*n,:] = Ad_pow[k+1]
            for j in range(k+1):
                Su[k*n:(k+1)*n, j*p:(j+1)*p] = Ad_pow[k-j] @ Bd
        return Sx, Su

    # ─────────────────────────────────────────────────────────────────────────
    # QP matrices (with normalisation)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_qp_matrices(self):
        n, p, N = 12, 4, self.N
        Su = self.Su

        Q_bar = np.zeros((n*N, n*N))
        for k in range(N-1):
            Q_bar[k*n:(k+1)*n, k*n:(k+1)*n] = self.Q
        Q_bar[(N-1)*n:, (N-1)*n:] = self.QN

        R_bar = np.zeros((p*N, p*N))
        for k in range(N):
            R_bar[k*p:(k+1)*p, k*p:(k+1)*p] = self.R

        H_raw = Su.T @ Q_bar @ Su + R_bar
        H_raw = 0.5 * (H_raw + H_raw.T)

        scale = max(np.trace(H_raw) / (p * N), 1.0)
        self.H        = H_raw / scale
        self.SuTQbar  = (Su.T @ Q_bar) / scale
        self.qp_scale = scale

        du_min = self.U_MIN - self.u_hover
        du_max = self.U_MAX - self.u_hover
        self.qp_bounds = [(float(du_min[i % p]), float(du_max[i % p]))
                          for i in range(p * N)]

    # ─────────────────────────────────────────────────────────────────────────
    # MPC solver
    # ─────────────────────────────────────────────────────────────────────────

    def _solve_mpc(self, x0, X_ref):
        e = self.Sx @ x0 - X_ref.flatten()
        c = self.SuTQbar @ e

        def cost(dU): return 0.5 * float(dU @ (self.H @ dU)) + float(c @ dU)
        def grad(dU): return self.H @ dU + c

        result = minimize(cost, self.dU_warm, jac=grad,
                          method='L-BFGS-B', bounds=self.qp_bounds,
                          options={'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-4})

        if not result.success:
            self.get_logger().debug(
                f'MPC: {result.message} (nit={result.nit} f={result.fun:.3e})')

        dU_opt = result.x
        self.dU_warm[:-4] = dU_opt[4:]
        self.dU_warm[-4:] = 0.0
        return self.u_hover + dU_opt[:4]

    # ─────────────────────────────────────────────────────────────────────────
    # Motor allocation
    # ─────────────────────────────────────────────────────────────────────────

    def _u_to_omega(self, u):
        forces = np.maximum(self.M_inv @ u, 0.0)
        omega  = np.sqrt(forces / self.K_F)
        return np.clip(omega, 0.0, self.OMEGA_MAX)

    # ─────────────────────────────────────────────────────────────────────────
    # Odometry callback
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_odom(self, msg):
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
    # Control callback
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_control(self):
        if not self.state_ready:
            self._publish_omega(np.zeros(4))
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # ── Check external path freshness ─────────────────────────────────────
        path_fresh = (
            self.ext_path is not None and
            self.ext_path_stamp is not None and
            (now_sec - self.ext_path_stamp) < self.EXTERNAL_TRAJ_TIMEOUT
        )

        # ── Phase state machine ────────────────────────────────────────────────
        if self.phase == Phase.IDLE:
            # Lock hover origin at first odometry position
            self.hover_origin  = np.array([self.state[0],
                                           self.state[1],
                                           self.state[5]])
            self.hover_entry_t = now_sec
            self.stable_since  = None
            self.phase         = Phase.HOVER
            self.get_logger().info(
                f'Phase -> HOVER  '
                f'origin=({self.hover_origin[0]:.2f}, {self.hover_origin[1]:.2f})  '
                f'target_alt={self.HOVER_ALT}m  '
                f'(will wait indefinitely for /reference_path)')

        elif self.phase == Phase.HOVER:
            # Check if hover is stable enough to accept a trajectory
            pos_err  = math.sqrt(
                (self.state[0] - self.hover_origin[0])**2 +
                (self.state[1] - self.hover_origin[1])**2 +
                (self.state[2] - self.HOVER_ALT)**2)
            vel_norm = float(np.linalg.norm(self.state[6:9]))
            is_stable = (pos_err < self.HOVER_POS_TOL and
                         vel_norm < self.HOVER_VEL_TOL)

            if is_stable:
                if self.stable_since is None:
                    self.stable_since = now_sec
            else:
                self.stable_since = None   # reset if disturbed

            hover_ready = (self.stable_since is not None and
                           (now_sec - self.stable_since) >= self.HOVER_STABLE_TIME)

            # Only enter TRAJECTORY when hover is stable AND path is available
            if hover_ready and path_fresh:
                self.traj_start_t = now_sec
                self.phase = Phase.TRAJECTORY
                self.get_logger().info(
                    f'Phase -> TRAJECTORY  '
                    f'(hover stable {now_sec - self.stable_since:.1f}s, '
                    f'external path received)')

        elif self.phase == Phase.TRAJECTORY:
            # Revert to HOVER if external path goes silent
            if not path_fresh:
                self.phase        = Phase.HOVER
                self.stable_since = None   # re-check stability on return
                self.get_logger().warn(
                    f'External /reference_path lost '
                    f'(>{self.EXTERNAL_TRAJ_TIMEOUT}s silence) — '
                    f'reverting to HOVER at '
                    f'({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f})')
                # Update hover_origin to current position so drone doesn't
                # snap back to its original takeoff spot
                self.hover_origin = np.array([self.state[0],
                                              self.state[1],
                                              self.state[5]])

        # ── Build reference & solve ────────────────────────────────────────────
        X_ref = self._get_reference(now_sec, path_fresh)

        # Align yaw reference to current heading (avoid pi-wrap cost spikes)
        cur_yaw = self.state[5]
        for k in range(self.N):
            X_ref[k, 5] = cur_yaw + wrap_angle(X_ref[k, 5] - cur_yaw)

        u_opt = self._solve_mpc(self.state, X_ref)
        omega = self._u_to_omega(u_opt)
        self._publish_omega(omega)


        torque = Float64MultiArray()
        torque.data = [float(u) for u in u_opt]
        self.control_pub.publish(torque)

        # ── Diagnostics (~2 Hz) ────────────────────────────────────────────────
        if int(now_sec * 2) % 4 == 0:
            pos  = self.state[:3]
            rpy  = np.degrees(self.state[3:6])
            ref0 = X_ref[0, :3]
            err  = float(np.linalg.norm(pos - ref0))
            du   = u_opt - self.u_hover
            src  = 'EXT' if (self.phase == Phase.TRAJECTORY and path_fresh) else 'HOV'
            self.get_logger().info(
                f'[{self.phase.name:10s}|{src}] '
                f'pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] '
                f'ref=[{ref0[0]:+.2f},{ref0[1]:+.2f},{ref0[2]:+.2f}] '
                f'err={err:.3f}m | '
                f'rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]deg | '
                f'dF={du[0]:+.2f}N | '
                f'omega={np.round(omega).astype(int)} rad/s'
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Reference trajectory builder
    # ─────────────────────────────────────────────────────────────────────────

    def _get_reference(self, now_sec, path_fresh):
        """
        Return X_ref (N, 12).

        IDLE / HOVER       → constant hover setpoint at hover_origin
        TRAJECTORY + fresh → horizon built from external /reference_path
        TRAJECTORY + stale → should not occur (phase reverted above), but
                             returns hover setpoint as a safety fallback
        """
        N, dt = self.N, self.DT
        hx, hy, hyaw = self.hover_origin

        if self.phase in (Phase.IDLE, Phase.HOVER):
            return self._hover_setpoint(hx, hy, hyaw, N)

        # TRAJECTORY phase — path is guaranteed fresh (phase reverted otherwise)
        if path_fresh:
            return self._build_horizon_from_path(self.ext_path, N, dt)

        # Safety fallback (should not be reached in normal operation)
        return self._hover_setpoint(hx, hy, hyaw, N)

    # ─────────────────────────────────────────────────────────────────────────
    # Constant hover setpoint
    # ─────────────────────────────────────────────────────────────────────────

    def _hover_setpoint(self, hx, hy, hyaw, N):
        """All N horizon steps fixed at (hx, hy, HOVER_ALT, hyaw), zero velocity."""
        refs = np.zeros((N, 12))
        for k in range(N):
            refs[k] = [hx, hy, self.HOVER_ALT,
                       0., 0., hyaw,
                       0., 0., 0.,
                       0., 0., 0.]
        return refs

    # ─────────────────────────────────────────────────────────────────────────
    # Build MPC horizon from nav_msgs/Path
    # ─────────────────────────────────────────────────────────────────────────

    def _build_horizon_from_path(self, path: Path, N: int, dt: float):
        """
        Convert a nav_msgs/Path into X_ref (N, 12).

        Pose k  → position + yaw for step k
        Finite-difference between poses k and k+1 → velocity estimates
        """
        poses   = path.poses
        n_poses = len(poses)
        refs    = np.zeros((N, 12))

        def get_pose(i):
            i = min(i, n_poses - 1)
            p   = poses[i].pose
            yaw = quat2yaw(p.orientation.z, p.orientation.w)
            return p.position.x, p.position.y, p.position.z, yaw

        for k in range(N):
            x0, y0, z0, yaw0 = get_pose(k)
            x1, y1, z1, yaw1 = get_pose(k + 1)

            vx     = (x1 - x0) / dt
            vy     = (y1 - y0) / dt
            vz     = (z1 - z0) / dt
            yawdot = wrap_angle(yaw1 - yaw0) / dt

            refs[k] = [x0,  y0,  z0,
                       0.,  0.,  yaw0,
                       vx,  vy,  vz,
                       0.,  0.,  yawdot]
        return refs

    # ─────────────────────────────────────────────────────────────────────────

    def _publish_omega(self, omega):
        msg = Actuators()
        msg.velocity = [float(w) for w in omega]
        self.pub_motors.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()