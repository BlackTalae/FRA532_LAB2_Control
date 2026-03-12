#!/usr/bin/env python3
"""
mpc_controller.py
=================
ROS 2 MPC node for quadrotor 3-D trajectory tracking.

Key behavior
------------
- Fixed hover reference at (0, 0, 2, yaw=0)
- Stay in HOVER until:
    1) startup delay has elapsed
    2) /start_trajectory service has been requested successfully
    3) /reference_path is fresh
- Track /reference_path only in TRAJECTORY phase
- Revert to HOVER if path times out

Important design choice
-----------------------
- The /start_trajectory service request is handled by a SEPARATE timer,
  not inside the control loop. This prevents blocking the 100 Hz MPC loop.
"""

import math
from enum import Enum, auto

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from actuator_msgs.msg import Actuators
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


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


class Phase(Enum):
    IDLE = auto()
    HOVER = auto()
    TRAJECTORY = auto()


class MPCController(Node):
    # ── Physical parameters ────────────────────────────────────────────────
    GRAVITY = 9.81
    MASS = 1.525
    IXX = 0.0356547
    IYY = 0.0705152
    IZZ = 0.0990924

    K_F = 8.54858e-06
    K_M = 0.06
    OMEGA_MAX = 1500.0

    ROTOR_POS = np.array([
        [ 0.13, -0.22],
        [-0.13,  0.20],
        [ 0.13,  0.22],
        [-0.13, -0.20],
    ])
    ROTOR_DIR = np.array([-1.0, -1.0, +1.0, +1.0])

    # ── MPC hyperparameters ────────────────────────────────────────────────
    DT = 0.01
    N = 20

    # Cooked!!!
    Q_DIAG = np.array([
        550., 550., 36.,
        75.,  55.,  5.5,
        30., 30., 3.5,
        2.,  2.,  1.5
        ])
    QN_SCALE = 0.0
    R_DIAG = np.array([0.01, 1.0, 1.0, 2.0])

    # Q_DIAG = np.array([
    #     600., 630., 36.,
    #     60.,  50.,  5.5,
    #     11., 10., 3.5,
    #     2.,  2.,  1.5
    # ])
    # QN_SCALE = 0.0
    # R_DIAG = np.array([0.01, 1.19, 1.0, 2.0])


    U_MIN = np.array([0.0, -5.0, -5.0, -3.0])
    U_MAX = np.array([4.0 * MASS * GRAVITY, 5.0, 5.0, 3.0])

    # ── Hover reference ────────────────────────────────────────────────────
    HOVER_X = 0.0
    HOVER_Y = 0.0
    HOVER_YAW = 0.0
    HOVER_ALT = 2.0

    # ── Trajectory management ───────────────────────────────────────────────
    EXTERNAL_TRAJ_TIMEOUT = 0.5
    START_DELAY = 8.0
    START_REQ_PERIOD = 0.25

    def __init__(self):
        super().__init__('quad_mpc_controller')
        self.get_logger().info('Initialising quadrotor MPC controller...')

        # Hover input
        m, g = self.MASS, self.GRAVITY
        self.u_hover = np.array([m * g, 0.0, 0.0, 0.0])

        # Optimize in delta-u
        self.du_min = self.U_MIN - self.u_hover
        self.du_max = self.U_MAX - self.u_hover

        self.Q = np.diag(self.Q_DIAG)
        self.QN = self.QN_SCALE * self.Q
        self.R = np.diag(self.R_DIAG)

        self.Ad, self.Bd = self._build_model()
        self.Sx, self.Su = self._build_prediction_matrices()
        self._build_qp_matrices()

        # Motor allocation
        px, py = self.ROTOR_POS[:, 0], self.ROTOR_POS[:, 1]
        M = np.array([
            np.ones(4),
            py,
            -px,
            self.ROTOR_DIR * self.K_M
        ])
        self.M_inv = np.linalg.pinv(M)

        # Controller state
        self.phase = Phase.IDLE
        self.state = np.zeros(12)
        self.state_ready = False
        self.dU_warm = np.zeros(4 * self.N)

        # External trajectory
        self.ext_path = None
        self.ext_path_stamp = None

        # Service-based start logic
        self.node_start_time = self.get_clock().now().nanoseconds * 1e-9
        self.start_requested = False
        self.traj_enabled = False
        self.start_future = None

        self._last_wait_log_t = -1e9
        self._last_service_warn_t = -1e9

        # ROS interfaces
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(Path, '/reference_path', self._cb_path, 1)

        self.pub_motors = self.create_publisher(Actuators, '/motor_commands', 10)
        self.control_pub = self.create_publisher(Float64MultiArray, '/control_u', 10)
        self.min_range_pub = self.create_publisher(Float64MultiArray, '/range_min', 10)
        self.max_range_pub = self.create_publisher(Float64MultiArray, '/range_max', 10)

        self.start_cli = self.create_client(Trigger, '/start_trajectory')

        # Timers
        self.control_timer = self.create_timer(self.DT, self._cb_control)
        self.start_req_timer = self.create_timer(self.START_REQ_PERIOD, self._cb_start_request_timer)

        self.get_logger().info(
            f'MPC ready | DT={self.DT:.3f}s | N={self.N} | '
            f'horizon={self.N * self.DT:.2f}s | hover=({self.HOVER_X}, {self.HOVER_Y}, {self.HOVER_ALT})'
        )

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        roll, pitch, yaw = quat2rpy(q.x, q.y, q.z, q.w)

        self.state = np.array([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw,
            lv.x, lv.y, lv.z,
            av.x, av.y, av.z
        ])
        self.state_ready = True

    def _cb_path(self, msg: Path):
        self.ext_path = msg
        self.ext_path_stamp = self.get_clock().now().nanoseconds * 1e-9

    def _cb_start_request_timer(self):
        """
        Separate timer for requesting /start_trajectory.
        This avoids blocking the high-rate MPC control loop.
        """
        if self.phase != Phase.HOVER:
            return
        if not self.state_ready:
            return
        if self.start_requested or self.traj_enabled:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        startup_elapsed = now_sec - self.node_start_time

        if startup_elapsed < self.START_DELAY:
            if (now_sec - self._last_wait_log_t) > 1.0:
                remain = self.START_DELAY - startup_elapsed
                self.get_logger().info(
                    f'Waiting {remain:.1f}s before requesting trajectory start...'
                )
                self._last_wait_log_t = now_sec
            return

        if not self.start_cli.service_is_ready():
            if (now_sec - self._last_service_warn_t) > 1.0:
                self.get_logger().warn('/start_trajectory service not available yet')
                self._last_service_warn_t = now_sec
            return

        req = Trigger.Request()
        self.start_future = self.start_cli.call_async(req)
        self.start_future.add_done_callback(self._on_start_response)
        self.start_requested = True
        self.get_logger().info('Requested /start_trajectory service')

    def _on_start_response(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.traj_enabled = True
                self.get_logger().info(f'Trajectory start confirmed: {resp.message}')
            else:
                self.start_requested = False
                self.get_logger().warn(f'Trajectory start rejected: {resp.message}')
        except Exception as e:
            self.start_requested = False
            self.get_logger().error(f'Start trajectory service failed: {e}')

    # ── Model / MPC build ───────────────────────────────────────────────────

    def _build_model(self):
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ
        n, p, dt = 12, 4, self.DT

        A = np.zeros((n, n))
        B = np.zeros((n, p))

        A[0, 6] = A[1, 7] = A[2, 8] = 1.0
        A[3, 9] = A[4,10] = A[5,11] = 1.0
        A[6, 4] = g
        A[7, 3] = -g

        B[8, 0] = 1.0 / m
        B[9, 1] = 1.0 / Ixx
        B[10,2] = 1.0 / Iyy
        B[11,3] = 1.0 / Izz

        AB = np.zeros((n + p, n + p))
        AB[:n, :n] = A * dt
        AB[:n, n:] = B * dt

        eAB = expm(AB)
        return eAB[:n, :n], eAB[:n, n:]

    def _build_prediction_matrices(self):
        n, p, N = 12, 4, self.N
        Ad, Bd = self.Ad, self.Bd

        Ad_pow = [np.eye(n)]
        for _ in range(N):
            Ad_pow.append(Ad @ Ad_pow[-1])

        Sx = np.zeros((n * N, n))
        Su = np.zeros((n * N, p * N))

        for k in range(N):
            Sx[k*n:(k+1)*n, :] = Ad_pow[k + 1]
            for j in range(k + 1):
                Su[k*n:(k+1)*n, j*p:(j+1)*p] = Ad_pow[k - j] @ Bd

        return Sx, Su

    def _build_qp_matrices(self):
        n, p, N = 12, 4, self.N

        Q_bar = np.zeros((n * N, n * N))
        for k in range(N - 1):
            Q_bar[k*n:(k+1)*n, k*n:(k+1)*n] = self.Q
        Q_bar[(N - 1)*n:, (N - 1)*n:] = self.QN

        R_bar = np.zeros((p * N, p * N))
        for k in range(N):
            R_bar[k*p:(k+1)*p, k*p:(k+1)*p] = self.R

        H_raw = self.Su.T @ Q_bar @ self.Su + R_bar
        H_raw = 0.5 * (H_raw + H_raw.T)

        scale = max(np.trace(H_raw) / (p * N), 1.0)
        self.H = H_raw / scale
        self.SuTQbar = (self.Su.T @ Q_bar) / scale
        self.qp_scale = scale

        self.qp_bounds = [
            (float(self.du_min[i % p]), float(self.du_max[i % p]))
            for i in range(p * N)
        ]

    # ── MPC solve ────────────────────────────────────────────────────────────

    def _solve_mpc(self, x0, X_ref):
        e = self.Sx @ x0 - X_ref.flatten()
        c = self.SuTQbar @ e

        dU_prev = self.dU_warm.copy()

        def cost(dU):
            return 0.5 * float(dU @ (self.H @ dU)) + float(c @ dU)

        def grad(dU):
            return self.H @ dU + c

        result = minimize(
            cost,
            self.dU_warm,
            jac=grad,
            method='L-BFGS-B',
            bounds=self.qp_bounds,
            options={'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-4}
        )

        if result.success and np.all(np.isfinite(result.x)):
            dU_opt = result.x
        elif hasattr(result, 'x') and result.x is not None and np.all(np.isfinite(result.x)):
            self.get_logger().warn(f'MPC solve not fully successful: {result.message}')
            dU_opt = result.x
        else:
            self.get_logger().warn('MPC solve failed badly, fallback to previous warm start')
            dU_opt = dU_prev

        self.dU_warm[:-4] = dU_opt[4:]
        self.dU_warm[-4:] = 0.0

        return self.u_hover + dU_opt[:4]

    # ── Control ──────────────────────────────────────────────────────────────

    def _cb_control(self):
        # Publish absolute bounds because /control_u publishes absolute u
        self.min_range_pub.publish(Float64MultiArray(data=[float(v) for v in self.U_MIN]))
        self.max_range_pub.publish(Float64MultiArray(data=[float(v) for v in self.U_MAX]))

        if not self.state_ready:
            self._publish_omega(np.zeros(4))
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        path_fresh = self._is_path_fresh(now_sec)

        # Phase logic
        if self.phase == Phase.IDLE:
            self.phase = Phase.HOVER
            self.get_logger().info('Phase -> HOVER')

        elif self.phase == Phase.HOVER:
            if self.traj_enabled and path_fresh:
                self.phase = Phase.TRAJECTORY
                self.get_logger().info('Phase -> TRAJECTORY')

        elif self.phase == Phase.TRAJECTORY:
            if not path_fresh:
                self.phase = Phase.HOVER
                self.get_logger().warn('Path timeout -> HOVER')

        X_ref = self._get_reference(path_fresh)

        # Align yaw reference near current yaw to avoid wrap jumps
        cur_yaw = self.state[5]
        for k in range(self.N):
            X_ref[k, 5] = cur_yaw + wrap_angle(X_ref[k, 5] - cur_yaw)

        u_opt = self._solve_mpc(self.state, X_ref)
        omega = self._u_to_omega(u_opt)

        self._publish_omega(omega)
        self.control_pub.publish(Float64MultiArray(data=[float(u) for u in u_opt]))

    def _is_path_fresh(self, now_sec: float) -> bool:
        return (
            self.ext_path is not None and
            self.ext_path_stamp is not None and
            (now_sec - self.ext_path_stamp) < self.EXTERNAL_TRAJ_TIMEOUT and
            len(self.ext_path.poses) > 0
        )

    def _get_reference(self, path_fresh: bool):
        if self.phase == Phase.TRAJECTORY and path_fresh:
            return self._build_horizon_from_path(self.ext_path, self.N, self.DT)
        return self._hover_setpoint(self.N)

    def _hover_setpoint(self, N: int):
        refs = np.zeros((N, 12))
        for k in range(N):
            refs[k] = [
                self.HOVER_X, self.HOVER_Y, self.HOVER_ALT,
                0.0, 0.0, self.HOVER_YAW,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            ]
        return refs

    def _build_horizon_from_path(self, path: Path, N: int, dt: float):
        poses = path.poses
        n_poses = len(poses)

        if n_poses == 0:
            return self._hover_setpoint(N)

        refs = np.zeros((N, 12))

        def get_pose(i):
            i = min(i, n_poses - 1)
            p = poses[i].pose
            yaw = quat2yaw(p.orientation.z, p.orientation.w)
            return p.position.x, p.position.y, p.position.z, yaw

        for k in range(N):
            x0, y0, z0, yaw0 = get_pose(k)
            x1, y1, z1, yaw1 = get_pose(k + 1)
            xp, yp, zp, yawp = get_pose(k - 1)

            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            vz = (z1 - z0) / dt

            # vx = (x0 - xp) / dt
            # vy = (y0 - yp) / dt
            # vz = (z0 - zp) / dt

            yawdot = wrap_angle(yaw1 - yaw0) / dt
            yaw = 0
            yawdot = 0
            refs[k] = [
                x0, y0, z0,
                0.0, 0.0, yaw0,
                vx, vy, vz,
                0.0, 0.0, yawdot
            ]

        return refs

    # ── Actuator output ──────────────────────────────────────────────────────

    def _u_to_omega(self, u):
        forces = np.maximum(self.M_inv @ u, 0.0)
        omega = np.sqrt(forces / self.K_F)
        return np.clip(omega, 0.0, self.OMEGA_MAX)

    def _publish_omega(self, omega):
        msg = Actuators()
        msg.velocity = [float(w) for w in omega]
        self.pub_motors.publish(msg)


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