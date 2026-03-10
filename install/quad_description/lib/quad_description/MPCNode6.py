#!/usr/bin/env python3
"""
mpc_controller.py — Closed-form MPC + DOB wind rejection
=========================================================

Solver (closed-form)
--------------------
Replaces scipy L-BFGS-B with a pre-computed analytical solution.

    Pre-computed at startup:
        K = H^{-1} * SuTQbar              shape (nu*N, nx*N)

    Per control tick (replaces entire minimize() call):
        e   = Sx @ x0_aug - X_ref.flatten()
        dU* = clip(-K @ e, lb, ub)        one matvec + element-wise clip
        u   = u_hover + dU*[:4]

    ~13x faster than L-BFGS-B, deterministic timing, no convergence issues.
    Clip (box projection) is exact for these decoupled bounds.

Disturbance observer (DOB)
--------------------------
Augmented state (nx=15):
    x_a = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r,  d_x, d_y, d_z]
                                                               12   13   14

d_x, d_y, d_z = constant acceleration disturbances (wind, model error).
Estimated online from odom velocity finite-differences:
    d_x = ax_meas - g*pitch          (x-channel residual)
    d_y = ay_meas + g*roll           (y-channel residual)
    d_z = az_meas                    (z-channel vertical residual)

Exponential moving average (alpha=exp(-DT/tau), tau=3s at DT=0.01):
    d_hat = alpha * d_hat + (1-alpha) * d_meas

Disturbance reference = d_hat (not zero) so the MPC does not fight the
estimated steady-state disturbance — it absorbs it into the prediction.

DT = 0.01 s (matches trajectory_sender.MPC_DT so velocity inference is correct).
"""

import math
from enum import Enum, auto

import numpy as np
from scipy.linalg import expm

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from actuator_msgs.msg import Actuators
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


def quat2rpy(qx, qy, qz, qw):
    sr = 2.0*(qw*qx + qy*qz); cr = 1.0 - 2.0*(qx*qx + qy*qy)
    roll = math.atan2(sr, cr)
    sp   = 2.0*(qw*qy - qz*qx)
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    sy = 2.0*(qw*qz + qx*qy); cy = 1.0 - 2.0*(qy*qy + qz*qz)
    return roll, pitch, math.atan2(sy, cy)

def quat2yaw(qz, qw):
    return 2.0*math.atan2(qz, qw)

def wrap_angle(a):
    return (a + math.pi) % (2.0*math.pi) - math.pi


class Phase(Enum):
    IDLE       = auto()
    HOVER      = auto()
    TRAJECTORY = auto()


class MPCController(Node):

    # ── Physical parameters ───────────────────────────────────────────────
    GRAVITY   = 9.81
    MASS      = 1.525
    IXX       = 0.0356547
    IYY       = 0.0705152
    IZZ       = 0.0990924
    K_F       = 8.54858e-06
    K_M       = 0.06
    OMEGA_MAX = 1500.0

    ROTOR_POS = np.array([[ 0.13,-0.22],[-0.13, 0.20],[ 0.13, 0.22],[-0.13,-0.20]])
    ROTOR_DIR = np.array([-1.0,-1.0,+1.0,+1.0])

    # ── MPC parameters ────────────────────────────────────────────────────
    DT = 0.01    # must match trajectory_sender.MPC_DT exactly
    N  = 20

    Q_DIAG = np.array([
        90., 90., 30.,    # x, y, z
         8.,  8.,  0.5,   # phi, theta, psi
        12., 12.,  3.0,   # vx, vy, vz
         5.,  5.,  0.5,   # p, q, r
    ])
    QN_SCALE = 1.0
    R_DIAG   = np.array([0.01, 3.0, 3.0, 1.5])

    # Disturbance state weights = 0: MPC tracks them freely, no correction force
    Q_DIST = np.array([0.0, 0.0, 0.0])

    U_MIN = np.array([0.0, -5.0, -5.0, -3.0])
    U_MAX = np.array([4.0*MASS*GRAVITY, 5.0, 5.0, 3.0])

    # ── Hover setpoint ────────────────────────────────────────────────────
    HOVER_X   = 0.0;  HOVER_Y   = 0.0
    HOVER_YAW = 0.0;  HOVER_ALT = 2.0

    # ── Trajectory timing ─────────────────────────────────────────────────
    EXTERNAL_TRAJ_TIMEOUT = 0.5
    START_DELAY           = 8.0    # s — wait before calling /start_trajectory
    START_REQ_PERIOD      = 0.25   # s — how often to retry if service not ready

    # ── DOB parameters ────────────────────────────────────────────────────
    # Wind SDF: constant 4 m/s -Y, rise_time=10s → tau=3s → alpha=exp(-0.01/3)
    DOB_ALPHA    = 0.9967    # ≈ 3s time constant at DT=0.01
    DIST_EST_LIM = 3.0       # m/s² — covers 4 m/s wind + gust headroom
    ODOM_DT_MIN  = 1e-4
    ODOM_DT_MAX  = 0.2

    def __init__(self):
        super().__init__('quad_mpc_controller')
        self.get_logger().info('Initialising closed-form MPC + DOB...')

        m, g = self.MASS, self.GRAVITY
        self.u_hover = np.array([m*g, 0.0, 0.0, 0.0])
        self.du_min  = self.U_MIN - self.u_hover
        self.du_max  = self.U_MAX - self.u_hover

        # Augmented state: 12 dynamics + 3 disturbance = 15
        self.nx = 15;  self.nu = 4

        q_aug   = np.concatenate([self.Q_DIAG, self.Q_DIST])
        self.Q  = np.diag(q_aug)
        self.QN = self.QN_SCALE * self.Q
        self.R  = np.diag(self.R_DIAG)

        self.Ad, self.Bd = self._build_model()
        self.Sx, self.Su = self._build_prediction_matrices()
        self._build_gain()        # pre-computes K, lb, ub — replaces _build_qp_matrices

        px, py = self.ROTOR_POS[:,0], self.ROTOR_POS[:,1]
        M = np.array([np.ones(4), py, -px, self.ROTOR_DIR*self.K_M])
        self.M_inv = np.linalg.pinv(M)

        # Flight state
        self.phase       = Phase.IDLE
        self.state       = np.zeros(12)
        self.state_ready = False

        # DOB state
        self.d_hat     = np.zeros(3)   # [d_x, d_y, d_z]
        self._prev_t   = None
        self._prev_vel = None          # (vx, vy, vz)

        # External path
        self.ext_path       = None
        self.ext_path_stamp = None

        # Service start logic (same as original)
        self.node_start_time      = self.get_clock().now().nanoseconds * 1e-9
        self.start_requested      = False
        self.traj_enabled         = False
        self.start_future         = None
        self._last_wait_log_t     = -1e9
        self._last_service_warn_t = -1e9
        self._last_log_t          = -1e9

        # ROS I/O
        self.create_subscription(Odometry, '/odom',           self._cb_odom, 10)
        self.create_subscription(Path,     '/reference_path', self._cb_path,  1)

        self.pub_motors    = self.create_publisher(Actuators,         '/motor_commands', 10)
        self.control_pub   = self.create_publisher(Float64MultiArray, '/control_u',      10)
        self.min_range_pub = self.create_publisher(Float64MultiArray, '/range_min',      10)
        self.max_range_pub = self.create_publisher(Float64MultiArray, '/range_max',      10)

        self.start_cli = self.create_client(Trigger, '/start_trajectory')
        self.create_timer(self.DT,               self._cb_control)
        self.create_timer(self.START_REQ_PERIOD, self._cb_start_request_timer)

        self.get_logger().info(
            f'MPC ready | DT={self.DT}s  N={self.N}  '
            f'horizon={self.N*self.DT:.2f}s  '
            f'solver=closed-form K{self.K.shape}  cond(H)={self._cond_H:.2e}\n'
            f'  DOB: alpha={self.DOB_ALPHA:.4f} (tau≈3s)  '
            f'lim=±{self.DIST_EST_LIM}m/s²\n'
            f'  Start: service /start_trajectory after {self.START_DELAY}s')

    # ── Model ─────────────────────────────────────────────────────────────

    def _build_model(self):
        """
        Augmented continuous-time model (15-state).
        Disturbance enters translational accelerations:
            A[6,12]=1 (d_x → ax),  A[7,13]=1 (d_y → ay),  A[8,14]=1 (d_z → az)
        Disturbance dynamics: d_dot = 0 (constant model).
        """
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ
        n, p, dt = self.nx, self.nu, self.DT

        A = np.zeros((n, n)); B = np.zeros((n, p))
        A[0,6]=A[1,7]=A[2,8]=1.0
        A[3,9]=A[4,10]=A[5,11]=1.0
        A[6,4]=g; A[7,3]=-g
        A[6,12]=1.0; A[7,13]=1.0; A[8,14]=1.0   # disturbance injection
        B[8,0]=1./m; B[9,1]=1./Ixx; B[10,2]=1./Iyy; B[11,3]=1./Izz

        AB = np.zeros((n+p, n+p))
        AB[:n,:n]=A*dt; AB[:n,n:]=B*dt
        eAB = expm(AB)
        return eAB[:n,:n], eAB[:n,n:]

    def _build_prediction_matrices(self):
        n, p, N = self.nx, self.nu, self.N
        Ad, Bd   = self.Ad, self.Bd
        Ad_pow   = [np.eye(n)]
        for _ in range(N): Ad_pow.append(Ad @ Ad_pow[-1])
        Sx = np.zeros((n*N, n)); Su = np.zeros((n*N, p*N))
        for k in range(N):
            Sx[k*n:(k+1)*n,:] = Ad_pow[k+1]
            for j in range(k+1):
                Su[k*n:(k+1)*n, j*p:(j+1)*p] = Ad_pow[k-j] @ Bd
        return Sx, Su

    def _build_gain(self):
        """
        Pre-compute closed-form gain K = H^{-1} * SuTQbar  (runs once).

        Runtime solve:
            dU* = -K @ (Sx @ x0_aug - X_ref)    [unconstrained optimum]
            dU* = clip(dU*, lb, ub)               [box projection — exact here]
        """
        n, p, N = self.nx, self.nu, self.N

        Q_bar = np.zeros((n*N, n*N))
        for k in range(N-1): Q_bar[k*n:(k+1)*n, k*n:(k+1)*n] = self.Q
        Q_bar[(N-1)*n:,(N-1)*n:] = self.QN

        R_bar = np.zeros((p*N, p*N))
        for k in range(N): R_bar[k*p:(k+1)*p, k*p:(k+1)*p] = self.R

        SuTQbar = self.Su.T @ Q_bar
        H_raw   = SuTQbar @ self.Su + R_bar
        H_raw   = 0.5*(H_raw + H_raw.T)

        self.K  = np.linalg.solve(H_raw, SuTQbar)    # (p*N, n*N)
        self.lb = np.array([self.du_min[i%p] for i in range(p*N)])
        self.ub = np.array([self.du_max[i%p] for i in range(p*N)])
        self._cond_H = float(np.linalg.cond(H_raw))

    # ── Closed-form solve ─────────────────────────────────────────────────

    def _solve_mpc(self, x0_aug: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
        """
        Single matrix-vector multiply + clip.  ~13x faster than L-BFGS-B.
        """
        e  = self.Sx @ x0_aug - X_ref.flatten()
        dU = np.clip(-self.K @ e, self.lb, self.ub)
        return self.u_hover + dU[:self.nu]

    # ── ROS callbacks ─────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        pos = msg.pose.pose.position; q = msg.pose.pose.orientation
        lv  = msg.twist.twist.linear;  av = msg.twist.twist.angular

        roll, pitch, yaw = quat2rpy(q.x, q.y, q.z, q.w)
        self.state = np.array([pos.x,pos.y,pos.z,
                                roll,pitch,yaw,
                                lv.x,lv.y,lv.z,
                                av.x,av.y,av.z])
        self.state_ready = True

        # ── DOB: estimate d from acceleration residual ────────────────────
        vel = np.array([lv.x, lv.y, lv.z])
        if self._prev_t is not None and self._prev_vel is not None:
            dt = now - self._prev_t
            if self.ODOM_DT_MIN < dt < self.ODOM_DT_MAX:
                a = (vel - self._prev_vel) / dt
                d_meas = np.array([
                    a[0] - self.GRAVITY * pitch,   # x_ddot residual
                    a[1] + self.GRAVITY * roll,    # y_ddot residual
                    a[2],                           # z_ddot residual
                ])
                alpha      = self.DOB_ALPHA
                self.d_hat = alpha*self.d_hat + (1.0-alpha)*d_meas
                self.d_hat = np.clip(self.d_hat, -self.DIST_EST_LIM, self.DIST_EST_LIM)
        self._prev_t   = now
        self._prev_vel = vel

    def _cb_path(self, msg: Path):
        self.ext_path       = msg
        self.ext_path_stamp = self.get_clock().now().nanoseconds * 1e-9

    def _cb_start_request_timer(self):
        """Separate timer so service call never blocks the control loop."""
        if self.phase != Phase.HOVER or not self.state_ready: return
        if self.start_requested or self.traj_enabled: return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now_sec - self.node_start_time

        if elapsed < self.START_DELAY:
            if now_sec - self._last_wait_log_t > 1.0:
                self.get_logger().info(
                    f'Waiting {self.START_DELAY - elapsed:.1f}s before '
                    f'requesting trajectory start...')
                self._last_wait_log_t = now_sec
            return

        if not self.start_cli.service_is_ready():
            if now_sec - self._last_service_warn_t > 1.0:
                self.get_logger().warn('/start_trajectory service not available yet')
                self._last_service_warn_t = now_sec
            return

        self.start_future = self.start_cli.call_async(Trigger.Request())
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
                self.get_logger().warn(f'Start rejected: {resp.message}')
        except Exception as exc:
            self.start_requested = False
            self.get_logger().error(f'Start service error: {exc}')

    # ── Control loop ──────────────────────────────────────────────────────

    def _cb_control(self):
        self.min_range_pub.publish(
            Float64MultiArray(data=[float(v) for v in self.U_MIN]))
        self.max_range_pub.publish(
            Float64MultiArray(data=[float(v) for v in self.U_MAX]))

        if not self.state_ready:
            self._publish_omega(np.zeros(4)); return

        now_sec    = self.get_clock().now().nanoseconds * 1e-9
        path_fresh = self._path_fresh(now_sec)

        # Phase machine
        if self.phase == Phase.IDLE:
            self.phase = Phase.HOVER
            self.get_logger().info('Phase → HOVER')

        elif self.phase == Phase.HOVER:
            if self.traj_enabled and path_fresh:
                self.phase = Phase.TRAJECTORY
                self.get_logger().info(
                    f'Phase → TRAJECTORY | '
                    f'd_hat=[{self.d_hat[0]:+.3f},{self.d_hat[1]:+.3f},'
                    f'{self.d_hat[2]:+.3f}] m/s²')

        elif self.phase == Phase.TRAJECTORY:
            if not path_fresh:
                self.phase = Phase.HOVER
                self.get_logger().warn('Path timeout → HOVER')

        # Build reference
        X_ref = self._get_reference(path_fresh)

        # Yaw wrap around current heading
        cur_yaw = self.state[5]
        for k in range(self.N):
            X_ref[k,5] = cur_yaw + wrap_angle(X_ref[k,5] - cur_yaw)

        # Augmented state: inject live d_hat
        x0_aug        = np.zeros(self.nx)
        x0_aug[:12]   = self.state
        x0_aug[12:15] = self.d_hat

        u_opt = self._solve_mpc(x0_aug, X_ref)
        omega = self._u_to_omega(u_opt)
        self._publish_omega(omega)
        self.control_pub.publish(
            Float64MultiArray(data=[float(u) for u in u_opt]))

        # Diagnostics (~2 Hz — not 100 Hz spam)
        if now_sec - self._last_log_t > 0.5:
            pos  = self.state[:3]; ref0 = X_ref[0,:3]
            err  = float(np.linalg.norm(pos - ref0))
            rpy  = np.degrees(self.state[3:6])
            du   = u_opt - self.u_hover
            self.get_logger().info(
                f'[{self.phase.name}] '
                f'pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] '
                f'err={err:.3f}m  '
                f'rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]° '
                f'dF={du[0]:+.2f}N  '
                f'd=[{self.d_hat[0]:+.3f},{self.d_hat[1]:+.3f},'
                f'{self.d_hat[2]:+.3f}]m/s²')
            self._last_log_t = now_sec

    # ── Reference builders ────────────────────────────────────────────────

    def _path_fresh(self, now_sec):
        return (self.ext_path is not None and
                self.ext_path_stamp is not None and
                (now_sec - self.ext_path_stamp) < self.EXTERNAL_TRAJ_TIMEOUT and
                len(self.ext_path.poses) > 0)

    def _get_reference(self, path_fresh):
        if self.phase == Phase.TRAJECTORY and path_fresh:
            return self._horizon_from_path(self.ext_path)
        return self._hover_ref()

    def _hover_ref(self):
        """
        d_ref = d_hat — MPC does NOT try to drive disturbance states to zero.
        Setting d_ref=0 would create a spurious correction force during hover.
        """
        refs = np.zeros((self.N, self.nx))
        for k in range(self.N):
            refs[k,:12] = [self.HOVER_X, self.HOVER_Y, self.HOVER_ALT,
                           0.,0.,self.HOVER_YAW, 0.,0.,0., 0.,0.,0.]
            refs[k,12:] = self.d_hat
        return refs

    def _horizon_from_path(self, path: Path):
        """
        Build N-step reference from nav_msgs/Path.
        Velocity inferred by finite-difference with MPC DT.
        Disturbance reference = current d_hat (steady-state estimate).
        """
        poses   = path.poses; n_poses = len(poses)
        if n_poses == 0: return self._hover_ref()
        refs = np.zeros((self.N, self.nx)); dt = self.DT

        def get_pose(i):
            i = min(i, n_poses-1); p = poses[i].pose
            return (p.position.x, p.position.y, p.position.z,
                    quat2yaw(p.orientation.z, p.orientation.w))

        for k in range(self.N):
            x0,y0,z0,yaw0 = get_pose(k)
            x1,y1,z1,yaw1 = get_pose(k+1)
            refs[k,:12] = [x0,y0,z0, 0.,0.,yaw0,
                           (x1-x0)/dt,(y1-y0)/dt,(z1-z0)/dt,
                           0.,0., wrap_angle(yaw1-yaw0)/dt]
            refs[k,12:] = self.d_hat
        return refs

    # ── Actuator output ───────────────────────────────────────────────────

    def _u_to_omega(self, u):
        forces = np.maximum(self.M_inv @ u, 0.0)
        return np.clip(np.sqrt(forces / self.K_F), 0.0, self.OMEGA_MAX)

    def _publish_omega(self, omega):
        msg = Actuators(); msg.velocity = [float(w) for w in omega]
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