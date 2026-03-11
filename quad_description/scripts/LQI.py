#!/usr/bin/env python3
"""
lqi_controller.py — Quadrotor Integral LQR (LQI)
=================================================

Bug fixes from original (doc7)
-------------------------------
BUG 1 — DUPLICATE create_timer (root cause of "LQI ≠ LQR even with Ki=0")
  Original had two identical `self.create_timer(0.01, self._cb_control)` calls
  in __init__, making the control loop run at 200 Hz instead of 100 Hz.

  Effect:
    tick A: TARGET_X just changed → v_target = (T - T_old) / 0.01  (correct)
    tick B: TARGET_X_ was just set = TARGET_X → v_target = 0        (wrong!)
  → error[6:9] alternates between correct and zero every other tick
  → output oscillates → "misbehaves" even with Ki = 0

  Fix: removed the second create_timer call.

BUG 2 — _cb_start_request_timer ignores START_DELAY
  Original called the /start_trajectory service immediately on the first timer
  tick regardless of how long the drone had been running.
  Fix: added the same elapsed-time guard used in mpc_controller.py.

Ki gains (now enabled)
----------------------
Wind SDF: constant 4 m/s in -Y → drone drifts in -Y → ey > 0.
To correct: need negative roll (φ < 0) → negative τ_roll.
    dτ_roll = -ki_r * ∫ey   (ki_r > 0, minus sign in Ki matrix)

Physics motivation:
    ÿ = -g·φ  →  steady-state φ = -d_y/g ≈ -0.06 rad for 4 m/s wind
    Required τ_roll ≈ K_lqr[φ] * 0.06 ≈ small, so ki_r = 0.005 converges
    in ~10-15 s (integration time constant = ki_r * dt * loop_gain)

Anti-windup limits are asymmetric for y (larger, to handle 4 m/s wind).
"""

import math
import numpy as np
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


def quat2rpy(qx, qy, qz, qw):
    sr = 2.0*(qw*qx + qy*qz); cr = 1.0 - 2.0*(qx*qx + qy*qy)
    roll = math.atan2(sr, cr)
    sp   = 2.0*(qw*qy - qz*qx)
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    sy = 2.0*(qw*qz + qx*qy); cy = 1.0 - 2.0*(qy*qy + qz*qz)
    return roll, pitch, math.atan2(sy, cy)

def wrap_angle(a):
    return (a + math.pi) % (2.0*math.pi) - math.pi

def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


# ─────────────────────────────────────────────────────────────────────────────
# LQI controller object
# ─────────────────────────────────────────────────────────────────────────────

class LQI:
    """
    u_corr = K @ error_12 + Ki @ integ_4

    integ_4 = integral of [ex, ey, ez, e_yaw] with anti-windup.
    """
    # Anti-windup: [∫ex, ∫ey, ∫ez, ∫e_yaw]  (m·s, m·s, m·s, rad·s)
    # ∫ey larger: 4 m/s wind in -Y needs more integral authority
    INTEG_LIM = np.array([5.0, 12.0, 3.0, 0.8])

    def __init__(self, K, Ki):
        self.K     = K               # (4, 12)
        self.Ki    = Ki              # (4,  4)
        self.integ = np.zeros(4)

    def reset(self):
        self.integ[:] = 0.0

    def compute(self, error: np.ndarray, dt: float) -> np.ndarray:
        """error shape (12,), dt in seconds."""
        self.integ += np.array([
            error[0],   # ∫(ex)
            error[1],   # ∫(ey)
            error[2],   # ∫(ez)
            error[5],   # ∫(e_yaw)
        ]) * dt
        self.integ = np.clip(self.integ, -self.INTEG_LIM, self.INTEG_LIM)
        return self.K @ error + self.Ki @ self.integ


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────

class QuadrotorLQINode(Node):

    GRAVITY   = 9.81
    K_F       = 8.54858e-06
    K_M       = 0.06
    OMEGA_MAX = 1500.0
    MASS      = 1.525
    IXX       = 0.0356547
    IYY       = 0.0705152
    IZZ       = 0.0990924
    L         = 0.22

    DEFAULT_TARGET_X   = 0.0
    DEFAULT_TARGET_Y   = 0.0
    DEFAULT_TARGET_Z   = 2.0
    DEFAULT_TARGET_YAW = 0.0

    START_DELAY      = 8.0
    START_REQ_PERIOD = 0.25
    DT               = 0.01   # control period (s) — single timer only

    U_MIN = np.array([0.0, -5.0, -5.0, -3.0])
    U_MAX = np.array([4.0 * MASS * GRAVITY, 5.0, 5.0, 3.0])

    def __init__(self):
        super().__init__('quadrotor_lqi_node')

        self.F_hover = self.MASS * self.GRAVITY
        self.u_hover = np.array([self.F_hover, 0.0, 0.0, 0.0])

        # Motor allocation
        kF, kM = self.K_F, self.K_M
        self.Gamma = np.array([
            [ kF,         kF,         kF,         kF        ],
            [-kF * 0.22,  kF * 0.20,  kF * 0.22, -kF * 0.20],
            [-kF * 0.13,  kF * 0.13, -kF * 0.13,  kF * 0.13],
            [-kF * kM,   -kF * kM,    kF * kM,    kF * kM   ],
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        A, B = self._build_model()

        # LQR weights (same as LQR baseline)
        Q = np.diag([
            400., 400.,  80.,
             60.,  60.,  15.,
             45.,  40.,  30.,
              5.,   5.,   4.,
        ])
        R = np.diag([0.1, 3.0, 3.0, 5.0])
        K = lqr(A, B, Q, R)

        # ── Integral gains ────────────────────────────────────────────────────
        # Ki sign convention:
        #   ÿ = -g·φ  → ∫ey > 0 → want φ < 0 → need τ_roll < 0
        #               → Ki[τ_roll, ∫ey] = -ki_r  (minus sign)
        #   ẍ = +g·θ  → ∫ex > 0 → want θ > 0 → need τ_pitch > 0
        #               → Ki[τ_pitch, ∫ex] = +ki_p
        #   z̈ = F/m   → ∫ez > 0 → want +F
        #               → Ki[F, ∫ez] = +ki_F
        #   ψ̈ = τy/Iz → ∫e_yaw > 0 → want +τ_yaw
        #               → Ki[τ_yaw, ∫e_yaw] = +ki_yaw
        #
        # Tuned for 4 m/s wind in -Y:
        #   ki_r = 0.005 → integrator converges in ~10-15 s
        #   ki_F = 2.0   → corrects altitude offset in ~3-5 s
        ki_F   = 3.0
        ki_r   = 0.1
        ki_p   = 0.005
        ki_yaw = 0.1

        Ki = np.array([
            [0.0,   0.0,   ki_F,   0.0   ],   # dF      ← ki_F  * ∫ez
            [0.0,  -ki_r,  0.0,    0.0   ],   # dτ_roll ← -ki_r * ∫ey
            [ki_p,  0.0,   0.0,    0.0   ],   # dτ_pitch← ki_p  * ∫ex
            [0.0,   0.0,   0.0,    ki_yaw],   # dτ_yaw  ← ki_yaw* ∫e_yaw
        ])

        self.lqi = LQI(K, Ki)
        self.get_logger().info(
            f'LQI ready | K{K.shape} | '
            f'ki_F={ki_F} ki_r={ki_r} ki_p={ki_p} ki_yaw={ki_yaw}')

        # Target setpoint
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # Previous target (for velocity feed-forward)
        self._prev_tx = self.TARGET_X
        self._prev_ty = self.TARGET_Y
        self._prev_tz = self.TARGET_Z

        self.state       = np.zeros(12)
        self.state_ready = False

        # Service start state
        self.node_start_time  = self.get_clock().now().nanoseconds * 1e-9
        self.start_requested  = False
        self._last_wait_log_t = -1e9
        self._last_svc_warn_t = -1e9
        self._last_log_t      = -1e9

        # ROS interfaces
        self.create_subscription(Odometry,    '/odom',        self._cb_odom,   10)
        self.create_subscription(PoseStamped, '/target_pose', self._cb_target, 10)

        self.cmd_pub       = self.create_publisher(Actuators,         '/motor_commands', 10)
        self.control_pub   = self.create_publisher(Float64MultiArray, '/control_u',      10)
        self.min_range_pub = self.create_publisher(Float64MultiArray, '/range_min',      10)
        self.max_range_pub = self.create_publisher(Float64MultiArray, '/range_max',      10)

        self.start_cli = self.create_client(Trigger, '/start_trajectory')

        # ONE control timer (bug fix: original had two)
        self.create_timer(self.DT,               self._cb_control)
        self.create_timer(self.START_REQ_PERIOD, self._cb_start_req)

        omega_hover = math.sqrt(self.F_hover / (4.0 * self.K_F))
        self.get_logger().info(
            f'LQI node ready | hover ω={omega_hover:.1f} rad/s '
            f'F={self.F_hover:.2f} N | '
            f'target=({self.TARGET_X},{self.TARGET_Y},{self.TARGET_Z})')

    # ── Model ─────────────────────────────────────────────────────────────

    def _build_model(self):
        m, g = self.MASS, self.GRAVITY
        Ixx, Iyy, Izz = self.IXX, self.IYY, self.IZZ
        A = np.zeros((12, 12)); B = np.zeros((12, 4))
        A[0,6]=A[1,7]=A[2,8]=1.0
        A[3,9]=A[4,10]=A[5,11]=1.0
        A[6,4]=g; A[7,3]=-g
        B[8,0]=1./m; B[9,1]=1./Ixx; B[10,2]=1./Iyy; B[11,3]=1./Izz
        return A, B

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_odom(self, msg: Odometry):
        pos = msg.pose.pose.position; q = msg.pose.pose.orientation
        lv  = msg.twist.twist.linear;  av = msg.twist.twist.angular
        roll, pitch, yaw = quat2rpy(q.x, q.y, q.z, q.w)
        self.state = np.array([pos.x,pos.y,pos.z,
                                roll,pitch,yaw,
                                lv.x,lv.y,lv.z,
                                av.x,av.y,av.z])
        self.state_ready = True

    def _cb_target(self, msg: PoseStamped):
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        new_z = msg.pose.position.z
        q = msg.pose.orientation
        new_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                              1.0 - 2.0*(q.y**2 + q.z**2))
        # Reset integrator on large target changes
        dist = math.sqrt((new_x-self.TARGET_X)**2 +
                         (new_y-self.TARGET_Y)**2 +
                         (new_z-self.TARGET_Z)**2)
        if dist > 0.3 or abs(wrap_angle(new_yaw - self.TARGET_YAW)) > 0.175:
            self.lqi.reset()
            self.get_logger().info(f'Target changed ({dist:.2f}m) → integrators reset')
        self.TARGET_X   = new_x
        self.TARGET_Y   = new_y
        self.TARGET_Z   = new_z
        self.TARGET_YAW = new_yaw

    def _cb_start_req(self):
        """Requests /start_trajectory after START_DELAY — never blocks control loop."""
        if self.start_requested:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now_sec - self.node_start_time

        # Bug fix: respect START_DELAY (original called service immediately)
        if elapsed < self.START_DELAY:
            if now_sec - self._last_wait_log_t > 1.0:
                self.get_logger().info(
                    f'Waiting {self.START_DELAY - elapsed:.1f}s before trajectory start...')
                self._last_wait_log_t = now_sec
            return

        if not self.start_cli.service_is_ready():
            if now_sec - self._last_svc_warn_t > 1.0:
                self.get_logger().warn('/start_trajectory not available')
                self._last_svc_warn_t = now_sec
            return

        future = self.start_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_start_resp)
        self.start_requested = True
        self.get_logger().info('Requested /start_trajectory')

    def _on_start_resp(self, future):
        try:
            resp = future.result()
            if resp.success:
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
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        dt      = self.DT

        # ── Velocity feed-forward from target motion ───────────────────────
        # Computed BEFORE updating _prev so delta is over exactly one DT.
        # Bug fix: original computed v AFTER updating TARGET_X_, giving v=0.
        v_tx = (self.TARGET_X - self._prev_tx) / dt
        v_ty = (self.TARGET_Y - self._prev_ty) / dt
        v_tz = (self.TARGET_Z - self._prev_tz) / dt
        self._prev_tx = self.TARGET_X
        self._prev_ty = self.TARGET_Y
        self._prev_tz = self.TARGET_Z

        # ── Error in body frame ────────────────────────────────────────────
        yaw    = self.state[5]
        cy, sy = math.cos(yaw), math.sin(yaw)

        ex_w = self.TARGET_X - self.state[0]
        ey_w = self.TARGET_Y - self.state[1]
        ex_b =  cy*ex_w + sy*ey_w   # world → body rotation
        ey_b = -sy*ex_w + cy*ey_w

        vx_b =  cy*self.state[6] + sy*self.state[7]
        vy_b = -sy*self.state[6] + cy*self.state[7]

        # Target velocity in body frame
        v_tx_b =  cy*v_tx + sy*v_ty
        v_ty_b = -sy*v_tx + cy*v_ty

        error = np.array([
            ex_b,
            ey_b,
            self.TARGET_Z  - self.state[2],
            0.0            - self.state[3],   # phi  ref = 0
            0.0            - self.state[4],   # theta ref = 0
            wrap_angle(self.TARGET_YAW - self.state[5]),
            v_tx_b - vx_b,
            v_ty_b - vy_b,
            v_tz   - self.state[8],
            0.0    - self.state[9],
            0.0    - self.state[10],
            0.0    - self.state[11],
        ])

        # ── LQI solve ─────────────────────────────────────────────────────
        u_corr   = self.lqi.compute(error, dt)
        u_opt    = self.u_hover + u_corr
        u_opt[0] = float(np.clip(u_opt[0], 0.0, 4.0*self.K_F*self.OMEGA_MAX**2))

        # ── Motor mixing ──────────────────────────────────────────────────
        omega_sq = np.clip(self.Gamma_inv @ u_opt, 0.0, self.OMEGA_MAX**2)
        omega    = np.sqrt(omega_sq)

        cmd = Actuators(); cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)
        self.control_pub.publish(
            Float64MultiArray(data=[float(u) for u in u_opt]))

        # Diagnostics (~2 Hz)
        if now_sec - self._last_log_t > 0.5:
            pos   = self.state[:3]
            rpy   = np.degrees(self.state[3:6])
            ref   = np.array([self.TARGET_X, self.TARGET_Y, self.TARGET_Z])
            err   = float(np.linalg.norm(pos - ref))
            du    = u_opt - self.u_hover
            integ = self.lqi.integ
            self.get_logger().info(
                f'pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] '
                f'err={err:.3f}m  '
                f'rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]°  '
                f'∫=[{integ[0]:+.3f},{integ[1]:+.3f},'
                f'{integ[2]:+.3f},{integ[3]:+.3f}]  '
                f'dF={du[0]:+.2f}N')
            self._last_log_t = now_sec


def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorLQINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()