#!/usr/bin/env python3
"""
Quadrotor Cascaded MPC Node — Multi-rate  (ROS 2)
==================================================
Each control layer runs at its own frequency via separate ROS timers:

    Layer               Timer rate    Period
    ─────────────────   ──────────    ──────
    Altitude MPC        50  Hz        20 ms
    Position MPC        50  Hz        20 ms   (same outer-loop timer)
    Attitude MPC        100 Hz        10 ms   (inner loop, faster)
    Motor publish       100 Hz        10 ms   (always uses latest signals)

Cascade block diagram (from reference image):

          x,y,z,phi,the,psi  (feedback to ALL layers)
    ┌──────────────────────────────────────────────────────┐
    │                                                      ▼
  zd  ──► [ Altitude MPC ] ─── U1 ──────────────────► [ Quadrotor ]
  50Hz                                                      ▲      │
  xd,yd ► [ Position MPC ] ─── phid, thed ──►             │      │ U2,U3,U4
  50Hz              │                                       │      │
  psid ─────────────►[ Attitude MPC ]─────────────────────-┘      │
                    100Hz          ◄──── dphi, dthe, dpsi ─────────┘
                         x,y,z,phi,the,psi (feedback)

Signal naming matches the diagram exactly:
  U1          = total thrust [N]          ← Altitude MPC
  phid, thed  = desired roll, pitch [rad] ← Position MPC  → Attitude MPC ref
  U2, U3, U4  = τ_φ, τ_θ, τ_ψ [N·m]     ← Attitude MPC

Models (ZOH-discretised, hover-trim linearisation)
──────────────────────────────────────────────────
Altitude   [z, vz]         → [ΔT]           Ac=[[0,1],[0,0]]   Bc=[[0],[1/m]]
Position   [x,y,vx,vy]     → [φd,θd]        gravity-coupling model
Attitude   [φ,θ,ψ,φ̇,θ̇,ψ̇] → [τφ,τθ,τψ]  integrator model

Subscriptions : /odom (nav_msgs/Odometry), /target_pose (geometry_msgs/PoseStamped)
Publications  : /motor_commands (actuator_msgs/Actuators)
                /mpc_u          (std_msgs/Float64MultiArray)  [U1,U2,U3,U4]
"""

from __future__ import annotations
import math
import numpy as np
from scipy.linalg import expm

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

from utils.MPCken2 import MPC_Opti   # ← UNCHANGED


# ──────────────────────────────────────────────────────────────────────────────
# ZOH discretisation helper
# ──────────────────────────────────────────────────────────────────────────────

def c2d(Ac: np.ndarray, Bc: np.ndarray, dt: float):
    """Exact Zero-Order Hold discretisation of (Ac, Bc) with step dt."""
    n, m = Ac.shape[0], Bc.shape[1]
    Z = np.zeros((n + m, n + m))
    Z[:n, :n] = Ac
    Z[:n, n:] = Bc
    Zd = expm(Z * dt)
    return Zd[:n, :n], Zd[:n, n:]


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

class CascadedMPCNode(Node):

    # ── Physical constants ─────────────────────────────────────────────────
    MASS      = 1.5          # kg
    GRAVITY   = 9.81         # m/s²
    K_F       = 8.54858e-06  # N/(rad/s)²
    K_M       = 0.06         # N·m/(rad/s)²
    OMEGA_MAX = 1000.0        # rad/s
    OMEGA_MIN = 500.0        # rad/s
    IXX       = 0.0347563    # kg·m²
    IYY       = 0.07         # kg·m²
    IZZ       = 0.0977       # kg·m²
    L         = 0.22         # m

    # ── MPC prediction horizons & time steps ──────────────────────────────
    ALT_DT = 0.05;  ALT_N = 20   # altitude   prediction step / horizon
    POS_DT = 0.05;  POS_N = 20   # position   prediction step / horizon
    ATT_DT = 0.02;  ATT_N = 20   # attitude   prediction step / horizon
    
    ALT_Nc = 20
    POS_Nc = 20
    ATT_Nc = 20
    # ── Control loop (timer) periods ──────────────────────────────────────
    #   Outer loop : Altitude + Position  @ 50 Hz  → dt = 0.02 s
    #   Inner loop : Attitude + Publish   @ 100 Hz → dt = 0.01 s
    OUTER_DT = 0.02   # s  — 50 Hz  (altitude & position)
    INNER_DT = 0.01   # s  — 100 Hz (attitude & motor publish)

    # ── Default setpoint ───────────────────────────────────────────────────
    DEFAULT_TARGET_X   = 0.0
    DEFAULT_TARGET_Y   = 0.0
    DEFAULT_TARGET_Z   = 5.0
    DEFAULT_TARGET_YAW = 0.0

    def __init__(self):
        super().__init__('quadrotor_cascaded_mpc_node')

        # ── Physical limits ─────────────────────────────────────────────────
        self.F_hover = self.MASS * self.GRAVITY
        self.F_max   = 4.0 * self.K_F * self.OMEGA_MAX ** 2
        self.F_min   = 4.0 * self.K_F * self.OMEGA_MIN ** 2
        print("Fmax : " , self.F_max)
        kF, kM, L    = self.K_F, self.K_M, self.L
        self.tau_max = kF * L * self.OMEGA_MAX ** 2
        # ── Motor allocation Γ ──────────────────────────────────────────────
        self.Gamma = np.array([
            [ kF,        kF,        kF,        kF       ],
            [-kF * L,    kF * 0.2,  kF * L,   -kF * 0.2],
            [-kF * L,    kF * 0.2, -kF * L,    kF * 0.2],
            [-kM,       -kM,        kM,         kM      ],
        ])
        self.Gamma_inv = np.linalg.inv(self.Gamma)

        # ── Build all three MPC_Opti controllers ────────────────────────────
        self._build_altitude_mpc()
        self._build_position_mpc()
        self._build_attitude_mpc()

        # ── Targets ─────────────────────────────────────────────────────────
        self.TARGET_X   = self.DEFAULT_TARGET_X
        self.TARGET_Y   = self.DEFAULT_TARGET_Y
        self.TARGET_Z   = self.DEFAULT_TARGET_Z
        self.TARGET_YAW = self.DEFAULT_TARGET_YAW

        # ── Odometry state ───────────────────────────────────────────────────
        self.pos_x = 0.0;  self.pos_y = 0.0;  self.pos_z = 0.0
        self.vel_x = 0.0;  self.vel_y = 0.0;  self.vel_z = 0.0
        self.roll  = 0.0;  self.pitch = 0.0;  self.yaw   = 0.0
        self.ang_vx = 0.0; self.ang_vy = 0.0; self.ang_vz = 0.0
        self._odom_received = False

        # ── Inter-layer signals (shared between timers, written by outer,
        #    read by inner) — initialised at hover trim
        self._U1   = self.F_hover   # Altitude MPC → motor mix
        self._phid = 0.0            # Position MPC → Attitude MPC ref (φ_d)
        self._thed = 0.0            # Position MPC → Attitude MPC ref (θ_d)
        self._U2   = 0.0            # Attitude MPC → motor mix (τ_φ)
        self._U3   = 0.0            # Attitude MPC → motor mix (τ_θ)
        self._U4   = 0.0            # Attitude MPC → motor mix (τ_ψ)

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.odom_sub   = self.create_subscription(
            Odometry,    '/odom',        self._odom_cb,   10)
        self.target_sub = self.create_subscription(
            PoseStamped, '/target_pose', self._target_cb, 10)
        self.cmd_pub = self.create_publisher(Actuators,         '/motor_commands', 10)
        self.u_pub   = self.create_publisher(Float64MultiArray, '/mpc_u',          10)

        # ── Two separate timers ──────────────────────────────────────────────
        #   outer_loop → Altitude MPC + Position MPC  (50 Hz)
        #   inner_loop → Attitude MPC + motor publish (100 Hz)
        self.create_timer(self.OUTER_DT, self._outer_loop)   # 50 Hz
        self.create_timer(self.INNER_DT, self._inner_loop)   # 100 Hz

        self.get_logger().info(
            f'Cascaded MPC ready — '
            f'outer(Alt+Pos)={1/self.OUTER_DT:.0f}Hz  '
            f'inner(Att+Pub)={1/self.INNER_DT:.0f}Hz'
        )

    # ──────────────────────────────────────────────────────────────────────
    # Controller builders
    # ──────────────────────────────────────────────────────────────────────

    def _build_altitude_mpc(self):
        """
        Altitude MPC
        ─────────────
        state  : [z, vz]     (n=2)
        input  : [ΔT]        (m=1)   delta-thrust from hover trim
        output : U1 = F_hover + ΔT
        """
        m = self.MASS
        Ac = np.array([[0., 1.],
                        [0., 0.]])
        Bc = np.array([[0.  ],
                        [1./m]])
        Ad, Bd = c2d(Ac, Bc, self.ALT_DT)

        Q  = np.diag([100.0, 20.0])
        R  = np.diag([1.0])
        Qf = 5.0 * Q

        dT_min = self.F_min - self.F_hover
        dT_max = self.F_max - self.F_hover

        # dT_min = -20.0
        # dT_max = 20.0

        self.alt_mpc = MPC_Opti(
            Ad=Ad, Bd=Bd, Q=Q, R=R, Qf=Qf,
            N=self.ALT_N,
            u_min=np.array([dT_min]),
            u_max=np.array([dT_max]),
            # state_idx=[1], # Constrain vel
            # state_lower_bound=np.array([-5.0]),
            # state_upper_bound=np.array([ 5.0]),
            constrain_initial_state=False,
            constrain_terminal_state=True,
        )
        self.alt_mpc.build()
        self.get_logger().info(f'AltitudeMPC  N={self.ALT_N} dt={self.ALT_DT}s  @ 50Hz timer')


    def _build_attitude_mpc(self):
        """
        Attitude MPC
        ─────────────
        state  : [φ, θ, ψ, φ̇, θ̇, ψ̇]   (n=6)
        input  : [τ_φ, τ_θ, τ_ψ]         (m=3)
        ref φ,θ : phid, thed from Position MPC
        ref ψ   : TARGET_YAW from user
        feedback: dphi, dthe, dpsi from odometry
        output : U2, U3, U4
        """
        Ix, Iy, Iz, la = self.IXX, self.IYY, self.IZZ, self.L
        Ac = np.zeros((6, 6))
        Ac[0, 3] = 1.0
        Ac[1, 4] = 1.0
        Ac[2, 5] = 1.0

        Bc = np.zeros((6, 3))
        Bc[3, 0] = la / Ix
        Bc[4, 1] = la / Iy
        Bc[5, 2] = 1.0 / Iz

        Ad, Bd = c2d(Ac, Bc, self.ATT_DT)

        Q  = np.diag([30.0, 30.0, 1.0, # r p y
                      5.0,  5.0,  1.0]) # rd pd yd
        R  = np.diag([5.0, 5.0, 1.0])
        Qf = 2.0 * Q

        rate_lim = math.pi / 2

        tau_limit = 1.0
        lim = 0.25
        self.att_mpc = MPC_Opti(
            Ad=Ad, Bd=Bd, Q=Q, R=R, Qf=Qf,
            N=self.ATT_N,
            u_min=np.array([-tau_limit, -tau_limit , -tau_limit]),
            u_max=np.array([ tau_limit,  tau_limit ,  tau_limit]),
            state_idx=[0, 1],
            state_lower_bound=np.array([-lim, -lim]),
            state_upper_bound=np.array([ lim,  lim]),
            constrain_initial_state=True,
            constrain_terminal_state=True,
        )
        self.att_mpc.build()
        self.get_logger().info(f'AttitudeMPC  N={self.ATT_N} dt={self.ATT_DT}s  @ 100Hz timer')

    def _build_position_mpc(self):
        """
        Position MPC
        ─────────────
        state  : [x, y, vx, vy]   (n=4)
        input  : [φ_d, θ_d]       (m=2)
        output : phid, thed → Attitude MPC reference angles
        """
        g = self.GRAVITY
        Ac = np.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ])
        Bc = np.array([
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  g ],   # v̇x =  g·θ_d
            [-g,   0.],   # v̇y = -g·φ_d
        ])
        Ad, Bd = c2d(Ac, Bc, self.POS_DT)

        Q  = np.diag([0.0, 0.0, 0.0, 0.0])
        # Q  = np.diag([60.0, 60.0, 3.0, 3.0])

        R  = np.diag([0.0, 0.0])
        Qf = 5.0 * Q

        phi_lim = math.radians(25.0)
        the_lim = math.radians(25.0)

        self.pos_mpc = MPC_Opti(
            Ad=Ad, Bd=Bd, Q=Q, R=R, Qf=Qf,
            N=self.POS_N,
            u_min=np.array([-phi_lim, -the_lim]),
            u_max=np.array([ phi_lim,  the_lim]),
            state_idx=[2, 3],
            state_lower_bound=np.array([-5.0, -5.0]),
            state_upper_bound=np.array([ 5.0,  5.0]),
            constrain_initial_state=False,
            constrain_terminal_state=True,
        )
        self.pos_mpc.build()
        self.get_logger().info(f'PositionMPC  N={self.POS_N} dt={self.POS_DT}s  @ 50Hz timer')

    # ──────────────────────────────────────────────────────────────────────
    # ROS callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        q = msg.pose.pose.orientation

        self.pos_x, self.pos_y, self.pos_z = p.x, p.y, p.z
        self.vel_x, self.vel_y, self.vel_z = v.x, v.y, v.z
        self.ang_vx, self.ang_vy, self.ang_vz = w.x, w.y, w.z

        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x ** 2 + q.y ** 2)
        self.roll = math.atan2(sinr, cosr)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self.pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.yaw = math.atan2(siny, cosy)

        self._odom_received = True

    def _target_cb(self, msg: PoseStamped):
        self.TARGET_X = msg.pose.position.x
        self.TARGET_Y = msg.pose.position.y
        self.TARGET_Z = msg.pose.position.z

        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.TARGET_YAW = math.atan2(siny, cosy)

        self.get_logger().info(
            f'New target: ({self.TARGET_X:.2f}, {self.TARGET_Y:.2f}, '
            f'{self.TARGET_Z:.2f}) m  ψ={math.degrees(self.TARGET_YAW):.1f}°'
        )

    # ──────────────────────────────────────────────────────────────────────
    # Reference builders — MPC_Opti expects (n, N+1) and (m, N)
    # ──────────────────────────────────────────────────────────────────────

    def _alt_ref(self):
        xref = np.tile([[self.TARGET_Z], [0.0]], (1, self.ALT_N + 1))  # (2, N+1)
        uref = np.zeros((1, self.ALT_N))                                 # (1, N)
        return xref, uref

    def _pos_ref(self):
        xref = np.tile(
            [[self.TARGET_X], [self.TARGET_Y], [0.0], [0.0]],
            (1, self.POS_N + 1)
        )  # (4, N+1)
        uref = np.zeros((2, self.POS_N))                                 # (2, N)
        return xref, uref

    def _att_ref(self, phid: float, thed: float):
        yaw_err = self.TARGET_YAW - self.yaw
        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi
        psid    = self.yaw + yaw_err

        xref = np.tile(
            [[phid], [thed], [psid], [0.0], [0.0], [0.0]],
            (1, self.ATT_N + 1)
        )  # (6, N+1)
        uref = np.zeros((3, self.ATT_N))                                 # (3, N)
        return xref, uref

    # ──────────────────────────────────────────────────────────────────────
    # OUTER LOOP — 50 Hz
    #   Runs: Altitude MPC + Position MPC
    #   Writes: self._U1, self._phid, self._thed
    # ──────────────────────────────────────────────────────────────────────

    def _outer_loop(self):
        if not self._odom_received:
            return

        # ── Altitude MPC  (zd → U1) ─────────────────────────────────────────
        alt_x0           = np.array([self.pos_z, self.vel_z])
        alt_xref, alt_uref = self._alt_ref()

        try:
            u_alt, _, _ = self.alt_mpc.solve(
                x0_val=alt_x0, xref_val=alt_xref, uref_val=alt_uref,
                allow_debug_fallback=True,
            )
            delta_T = float(u_alt[0])
        except Exception as exc:
            self.get_logger().warn(f'AltMPC failed: {exc}')
            delta_T = 0.0

        self._U1 = float(np.clip(self.F_hover + delta_T, self.F_min, self.F_max))

        # ── Position MPC  (xd, yd → phid, thed) ────────────────────────────
        pos_x0           = np.array([self.pos_x, self.pos_y,
                                      self.vel_x, self.vel_y])
        pos_xref, pos_uref = self._pos_ref()

        try:
            u_pos, _, _ = self.pos_mpc.solve(
                x0_val=pos_x0, xref_val=pos_xref, uref_val=pos_uref,
                allow_debug_fallback=True,
            )
            self._phid = float(u_pos[0])
            self._thed = float(u_pos[1])
        except Exception as exc:
            self.get_logger().warn(f'PosMPC failed: {exc}')
            self._phid = 0.0
            self._thed = 0.0

        self.get_logger().debug(
            f'[OUTER 50Hz] U1={self._U1:.2f}N  '
            f'phid={math.degrees(self._phid):.1f}°  '
            f'thed={math.degrees(self._thed):.1f}°'
        )

    # ──────────────────────────────────────────────────────────────────────
    # INNER LOOP — 100 Hz
    #   Runs: Attitude MPC + motor mixing + publish
    #   Reads:  self._U1, self._phid, self._thed  (latest from outer loop)
    #   Writes: self._U2, self._U3, self._U4
    # ──────────────────────────────────────────────────────────────────────

    def _inner_loop(self):
        if not self._odom_received:
            return

        # ── Attitude MPC  (phid, thed, psid → U2, U3, U4) ──────────────────
        att_x0           = np.array([self.roll,   self.pitch,  self.yaw,
                                      self.ang_vx, self.ang_vy, self.ang_vz])
        att_xref, att_uref = self._att_ref(self._phid, self._thed)

        try:
            u_att, _, _ = self.att_mpc.solve(
                x0_val=att_x0, xref_val=att_xref, uref_val=att_uref,
                allow_debug_fallback=True,
            )
            self._U2 = float(u_att[0])   # τ_φ
            self._U3 = float(u_att[1])   # τ_θ
            self._U4 = float(u_att[2])   # τ_ψ
        except Exception as exc:
            self.get_logger().warn(f'AttMPC failed: {exc}')
            self._U2 = self._U3 = self._U4 = 0.0

        # ── Motor mixing  [U1, U2, U3, U4] → [ω₀, ω₁, ω₂, ω₃] ─────────────
        wrench   = np.array([self._U1, self._U2, self._U3, self._U4], dtype=float)
        omega_sq = self.Gamma_inv @ wrench
        omega_sq = np.clip(omega_sq, self.OMEGA_MIN ** 2, self.OMEGA_MAX ** 2)
        omega    = np.sqrt(omega_sq)


        # ── Publish motor commands ────────────────────────────────────────────
        cmd = Actuators()
        cmd.velocity = [float(w) for w in omega]
        self.cmd_pub.publish(cmd)

        # ── Publish debug: [U1, U2, U3, U4] ──────────────────────────────────
        u_msg = Float64MultiArray()
        u_msg.data = [self._U1, self._U2, self._U3, self._U4]
        self.u_pub.publish(u_msg)

        # self.get_logger().debug(
        #     f'[INNER 100Hz] '
        #     f'U1={self._U1:.2f}N  '
        #     f'U2={self._U2:.4f}  U3={self._U3:.4f}  U4={self._U4:.4f}'
        # )


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CascadedMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()