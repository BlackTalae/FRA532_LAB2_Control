#!/usr/bin/env python3
"""
trajectory_visualizer.py
========================
Main entry point — spins one ROS 2 node and drives two matplotlib windows.

  Window 1  dashboard_realtime.py  — Live position + control inputs
  Window 2  dashboard_analysis.py  — Error / cumulative / robustness

Control-input plot limits are received live from the MPC controller via:
  /range_min  (Float64MultiArray)  — U_MIN per channel [F, τ_roll, τ_pitch, τ_yaw]
  /range_max  (Float64MultiArray)  — U_MAX per channel

The visualizer waits up to RANGE_WAIT_S seconds for those topics before
falling back to the ROS-parameter defaults (u_min / u_max).

Usage
─────
  ros2 run <pkg> trajectory_visualizer.py
  ros2 run <pkg> trajectory_visualizer.py \\
      --ros-args -p robustness_mode:=true -p num_laps:=5
  # Override fallback limits if controller is not running:
  ros2 run <pkg> trajectory_visualizer.py \\
      --ros-args -p u_min:=0.0 -p u_max:=60.0

Topics subscribed
─────────────────
  /odom                     nav_msgs/Odometry
  /reference_path           nav_msgs/Path
  /control_u                std_msgs/Float64MultiArray
  /range_min                std_msgs/Float64MultiArray   ← from MPC controller
  /range_max                std_msgs/Float64MultiArray   ← from MPC controller
  /trajectory_lap_complete  std_msgs/Bool  (robustness mode only)
"""

import math
import os
import threading
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use(os.environ.get('MPLBACKEND', 'TkAgg'))
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import PoseStamped

import dashboard_realtime as rt
import dashboard_analysis  as an

MAX_HIST     = 12_000
ANIM_MS      = 200
RANGE_WAIT_S = 3.0     # seconds to wait for /range_min|max before using defaults


# ═════════════════════════════════════════════════════════════════════════════
# Callbacks-only ROS 2 node
# ═════════════════════════════════════════════════════════════════════════════

class TrajectoryVisualizer(Node):

    def __init__(self):
        super().__init__('trajectory_visualizer')

        self.declare_parameter('robustness_mode', False)
        self.declare_parameter('num_laps', 1)
        # Fallback limits used only if /range_min|max never arrive
        self.declare_parameter('u_min', 0.0)
        self.declare_parameter('u_max', 1500.0)

        self.declare_parameter('auto_save', True)
        self.declare_parameter('save_dir', 'trajectory_results')

        self.auto_save = bool(self.get_parameter('auto_save').value)
        self.save_dir = str(self.get_parameter('save_dir').value)
        self._final_save_data = None
        self._save_requested = False
        self._save_done = False


        self.robustness_mode = self.get_parameter('robustness_mode').value
        self.num_laps        = int(self.get_parameter('num_laps').value)
        self._u_min_default  = float(self.get_parameter('u_min').value)
        self._u_max_default  = float(self.get_parameter('u_max').value)

        self._lock = threading.Lock()
        H = MAX_HIST

        # Odom / position / error buffers
        self._t  = deque(maxlen=H)
        self._ox = deque(maxlen=H); self._oy = deque(maxlen=H); self._oz = deque(maxlen=H)
        self._rx = deque(maxlen=H); self._ry = deque(maxlen=H); self._rz = deque(maxlen=H)
        self._ex = deque(maxlen=H); self._ey = deque(maxlen=H)
        self._ez = deque(maxlen=H); self._et = deque(maxlen=H)
        self._cx = deque(maxlen=H); self._cy = deque(maxlen=H)
        self._cz = deque(maxlen=H); self._ct = deque(maxlen=H)

        # Control input buffers  (4 channels: F, τ_roll, τ_pitch, τ_yaw)
        self._mt     = deque(maxlen=H)
        self._motors = [deque(maxlen=H) for _ in range(4)]

        # Per-channel control bounds — updated by /range_min|max callbacks
        # Shape: (4,) each — one entry per control channel
        self._u_min = None   # None = not yet received
        self._u_max = None
        self._range_received = threading.Event()

        # Accumulated state
        self._t0      = None; self._last_t  = None; self._cur_ref = None
        self._cx_acc  = self._cy_acc = self._cz_acc = self._ct_acc = 0.0
        self._max_ex  = self._max_ey = self._max_ez = self._max_et = 0.0
        self._laps    = []; self._lap_new = False

        # Subscriptions
        self.create_subscription(Odometry,         '/odom',            self._cb_odom,   10)
        self.create_subscription(PoseStamped,      '/target_pose',  self._cb_target, 10)
        # self.create_subscription(Path,             '/reference_path',  self._cb_path,    1)
        self.create_subscription(Float64MultiArray,'/control_u',       self._cb_motor,  10)
        self.create_subscription(Float64MultiArray,'/range_min',       self._cb_range_min, 1)
        self.create_subscription(Float64MultiArray,'/range_max',       self._cb_range_max, 1)
        self.create_subscription(Bool, '/trajectory_started', self._cb_start, 10)


        self._start_received = threading.Event()
        self._started = False

        self.create_subscription(
            Bool, '/trajectory_lap_complete', self._cb_lap, 10)

        if self.robustness_mode:
            self.get_logger().info(
                f'[Viz] Robustness ON — {self.num_laps} laps expected')
        else:
            self.get_logger().info('[Viz] Normal mode')

        self.get_logger().info(
            f'[Viz] Waiting up to {RANGE_WAIT_S}s for /range_min|max '
            f'(fallback: [{self._u_min_default}, {self._u_max_default}])')

    # ── /range_min and /range_max ─────────────────────────────────────────────
    def _cb_range_min(self, msg: Float64MultiArray):
        with self._lock:
            self._u_min = np.asarray(msg.data, dtype=float)
        if not self._range_received.is_set() and self._u_max is not None:
            self._range_received.set()
            self.get_logger().info(
                f'[Viz] Control bounds received  '
                f'u_min={np.round(self._u_min, 3).tolist()}')

    def _cb_range_max(self, msg: Float64MultiArray):
        with self._lock:
            self._u_max = np.asarray(msg.data, dtype=float)
        if not self._range_received.is_set() and self._u_min is not None:
            self._range_received.set()
            self.get_logger().info(
                f'[Viz] Control bounds received  '
                f'u_max={np.round(self._u_max, 3).tolist()}')

    def get_control_bounds(self):
        """
        Block until bounds arrive (or timeout), then return (u_min, u_max).
        Each is a numpy array of shape (4,) — one limit per control channel.
        Falls back to scalar defaults if controller never responds.
        """
        self._range_received.wait(timeout=RANGE_WAIT_S)
        with self._lock:
            if self._u_min is not None and self._u_max is not None:
                return self._u_min.copy(), self._u_max.copy()
        # Fallback: broadcast scalar to all 4 channels
        self.get_logger().warn(
            f'[Viz] /range_min|max not received after {RANGE_WAIT_S}s — '
            f'using fallback [{self._u_min_default}, {self._u_max_default}]')
        return (np.full(4, self._u_min_default),
                np.full(4, self._u_max_default))

    # ── /odom ─────────────────────────────────────────────────────────────────
    def _cb_odom(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            if self._t0 is None: self._t0 = now
            t  = now - self._t0
            ox = msg.pose.pose.position.x
            oy = msg.pose.pose.position.y
            oz = msg.pose.pose.position.z

            # ox = 0.0
            # oy = 0.0
            # oz = 2.0
            
            rx, ry, rz = self._cur_ref if self._cur_ref else (0.0, 0.0, 2.0)
            ex = ox-rx; ey = oy-ry; ez = oz-rz
            et = math.sqrt(ex*ex + ey*ey + ez*ez)
            self._max_ex = max(self._max_ex, abs(ex))
            self._max_ey = max(self._max_ey, abs(ey))
            self._max_ez = max(self._max_ez, abs(ez))
            self._max_et = max(self._max_et, et)
            if self._last_t is not None:
                dt = max(t - self._last_t, 0.0)
                self._cx_acc += abs(ex)*dt; self._cy_acc += abs(ey)*dt
                self._cz_acc += abs(ez)*dt; self._ct_acc += et*dt
            self._last_t = t
            self._t.append(t)
            self._ox.append(ox); self._oy.append(oy); self._oz.append(oz)
            self._rx.append(rx); self._ry.append(ry); self._rz.append(rz)
            self._ex.append(ex); self._ey.append(ey); self._ez.append(ez)
            self._et.append(et)
            self._cx.append(self._cx_acc); self._cy.append(self._cy_acc)
            self._cz.append(self._cz_acc); self._ct.append(self._ct_acc)

    # ── /reference_path ───────────────────────────────────────────────────────
    # def _cb_path(self, msg: Path):
    #     if not msg.poses: return
    #     p = msg.poses[0].pose.position
    #     with self._lock: self._cur_ref = (p.x, p.y, p.z)

    def _cb_target(self, msg: PoseStamped):
        p = msg.pose.position
        with self._lock:
            self._cur_ref = (p.x, p.y, p.z)
            

    # ── /control_u ────────────────────────────────────────────────────────────
    def _cb_motor(self, msg: Float64MultiArray):
        if len(msg.data) < 4: return
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            if self._t0 is None: return
            self._mt.append(now - self._t0)
            for i in range(4):
                self._motors[i].append(float(msg.data[i]))

    def _cb_start(self, msg: Bool):
        if not msg.data:
            return
        if not self._start_received.is_set():
            self._started = True
            self._start_received.set()
            self.get_logger().info('[Viz] Start signal received from trajectory generator')

    # ── /trajectory_lap_complete ──────────────────────────────────────────────
    def _cb_lap(self, msg: Bool):
        if not msg.data:
            return

        with self._lock:
            lap_num = len(self._laps) + 1
            snap = dict(
                lap=lap_num, t=list(self._t),
                ox=list(self._ox), oy=list(self._oy), oz=list(self._oz),
                rx=list(self._rx), ry=list(self._ry), rz=list(self._rz),
                ex=list(self._ex), ey=list(self._ey),
                ez=list(self._ez), et=list(self._et),
                cx=list(self._cx), cy=list(self._cy),
                cz=list(self._cz), ct=list(self._ct),
                mt=list(self._mt),
                motors=[list(m) for m in self._motors],
                max_ex=self._max_ex, max_ey=self._max_ey,
                max_ez=self._max_ez, max_et=self._max_et,
                total_cum=self._ct_acc,
                u_min=self._u_min.copy() if self._u_min is not None else None,
                u_max=self._u_max.copy() if self._u_max is not None else None,
            )

            self._laps.append(snap)
            self._lap_new = True

            if self.auto_save and lap_num >= self.num_laps:
                self._final_save_data = dict(
                    t=np.asarray(snap['t'], float),
                    ox=np.asarray(snap['ox'], float),
                    oy=np.asarray(snap['oy'], float),
                    oz=np.asarray(snap['oz'], float),
                    rx=np.asarray(snap['rx'], float),
                    ry=np.asarray(snap['ry'], float),
                    rz=np.asarray(snap['rz'], float),
                    ex=np.asarray(snap['ex'], float),
                    ey=np.asarray(snap['ey'], float),
                    ez=np.asarray(snap['ez'], float),
                    et=np.asarray(snap['et'], float),
                    cx=np.asarray(snap['cx'], float),
                    cy=np.asarray(snap['cy'], float),
                    cz=np.asarray(snap['cz'], float),
                    ct=np.asarray(snap['ct'], float),
                    mt=np.asarray(snap['mt'], float),
                    motors=[np.asarray(m, float) for m in snap['motors']],
                    max_ex=snap['max_ex'],
                    max_ey=snap['max_ey'],
                    max_ez=snap['max_ez'],
                    max_et=snap['max_et'],
                    laps=list(self._laps),
                    lap_new=True,
                    u_min=snap['u_min'],
                    u_max=snap['u_max'],
                )
                self._save_requested = True
                self.get_logger().info('[Viz] Final lap reached — frozen snapshot prepared for save')

            for q in (
                self._t, self._ox, self._oy, self._oz,
                self._rx, self._ry, self._rz,
                self._ex, self._ey, self._ez, self._et,
                self._cx, self._cy, self._cz, self._ct,
                self._mt, *self._motors
            ):
                q.clear()

            self._cx_acc = self._cy_acc = self._cz_acc = self._ct_acc = 0.0
            self._last_t = None
            self._t0 = None
            self._max_ex = self._max_ey = self._max_ez = self._max_et = 0.0

            self.get_logger().info(
                f'[Viz] Lap {lap_num} saved | cum={snap["total_cum"]:.3f} m·s | '
                f'{max(0, self.num_laps-lap_num)} remaining'
            )

    # ── Thread-safe snapshot ──────────────────────────────────────────────────
    def get_data(self) -> dict:
        with self._lock:
            u_min = self._u_min.copy() if self._u_min is not None else None
            u_max = self._u_max.copy() if self._u_max is not None else None
            d = dict(
                t =np.asarray(self._t,  float),
                ox=np.asarray(self._ox, float), oy=np.asarray(self._oy, float),
                oz=np.asarray(self._oz, float),
                rx=np.asarray(self._rx, float), ry=np.asarray(self._ry, float),
                rz=np.asarray(self._rz, float),
                ex=np.asarray(self._ex, float), ey=np.asarray(self._ey, float),
                ez=np.asarray(self._ez, float), et=np.asarray(self._et, float),
                cx=np.asarray(self._cx, float), cy=np.asarray(self._cy, float),
                cz=np.asarray(self._cz, float), ct=np.asarray(self._ct, float),
                mt=np.asarray(self._mt, float),
                motors=[np.asarray(m, float) for m in self._motors],
                max_ex=self._max_ex, max_ey=self._max_ey,
                max_ez=self._max_ez, max_et=self._max_et,
                laps=list(self._laps), lap_new=self._lap_new,
                u_min=u_min, u_max=u_max,   # per-channel arrays or None
            )
            self._lap_new = False
            return d

    def wait_for_start_signal(self, timeout=None):
        """
        Wait until /trajectory_started is received.
        timeout=None means wait forever.
        """
        self.get_logger().info('[Viz] Waiting for /trajectory_started before opening plots...')
        ok = self._start_received.wait(timeout=timeout)
        if ok:
            self.get_logger().info('[Viz] Start signal received — opening plots')
        else:
            self.get_logger().warn('[Viz] Start signal wait timed out')
        return ok
    
    def request_save(self):
        with self._lock:
            if self.auto_save and not self._save_done:
                self._save_requested = True

    def consume_save_request(self):
        with self._lock:
            if self._save_requested and not self._save_done:
                self._save_requested = False
                self._save_done = True
                return True
            return False
        
    def consume_final_save_data(self):
        with self._lock:
            data = self._final_save_data
            self._final_save_data = None
            return data
    
    
# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizer()

    # ROS spins in background; matplotlib owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()



    # Wait for control bounds (blocks up to RANGE_WAIT_S)
    u_min, u_max = node.get_control_bounds()
    node.get_logger().info(
        f'[Viz] Building plots with '
        f'u_min={np.round(u_min,2).tolist()}  '
        f'u_max={np.round(u_max,2).tolist()}')

    # Build both windows
    fig_rt, artists_rt, info_rt = rt.build(u_min=u_min, u_max=u_max)


    if node.robustness_mode:
        fig_an, axes_an, info_an = an.build_robustness()
        artists_an = None; info_an = None
    else:
        fig_an, artists_an, info_an = an.build_normal()

    def _anim(frame):
        d = node.get_data()

        if d['u_min'] is not None:
            artists_rt['u_min'] = d['u_min']
            artists_rt['u_max'] = d['u_max']

        try:
            rt.update(d, artists_rt, info_rt)
        except Exception as exc:
            node.get_logger().error(f'[Viz/RT] {exc}')

        try:
            if node.robustness_mode:
                if d['laps']:
                    an.update_robustness(axes_an, d['laps'], info_an)
                    fig_an.canvas.draw_idle()
            else:
                an.update_normal(d, artists_an, info_an)
                fig_an.canvas.draw_idle()
        except Exception as exc:
            node.get_logger().error(f'[Viz/AN] {exc}')

        if node.consume_save_request():
            try:
                frozen = node.consume_final_save_data()

                if frozen is not None:
                    if frozen['u_min'] is not None:
                        artists_rt['u_min'] = frozen['u_min']
                        artists_rt['u_max'] = frozen['u_max']

                    rt.update(frozen, artists_rt, info_rt)

                    if node.robustness_mode:
                        an.update_robustness(axes_an, frozen['laps'])
                    else:
                        an.update_normal(frozen, artists_an, info_an)

                fig_rt.canvas.draw()
                fig_an.canvas.draw()
                _save_figures()
            except Exception as exc:
                node.get_logger().error(f'[Viz/SAVE] {exc}')

        return []
    
    def _save_figures():
        os.makedirs(node.save_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        mode = 'robustness' if node.robustness_mode else 'normal'

        path_rt = os.path.join(node.save_dir, f'{timestamp}_{mode}_realtime.png')
        path_an = os.path.join(node.save_dir, f'{timestamp}_{mode}_analysis.png')

        fig_rt.savefig(path_rt, dpi=200, bbox_inches='tight')
        fig_an.savefig(path_an, dpi=200, bbox_inches='tight')

        node.get_logger().info(f'[Viz] Saved realtime window  -> {path_rt}')
        node.get_logger().info(f'[Viz] Saved analysis window -> {path_an}')
        


    anim = FuncAnimation(fig_rt, _anim, interval=ANIM_MS,  # noqa: F841
                         blit=False, cache_frame_data=False)

    node.get_logger().info('[Viz] Both windows open — close either to quit.')
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()