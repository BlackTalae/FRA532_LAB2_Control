#!/usr/bin/env python3
"""
trajectory_visualizer.py
========================
Main entry point — spins one ROS 2 node and drives two matplotlib windows:

  Window 1  dashboard_realtime.py   — Live position + error + motors
  Window 2  dashboard_analysis.py   — Cumulative errors + robustness stats

Usage
─────
  # Normal mode (both windows)
  ros2 run <pkg> trajectory_visualizer.py

  # Robustness mode — 5 laps
  ros2 run <pkg> trajectory_visualizer.py \
      --ros-args -p robustness_mode:=true -p num_laps:=5

  # Custom matplotlib backend
  MPLBACKEND=Qt5Agg ros2 run <pkg> trajectory_visualizer.py

Topics subscribed
─────────────────
  /odom                     nav_msgs/Odometry
  /reference_path           nav_msgs/Path
  /motor_commands           actuator_msgs/Actuators
  /trajectory_lap_complete  std_msgs/Bool  (robustness mode only)
"""

import math
import os
import threading
from collections import deque

import numpy as np
import matplotlib
matplotlib.use(os.environ.get('MPLBACKEND', 'TkAgg'))
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from actuator_msgs.msg import Actuators
from std_msgs.msg import Bool, Float64MultiArray

import dashboard_realtime  as rt
import dashboard_analysis  as an

MAX_HIST = 12_000
ANIM_MS  = 100


# ═════════════════════════════════════════════════════════════════════════════
# Callbacks-only ROS 2 node
# ═════════════════════════════════════════════════════════════════════════════

class TrajectoryVisualizer(Node):

    def __init__(self):
        super().__init__('trajectory_visualizer')
        self.declare_parameter('robustness_mode', True)
        self.declare_parameter('num_laps', 5)
        self.robustness_mode = self.get_parameter('robustness_mode').value
        self.num_laps        = int(self.get_parameter('num_laps').value)

        self._lock = threading.Lock()
        H = MAX_HIST

        self._t  = deque(maxlen=H)
        self._ox = deque(maxlen=H); self._oy = deque(maxlen=H); self._oz = deque(maxlen=H)
        self._rx = deque(maxlen=H); self._ry = deque(maxlen=H); self._rz = deque(maxlen=H)
        self._ex = deque(maxlen=H); self._ey = deque(maxlen=H)
        self._ez = deque(maxlen=H); self._et = deque(maxlen=H)
        self._cx = deque(maxlen=H); self._cy = deque(maxlen=H)
        self._cz = deque(maxlen=H); self._ct = deque(maxlen=H)
        self._mt = deque(maxlen=H)
        self._motors = [deque(maxlen=H) for _ in range(4)]

        self._t0      = None; self._last_t  = None; self._cur_ref = None
        self._cx_acc  = self._cy_acc = self._cz_acc = self._ct_acc = 0.0
        self._max_ex  = self._max_ey = self._max_ez = self._max_et = 0.0
        self._laps    = []; self._lap_new = False

        self.create_subscription(Odometry,  '/odom',           self._cb_odom,  10)
        self.create_subscription(Path,      '/reference_path', self._cb_path,   1)
        self.create_subscription(Float64MultiArray, '/control_u', self._cb_motor, 10)
        if self.robustness_mode:
            self.create_subscription(
                Bool, '/trajectory_lap_complete', self._cb_lap, 10)
            self.get_logger().info(
                f'[Viz] Robustness ON — {self.num_laps} laps expected')
        else:
            self.get_logger().info('[Viz] Normal mode')

    # ── /odom ─────────────────────────────────────────────────────────────────
    def _cb_odom(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            if self._t0 is None: self._t0 = now
            t  = now - self._t0
            ox = msg.pose.pose.position.x
            oy = msg.pose.pose.position.y
            oz = msg.pose.pose.position.z
            rx, ry, rz = self._cur_ref if self._cur_ref else (ox, oy, oz)
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
    def _cb_path(self, msg: Path):
        if not msg.poses: return
        p = msg.poses[0].pose.position
        with self._lock: self._cur_ref = (p.x, p.y, p.z)

    # ── /motor_commands ───────────────────────────────────────────────────────
    def _cb_motor(self, msg: Float64MultiArray):
        if len(msg.data) < 4: return
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            if self._t0 is None: return
            self._mt.append(now - self._t0)
            for i in range(4): self._motors[i].append(float(msg.data[i]))

    # ── /trajectory_lap_complete ──────────────────────────────────────────────
    def _cb_lap(self, msg: Bool):
        if not msg.data: return
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
                max_ex=self._max_ex, max_ey=self._max_ey,
                max_ez=self._max_ez, max_et=self._max_et,
                total_cum=self._ct_acc,
            )
            self._laps.append(snap); self._lap_new = True
            for q in (self._t, self._ox, self._oy, self._oz,
                      self._rx, self._ry, self._rz,
                      self._ex, self._ey, self._ez, self._et,
                      self._cx, self._cy, self._cz, self._ct,
                      self._mt, *self._motors):
                q.clear()
            self._cx_acc = self._cy_acc = self._cz_acc = self._ct_acc = 0.0
            self._last_t = None; self._t0 = None
            self._max_ex = self._max_ey = self._max_ez = self._max_et = 0.0
            self.get_logger().info(
                f'[Viz] Lap {lap_num} saved | '
                f'cum={snap["total_cum"]:.3f} m·s | '
                f'{max(0, self.num_laps-lap_num)} remaining')

    # ── thread-safe snapshot ──────────────────────────────────────────────────
    def get_data(self) -> dict:
        with self._lock:
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
            )
            self._lap_new = False
            return d


# ═════════════════════════════════════════════════════════════════════════════
# Entry point — builds both windows, single FuncAnimation drives both
# ═════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizer()

    # ROS spins in a background thread; matplotlib owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Window 1: Live flight monitor ────────────────────────────────────────
    fig_rt, artists_rt,info_rt = rt.build()

    # ── Window 2: Error analysis ─────────────────────────────────────────────
    if node.robustness_mode:
        fig_an, axes_an = an.build_robustness()
        artists_an = None          # robustness figure has no persistent artists
        info_an    = None
    else:
        fig_an, artists_an, info_an = an.build_normal()

    # ── Single animation callback drives both figures ─────────────────────────
    def _anim(frame):
        d = node.get_data()

        # Window 1 — always update
        try:
            rt.update(d, artists_rt,info_rt)
        except Exception as exc:
            node.get_logger().error(f'[Viz/RT] {exc}')

        # Window 2
        try:
            if node.robustness_mode:
                if d['laps']:
                    an.update_robustness(axes_an, d['laps'])
                    fig_an.canvas.draw_idle()
            else:
                an.update_normal(d, artists_an,info_an)
                fig_an.canvas.draw_idle()
        except Exception as exc:
            node.get_logger().error(f'[Viz/AN] {exc}')

        return []

    anim = FuncAnimation(fig_rt, _anim, interval=ANIM_MS,   # noqa: F841
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
