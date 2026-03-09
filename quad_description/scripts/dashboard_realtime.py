#!/usr/bin/env python3
"""
dashboard_realtime.py
=====================
Window 1 — Live Flight Monitor
───────────────────────────────
Row 1 │ X Position vs Ref  │ Y Position vs Ref  │ Z Position vs Ref
Row 2 │ Total Error ‖e‖    │ X Error            │ Y Error            │ Z Error
Row 3 │ Motor 1 ω          │ Motor 2 ω          │ Motor 3 ω          │ Motor 4 ω

Gold info-bar: running max errors.

Imports the callbacks-only TrajectoryVisualizer node from trajectory_visualizer.py.
Run alongside dashboard_analysis.py from the same process via main() here.
"""

import os, threading
import numpy as np
import matplotlib
matplotlib.use(os.environ.get('MPLBACKEND', 'TkAgg'))
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ── Theme ─────────────────────────────────────────────────────────────────────
BG_FIG   = '#0D1117'; BG_AX    = '#161B22'; BG_AX2   = '#1C2128'
COL_GRID = '#21262D'; COL_SPINE= '#30363D'; COL_TICK = '#8B949E'
COL_TITLE= '#58A6FF'; COL_ODOM = '#00D4FF'; COL_REF  = '#FF8C42'
COL_ERR  = '#FF4757'; COL_INFO = '#FFD700'
COL_MOTORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

ALPHA_FILL = 0.15
ANIM_MS    = 33
WINDOW_S   = 30.0
OMEGA_MAX  = 1500.0


# ── Styling helpers ───────────────────────────────────────────────────────────

def _setup_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COL_TICK, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(COL_SPINE); sp.set_linewidth(0.6)
    ax.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.9)

def _lbl(ax, title, xlabel='t (s)', ylabel='', tc=COL_TITLE):
    ax.set_title(title,   color=tc,       fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel(xlabel, color=COL_TICK, fontsize=9,  labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, color=COL_TICK, fontsize=9, labelpad=3)

def _leg(ax):
    ax.legend(fontsize=8, facecolor=BG_AX2, edgecolor=COL_SPINE,
              labelcolor='white', framealpha=0.9, handlelength=1.5)

def _fill(fills, key, ax, t, y1, y2=None, color=COL_ERR):
    old = fills.get(key)
    if old is not None:
        try: old.remove()
        except ValueError: pass
    if len(t) > 1:
        fills[key] = ax.fill_between(
            t, y1, y2 if y2 is not None else 0, alpha=ALPHA_FILL, color=color)
    else:
        fills[key] = None

def _scroll(ax, t):
    if len(t) < 2: return
    t_max = float(t[-1])
    ax.set_xlim(max(float(t[0]), t_max - WINDOW_S), t_max + 0.3)
    ax.relim(); ax.autoscale_view(scalex=False, scaley=True)


# ── Build figure ──────────────────────────────────────────────────────────────

def build(fig_title='Live Flight Monitor'):
    plt.style.use('dark_background')
    fig = plt.figure(fig_title, figsize=(32, 18))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle('Quadrotor Controller — Live Flight Monitor',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[1.0, 0.85],
                              hspace=0.30,
                              left=0.045, right=0.98,
                              top=0.925,  bottom=0.075)

    # Row 0 — position
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.18)
    ax_xpos = fig.add_subplot(gs0[0])
    ax_ypos = fig.add_subplot(gs0[1])
    ax_zpos = fig.add_subplot(gs0[2])

    # Row 2 — motors
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.22)
    ax_m = [fig.add_subplot(gs2[i]) for i in range(4)]

    # Style + decorate (once)
    for ax in [ax_xpos, ax_ypos, ax_zpos,] + ax_m:
        _setup_ax(ax)

    _lbl(ax_xpos, 'X Position',      ylabel='X (m)')
    _lbl(ax_ypos, 'Y Position',      ylabel='Y (m)')
    _lbl(ax_zpos, 'Z Position',      ylabel='Z (m)')

    for i, ax in enumerate(ax_m):
        _lbl(ax, f'Motor {i+1}  ω', ylabel='ω (rad/s)', tc=COL_MOTORS[i])
        ax.axhline(OMEGA_MAX, color='#FF4757', ls='--', lw=1.1, alpha=0.85,
                   label=f'Limit {OMEGA_MAX:.0f} rad/s')
        ax.axhline(0.0, color='#2ECC71', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylim(-40, OMEGA_MAX * 1.09)
        ax.legend(fontsize=7.5, facecolor=BG_AX2, edgecolor=COL_SPINE,
                  labelcolor='white', framealpha=0.9)

    # Persistent artists
    lx_act, = ax_xpos.plot([], [], color=COL_ODOM, lw=2.0, alpha=0.95, label='Actual')
    lx_ref, = ax_xpos.plot([], [], color=COL_REF,  lw=1.4, alpha=0.85, ls='--', label='Reference')
    ly_act, = ax_ypos.plot([], [], color=COL_ODOM, lw=2.0, alpha=0.95, label='Actual')
    ly_ref, = ax_ypos.plot([], [], color=COL_REF,  lw=1.4, alpha=0.85, ls='--', label='Reference')
    lz_act, = ax_zpos.plot([], [], color=COL_ODOM, lw=2.0, alpha=0.95, label='Actual')
    lz_ref, = ax_zpos.plot([], [], color=COL_REF,  lw=1.4, alpha=0.85, ls='--', label='Reference')
    for ax in [ax_xpos, ax_ypos, ax_zpos]: _leg(ax)


    lm = [ax_m[i].plot([], [], color=COL_MOTORS[i], lw=1.8, alpha=0.93)[0]
          for i in range(4)]

    info = fig.text(
        0.5, 0.004, 'Waiting for data…',
        ha='center', va='bottom', color=COL_INFO, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.55', facecolor=BG_AX,
                  edgecolor=COL_INFO, linewidth=1.5, alpha=0.95))

    for fx, fy, ftxt in [
        (0.005, 0.84, 'Position'),
        (0.005, 0.54, 'Error'),
        (0.005, 0.16, 'Motors'),
    ]:
        fig.text(fx, fy, ftxt, color='#484F58', fontsize=10,
                 fontweight='bold', rotation=90, va='center',
                 transform=fig.transFigure)

    axes_d = dict(ax_xpos=ax_xpos, ax_ypos=ax_ypos, ax_zpos=ax_zpos,
                  ax_m=ax_m)
    artists = dict(
        lx_act=lx_act, lx_ref=lx_ref,
        ly_act=ly_act, ly_ref=ly_ref,
        lz_act=lz_act, lz_ref=lz_ref,
        lm=lm,
        fills={k: None for k in [
            'pos_x','pos_y','pos_z',
            'err_t','err_x','err_y','err_z',
            'm0','m1','m2','m3',
        ]},
        axes=axes_d,
    )
    return fig, artists, info


# ── Per-frame update ──────────────────────────────────────────────────────────

def update(d: dict, artists: dict, info_text):
    t = d['t']
    if len(t) < 2: return

    ox=d['ox']; oy=d['oy']; oz=d['oz']
    rx=d['rx']; ry=d['ry']; rz=d['rz']
    ex=d['ex']; ey=d['ey']; ez=d['ez']; et=d['et']
    mt=d['mt']
    fills=artists['fills']; axes=artists['axes']
    aex=np.abs(ex); aey=np.abs(ey); aez=np.abs(ez)

    # Position
    artists['lx_act'].set_data(t, ox); artists['lx_ref'].set_data(t, rx)
    artists['ly_act'].set_data(t, oy); artists['ly_ref'].set_data(t, ry)
    artists['lz_act'].set_data(t, oz); artists['lz_ref'].set_data(t, rz)
    _fill(fills,'pos_x', axes['ax_xpos'], t, ox, rx)
    _fill(fills,'pos_y', axes['ax_ypos'], t, oy, ry)
    _fill(fills,'pos_z', axes['ax_zpos'], t, oz, rz)
    for ax in [axes['ax_xpos'], axes['ax_ypos'], axes['ax_zpos']]:
        _scroll(ax, t)

    # Motors
    nm = len(mt)
    for i, lm in enumerate(artists['lm']):
        md = d['motors'][i]; n_pts = min(nm, len(md))
        if n_pts > 1:
            mts = mt[-n_pts:]; mds = md[-n_pts:]
            lm.set_data(mts, mds)
            _fill(fills, f'm{i}', axes['ax_m'][i], mts, mds, color=COL_MOTORS[i])
            _scroll(axes['ax_m'][i], mts)
            axes['ax_m'][i].set_ylim(-40, OMEGA_MAX * 1.09)

    info_text.set_text(
        f'  ▶  MAX ERRORS ——'
        f'  ‖e‖ : {d["max_et"]:.4f} m'
        f'  |  X : {d["max_ex"]:.4f} m'
        f'  |  Y : {d["max_ey"]:.4f} m'
        f'  |  Z : {d["max_ez"]:.4f} m  '
    )
