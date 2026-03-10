#!/usr/bin/env python3
"""
dashboard_realtime.py
=====================
Window 1 — Live Position & Control Inputs
──────────────────────────────────────────
Row 1 │ X Position vs Ref  │ Y Position vs Ref  │ Z Position vs Ref
Row 2 │ Control u1 (F)     │ Control u2 (τroll) │ Control u3 (τpitch) │ Control u4 (τyaw)

Control y-axes are set from per-channel u_min / u_max arrays received from
the MPC controller via /range_min and /range_max topics.
Each channel has its own scale (e.g. u1 is 0–60 N, u2-u4 are ±8 N·m).
"""

import os
import numpy as np
import matplotlib
matplotlib.use(os.environ.get('MPLBACKEND', 'TkAgg'))
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BG_FIG   = '#0D1117'; BG_AX    = '#161B22'; BG_AX2   = '#1C2128'
COL_GRID = '#21262D'; COL_SPINE= '#30363D'; COL_TICK = '#8B949E'
COL_TITLE= '#58A6FF'; COL_ODOM = '#00D4FF'; COL_REF  = '#FF8C42'
COL_ERR  = '#FF4757'; COL_INFO = '#FFD700'
COL_MOTORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
# Human-readable channel labels
U_LABELS = ['F_total (N)', 'τ_roll (N·m)', 'τ_pitch (N·m)', 'τ_yaw (N·m)']

ALPHA_FILL = 0.15
WINDOW_S   = 50.0


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
    fills[key] = (ax.fill_between(t, y1, y2 if y2 is not None else 0,
                                  alpha=ALPHA_FILL, color=color)
                  if len(t) > 1 else None)

def _scroll(ax, t):
    if len(t) < 2: return
    t_max = float(t[-1])
    ax.set_xlim(max(float(t[0]), t_max - WINDOW_S), t_max + 0.3)
    ax.relim(); ax.autoscale_view(scalex=False, scaley=True)

def _ylim_for_channel(u_min_arr, u_max_arr, i):
    """Return (y_lo, y_hi) for channel i with 9 % padding."""
    lo = float(u_min_arr[i]); hi = float(u_max_arr[i])
    pad = (hi - lo) * 0.09
    return lo - pad, hi + pad


# ── Build ─────────────────────────────────────────────────────────────────────

def build(u_min: np.ndarray, u_max: np.ndarray,
          fig_title: str = 'Live Position & Control'):
    """
    Parameters
    ----------
    u_min : array-like, shape (4,)
        Lower bound for each control channel [F, τ_roll, τ_pitch, τ_yaw].
    u_max : array-like, shape (4,)
        Upper bound for each control channel.
    """
    u_min = np.asarray(u_min, float)
    u_max = np.asarray(u_max, float)

    plt.style.use('dark_background')
    fig = plt.figure(fig_title, figsize=(32, 16))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle('Quadrotor MPC — Live Position & Control Inputs',
                 color='white', fontsize=16, fontweight='bold', y=0.998)

    # Range annotation in top-right corner
    range_str = '  |  '.join(
        f'u{i+1}: [{u_min[i]:.1f}, {u_max[i]:.1f}]' for i in range(4))
    fig.text(0.99, 0.988, f'Control bounds — {range_str}',
             ha='right', va='top', color=COL_INFO, fontsize=8.5,
             transform=fig.transFigure)

    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[1.15, 0.85],
                              hspace=0.42,
                              left=0.045, right=0.98,
                              top=0.925, bottom=0.060)

    # Row 0 — X / Y / Z position vs reference
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.18)
    ax_xpos = fig.add_subplot(gs0[0])
    ax_ypos = fig.add_subplot(gs0[1])
    ax_zpos = fig.add_subplot(gs0[2])

    # Row 1 — 4 control input channels
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.25)
    ax_m = [fig.add_subplot(gs1[i]) for i in range(4)]

    for ax in [ax_xpos, ax_ypos, ax_zpos] + ax_m:
        _setup_ax(ax)

    _lbl(ax_xpos, 'X Position', ylabel='X (m)')
    _lbl(ax_ypos, 'Y Position', ylabel='Y (m)')
    _lbl(ax_zpos, 'Z Position', ylabel='Z (m)')

    for i, ax in enumerate(ax_m):
        lo, hi = _ylim_for_channel(u_min, u_max, i)
        _lbl(ax, f'Control u{i+1}', ylabel=U_LABELS[i], tc=COL_MOTORS[i])
        # Upper saturation line
        ax.axhline(u_max[i], color='#FF4757', ls='--', lw=1.2, alpha=0.85,
                   label=f'max = {u_max[i]:.2g}')
        # Lower saturation line
        ax.axhline(u_min[i], color='#2ECC71', ls='--', lw=1.0, alpha=0.75,
                   label=f'min = {u_min[i]:.2g}')
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=7.5, facecolor=BG_AX2, edgecolor=COL_SPINE,
                  labelcolor='white', framealpha=0.9)

    # Persistent position artists
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
        (0.005, 0.76, 'Position'),
        (0.005, 0.22, 'Control u'),
    ]:
        fig.text(fx, fy, ftxt, color='#484F58', fontsize=10,
                 fontweight='bold', rotation=90, va='center',
                 transform=fig.transFigure)

    axes_d = dict(ax_xpos=ax_xpos, ax_ypos=ax_ypos, ax_zpos=ax_zpos, ax_m=ax_m)
    artists = dict(
        lx_act=lx_act, lx_ref=lx_ref,
        ly_act=ly_act, ly_ref=ly_ref,
        lz_act=lz_act, lz_ref=lz_ref,
        lm=lm,
        u_min=u_min.copy(),   # per-channel arrays kept in artists
        u_max=u_max.copy(),
        fills={k: None for k in ['pos_x','pos_y','pos_z',
                                  'm0','m1','m2','m3']},
        axes=axes_d,
    )
    return fig, artists, info


# ── Per-frame update ──────────────────────────────────────────────────────────

def update(d: dict, artists: dict, info_text):
    t = d['t']
    if len(t) < 2: return

    ox=d['ox']; oy=d['oy']; oz=d['oz']
    rx=d['rx']; ry=d['ry']; rz=d['rz']
    mt=d['mt']
    fills=artists['fills']; axes=artists['axes']
    u_min=artists['u_min']; u_max=artists['u_max']

    # Position
    artists['lx_act'].set_data(t, ox); artists['lx_ref'].set_data(t, rx)
    artists['ly_act'].set_data(t, oy); artists['ly_ref'].set_data(t, ry)
    artists['lz_act'].set_data(t, oz); artists['lz_ref'].set_data(t, rz)
    _fill(fills, 'pos_x', axes['ax_xpos'], t, ox, rx)
    _fill(fills, 'pos_y', axes['ax_ypos'], t, oy, ry)
    _fill(fills, 'pos_z', axes['ax_zpos'], t, oz, rz)
    for ax in [axes['ax_xpos'], axes['ax_ypos'], axes['ax_zpos']]:
        _scroll(ax, t)

    # Control inputs — clamp y-axis to controller bounds per channel
    nm = len(mt)
    for i, lm in enumerate(artists['lm']):
        md = d['motors'][i]; n_pts = min(nm, len(md))
        if n_pts > 1:
            mts = mt[-n_pts:]; mds = md[-n_pts:]
            lm.set_data(mts, mds)
            _fill(fills, f'm{i}', axes['ax_m'][i], mts, mds, color=COL_MOTORS[i])
            _scroll(axes['ax_m'][i], mts)
            # Re-apply bounds every frame (handles live updates from controller)
            lo, hi = _ylim_for_channel(u_min, u_max, i)
            axes['ax_m'][i].set_ylim(lo, hi)

    info_text.set_text(
        f'  ▶  MAX ERRORS ——'
        f'  ‖e‖ : {d["max_et"]:.4f} m'
        f'  |  X : {d["max_ex"]:.4f} m'
        f'  |  Y : {d["max_ey"]:.4f} m'
        f'  |  Z : {d["max_ez"]:.4f} m  '
    )