#!/usr/bin/env python3
"""
dashboard_analysis.py
=====================
Window 2 — Error Analysis & Robustness
──────────────────────────────────────
Normal mode (single run)
  Row 1 │ Cumulative Total  │ Cumulative X   │ Cumulative Y   │ Cumulative Z
  Row 2 │ Total Error ‖e‖   │ X Error        │ Y Error        │ Z Error

Robustness mode (--ros-args -p robustness_mode:=true)
  Row 1 │ Total Error / Lap  │ X Error / Lap  │ Y Error / Lap  │ Z Error / Lap
  Row 2 │ Cum Total / Lap    │ Cum X / Lap    │ Cum Y / Lap    │ Cum Z / Lap
  Row 3 │ Max ‖e‖ bar        │ Max X bar      │ Max Y bar      │ Max Z bar
"""

import os
import numpy as np
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Theme ─────────────────────────────────────────────────────────────────────
BG_FIG = "#0D1117"
BG_AX = "#161B22"
BG_AX2 = "#1C2128"
COL_GRID = "#21262D"
COL_SPINE = "#30363D"
COL_TICK = "#8B949E"
COL_TITLE = "#58A6FF"
COL_ERR = "#FF4757"
COL_CUM = "#FFA502"
COL_INFO = "#FFD700"
COL_MOTORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
COL_LAPS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#A29BFE",
    "#FD79A8",
    "#FDCB6E",
    "#55EFC4",
    "#74B9FF",
    "#B2BEC3",
]

ALPHA_FILL = 0.15
WINDOW_S = 30.0


# ── Styling helpers ───────────────────────────────────────────────────────────
def _setup_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COL_TICK, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(COL_SPINE)
        sp.set_linewidth(0.6)
    ax.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.9)


def _lbl(ax, title, xlabel="t (s)", ylabel="", tc=COL_TITLE):
    ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, color=COL_TICK, fontsize=9, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, color=COL_TICK, fontsize=9, labelpad=3)


def _leg(ax):
    ax.legend(
        fontsize=8,
        facecolor=BG_AX2,
        edgecolor=COL_SPINE,
        labelcolor="white",
        framealpha=0.9,
        handlelength=1.5,
    )


def _fill(fills, key, ax, t, y, color=COL_CUM):
    old = fills.get(key)
    if old is not None:
        try:
            old.remove()
        except ValueError:
            pass

    if len(t) > 1:
        fills[key] = ax.fill_between(t, 0, y, alpha=ALPHA_FILL, color=color)
    else:
        fills[key] = None


def _scroll(ax, t):
    if len(t) < 2:
        return
    t_max = float(t[-1])
    ax.set_xlim(max(float(t[0]), t_max - WINDOW_S), t_max + 0.3)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)


# ═════════════════════════════════════════════════════════════════════════════
# Normal mode — cumulative errors + instantaneous error plots
# ═════════════════════════════════════════════════════════════════════════════
def build_normal():
    plt.style.use("dark_background")
    fig = plt.figure("Error Analysis", figsize=(32, 14))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle(
        "Quadrotor Controller — Error Analysis",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.30,
        left=0.045,
        right=0.98,
        top=0.925,
        bottom=0.065,
    )

    # Row 0 — cumulative errors
    gs0 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[0], wspace=0.22
    )
    ax_ctot = fig.add_subplot(gs0[0])
    ax_cx = fig.add_subplot(gs0[1])
    ax_cy = fig.add_subplot(gs0[2])
    ax_cz = fig.add_subplot(gs0[3])

    # Row 1 — instantaneous errors
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], wspace=0.22
    )
    ax_etot = fig.add_subplot(gs1[0])
    ax_ex = fig.add_subplot(gs1[1])
    ax_ey = fig.add_subplot(gs1[2])
    ax_ez = fig.add_subplot(gs1[3])

    for ax in [ax_ctot, ax_cx, ax_cy, ax_cz, ax_etot, ax_ex, ax_ey, ax_ez]:
        _setup_ax(ax)

    _lbl(ax_ctot, "Cumulative Total Error", ylabel="∫‖e‖dt (m·s)", tc=COL_CUM)
    _lbl(ax_cx, "Cumulative X Error", ylabel="∫|eₓ|dt (m·s)", tc=COL_CUM)
    _lbl(ax_cy, "Cumulative Y Error", ylabel="∫|eᵧ|dt (m·s)", tc=COL_CUM)
    _lbl(ax_cz, "Cumulative Z Error", ylabel="∫|e_z|dt (m·s)", tc=COL_CUM)

    _lbl(ax_etot, "Total Error ‖e‖", ylabel="‖e‖ (m)", tc=COL_ERR)
    _lbl(ax_ex, "X Error", ylabel="|eₓ| (m)", tc=COL_ERR)
    _lbl(ax_ey, "Y Error", ylabel="|eᵧ| (m)", tc=COL_ERR)
    _lbl(ax_ez, "Z Error", ylabel="|e_z| (m)", tc=COL_ERR)

    # Persistent cumulative lines
    lc_tot, = ax_ctot.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)
    lc_x, = ax_cx.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)
    lc_y, = ax_cy.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)
    lc_z, = ax_cz.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)

    # Persistent error lines
    le_tot, = ax_etot.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)
    le_x, = ax_ex.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)
    le_y, = ax_ey.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)
    le_z, = ax_ez.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)

    cum_axes = dict(ct=ax_ctot, cx=ax_cx, cy=ax_cy, cz=ax_cz)
    err_axes = dict(et=ax_etot, ex=ax_ex, ey=ax_ey, ez=ax_ez)

    info = fig.text(
        0.5,
        0.004,
        "Waiting for data…",
        ha="center",
        va="bottom",
        color=COL_INFO,
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor=BG_AX,
            edgecolor=COL_INFO,
            linewidth=1.5,
            alpha=0.95,
        ),
    )

    for fx, fy, ftxt in [
        (0.005, 0.78, "Cumulative"),
        (0.005, 0.27, "Error"),
    ]:
        fig.text(
            fx,
            fy,
            ftxt,
            color="#484F58",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            va="center",
            transform=fig.transFigure,
        )

    artists = dict(
        lc_tot=lc_tot,
        lc_x=lc_x,
        lc_y=lc_y,
        lc_z=lc_z,
        le_tot=le_tot,
        le_x=le_x,
        le_y=le_y,
        le_z=le_z,
        fills={
            k: None
            for k in [
                "cum_t",
                "cum_x",
                "cum_y",
                "cum_z",
                "err_t",
                "err_x",
                "err_y",
                "err_z",
            ]
        },
        cum_axes=cum_axes,
        err_axes=err_axes,
    )
    return fig, artists, info


def update_normal(d: dict, artists: dict, info_text):
    t = d["t"]
    if len(t) < 2:
        return

    cx = d["cx"]
    cy = d["cy"]
    cz = d["cz"]
    ct = d["ct"]

    ex = d["ex"]
    ey = d["ey"]
    ez = d["ez"]
    et = d["et"]

    aex = np.abs(ex)
    aey = np.abs(ey)
    aez = np.abs(ez)

    fills = artists["fills"]
    ca = artists["cum_axes"]
    ea = artists["err_axes"]

    # Row 1 — cumulative
    artists["lc_tot"].set_data(t, ct)
    artists["lc_x"].set_data(t, cx)
    artists["lc_y"].set_data(t, cy)
    artists["lc_z"].set_data(t, cz)

    _fill(fills, "cum_t", ca["ct"], t, ct, color=COL_CUM)
    _fill(fills, "cum_x", ca["cx"], t, cx, color=COL_CUM)
    _fill(fills, "cum_y", ca["cy"], t, cy, color=COL_CUM)
    _fill(fills, "cum_z", ca["cz"], t, cz, color=COL_CUM)

    _scroll(ca["ct"], t)
    _set_dynamic_ylim(ca["ct"], ct, bottom_zero=True)

    _scroll(ca["cx"], t)
    _set_dynamic_ylim(ca["cx"], cx, bottom_zero=True)

    _scroll(ca["cy"], t)
    _set_dynamic_ylim(ca["cy"], cy, bottom_zero=True)

    _scroll(ca["cz"], t)
    _set_dynamic_ylim(ca["cz"], cz, bottom_zero=True)

    # Row 2 — error plots
    artists["le_tot"].set_data(t, et)
    artists["le_x"].set_data(t, aex)
    artists["le_y"].set_data(t, aey)
    artists["le_z"].set_data(t, aez)

    _fill(fills, "err_t", ea["et"], t, et, color=COL_ERR)
    _fill(fills, "err_x", ea["ex"], t, aex, color=COL_ERR)
    _fill(fills, "err_y", ea["ey"], t, aey, color=COL_ERR)
    _fill(fills, "err_z", ea["ez"], t, aez, color=COL_ERR)

    _scroll(ea["et"], t)
    _set_dynamic_ylim(ea["et"], _visible_window(t, et), bottom_zero=True)

    _scroll(ea["ex"], t)
    _set_dynamic_ylim(ea["ex"], _visible_window(t, aex), bottom_zero=True)

    _scroll(ea["ey"], t)
    _set_dynamic_ylim(ea["ey"], _visible_window(t, aey), bottom_zero=True)

    _scroll(ea["ez"], t)
    _set_dynamic_ylim(ea["ez"], _visible_window(t, aez), bottom_zero=True)

    info_text.set_text(
        f'  ▶  MAX ERRORS ——'
        f'  ‖e‖ : {d["max_et"]:.4f} m'
        f'  |  X : {d["max_ex"]:.4f} m'
        f'  |  Y : {d["max_ey"]:.4f} m'
        f'  |  Z : {d["max_ez"]:.4f} m  '
    )


# ═════════════════════════════════════════════════════════════════════════════
# Robustness mode — per-lap overlays + bar charts
# ═════════════════════════════════════════════════════════════════════════════

#!/usr/bin/env python3
"""
dashboard_analysis.py
=====================
Window 2 — Error Analysis & Robustness
──────────────────────────────────────
Normal mode (single run)
  Row 1 │ Cumulative Total  │ Cumulative X   │ Cumulative Y   │ Cumulative Z
  Row 2 │ Total Error ‖e‖   │ X Error        │ Y Error        │ Z Error

Robustness mode (--ros-args -p robustness_mode:=true)
  Row 1 │ Total Error / Lap  │ X Error / Lap  │ Y Error / Lap  │ Z Error / Lap
  Row 2 │ Cum Total / Lap    │ Cum X / Lap    │ Cum Y / Lap    │ Cum Z / Lap
  Row 3 │ Max ‖e‖ bar        │ Max X bar      │ Max Y bar      │ Max Z bar
"""

import os
import numpy as np
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Theme ─────────────────────────────────────────────────────────────────────
BG_FIG = "#0D1117"
BG_AX = "#161B22"
BG_AX2 = "#1C2128"
COL_GRID = "#21262D"
COL_SPINE = "#30363D"
COL_TICK = "#8B949E"
COL_TITLE = "#58A6FF"
COL_ERR = "#FF4757"
COL_CUM = "#FFA502"
COL_INFO = "#FFD700"
COL_MOTORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
COL_LAPS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#A29BFE",
    "#FD79A8",
    "#FDCB6E",
    "#55EFC4",
    "#74B9FF",
    "#B2BEC3",
]

ALPHA_FILL = 0.15
WINDOW_S = 30.0


# ── Styling helpers ───────────────────────────────────────────────────────────
def _setup_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COL_TICK, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(COL_SPINE)
        sp.set_linewidth(0.6)
    ax.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.9)


def _lbl(ax, title, xlabel="t (s)", ylabel="", tc=COL_TITLE):
    ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, color=COL_TICK, fontsize=9, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, color=COL_TICK, fontsize=9, labelpad=3)


def _leg(ax):
    ax.legend(
        fontsize=8,
        facecolor=BG_AX2,
        edgecolor=COL_SPINE,
        labelcolor="white",
        framealpha=0.9,
        handlelength=1.5,
    )


def _fill(fills, key, ax, t, y, color=COL_CUM):
    old = fills.get(key)
    if old is not None:
        try:
            old.remove()
        except ValueError:
            pass

    if len(t) > 1:
        fills[key] = ax.fill_between(t, 0, y, alpha=ALPHA_FILL, color=color)
    else:
        fills[key] = None


def _scroll(ax, t):
    if len(t) < 2:
        return
    t_max = float(t[-1])
    ax.set_xlim(max(float(t[0]), t_max - WINDOW_S), t_max + 0.3)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)


# ═════════════════════════════════════════════════════════════════════════════
# Normal mode — cumulative errors + instantaneous error plots
# ═════════════════════════════════════════════════════════════════════════════
def build_normal():
    plt.style.use("dark_background")
    fig = plt.figure("Error Analysis", figsize=(32, 14))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle(
        "Quadrotor Controller — Error Analysis",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.30,
        left=0.045,
        right=0.98,
        top=0.925,
        bottom=0.065,
    )

    # Row 0 — cumulative errors
    gs0 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[0], wspace=0.22
    )
    ax_ctot = fig.add_subplot(gs0[0])
    ax_cx = fig.add_subplot(gs0[1])
    ax_cy = fig.add_subplot(gs0[2])
    ax_cz = fig.add_subplot(gs0[3])

    # Row 1 — instantaneous errors
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], wspace=0.22
    )
    ax_etot = fig.add_subplot(gs1[0])
    ax_ex = fig.add_subplot(gs1[1])
    ax_ey = fig.add_subplot(gs1[2])
    ax_ez = fig.add_subplot(gs1[3])

    for ax in [ax_ctot, ax_cx, ax_cy, ax_cz, ax_etot, ax_ex, ax_ey, ax_ez]:
        _setup_ax(ax)

    _lbl(ax_ctot, "Cumulative Total Error", ylabel="∫‖e‖dt (m·s)", tc=COL_CUM)
    _lbl(ax_cx, "Cumulative X Error", ylabel="∫|eₓ|dt (m·s)", tc=COL_CUM)
    _lbl(ax_cy, "Cumulative Y Error", ylabel="∫|eᵧ|dt (m·s)", tc=COL_CUM)
    _lbl(ax_cz, "Cumulative Z Error", ylabel="∫|e_z|dt (m·s)", tc=COL_CUM)

    _lbl(ax_etot, "Total Error ‖e‖", ylabel="‖e‖ (m)", tc=COL_ERR)
    _lbl(ax_ex, "X Error", ylabel="|eₓ| (m)", tc=COL_ERR)
    _lbl(ax_ey, "Y Error", ylabel="|eᵧ| (m)", tc=COL_ERR)
    _lbl(ax_ez, "Z Error", ylabel="|e_z| (m)", tc=COL_ERR)

    # Persistent cumulative lines
    lc_tot, = ax_ctot.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)
    lc_x, = ax_cx.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)
    lc_y, = ax_cy.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)
    lc_z, = ax_cz.plot([], [], color=COL_CUM, lw=2.0, alpha=0.93)

    # Persistent error lines
    le_tot, = ax_etot.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)
    le_x, = ax_ex.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)
    le_y, = ax_ey.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)
    le_z, = ax_ez.plot([], [], color=COL_ERR, lw=2.0, alpha=0.93)

    cum_axes = dict(ct=ax_ctot, cx=ax_cx, cy=ax_cy, cz=ax_cz)
    err_axes = dict(et=ax_etot, ex=ax_ex, ey=ax_ey, ez=ax_ez)

    info = fig.text(
        0.5,
        0.004,
        "Waiting for data…",
        ha="center",
        va="bottom",
        color=COL_INFO,
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor=BG_AX,
            edgecolor=COL_INFO,
            linewidth=1.5,
            alpha=0.95,
        ),
    )

    for fx, fy, ftxt in [
        (0.005, 0.78, "Cumulative"),
        (0.005, 0.27, "Error"),
    ]:
        fig.text(
            fx,
            fy,
            ftxt,
            color="#484F58",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            va="center",
            transform=fig.transFigure,
        )

    artists = dict(
        lc_tot=lc_tot,
        lc_x=lc_x,
        lc_y=lc_y,
        lc_z=lc_z,
        le_tot=le_tot,
        le_x=le_x,
        le_y=le_y,
        le_z=le_z,
        fills={
            k: None
            for k in [
                "cum_t",
                "cum_x",
                "cum_y",
                "cum_z",
                "err_t",
                "err_x",
                "err_y",
                "err_z",
            ]
        },
        cum_axes=cum_axes,
        err_axes=err_axes,
    )
    return fig, artists, info


def update_normal(d: dict, artists: dict, info_text):
    t = d["t"]
    if len(t) < 2:
        return

    cx = d["cx"]
    cy = d["cy"]
    cz = d["cz"]
    ct = d["ct"]

    ex = d["ex"]
    ey = d["ey"]
    ez = d["ez"]
    et = d["et"]

    aex = np.abs(ex)
    aey = np.abs(ey)
    aez = np.abs(ez)

    fills = artists["fills"]
    ca = artists["cum_axes"]
    ea = artists["err_axes"]

    # Row 1 — cumulative
    artists["lc_tot"].set_data(t, ct)
    artists["lc_x"].set_data(t, cx)
    artists["lc_y"].set_data(t, cy)
    artists["lc_z"].set_data(t, cz)

    _fill(fills, "cum_t", ca["ct"], t, ct, color=COL_CUM)
    _fill(fills, "cum_x", ca["cx"], t, cx, color=COL_CUM)
    _fill(fills, "cum_y", ca["cy"], t, cy, color=COL_CUM)
    _fill(fills, "cum_z", ca["cz"], t, cz, color=COL_CUM)

    _scroll(ca["ct"], t)
    _set_dynamic_ylim(ca["ct"], ct, bottom_zero=True)

    _scroll(ca["cx"], t)
    _set_dynamic_ylim(ca["cx"], cx, bottom_zero=True)

    _scroll(ca["cy"], t)
    _set_dynamic_ylim(ca["cy"], cy, bottom_zero=True)

    _scroll(ca["cz"], t)
    _set_dynamic_ylim(ca["cz"], cz, bottom_zero=True)

    # Row 2 — error plots
    artists["le_tot"].set_data(t, et)
    artists["le_x"].set_data(t, aex)
    artists["le_y"].set_data(t, aey)
    artists["le_z"].set_data(t, aez)

    _fill(fills, "err_t", ea["et"], t, et, color=COL_ERR)
    _fill(fills, "err_x", ea["ex"], t, aex, color=COL_ERR)
    _fill(fills, "err_y", ea["ey"], t, aey, color=COL_ERR)
    _fill(fills, "err_z", ea["ez"], t, aez, color=COL_ERR)

    _scroll(ea["et"], t)
    _set_dynamic_ylim(ea["et"], _visible_window(t, et), bottom_zero=True)

    _scroll(ea["ex"], t)
    _set_dynamic_ylim(ea["ex"], _visible_window(t, aex), bottom_zero=True)

    _scroll(ea["ey"], t)
    _set_dynamic_ylim(ea["ey"], _visible_window(t, aey), bottom_zero=True)

    _scroll(ea["ez"], t)
    _set_dynamic_ylim(ea["ez"], _visible_window(t, aez), bottom_zero=True)

    info_text.set_text(
        f'  ▶  MAX ERRORS ——'
        f'  ‖e‖ : {d["max_et"]:.4f} m'
        f'  |  X : {d["max_ex"]:.4f} m'
        f'  |  Y : {d["max_ey"]:.4f} m'
        f'  |  Z : {d["max_ez"]:.4f} m  '
    )


# ═════════════════════════════════════════════════════════════════════════════
# Robustness mode — per-lap overlays + bar charts
# ═════════════════════════════════════════════════════════════════════════════

def build_robustness():
    plt.style.use("dark_background")
    fig = plt.figure("Robustness Analysis", figsize=(32, 20))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle(
        "Robustness Analysis — Multi-Lap Error Comparison",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    outer = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.9],
        hspace=0.32,
        left=0.045,
        right=0.98,
        top=0.925,
        bottom=0.065,
    )

    # Row 1 — instantaneous error per lap
    gs0 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[0], wspace=0.22
    )
    ax_et_l = fig.add_subplot(gs0[0])
    ax_ex_l = fig.add_subplot(gs0[1])
    ax_ey_l = fig.add_subplot(gs0[2])
    ax_ez_l = fig.add_subplot(gs0[3])

    # Row 2 — cumulative error per lap
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], wspace=0.22
    )
    ax_ct_l = fig.add_subplot(gs1[0])
    ax_cx_l = fig.add_subplot(gs1[1])
    ax_cy_l = fig.add_subplot(gs1[2])
    ax_cz_l = fig.add_subplot(gs1[3])

    # Row 3 — max error bars
    gs2 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[2], wspace=0.22
    )
    ax_be  = fig.add_subplot(gs2[0])
    ax_bex = fig.add_subplot(gs2[1])
    ax_bey = fig.add_subplot(gs2[2])
    ax_bez = fig.add_subplot(gs2[3])

    axes_r = {
        "et_l": ax_et_l,
        "ex_l": ax_ex_l,
        "ey_l": ax_ey_l,
        "ez_l": ax_ez_l,
        "ct_l": ax_ct_l,
        "cx_l": ax_cx_l,
        "cy_l": ax_cy_l,
        "cz_l": ax_cz_l,
        "be": ax_be,
        "bex": ax_bex,
        "bey": ax_bey,
        "bez": ax_bez,
    }

    for ax in axes_r.values():
        _setup_ax(ax)

    # Row 1 labels
    _lbl(ax_et_l, "Total Error / Lap", ylabel="‖e‖ (m)", tc=COL_ERR)
    _lbl(ax_ex_l, "X Error / Lap", ylabel="|eₓ| (m)", tc=COL_MOTORS[0])
    _lbl(ax_ey_l, "Y Error / Lap", ylabel="|eᵧ| (m)", tc=COL_MOTORS[1])
    _lbl(ax_ez_l, "Z Error / Lap", ylabel="|e_z| (m)", tc=COL_MOTORS[2])

    # Row 2 labels
    _lbl(ax_ct_l, "Cum Total / Lap", ylabel="∫‖e‖dt", tc=COL_CUM)
    _lbl(ax_cx_l, "Cum X / Lap", ylabel="∫|eₓ|dt", tc="#FDCB6E")
    _lbl(ax_cy_l, "Cum Y / Lap", ylabel="∫|eᵧ|dt", tc="#55EFC4")
    _lbl(ax_cz_l, "Cum Z / Lap", ylabel="∫|e_z|dt", tc="#74B9FF")

    # Row 3 labels
    _lbl(ax_be,  "Max ‖e‖ / Lap", xlabel="Lap", ylabel="‖e‖ (m)", tc=COL_ERR)
    _lbl(ax_bex, "Max |eₓ| / Lap", xlabel="Lap", ylabel="|eₓ| (m)", tc=COL_MOTORS[0])
    _lbl(ax_bey, "Max |eᵧ| / Lap", xlabel="Lap", ylabel="|eᵧ| (m)", tc=COL_MOTORS[1])
    _lbl(ax_bez, "Max |e_z| / Lap", xlabel="Lap", ylabel="|e_z| (m)", tc=COL_MOTORS[2])

    # Left-side row labels
    for fx, fy, ftxt in [
        (0.005, 0.82, "Error / Lap"),
        (0.005, 0.52, "Cum / Lap"),
        (0.005, 0.18, "Max / Lap"),
    ]:
        fig.text(
            fx,
            fy,
            ftxt,
            color="#484F58",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            va="center",
            transform=fig.transFigure,
        )

    return fig, axes_r


def update_robustness(axes_r: dict, laps: list):
    if not laps:
        return

    cols = COL_LAPS

    # Instantaneous error per lap
    for ax_k, dkey, ylabel, title, tc in [
        ("et_l", "et", "‖e‖ (m)", "Total Error / Lap", COL_ERR),
        ("ex_l", "ex", "|eₓ| (m)", "X Error / Lap", COL_MOTORS[0]),
        ("ey_l", "ey", "|eᵧ| (m)", "Y Error / Lap", COL_MOTORS[1]),
        ("ez_l", "ez", "|e_z| (m)", "Z Error / Lap", COL_MOTORS[2]),
    ]:
        ax = axes_r[ax_k]
        ax.cla()
        _setup_ax(ax)

        # ax.set_ylim(bottom=0)
        all_e = []
        for i, lap in enumerate(laps):
            t = np.asarray(lap["t"], dtype=float)
            e = np.abs(np.asarray(lap[dkey], dtype=float))
            all_e.append(e)
            c = cols[i % len(cols)]
            ax.plot(t, e, color=c, lw=1.8, alpha=0.88, label=f'Lap {lap["lap"]}')
            ax.fill_between(t, 0, e, alpha=0.08, color=c)

        _set_dynamic_ylim_multi(ax, all_e, bottom_zero=True)

        _lbl(ax, title, ylabel=ylabel, tc=tc)
        _leg(ax)

    # Cumulative per lap
    for ax_k, dkey, ylabel, title, tc in [
        ("ct_l", "ct", "∫‖e‖dt", "Cum Total / Lap", COL_CUM),
        ("cx_l", "cx", "∫|eₓ|dt", "Cum X / Lap", "#FDCB6E"),
        ("cy_l", "cy", "∫|eᵧ|dt", "Cum Y / Lap", "#55EFC4"),
        ("cz_l", "cz", "∫|e_z|dt", "Cum Z / Lap", "#74B9FF"),
    ]:
        ax = axes_r[ax_k]
        ax.cla()
        _setup_ax(ax)

        all_c = []
        for i, lap in enumerate(laps):
            t = np.asarray(lap["t"], dtype=float)
            c_ = np.asarray(lap[dkey], dtype=float)
            all_c.append(c_)
            ax.plot(
                t,
                c_,
                color=cols[i % len(cols)],
                lw=2.0,
                alpha=0.90,
                label=f'Lap {lap["lap"]}',
            )

        # ax.set_ylim(bottom=0)

        _set_dynamic_ylim_multi(ax, all_c, bottom_zero=True)
        _lbl(ax, title, ylabel=ylabel, tc=tc)
        _leg(ax)

    # Bar charts — max error per lap
    lap_nums = [l["lap"] for l in laps]
    bar_colors = [cols[i % len(cols)] for i in range(len(laps))]

    for ax_k, dkey, title, ylabel, tc in [
        ("be", "max_et", "Max ‖e‖ / Lap", "‖e‖ (m)", COL_ERR),
        ("bex", "max_ex", "Max |eₓ| / Lap", "|eₓ| (m)", COL_MOTORS[0]),
        ("bey", "max_ey", "Max |eᵧ| / Lap", "|eᵧ| (m)", COL_MOTORS[1]),
        ("bez", "max_ez", "Max |e_z| / Lap", "|e_z| (m)", COL_MOTORS[2]),
    ]:
        ax = axes_r[ax_k]
        ax.cla()
        _setup_ax(ax)

        vals = [l[dkey] for l in laps]
        bars = ax.bar(
            range(len(laps)),
            vals,
            color=bar_colors,
            width=0.62,
            edgecolor=COL_SPINE,
            linewidth=0.7,
        )

        vmax = max(vals) if vals else 0.0
        y_top = max(vmax * 1.22, 0.01)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vmax * 0.025, 0.002),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xticks(range(len(laps)))
        ax.set_xticklabels(
            [f"Lap {n}" for n in lap_nums],
            fontsize=8.5,
            color=COL_TICK,
        )
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(True, color=COL_GRID, linewidth=0.4, alpha=0.9, axis="y")
        _lbl(ax, title, xlabel="Lap", ylabel=ylabel, tc=tc)

        if len(vals) > 1:
            mean_v = float(np.mean(vals))
            ax.axhline(
                mean_v,
                color="white",
                ls=":",
                lw=1.0,
                alpha=0.65,
                label=f"Mean = {mean_v:.4f}",
            )
            ax.legend(
                fontsize=8,
                facecolor=BG_AX2,
                edgecolor=COL_SPINE,
                labelcolor="white",
                framealpha=0.9,
            )

def _set_dynamic_ylim(ax, y, bottom_zero=False, pad_ratio=0.10, min_span=1e-3):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if bottom_zero:
        y_min = 0.0

    span = y_max - y_min
    if span < min_span:
        span = min_span

    pad = span * pad_ratio
    ax.set_ylim(y_min - (0.0 if bottom_zero else pad), y_max + pad)


def _set_dynamic_ylim_multi(ax, ys, bottom_zero=False, pad_ratio=0.10, min_span=1e-3):
    arrs = []
    for y in ys:
        y = np.asarray(y, dtype=float)
        if y.size > 0:
            arrs.append(y)

    if not arrs:
        return

    y_all = np.concatenate(arrs)
    _set_dynamic_ylim(ax, y_all, bottom_zero=bottom_zero, pad_ratio=pad_ratio, min_span=min_span)

def _visible_window(t, y, window_s=WINDOW_S):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0 or y.size == 0:
        return y

    t_max = float(t[-1])
    mask = t >= (t_max - window_s)
    return y[mask]