"""
interactive.py

Interactive CLI/GUI helpers for the SpikePath pipeline.

Functions
---------
confirm_axis          : matplotlib window to preview and confirm the principal axis
confirm_intermediates : text prompt to accept or override auto-selected intermediates
prompt_n_traces       : text prompt to choose how many waveform traces to extract
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from .filtering import PITCH, GRID_ROWS, GRID_COLS

__all__ = [
    "confirm_axis",
    "confirm_intermediates",
    "prompt_n_traces",
]


def confirm_axis(rec, axis, overlay):
    """
    Show the principal axis, search band, and candidate electrodes.
    Press ENTER or SPACE to confirm and continue.
    Press Q or Escape to abort (calls sys.exit).

    Parameters
    ----------
    rec     : Recording
    axis    : AxisSelection
    overlay : ndarray or None  — RGBA overlay image
    """
    gc = rec.gc
    gr = rec.gr

    px, py    = -axis.uy, axis.ux          # perpendicular unit vector
    offset_gu = axis.search_band_um / PITCH

    src_gu = axis.sx / PITCH;  src_gr = axis.sy / PITCH
    tgt_gu = axis.tx / PITCH;  tgt_gr = axis.ty / PITCH

    ch_gr_i = np.round(gr).astype(int)
    ch_gc_i = np.round(gc).astype(int)
    pad = 3
    r_min = max(ch_gr_i.min() - pad, 0);  r_max = min(ch_gr_i.max() + pad, GRID_ROWS - 1)
    c_min = max(ch_gc_i.min() - pad, 0);  c_max = min(ch_gc_i.max() + pad, GRID_COLS - 1)

    fig, ax = plt.subplots(figsize=(14, 9))

    if overlay is not None:
        ax.imshow(overlay, extent=[0, GRID_COLS, GRID_ROWS, 0],
                  origin='upper', aspect='auto', alpha=0.5, zorder=0)

    ax.plot(gc, gr, '.', color='cornflowerblue', markersize=3, alpha=0.4,
            zorder=1, linestyle='none')

    if len(axis.cands) > 0:
        cmap_axis = plt.cm.plasma
        norm_axis = plt.Normalize(axis.d_lo, axis.d_hi)
        for c in axis.cands:
            col = cmap_axis(norm_axis(axis.d_along[c]))
            ax.plot(rec.xy[c, 0] / PITCH, rec.xy[c, 1] / PITCH,
                    'o', color=col, markersize=8,
                    markeredgecolor='white', markeredgewidth=0.6,
                    alpha=0.9, zorder=5)
        sm = plt.cm.ScalarMappable(cmap=cmap_axis, norm=norm_axis)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label('Distance along axis (µm)', fontsize=9)

    for sign in (+1, -1):
        bx1 = src_gu + sign * px * offset_gu
        by1 = src_gr + sign * py * offset_gu
        bx2 = tgt_gu + sign * px * offset_gu
        by2 = tgt_gr + sign * py * offset_gu
        ax.plot([bx1, bx2], [by1, by2], '--', color='yellow',
                lw=1.4, alpha=0.9, zorder=4)

    ax.plot([src_gu, tgt_gu], [src_gr, tgt_gr],
            color='white', lw=1.8, ls='--', alpha=0.7, zorder=6)

    ax.plot(src_gu, src_gr, 'o', color='lime', markersize=13,
            markeredgecolor='white', markeredgewidth=1.5, zorder=8)
    ax.text(src_gu + offset_gu * 1.1, src_gr, 'SRC',
            color='lime', fontsize=10, fontweight='bold', va='center', zorder=8)
    ax.plot(tgt_gu, tgt_gr, 'o', color='tomato', markersize=13,
            markeredgecolor='white', markeredgewidth=1.5, zorder=8)
    ax.text(tgt_gu + offset_gu * 1.1, tgt_gr, 'TGT',
            color='tomato', fontsize=10, fontweight='bold', va='center', zorder=8)

    ax.set_xlim(c_min, c_max + 1);  ax.set_ylim(r_max + 1, r_min)
    ax.set_xlabel('Grid column (x17.5 µm)');  ax.set_ylabel('Grid row (x17.5 µm)')
    ax.grid(True, alpha=0.12)
    ax.set_title(
        f'Axis preview  |  SRC=ch{axis.src_ch}  TGT=ch{axis.tgt_ch}  '
        f'|  axis={axis.axis_len:.0f} µm  |  band=±{axis.search_band_um:.0f} µm  '
        f'|  {len(axis.cands)} candidates\n'
        f'Press  ENTER / SPACE  to confirm      Q / Escape  to abort',
        fontsize=11)
    plt.tight_layout()

    confirmed = [False]

    def _on_key(event):
        if event.key in ('enter', ' '):
            confirmed[0] = True
            plt.close(fig)
        elif event.key in ('q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', _on_key)
    plt.show()   # blocking

    if not confirmed[0]:
        sys.exit("Axis rejected — re-run with different --src / --tgt or a different --search_band_um.")


def confirm_intermediates(rec, axis, auto_inter):
    """
    Print the auto-selected intermediate electrodes and let the user accept or
    override them.  Returns the final list of channel indices, sorted by
    position along the axis.

    Parameters
    ----------
    rec        : Recording
    axis       : AxisSelection
    auto_inter : list[int]  — channel indices from select_intermediates()

    Returns
    -------
    list[int]
    """
    d_lo, d_hi = axis.d_lo, axis.d_hi

    print(f"\nAuto-selected {len(auto_inter)} intermediate(s):")
    for i, c in enumerate(auto_inter):
        frac = (axis.d_along[c] - d_lo) / (d_hi - d_lo) * 100 if d_hi > d_lo else 0.0
        print(f"  M{i+1}  ch={c}  "
              f"({rec.xy[c, 0]:.0f}, {rec.xy[c, 1]:.0f}) µm  "
              f"{frac:.0f}% along axis")

    try:
        resp = input(
            "\nPress Enter to accept, or type comma-separated channel IDs to override: "
        ).strip()
    except KeyboardInterrupt:
        sys.exit("\nAborted.")

    if resp == '':
        print("Accepted auto-selected intermediates.")
        return auto_inter

    try:
        manual = [int(x.strip()) for x in resp.split(',') if x.strip()]
        manual = sorted(manual, key=lambda c: axis.d_along[c])
        print(f"Using manual intermediates: {manual}")
        return manual
    except ValueError:
        print("[warn] Could not parse input — using auto-selected intermediates.")
        return auto_inter


def prompt_n_traces(rec, axis, mode, link_min_ms, link_max_ms, default):
    """
    Count available spike events and let the user choose how many traces to
    extract.  Returns the chosen integer.

    Parameters
    ----------
    rec          : Recording
    axis         : AxisSelection
    mode         : str    — 'src_tgt' or 'src_only'
    link_min_ms  : float
    link_max_ms  : float
    default      : int    — value used when the user presses Enter

    Returns
    -------
    int
    """
    sp_times = rec.sp_times
    sp_ch    = rec.sp_ch
    fs       = rec.fs
    src_ch   = axis.src_ch
    tgt_ch   = axis.tgt_ch

    src_evt = rec.spike_array[sp_ch == src_ch]

    if mode == 'src_only':
        n_available = len(src_evt)
    else:
        tgt_times = sp_times[sp_ch == tgt_ch]
        link_lo   = int(link_min_ms / 1000 * fs)
        link_hi   = int(link_max_ms / 1000 * fs)
        n_available = sum(
            1 for sf in src_evt[:, 0]
            if np.searchsorted(tgt_times, sf + link_lo, 'left') <
               np.searchsorted(tgt_times, sf + link_hi, 'right')
        )

    print(f"\nAvailable events (mode='{mode}'): {n_available}")

    try:
        resp = input(f"Number of traces to extract [{default}]: ").strip()
    except KeyboardInterrupt:
        sys.exit("\nAborted.")

    if resp == '':
        return default
    try:
        n = int(resp)
        if n < 1:
            print("[warn] Must be at least 1 — using default.")
            return default
        if n > n_available:
            print(f"[warn] Only {n_available} available — clamping.")
            return n_available
        return n
    except ValueError:
        print("[warn] Invalid input — using default.")
        return default
