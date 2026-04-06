"""
spikepath/selection.py

Electrode and axis selection for MaxWell HD-MEA recordings.

Loads recordings and overlays, selects the principal propagation axis,
counts spikes along that axis, and scores intermediate electrodes.

Data classes
------------
Recording      : all data loaded from an H5 file and its spike cache
AxisSelection  : principal axis geometry and on-axis candidate electrodes

Functions
---------
load_recording(h5_path, mapping_path, ...)    -> Recording
load_overlay(overlay_path, overlay_cmap)       -> np.ndarray | None
select_axis(rec, overlay, ...)                 -> AxisSelection
run_spike_count_axis(rec, axis, out_dir, ...)  -> dict
select_intermediates(rec, axis, ...)           -> list[int]
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as _PILImage
from dataclasses import dataclass

from .filtering import (
    GRID_ROWS, GRID_COLS, PITCH,
    load_h5_mapping, compute_grid_positions, refractory_filter,
)


# ==========================================================================
# Data classes
# ==========================================================================

@dataclass
class Recording:
    """All data loaded from an H5 file and its spike cache."""
    xy          : object   # np.ndarray (n_channels, 2) — channel positions in µm
    gc          : object   # np.ndarray — grid column positions
    gr          : object   # np.ndarray — grid row positions
    n_channels  : int
    outlier_idx : int
    spike_array : object   # np.ndarray (N, 3) — [frame, amplitude, channel]
    sp_times    : object   # np.ndarray — spike frame numbers
    sp_ch       : object   # np.ndarray — spike channel indices
    duration_s  : float
    fs          : float
    h5_path     : str
    rec_f       : object = None   # bandpass-filtered SpikeInterface recording (optional)


@dataclass
class AxisSelection:
    """Result of select_axis() — principal axis geometry and on-axis candidates."""
    src_ch         : int
    tgt_ch         : int
    sx             : float    # SRC x position (µm)
    sy             : float    # SRC y position (µm)
    tx             : float    # TGT x position (µm)
    ty             : float    # TGT y position (µm)
    ux             : float    # axis unit vector x
    uy             : float    # axis unit vector y
    d_along        : object   # np.ndarray — along-axis distance for every channel (µm)
    d_perp         : object   # np.ndarray — perpendicular distance for every channel (µm)
    d_lo           : float    # min(src_d, tgt_d)
    d_hi           : float    # max(src_d, tgt_d)
    axis_len       : float    # d_hi - d_lo
    cands          : object   # np.ndarray — indices of on-axis intermediate candidates
    search_band_um : float


# ==========================================================================
# load_recording
# ==========================================================================

def load_recording(
    h5_path       : str,
    mapping_path  : str,
    fs            : float = 20_000,
    refractory_ms : float = 1.0,
    load_raw      : bool  = False,
) -> Recording:
    """
    Load channel map and spike cache from a MaxWell H5 file.

    Parameters
    ----------
    h5_path       : path to .raw.h5 file
    mapping_path  : path to MaxWell mapping directory
    fs            : sampling rate in Hz (default 20 000)
    refractory_ms : refractory period for spike de-duplication (ms)
    load_raw      : if True, also load and bandpass-filter the raw recording
                    via SpikeInterface — required for extract_waveforms()

    Returns
    -------
    Recording dataclass
    """
    x_arr, y_arr, n_channels, outlier_idx = load_h5_mapping(h5_path, mapping_path)
    xy = np.column_stack([x_arr, y_arr])
    gc, gr, _ = compute_grid_positions(x_arr, y_arr, PITCH, GRID_ROWS, GRID_COLS)

    spike_cache = h5_path.replace('.raw.h5', '_spikes.npy')
    if not os.path.exists(spike_cache):
        raise FileNotFoundError(
            f"Spike cache not found: {spike_cache}\n"
            "Run spikepath.detect_spikes() first to generate it."
        )
    spike_array = np.load(spike_cache)
    print(f"Loaded {len(spike_array)} spikes from: {os.path.basename(spike_cache)}")

    spike_array  = refractory_filter(spike_array, refractory_ms, fs)
    total_frames = spike_array[:, 0].max() - spike_array[:, 0].min()
    duration_s   = total_frames / fs
    print(f"Recording span : {duration_s:.1f} s")

    rec_f = None
    if load_raw:
        from spikeinterface.extractors import read_maxwell
        from spikeinterface.preprocessing import bandpass_filter, unsigned_to_signed
        rec_f = bandpass_filter(
            unsigned_to_signed(read_maxwell(h5_path)),
            freq_min=200, freq_max=6000, filter_order=3,
        )
        print("Raw recording loaded and bandpass-filtered (200–6000 Hz).")

    return Recording(
        xy=xy, gc=gc, gr=gr,
        n_channels=n_channels, outlier_idx=outlier_idx,
        spike_array=spike_array,
        sp_times=spike_array[:, 0],
        sp_ch=spike_array[:, 2].astype(int),
        duration_s=duration_s,
        fs=fs,
        h5_path=h5_path,
        rec_f=rec_f,
    )


# ==========================================================================
# load_overlay
# ==========================================================================

def load_overlay(
    overlay_path : str,
    overlay_cmap : str = 'hot',
) -> 'np.ndarray | None':
    """
    Load and resize an impedance map overlay image to the chip grid dimensions.

    Accepts PNG/JPG (returned as RGBA) or .npy scalar arrays (mapped via
    overlay_cmap).  Returns None if path is None or file not found.

    Parameters
    ----------
    overlay_path : path to .png, .jpg, or .npy file
    overlay_cmap : colormap used when loading a .npy scalar array

    Returns
    -------
    np.ndarray (GRID_ROWS, GRID_COLS, 4) uint8 RGBA, or None
    """
    if not overlay_path:
        return None
    if not os.path.exists(overlay_path):
        print(f"[warn] Overlay not found: {overlay_path} — skipping.")
        return None

    ext = os.path.splitext(overlay_path)[1].lower()
    if ext == '.npy':
        raw = np.load(overlay_path).astype(float)
        raw -= raw.min()
        if raw.max() > 0:
            raw /= raw.max()
        rgba = (plt.get_cmap(overlay_cmap)(raw) * 255).astype(np.uint8)
        pil  = _PILImage.fromarray(rgba, mode='RGBA')
    else:
        pil = _PILImage.open(overlay_path).convert('RGBA')

    pil = pil.resize((GRID_COLS, GRID_ROWS), _PILImage.LANCZOS)
    print(f"Overlay loaded: {os.path.basename(overlay_path)}")
    return np.array(pil)


# ==========================================================================
# select_axis
# ==========================================================================

def select_axis(
    rec            : Recording,
    overlay        : 'np.ndarray | None' = None,
    manual_src     : 'int | None'        = None,
    manual_tgt     : 'int | None'        = None,
    search_band_um : float               = 25.0,
    overlay_alpha  : float               = 0.55,
    figsize        : tuple               = (14, 9),
) -> AxisSelection:
    """
    Display an electrode map and select SOURCE and TARGET electrodes, then
    compute the principal axis geometry and collect on-axis candidate electrodes.

    If manual_src and manual_tgt are both provided the map is shown briefly
    for confirmation and closed automatically — no mouse clicks required.

    Parameters
    ----------
    rec            : Recording from load_recording()
    overlay        : overlay image array from load_overlay() (or None)
    manual_src     : source channel index — None triggers interactive click
    manual_tgt     : target channel index — None triggers interactive click
    search_band_um : max perpendicular distance from axis for candidate collection (µm)
    overlay_alpha  : overlay image transparency (0–1)
    figsize        : figure size for the electrode selection map

    Returns
    -------
    AxisSelection dataclass with axis geometry and candidate indices
    """
    xy, gc, gr = rec.xy, rec.gc, rec.gr

    pad     = 3
    ch_gr_i = np.round(gr).astype(int)
    ch_gc_i = np.round(gc).astype(int)
    r_min = max(ch_gr_i.min() - pad, 0)
    r_max = min(ch_gr_i.max() + pad, GRID_ROWS - 1)
    c_min = max(ch_gc_i.min() - pad, 0)
    c_max = min(ch_gc_i.max() + pad, GRID_COLS - 1)

    fig_map, ax_map = plt.subplots(figsize=figsize)
    if overlay is not None:
        ax_map.imshow(overlay, extent=[0, GRID_COLS, GRID_ROWS, 0],
                      origin='upper', aspect='auto', alpha=overlay_alpha, zorder=0)
    ax_map.plot(gc, gr, '.', color='cornflowerblue', markersize=3,
                alpha=0.6, zorder=2, linestyle='none')
    ax_map.set_xlim(c_min, c_max + 1)
    ax_map.set_ylim(r_max + 1, r_min)
    ax_map.set_xlabel('Grid column  (x17.5 um)')
    ax_map.set_ylabel('Grid row     (x17.5 um)')
    ax_map.grid(True, alpha=0.15)

    if manual_src is not None and manual_tgt is not None:
        src_ch = int(manual_src)
        tgt_ch = int(manual_tgt)
        sx, sy = float(xy[src_ch, 0]), float(xy[src_ch, 1])
        tx, ty = float(xy[tgt_ch, 0]), float(xy[tgt_ch, 1])
        ax_map.set_title(
            f'Manual selection: SRC=ch{src_ch}  TGT=ch{tgt_ch}\nClose to continue.',
            fontsize=12)
        fig_map.canvas.draw()
        plt.pause(0.5)
        plt.close(fig_map)
    else:
        ax_map.set_title(
            'Click 1 -> SOURCE electrode     Click 2 -> TARGET electrode\n'
            'Right-click or Enter to confirm', fontsize=12)
        fig_map.canvas.draw()
        pts = plt.ginput(2, timeout=120, show_clicks=True)
        plt.close(fig_map)
        if len(pts) < 2:
            raise SystemExit("Need exactly 2 clicks — re-run.")
        (s_gc, s_gr), (t_gc, t_gr) = pts
        sx, sy = s_gc * PITCH, s_gr * PITCH
        tx, ty = t_gc * PITCH, t_gr * PITCH
        src_ch = int(np.argmin(np.hypot(xy[:, 0] - sx, xy[:, 1] - sy)))
        tgt_ch = int(np.argmin(np.hypot(xy[:, 0] - tx, xy[:, 1] - ty)))

    print(f"SRC: ch={src_ch}  ({xy[src_ch,0]:.1f}, {xy[src_ch,1]:.1f}) um")
    print(f"TGT: ch={tgt_ch}  ({xy[tgt_ch,0]:.1f}, {xy[tgt_ch,1]:.1f}) um")

    # Principal axis geometry
    adx, ady = tx - sx, ty - sy
    alen     = float(np.hypot(adx, ady))
    ux, uy   = (adx / alen, ady / alen) if alen > 0 else (1.0, 0.0)

    d_along = (xy[:, 0] - sx) * ux + (xy[:, 1] - sy) * uy
    d_perp  = np.abs((xy[:, 0] - sx) * uy - (xy[:, 1] - sy) * ux)

    src_d = float(d_along[src_ch])
    tgt_d = float(d_along[tgt_ch])
    d_lo, d_hi = min(src_d, tgt_d), max(src_d, tgt_d)
    axis_len   = d_hi - d_lo

    cand_mask = ((d_perp  <= search_band_um) &
                 (d_along >  d_lo) &
                 (d_along <  d_hi))
    cand_mask[src_ch] = False
    cand_mask[tgt_ch] = False
    cands = np.flatnonzero(cand_mask)

    print(f"Axis length    : {axis_len:.1f} um")
    print(f"Search band    : +/-{search_band_um:.1f} um perpendicular")
    print(f"Candidates     : {len(cands)} electrodes (excl. SRC and TGT)")

    return AxisSelection(
        src_ch=src_ch, tgt_ch=tgt_ch,
        sx=sx, sy=sy, tx=tx, ty=ty,
        ux=ux, uy=uy,
        d_along=d_along, d_perp=d_perp,
        d_lo=d_lo, d_hi=d_hi, axis_len=axis_len,
        cands=cands,
        search_band_um=search_band_um,
    )


# ==========================================================================
# _Tee  (internal helper)
# ==========================================================================

class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path):
        self._file   = open(path, 'w', encoding='utf-8')
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# ==========================================================================
# run_spike_count_axis
# ==========================================================================

def run_spike_count_axis(
    rec           : Recording,
    axis          : AxisSelection,
    out_dir       : str,
    group_frac    : float               = 0.2,
    overlay       : 'np.ndarray | None' = None,
    overlay_alpha : float               = 0.55,
    save_log      : bool                = True,
) -> dict:
    """
    Count spikes along the principal axis, group into distance bins, plot,
    and save CSVs.

    Parameters
    ----------
    rec           : Recording from load_recording()
    axis          : AxisSelection from select_axis()
    out_dir       : directory to write outputs (created if missing)
    group_frac    : bin width as fraction of total axis length (default 0.2 -> 5 bins)
    overlay       : RGBA overlay array from load_overlay() (or None)
    overlay_alpha : overlay transparency
    save_log      : if True, tee stdout to summary.txt in out_dir

    Returns
    -------
    dict with keys:
        all_chs, labels, spike_counts, spike_rate_hz,
        bins_chs, bins_labels, bins_counts, bins_norm_rate,
        bin_edges, n_bins, bin_width,
        elec_csv_path, bin_csv_path
    """
    os.makedirs(out_dir, exist_ok=True)
    log = _Tee(os.path.join(out_dir, 'summary.txt')) if save_log else None

    xy         = rec.xy
    gc         = rec.gc
    gr         = rec.gr
    sp_ch      = rec.sp_ch
    duration_s = rec.duration_s

    src_ch   = axis.src_ch
    tgt_ch   = axis.tgt_ch
    d_along  = axis.d_along
    d_perp   = axis.d_perp
    d_lo     = axis.d_lo
    d_hi     = axis.d_hi
    axis_len = axis.axis_len
    cands    = axis.cands

    # Electrode list: SRC + sorted intermediates + TGT
    sorted_cands = sorted(cands.tolist(), key=lambda c: d_along[c])
    all_chs = np.array([src_ch] + sorted_cands + [tgt_ch])
    labels  = ['SRC'] + [f'C{i+1}' for i in range(len(sorted_cands))] + ['TGT']

    # Spike counts
    spike_counts  = {int(ch): int(np.sum(sp_ch == ch)) for ch in all_chs}
    spike_rate_hz = {ch: spike_counts[ch] / duration_s for ch in spike_counts}

    print(f"\nPrincipal axis: {axis_len:.1f} um  |  "
          f"band: +/-{axis.search_band_um:.1f} um  |  "
          f"{len(all_chs)} electrodes")
    print(f"\n{'label':>5}  {'ch':>5}  {'x(um)':>8}  {'y(um)':>8}  "
          f"{'d_along(um)':>12}  {'d%':>6}  {'d_perp(um)':>11}  "
          f"{'n_spikes':>9}  {'rate(Hz)':>9}")
    for ch, lbl in zip(all_chs, labels):
        d   = float(d_along[ch])
        pct = (d - d_lo) / axis_len * 100 if axis_len > 0 else 0.0
        print(f"  {lbl:>5}  {ch:5d}  {xy[ch,0]:8.1f}  {xy[ch,1]:8.1f}  "
              f"{d:12.1f}  {pct:5.1f}%  {d_perp[ch]:11.2f}  "
              f"{spike_counts[ch]:9d}  {spike_rate_hz[ch]:9.3f}")

    # Distance bins
    bin_width = group_frac * axis_len
    n_bins    = int(np.ceil(1.0 / group_frac))
    bin_edges = [d_lo + i * bin_width for i in range(n_bins + 1)]

    def _bin_index(d):
        idx = int((d - d_lo) / bin_width) if bin_width > 0 else 0
        return min(max(idx, 0), n_bins - 1)

    bins_chs    = [[] for _ in range(n_bins)]
    bins_labels = [[] for _ in range(n_bins)]
    bins_counts = [0  for _ in range(n_bins)]

    for ch, lbl in zip(all_chs, labels):
        b = _bin_index(float(d_along[ch]))
        bins_chs[b].append(int(ch))
        bins_labels[b].append(lbl)
        bins_counts[b] += spike_counts[int(ch)]

    bins_norm_rate = [
        (bins_counts[b] / len(bins_chs[b]) / duration_s)
        if (len(bins_chs[b]) > 0 and duration_s > 0) else 0.0
        for b in range(n_bins)
    ]

    print(f"\nBin width: {group_frac*100:.0f}% of axis = {bin_width:.1f} um -> {n_bins} bins")
    for b in range(n_bins):
        e_lo   = bin_edges[b]; e_hi = bin_edges[b+1]
        pct_lo = (e_lo - d_lo) / axis_len * 100 if axis_len > 0 else 0.0
        pct_hi = min((e_hi - d_lo) / axis_len * 100, 100.0) if axis_len > 0 else 0.0
        print(f"  Bin {b+1}  {pct_lo:4.0f}%-{pct_hi:4.0f}%  "
              f"n={len(bins_chs[b])}  spikes={bins_counts[b]}  "
              f"norm_rate={bins_norm_rate[b]:.4f} Hz  {bins_labels[b]}")

    # Figures
    cmap_bins = plt.cm.plasma
    norm_bins = plt.Normalize(0, max(n_bins - 1, 1))
    RADIUS_GU = axis.search_band_um / PITCH

    src_gu_f = xy[src_ch, 0] / PITCH
    src_gr_f = xy[src_ch, 1] / PITCH
    tgt_gu_f = xy[tgt_ch, 0] / PITCH
    tgt_gr_f = xy[tgt_ch, 1] / PITCH

    # Figure 1: electrode map
    fig1, ax1 = plt.subplots(figsize=(14, 9))
    if overlay is not None:
        ax1.imshow(overlay, extent=[0, GRID_COLS, GRID_ROWS, 0],
                   origin='upper', aspect='auto', alpha=overlay_alpha, zorder=0)
    ax1.plot(gc, gr, '.', color='cornflowerblue', markersize=3,
             alpha=0.35, zorder=1, linestyle='none')
    ax1.plot([src_gu_f, tgt_gu_f], [src_gr_f, tgt_gr_f],
             color='white', lw=1.5, ls='--', alpha=0.7, zorder=5)
    for b in range(n_bins):
        col = cmap_bins(norm_bins(b))
        for ch in bins_chs[b]:
            if ch in (src_ch, tgt_ch):
                continue
            ax1.plot(xy[ch, 0] / PITCH, xy[ch, 1] / PITCH, 'o', color=col,
                     markersize=8, markeredgecolor='white', markeredgewidth=0.8, zorder=6)
    ax1.plot(src_gu_f, src_gr_f, 'o', color='lime', markersize=13,
             markeredgecolor='white', markeredgewidth=1.5, zorder=8)
    ax1.text(src_gu_f + RADIUS_GU * 1.1, src_gr_f, 'SRC',
             color='lime', fontsize=10, fontweight='bold', va='center', zorder=8)
    ax1.plot(tgt_gu_f, tgt_gr_f, 'o', color='tomato', markersize=13,
             markeredgecolor='white', markeredgewidth=1.5, zorder=8)
    ax1.text(tgt_gu_f + RADIUS_GU * 1.1, tgt_gr_f, 'TGT',
             color='tomato', fontsize=10, fontweight='bold', va='center', zorder=8)
    sm = plt.cm.ScalarMappable(cmap=cmap_bins, norm=norm_bins)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1, fraction=0.02, pad=0.01)
    cbar.set_label('Distance bin', fontsize=9)
    cbar.set_ticks(range(n_bins))
    cbar.set_ticklabels([
        f'Bin {b+1} ({(bin_edges[b]-d_lo)/axis_len*100:.0f}-'
        f'{min((bin_edges[b+1]-d_lo)/axis_len*100, 100):.0f}%)'
        for b in range(n_bins)], fontsize=7)
    ax1.set_title(
        f'Principal-axis spike count  |  SRC=ch{src_ch}  TGT=ch{tgt_ch}  '
        f'|  {len(all_chs)} electrodes  |  band=+/-{axis.search_band_um:.0f} um',
        fontsize=11)
    ax1.set_xlabel('Grid column (x17.5 um)'); ax1.set_ylabel('Grid row (x17.5 um)')
    ax1.grid(True, alpha=0.12)
    plt.tight_layout()
    map_path = os.path.join(out_dir, 'axis_spike_count_map.png')
    fig1.savefig(map_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved map: {map_path}")

    # Figure 2: scatter + bars + normalised rate
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 6))
    d_vals    = [float(d_along[ch]) for ch in all_chs]
    s_vals    = [spike_counts[int(ch)] for ch in all_chs]
    pt_colors = [cmap_bins(norm_bins(_bin_index(d))) for d in d_vals]

    ax_L = axes2[0]
    ax_L.scatter(d_vals, s_vals, c=pt_colors, s=70, zorder=5,
                 edgecolors='k', linewidths=0.5)
    ax_L.annotate('SRC', xy=(float(d_along[src_ch]), spike_counts[src_ch]),
                  xytext=(5, 5), textcoords='offset points', fontsize=8,
                  color='lime', fontweight='bold')
    ax_L.annotate('TGT', xy=(float(d_along[tgt_ch]), spike_counts[tgt_ch]),
                  xytext=(5, 5), textcoords='offset points', fontsize=8,
                  color='tomato', fontweight='bold')
    ax_L.set_xlabel('Distance along axis (um)', fontsize=10)
    ax_L.set_ylabel('Total spikes', fontsize=10)
    ax_L.set_title('Per-electrode spike count vs axis position', fontsize=10)
    ax_L.grid(True, alpha=0.2); ax_L.spines[['top','right']].set_visible(False)

    bin_colors     = [cmap_bins(norm_bins(b)) for b in range(n_bins)]
    bin_tick_labels = [
        f'Bin {b+1}\n({(bin_edges[b]-d_lo)/axis_len*100:.0f}-'
        f'{min((bin_edges[b+1]-d_lo)/axis_len*100,100):.0f}%)\nn={len(bins_chs[b])}'
        for b in range(n_bins)]

    ax_R  = axes2[1]
    bars  = ax_R.bar(range(n_bins), bins_counts, color=bin_colors, edgecolor='k', lw=0.6)
    ax_R.set_xticks(range(n_bins)); ax_R.set_xticklabels(bin_tick_labels, fontsize=8)
    ax_R.set_ylabel('Cumulative spikes', fontsize=10)
    ax_R.set_title(f'Spike count per bin  (bin={group_frac*100:.0f}% of {axis_len:.0f} um)', fontsize=10)
    ax_R.grid(True, alpha=0.2, axis='y'); ax_R.spines[['top','right']].set_visible(False)
    for b, bar in enumerate(bars):
        ax_R.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                  str(bins_counts[b]), ha='center', va='bottom', fontsize=9)

    ax_NR    = axes2[2]
    nr_bars  = ax_NR.bar(range(n_bins), bins_norm_rate, color=bin_colors, edgecolor='k', lw=0.6)
    ax_NR.set_xticks(range(n_bins)); ax_NR.set_xticklabels(bin_tick_labels, fontsize=8)
    ax_NR.set_ylabel('Norm. avg firing rate (Hz)', fontsize=10)
    ax_NR.set_title('Normalised avg rate per bin  [(spikes/n_elec)/duration]', fontsize=10)
    ax_NR.grid(True, alpha=0.2, axis='y'); ax_NR.spines[['top','right']].set_visible(False)
    for b, bar in enumerate(nr_bars):
        ax_NR.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                   f'{bins_norm_rate[b]:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(
        f'Principal-axis spike count  |  SRC=ch{src_ch}  TGT=ch{tgt_ch}\n'
        f'{os.path.basename(rec.h5_path)}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    bar_path = os.path.join(out_dir, 'axis_spike_count_bars.png')
    fig2.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar chart: {bar_path}")

    # CSV output
    elec_csv_path = os.path.join(out_dir, 'axis_spike_count_electrodes.csv')
    with open(elec_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'label', 'channel', 'x_um', 'y_um',
            'd_along_um', 'd_along_pct', 'd_perp_um',
            'bin', 'n_spikes', 'spike_rate_hz'])
        writer.writeheader()
        for ch, lbl in zip(all_chs, labels):
            d   = float(d_along[ch])
            pct = (d - d_lo) / axis_len * 100 if axis_len > 0 else 0.0
            b   = _bin_index(d)
            writer.writerow({
                'label':         lbl,
                'channel':       int(ch),
                'x_um':          round(float(xy[ch, 0]), 2),
                'y_um':          round(float(xy[ch, 1]), 2),
                'd_along_um':    round(d, 2),
                'd_along_pct':   round(pct, 2),
                'd_perp_um':     round(float(d_perp[ch]), 2),
                'bin':           b + 1,
                'n_spikes':      spike_counts[int(ch)],
                'spike_rate_hz': round(spike_counts[int(ch)] / duration_s, 4)
                                 if duration_s > 0 else 0,
            })
    print(f"Saved electrode CSV: {elec_csv_path}")

    bin_csv_path = os.path.join(out_dir, 'axis_spike_count_bins.csv')
    with open(bin_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'bin', 'd_lo_um', 'd_hi_um', 'd_lo_pct', 'd_hi_pct',
            'n_electrodes', 'channel_ids', 'labels',
            'n_spikes', 'spike_rate_hz', 'norm_avg_rate_hz'])
        writer.writeheader()
        for b in range(n_bins):
            e_lo   = bin_edges[b]; e_hi = bin_edges[b+1]
            pct_lo = (e_lo - d_lo) / axis_len * 100 if axis_len > 0 else 0.0
            pct_hi = min((e_hi - d_lo) / axis_len * 100, 100.0) if axis_len > 0 else 0.0
            writer.writerow({
                'bin':              b + 1,
                'd_lo_um':          round(e_lo, 2),
                'd_hi_um':          round(e_hi, 2),
                'd_lo_pct':         round(pct_lo, 1),
                'd_hi_pct':         round(pct_hi, 1),
                'n_electrodes':     len(bins_chs[b]),
                'channel_ids':      str(bins_chs[b]),
                'labels':           str(bins_labels[b]),
                'n_spikes':         bins_counts[b],
                'spike_rate_hz':    round(bins_counts[b] / duration_s, 4)
                                    if duration_s > 0 else 0,
                'norm_avg_rate_hz': round(bins_norm_rate[b], 6),
            })
    print(f"Saved bin CSV: {bin_csv_path}")

    if log:
        log.close()

    return dict(
        all_chs=all_chs, labels=labels,
        spike_counts=spike_counts, spike_rate_hz=spike_rate_hz,
        bins_chs=bins_chs, bins_labels=bins_labels,
        bins_counts=bins_counts, bins_norm_rate=bins_norm_rate,
        bin_edges=bin_edges, n_bins=n_bins, bin_width=bin_width,
        elec_csv_path=elec_csv_path, bin_csv_path=bin_csv_path,
    )


# ==========================================================================
# select_intermediates
# ==========================================================================

def select_intermediates(
    rec           : Recording,
    axis          : AxisSelection,
    n_inter       : int           = 3,
    manual_inter  : 'list | None' = None,
    link_min_ms   : float         = 0.5,
    link_max_ms   : float         = 1.5,
) -> list:
    """
    Score on-axis candidate electrodes by how often they fire between SRC
    and TGT across all paired SRC->TGT events, then return the top N.

    Parameters
    ----------
    rec          : Recording from load_recording()
    axis         : AxisSelection from select_axis()
    n_inter      : number of intermediate electrodes to select (auto mode)
    manual_inter : fixed list of channel indices — skips scoring if provided
    link_min_ms  : min SRC->TGT delay defining a paired event (ms)
    link_max_ms  : max SRC->TGT delay defining a paired event (ms)

    Returns
    -------
    list of intermediate channel indices, sorted by axis position
    """
    if manual_inter is not None:
        print(f"Using manual intermediate channels: {manual_inter}")
        return [int(c) for c in manual_inter]

    sp_times = rec.sp_times
    sp_ch    = rec.sp_ch
    fs       = rec.fs
    src_ch   = axis.src_ch
    tgt_ch   = axis.tgt_ch
    cands    = axis.cands
    d_along  = axis.d_along
    d_lo     = axis.d_lo
    d_hi     = axis.d_hi

    link_lo = int(link_min_ms / 1000 * fs)
    link_hi = int(link_max_ms / 1000 * fs)

    tgt_times   = sp_times[sp_ch == tgt_ch]
    src_evt_all = rec.spike_array[sp_ch == src_ch]

    all_pairs = [
        (int(sf), int(tgt_times[np.searchsorted(tgt_times, sf + link_lo, side='left')]))
        for sf in src_evt_all[:, 0]
        if np.searchsorted(tgt_times, sf + link_lo, side='left') <
           np.searchsorted(tgt_times, sf + link_hi, side='right')
    ]
    print(f"Total paired SRC->TGT events: {len(all_pairs)}")
    if len(all_pairs) == 0:
        raise ValueError(
            "No paired SRC->TGT events found. "
            "Adjust link_min_ms / link_max_ms or check channel selection."
        )

    cand_hits   = {}
    cand_counts = {}
    for c in cands:
        c_times     = sp_times[sp_ch == c]
        hits = spike_count = 0
        for sf, tgt_t in all_pairs:
            lo = np.searchsorted(c_times, sf,    side='right')
            hi = np.searchsorted(c_times, tgt_t, side='right')
            n  = hi - lo
            if n > 0:
                hits        += 1
                spike_count += n
        cand_hits[c]   = hits
        cand_counts[c] = spike_count

    print(f"\nOn-axis candidates ({len(cands)}):")
    print(f"  {'ch':>5}  {'x(um)':>8}  {'y(um)':>8}  {'along%':>7}  "
          f"{'hits':>6}  {'spikes':>8}")
    for c in sorted(cands, key=lambda c: d_along[c]):
        frac = (d_along[c] - d_lo) / (d_hi - d_lo) * 100 if d_hi > d_lo else 0
        print(f"  {c:>5d}  {rec.xy[c,0]:8.1f}  {rec.xy[c,1]:8.1f}  "
              f"{frac:6.1f}%  {cand_hits[c]:6d}  {cand_counts[c]:8d}")

    ideal_fracs  = np.linspace(d_lo, d_hi, n_inter + 2)[1:-1]
    cands_ranked = sorted(
        [(c, cand_hits[c]) for c in cands if cand_hits[c] > 0],
        key=lambda x: (-x[1],
                       min(abs(d_along[x[0]] - t) for t in ideal_fracs))
    )
    inter_chs = sorted([c for c, _ in cands_ranked[:n_inter]],
                       key=lambda c: d_along[c])

    print(f"\nSelected {len(inter_chs)} intermediate(s):")
    for i, c in enumerate(inter_chs):
        frac = (d_along[c] - d_lo) / (d_hi - d_lo) if d_hi > d_lo else 0
        print(f"  M{i+1}  ch={c}  ({rec.xy[c,0]:.0f},{rec.xy[c,1]:.0f}) um  "
              f"along={frac*100:.0f}%  hits={cand_hits[c]}/{len(all_pairs)}")

    return inter_chs
