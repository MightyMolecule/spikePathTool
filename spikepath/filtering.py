"""
spikepath/filtering.py

Raw data preparation for MaxWell HD-MEA recordings.

Provides chip geometry constants and all data-filtering utilities:
H5 mapping, grid position computation, heatmap loading/alignment,
interactive channel selection, spike detection, and refractory filtering.

Constants
---------
GRID_ROWS, GRID_COLS, PITCH

Functions
---------
load_h5_mapping(h5_path, mapping_path)                          -> x_arr, y_arr, n_channels, outlier_idx
compute_grid_positions(x_arr, y_arr, pitch, grid_rows, ...)     -> gc, gr, grid_positions
load_heatmap(npy_path)                                          -> np.ndarray
align_heatmap_to_grid(heatmap, grid_rows, grid_cols)            -> np.ndarray
select_channels_by_heatmap(grid_heatmap, ..., png_path)         -> list[int]
select_channels_along_line(selected_channels, ..., png_path)    -> list[int]
detect_spikes(h5_path, n_channels, outlier_idx, ...)            -> np.ndarray (N, 3)
refractory_filter(spike_array, refractory_ms, fs)               -> np.ndarray
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks
from PIL import Image


# MaxWell HD-MEA chip geometry
GRID_ROWS = 120
GRID_COLS  = 220
PITCH      = 17.5   # µm per electrode


# ==========================================================================
# load_h5_mapping
# ==========================================================================

def load_h5_mapping(h5_path, mapping_path="/data_store/data0000/settings/mapping"):
    """Load channel (x, y) positions from a MaxWell H5 file and remove the
    single furthest-from-centroid outlier.

    Parameters
    ----------
    h5_path      : path to .raw.h5 file
    mapping_path : internal HDF5 path to the mapping dataset

    Returns
    -------
    x_arr       : list of x positions in µm (outlier removed)
    y_arr       : list of y positions in µm (outlier removed)
    n_channels  : number of channels after outlier removal
    outlier_idx : index of the removed outlier in the original mapping
    """
    x_arr, y_arr = [], []
    with h5py.File(h5_path, "r") as f:
        for row in f[mapping_path][:]:
            x_arr.append(float(row["x"]))
            y_arr.append(float(row["y"]))

    all_xy      = np.column_stack([x_arr, y_arr])
    centroid    = all_xy.mean(axis=0)
    dists       = np.linalg.norm(all_xy - centroid, axis=1)
    outlier_idx = int(np.argmax(dists))
    print(f"Excluded outlier electrode {outlier_idx} "
          f"({x_arr[outlier_idx]:.0f}, {y_arr[outlier_idx]:.0f}), "
          f"{dists[outlier_idx]:.0f} µm from centroid")

    x_arr      = [x for i, x in enumerate(x_arr) if i != outlier_idx]
    y_arr      = [y for i, y in enumerate(y_arr) if i != outlier_idx]
    n_channels = len(x_arr)
    print(f"H5 recording has {n_channels} channels (sub-selection of full chip)")
    return x_arr, y_arr, n_channels, outlier_idx


# ==========================================================================
# compute_grid_positions
# ==========================================================================

def compute_grid_positions(x_arr, y_arr, pitch=17.5, grid_rows=120, grid_cols=220):
    """Convert channel µm positions to fractional grid coordinates.

    Parameters
    ----------
    x_arr, y_arr : channel positions in µm
    pitch        : electrode pitch in µm
    grid_rows, grid_cols : full chip dimensions in grid units

    Returns
    -------
    h5_grid_cols      : ndarray of fractional column positions
    h5_grid_rows      : ndarray of fractional row positions
    h5_grid_positions : set of (row, col) integer grid positions with channels
    """
    h5_grid_cols = np.array([x / pitch for x in x_arr])
    h5_grid_rows = np.array([y / pitch for y in y_arr])

    print(f"Estimated chip extent from H5 positions:")
    print(f"  x: {min(x_arr):.1f} – {max(x_arr):.1f} µm  "
          f"(cols {h5_grid_cols.min():.1f} – {h5_grid_cols.max():.1f})")
    print(f"  y: {min(y_arr):.1f} – {max(y_arr):.1f} µm  "
          f"(rows {h5_grid_rows.min():.1f} – {h5_grid_rows.max():.1f})")
    print(f"  Full chip: {grid_cols}x{grid_rows} = "
          f"{grid_cols * pitch:.0f}x{grid_rows * pitch:.0f} µm")

    h5_grid_positions = set()
    for ch_idx in range(len(x_arr)):
        gc = int(round(x_arr[ch_idx] / pitch))
        gr = int(round(y_arr[ch_idx] / pitch))
        if 0 <= gr < grid_rows and 0 <= gc < grid_cols:
            h5_grid_positions.add((gr, gc))
    print(f"H5 channels map to {len(h5_grid_positions)} grid positions on the full chip")

    return h5_grid_cols, h5_grid_rows, h5_grid_positions


# ==========================================================================
# load_heatmap
# ==========================================================================

def load_heatmap(npy_path):
    """Load an impedance heatmap numpy array.

    Parameters
    ----------
    npy_path : path to .npy heatmap file

    Returns
    -------
    heatmap : ndarray (2D or 3D)
    """
    heatmap = np.load(npy_path)
    print(f"Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
    print(f"  value range: [{heatmap.min():.2f}, {heatmap.max():.2f}]")
    return heatmap


# ==========================================================================
# align_heatmap_to_grid
# ==========================================================================

def align_heatmap_to_grid(heatmap, grid_rows=120, grid_cols=220):
    """Resize heatmap to match the chip grid.

    If heatmap is 3D (RGB image), converts to single-channel intensity first.

    Parameters
    ----------
    heatmap             : ndarray (2D or 3D)
    grid_rows, grid_cols : target chip grid dimensions

    Returns
    -------
    grid_heatmap : 2D float ndarray of shape (grid_rows, grid_cols)
    """
    if heatmap.ndim == 3:
        heatmap = np.dot(heatmap[..., :3], [0.299, 0.587, 0.114])

    img         = Image.fromarray(heatmap.astype(np.float32), mode='F')
    img_resized = img.resize((grid_cols, grid_rows), Image.LANCZOS)
    grid_heatmap = np.array(img_resized)

    print(f"Aligned heatmap to grid: {grid_heatmap.shape}")
    print(f"  aligned range: [{grid_heatmap.min():.2f}, {grid_heatmap.max():.2f}]")
    return grid_heatmap


# ==========================================================================
# select_channels_by_heatmap
# ==========================================================================

def select_channels_by_heatmap(grid_heatmap, h5_grid_cols, h5_grid_rows,
                                x_arr, y_arr, png_path,
                                grid_rows=120, grid_cols=220, pitch=17.5):
    """Interactive percentile-threshold selection over the recording area.

    Three-panel figure:
      1. PNG + all channels (blue)
      2. Cropped heatmap with threshold contour
      3. PNG + all channels (blue) + selected channels (red)

    Percentile is computed from heatmap values in the recording area only.

    Parameters
    ----------
    grid_heatmap             : aligned heatmap array from align_heatmap_to_grid()
    h5_grid_cols, h5_grid_rows : channel grid coordinates
    x_arr, y_arr             : channel µm positions
    png_path                 : path to impedance overlay image
    grid_rows, grid_cols     : chip grid dimensions
    pitch                    : electrode pitch in µm

    Returns
    -------
    selected_channels : list of channel indices above the chosen threshold
    """
    img       = Image.open(png_path).convert("RGB")
    img_array = np.array(img.resize((grid_cols, grid_rows), Image.LANCZOS))

    ch_grid_r = np.clip(np.round(h5_grid_rows).astype(int), 0, grid_rows - 1)
    ch_grid_c = np.clip(np.round(h5_grid_cols).astype(int), 0, grid_cols - 1)

    pad   = 3
    r_min = max(ch_grid_r.min() - pad, 0)
    r_max = min(ch_grid_r.max() + pad, grid_rows - 1)
    c_min = max(ch_grid_c.min() - pad, 0)
    c_max = min(ch_grid_c.max() + pad, grid_cols - 1)
    print(f"Recording footprint: rows [{r_min}, {r_max}], cols [{c_min}, {c_max}]")

    cropped_heatmap = grid_heatmap[r_min:r_max+1, c_min:c_max+1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    plt.subplots_adjust(bottom=0.22)
    crop_extent = [c_min, c_max+1, r_max+1, r_min]

    axes[0].imshow(img_array, extent=[0, grid_cols, grid_rows, 0], aspect="auto")
    axes[0].scatter(h5_grid_cols, h5_grid_rows, s=6, c='dodgerblue',
                    alpha=0.6, linewidths=0)
    axes[0].set_xlim(c_min, c_max+1); axes[0].set_ylim(r_max+1, r_min)
    axes[0].set_title(f"Impedance image + channels ({len(x_arr)})")
    axes[0].set_xlabel("Column"); axes[0].set_ylabel("Row")

    hm_im = axes[1].imshow(cropped_heatmap, extent=crop_extent,
                            aspect="auto", cmap="hot")
    axes[1].scatter(h5_grid_cols, h5_grid_rows, s=3, c='cyan',
                    alpha=0.3, linewidths=0)
    plt.colorbar(hm_im, ax=axes[1], fraction=0.046)
    contour_artist = [None]
    axes[1].set_title("Heatmap + threshold (recording area only)")
    axes[1].set_xlabel("Column"); axes[1].set_ylabel("Row")

    axes[2].imshow(img_array, extent=[0, grid_cols, grid_rows, 0], aspect="auto")
    axes[2].scatter(h5_grid_cols, h5_grid_rows, s=6, c='dodgerblue',
                    alpha=0.4, linewidths=0, label="all channels")
    sel_scatter = axes[2].scatter([], [], s=18, c='red', alpha=0.85,
                                   linewidths=0.3, edgecolors='yellow',
                                   label="selected", zorder=5)
    axes[2].set_xlim(c_min, c_max+1); axes[2].set_ylim(r_max+1, r_min)
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].set_title("Selected channels")
    axes[2].set_xlabel("Column"); axes[2].set_ylabel("Row")

    slider_ax  = fig.add_axes([0.15, 0.08, 0.55, 0.04])
    pct_slider = Slider(slider_ax, "Percentile threshold", 0, 100,
                        valinit=75, valstep=1)

    selected_channels = []

    def update(val):
        nonlocal selected_channels
        pct    = pct_slider.val
        thresh = np.percentile(cropped_heatmap, pct)

        heatmap_mask      = grid_heatmap >= thresh
        mask              = heatmap_mask[ch_grid_r, ch_grid_c]
        selected_channels = np.flatnonzero(mask).tolist()

        if contour_artist[0] is not None:
            for c in contour_artist[0].collections:
                c.remove()
        contour_artist[0] = axes[1].contour(
            cropped_heatmap, levels=[thresh],
            extent=crop_extent,
            colors='cyan', linewidths=1.5)
        axes[1].set_title(f"Heatmap — thresh={thresh:.2f}  (P{pct:.0f} of recording area)")

        if len(selected_channels) > 0:
            sel_scatter.set_offsets(
                np.column_stack([h5_grid_cols[mask], h5_grid_rows[mask]]))
        else:
            sel_scatter.set_offsets(np.empty((0, 2)))
        axes[2].set_title(f"Selected: {len(selected_channels)} channels")

        fig.canvas.draw_idle()

    pct_slider.on_changed(update)
    update(75)

    plt.suptitle("Heatmap-based channel selection — adjust slider, then close window",
                 fontsize=12, y=0.98)
    plt.show()

    print(f"\n--- Heatmap selection result ---")
    print(f"Percentile: {pct_slider.val:.0f}")
    print(f"Channels selected: {len(selected_channels)} / {len(x_arr)}")
    if selected_channels:
        sel_x = [x_arr[c] for c in selected_channels]
        sel_y = [y_arr[c] for c in selected_channels]
        print(f"  X range: [{min(sel_x):.1f}, {max(sel_x):.1f}] µm  "
              f"(span {max(sel_x)-min(sel_x):.1f} µm)")
        print(f"  Y range: [{min(sel_y):.1f}, {max(sel_y):.1f}] µm  "
              f"(span {max(sel_y)-min(sel_y):.1f} µm)")
    return selected_channels


# ==========================================================================
# select_channels_along_line
# ==========================================================================

def select_channels_along_line(selected_channels, h5_grid_cols, h5_grid_rows,
                                x_arr, y_arr, png_path,
                                grid_rows=120, grid_cols=220, pitch=17.5):
    """Draw a line across threshold-selected channels; channels within an
    adjustable perpendicular half-width are kept for flow analysis.

    Parameters
    ----------
    selected_channels        : channel indices from select_channels_by_heatmap()
    h5_grid_cols, h5_grid_rows : channel grid coordinates
    x_arr, y_arr             : channel µm positions
    png_path                 : path to impedance overlay image
    grid_rows, grid_cols     : chip grid dimensions
    pitch                    : electrode pitch in µm

    Returns
    -------
    line_channels : list of channel indices intersecting the chosen line band
    """
    img       = Image.open(png_path).convert("RGB")
    img_array = np.array(img.resize((grid_cols, grid_rows), Image.LANCZOS))

    sel_cols = h5_grid_cols[selected_channels]
    sel_rows = h5_grid_rows[selected_channels]

    pad       = 3
    ch_grid_r = np.round(h5_grid_rows).astype(int)
    ch_grid_c = np.round(h5_grid_cols).astype(int)
    r_min = max(ch_grid_r.min() - pad, 0)
    r_max = min(ch_grid_r.max() + pad, grid_rows - 1)
    c_min = max(ch_grid_c.min() - pad, 0)
    c_max = min(ch_grid_c.max() + pad, grid_cols - 1)

    # Window 1: click two endpoints
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.imshow(img_array, extent=[0, grid_cols, grid_rows, 0], aspect="auto")
    ax1.scatter(h5_grid_cols, h5_grid_rows, s=4, c='dodgerblue', alpha=0.3,
                linewidths=0, label="all channels")
    ax1.scatter(sel_cols, sel_rows, s=14, c='red', alpha=0.7,
                linewidths=0, label="threshold-selected")
    ax1.set_xlim(c_min, c_max+1); ax1.set_ylim(r_max+1, r_min)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title("Click TWO points to define a line through the channels")
    ax1.set_xlabel("Column"); ax1.set_ylabel("Row")

    points = plt.ginput(2, timeout=0)
    plt.close(fig1)

    if len(points) < 2:
        raise SystemExit("Need 2 points for line selection.")

    p1, p2   = sorted(points, key=lambda p: p[0])
    col1, row1 = p1
    col2, row2 = p2
    print(f"Line endpoints: ({col1:.1f}, {row1:.1f}) -> ({col2:.1f}, {row2:.1f})")

    line_vec = np.array([col2 - col1, row2 - row1])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        raise SystemExit("Degenerate line (zero length).")
    line_dir    = line_vec / line_len
    line_normal = np.array([-line_dir[1], line_dir[0]])

    def _perp_distance(cols, rows):
        return (cols - col1) * line_normal[0] + (rows - row1) * line_normal[1]

    def _along_distance(cols, rows):
        return (cols - col1) * line_dir[0] + (rows - row1) * line_dir[1]

    # Window 2: slider for half-width + confirm
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.28)
    ax2.imshow(img_array, extent=[0, grid_cols, grid_rows, 0], aspect="auto")
    ax2.scatter(h5_grid_cols, h5_grid_rows, s=4, c='dodgerblue', alpha=0.25,
                linewidths=0, label="all channels")
    ax2.scatter(sel_cols, sel_rows, s=10, c='red', alpha=0.5,
                linewidths=0, label="threshold-selected")
    ax2.plot([col1, col2], [row1, row2], 'c-', linewidth=1.5)
    ax2.plot([col1, col2], [row1, row2], 'cx', markersize=8)

    band_lines = [
        ax2.plot([], [], 'c--', linewidth=1, alpha=0.6)[0],
        ax2.plot([], [], 'c--', linewidth=1, alpha=0.6)[0],
    ]
    line_scatter = ax2.scatter([], [], s=30, c='lime', alpha=0.9,
                                linewidths=0.5, edgecolors='white',
                                label="line-selected", zorder=5)
    ax2.set_xlim(c_min, c_max+1); ax2.set_ylim(r_max+1, r_min)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlabel("Column"); ax2.set_ylabel("Row")

    slider_ax = fig2.add_axes([0.15, 0.12, 0.55, 0.04])
    hw_slider = Slider(slider_ax, "Half-width (grid)", 0.5, 20, valinit=3, valstep=0.5)

    line_channels = []

    def update_line(val):
        nonlocal line_channels
        hw    = hw_slider.val
        perp  = _perp_distance(sel_cols, sel_rows)
        along = _along_distance(sel_cols, sel_rows)
        margin  = 1.0
        in_band = (np.abs(perp) <= hw) & (along >= -margin) & (along <= line_len + margin)
        line_channels = [selected_channels[i] for i in np.flatnonzero(in_band)]

        if line_channels:
            lc_cols = h5_grid_cols[line_channels]
            lc_rows = h5_grid_rows[line_channels]
            line_scatter.set_offsets(np.column_stack([lc_cols, lc_rows]))
        else:
            line_scatter.set_offsets(np.empty((0, 2)))

        offset = hw * line_normal
        for sign, bl in zip([1, -1], band_lines):
            bl.set_data(
                [col1 + sign * offset[0], col2 + sign * offset[0]],
                [row1 + sign * offset[1], row2 + sign * offset[1]])

        band_um = hw * 2 * pitch
        ax2.set_title(f"Line selection: {len(line_channels)} channels  |  "
                      f"half-width={hw:.1f} grid = {band_um:.0f} µm band")
        fig2.canvas.draw_idle()

    hw_slider.on_changed(update_line)
    update_line(3)

    confirmed = [False]
    btn_ax     = fig2.add_axes([0.78, 0.12, 0.12, 0.04])
    confirm_btn = Button(btn_ax, "Confirm")

    def _on_confirm(event):
        confirmed[0] = True
        plt.close(fig2)

    confirm_btn.on_clicked(_on_confirm)
    plt.show()

    if not confirmed[0] or not line_channels:
        raise SystemExit("Line selection cancelled.")

    hw             = hw_slider.val
    band_width_um  = hw * 2 * pitch
    line_length_um = line_len * pitch
    line_angle_deg = np.degrees(np.arctan2(row2 - row1, col2 - col1))

    lc_cols = h5_grid_cols[line_channels]
    lc_rows = h5_grid_rows[line_channels]
    perp_um = _perp_distance(lc_cols, lc_rows) * pitch

    print(f"\n--- Line selection result ---")
    print(f"Channels along line: {len(line_channels)}")
    lx = [x_arr[c] for c in line_channels]
    ly = [y_arr[c] for c in line_channels]
    print(f"  X range: [{min(lx):.1f}, {max(lx):.1f}] µm  "
          f"(span {max(lx)-min(lx):.1f} µm)")
    print(f"  Y range: [{min(ly):.1f}, {max(ly):.1f}] µm  "
          f"(span {max(ly)-min(ly):.1f} µm)")
    print(f"  Line length: {line_length_um:.1f} µm  |  "
          f"Band width: {band_width_um:.1f} µm  (hw={hw:.1f} grid)")
    print(f"  Line angle: {line_angle_deg:.1f}°")
    print(f"  Perp distance from line center (µm):")
    print(f"    std={np.std(perp_um):.1f}  var={np.var(perp_um):.1f}  "
          f"max|d|={np.max(np.abs(perp_um)):.1f}  "
          f"mean={np.mean(perp_um):.1f}  median={np.median(perp_um):.1f}")
    return line_channels


# ==========================================================================
# detect_spikes
# ==========================================================================

def detect_spikes(h5_path, n_channels, outlier_idx,
                  fs=20000, gain=3.14, chunk_frames=200000, frame_range=None):
    """Run chunked spike detection with caching.

    Results are written to a .npy file alongside the H5 file and reloaded
    on subsequent calls.

    Parameters
    ----------
    h5_path       : path to .raw.h5 file
    n_channels    : number of channels (after outlier removal)
    outlier_idx   : outlier electrode index to exclude from traces
    fs            : sampling rate in Hz
    gain          : au -> µV conversion factor
    chunk_frames  : number of frames per processing chunk
    frame_range   : (start_frame, end_frame) tuple, or None for full recording

    Returns
    -------
    spike_array : ndarray (N, 3) — [frame, amplitude, channel_idx]
    """
    from spikeinterface.extractors import read_maxwell
    from spikeinterface.preprocessing import bandpass_filter, unsigned_to_signed

    spike_cache = h5_path.replace('.raw.h5', '_spikes.npy')

    if os.path.exists(spike_cache):
        print(f"Loading cached spikes from {spike_cache}")
        spike_array = np.load(spike_cache)
        print(f"Loaded {len(spike_array)} spikes.")
        return spike_array

    rec     = read_maxwell(h5_path)
    n_total = rec.get_num_frames()
    start   = frame_range[0] if frame_range else 0
    end     = frame_range[1] if frame_range else n_total
    rec_f   = bandpass_filter(unsigned_to_signed(rec),
                               freq_min=200, freq_max=6000, filter_order=3)

    chunk_files, n_chunks = [], 0
    for cs in range(start, end, chunk_frames):
        ce       = min(cs + chunk_frames, end)
        n_chunks += 1
        print(f"  Chunk {n_chunks}: {cs/fs:.1f}–{ce/fs:.1f} s")
        tr = rec_f.get_traces(start_frame=cs, end_frame=ce, return_in_uV=True) * gain
        tr = np.delete(tr, outlier_idx, axis=1)
        spks = []
        for idx in range(n_channels):
            sig  = tr[:, idx]
            rms  = np.sqrt(np.mean(sig ** 2))
            peaks, _ = find_peaks(-sig, height=3.6 * rms)
            for p in peaks:
                spks.append([float(p + cs), float(tr[p, idx]), float(idx)])
        del tr
        if spks:
            cf = spike_cache.replace('_spikes.npy', f'_spikes_tmp{n_chunks:04d}.npy')
            np.save(cf, np.array(spks, dtype=np.float32))
            chunk_files.append(cf)

    spike_array = np.concatenate([np.load(f) for f in chunk_files], axis=0)
    spike_array = spike_array[spike_array[:, 0].argsort()]
    np.save(spike_cache, spike_array)
    for f in chunk_files:
        os.remove(f)
    print(f"Saved {len(spike_array)} spikes to {spike_cache}")
    return spike_array


# ==========================================================================
# refractory_filter
# ==========================================================================

def refractory_filter(spike_array, refractory_ms=1.0, fs=20000):
    """Remove spikes on the same electrode within refractory_ms of each other.

    Parameters
    ----------
    spike_array    : ndarray (N, 3) — [frame, amplitude, channel_idx]
    refractory_ms  : refractory period in ms
    fs             : sampling rate in Hz

    Returns
    -------
    filtered spike_array (copy, same dtype)
    """
    refractory_frames = refractory_ms / 1000.0 * fs
    n_before = len(spike_array)
    keep     = np.ones(n_before, dtype=bool)
    for elec in np.unique(spike_array[:, 2]):
        idx = np.flatnonzero(spike_array[:, 2] == elec)
        isi = np.diff(spike_array[idx, 0])
        keep[idx[np.flatnonzero(isi < refractory_frames) + 1]] = False
    filtered = spike_array[keep]
    print(f"Refractory filter: {n_before} -> {len(filtered)} spikes")
    return filtered
