"""
spikepath/analysis.py

Waveform extraction, propagation analysis, and multi-recording statistics
for MaxWell HD-MEA principal-axis experiments.

Functions
---------
extract_waveforms(rec, axis, intermediates, ...)  -> dict
    Extract waveform snippets triggered on source (or paired SRC->TGT) events.

plot_waveforms(rec, axis, intermediates, waveform_data, ...)  -> Figure
    Stacked waveform traces with mean +/- SD and inset electrode map.

save_speed_csv(waveform_data, out_path)
    Write per-event propagation latency and speed to CSV.

concatenate_axis_csvs(base_dir, out_path)  -> pd.DataFrame
    Walk SpikeCountAxisTool directory structure and build a combined CSV.

compute_stats(df)  -> pd.DataFrame
    Mean +/- SEM of spike_rate_hz per channel_type x bin.

plot_firing_rate(df, out_path)  -> Figure
    Mean +/- SEM firing rate vs bin, coloured by channel type.

run_lme(df)  -> MixedLMResults
    Linear mixed-effects model: spike_rate_hz ~ channel_type * bin.
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D as _Line2D

from .filtering import GRID_ROWS, GRID_COLS, PITCH
from .selection import Recording, AxisSelection

# Default electrode colours; labels not listed fall back to coolwarm gradient
DEFAULT_COLORS = {
    'SRC': '#1f1f9e',
    'TGT': '#6e1f98',
    'M1':  '#d5306c',
    'M2':  '#ff5d38',
    'M3':  '#ffbd42',
}


# ==========================================================================
# extract_waveforms
# ==========================================================================

def extract_waveforms(
    rec           : Recording,
    axis          : AxisSelection,
    intermediates : list,
    mode          : str   = 'src_tgt',
    n_traces      : int   = 30,
    start_time_s  : float = 0.0,
    pre_ms        : float = 1.0,
    post_ms       : float = 2.5,
    link_min_ms   : float = 0.5,
    link_max_ms   : float = 1.5,
    gain          : float = 3.14,
) -> dict:
    """
    Extract waveform snippets for SRC, intermediates, and TGT.

    Parameters
    ----------
    rec           : Recording with rec_f loaded (load_raw=True)
    axis          : AxisSelection from select_axis()
    intermediates : list of intermediate channel indices from select_intermediates()
    mode          : 'src_tgt'  — only paired SRC->TGT events (default)
                    'src_only' — all source spikes as trigger
    n_traces      : number of consecutive events to extract
    start_time_s  : start position in the recording (seconds)
    pre_ms        : window before spike (ms)
    post_ms       : window after spike (ms)
    link_min_ms   : min SRC->TGT delay for 'src_tgt' mode (ms)
    link_max_ms   : max SRC->TGT delay for 'src_tgt' mode (ms)
    gain          : au -> µV conversion factor

    Returns
    -------
    dict with keys:
        arr_list, t_ms, axis_chs, axis_labels, axis_xy,
        valid_evts, valid_spk_t, n_frames, mode,
        pre_ms, post_ms, fs
    """
    if rec.rec_f is None:
        raise RuntimeError(
            "Raw recording not loaded. Re-run load_recording() with load_raw=True."
        )

    fs       = rec.fs
    sp_times = rec.sp_times
    sp_ch    = rec.sp_ch
    xy       = rec.xy
    src_ch   = axis.src_ch
    tgt_ch   = axis.tgt_ch
    d_along  = axis.d_along
    d_lo     = axis.d_lo
    d_hi     = axis.d_hi

    pre_f   = int(pre_ms  / 1000 * fs)
    post_f  = int(post_ms / 1000 * fs)
    win_f   = pre_f + post_f
    t_ms    = np.arange(win_f) / fs * 1000 - pre_ms
    link_lo = int(link_min_ms / 1000 * fs)
    link_hi = int(link_max_ms / 1000 * fs)
    n_frames = rec.rec_f.get_num_frames()

    inter_labels = [f'M{i+1}' for i in range(len(intermediates))]
    axis_chs     = [src_ch] + intermediates + [tgt_ch]
    axis_labels  = ['SRC'] + inter_labels + ['TGT']
    axis_xy      = [tuple(xy[c]) for c in axis_chs]

    def _snippet(spike_frame, ch):
        f0 = int(spike_frame) - pre_f
        f1 = f0 + win_f
        if f0 < 0 or f1 > n_frames:
            return None
        raw = rec.rec_f.get_traces(start_frame=f0, end_frame=f1,
                                    return_in_uV=True) * gain
        raw = np.delete(raw, rec.outlier_idx, axis=1)
        return raw[:, ch].copy()

    tgt_times   = sp_times[sp_ch == tgt_ch]
    inter_times = [sp_times[sp_ch == c] for c in intermediates]

    # Select events according to mode
    src_evt = rec.spike_array[sp_ch == src_ch]
    if mode == 'src_tgt':
        def _is_paired(sf):
            lo = np.searchsorted(tgt_times, sf + link_lo, side='left')
            hi = np.searchsorted(tgt_times, sf + link_hi, side='right')
            return lo < hi
        mask   = np.array([_is_paired(sf) for sf in src_evt[:, 0]], dtype=bool)
        events = src_evt[mask]
        print(f"Paired SRC->TGT events ({link_min_ms}-{link_max_ms} ms): {len(events)}")
    else:  # src_only
        events = src_evt
        print(f"Source-only mode: {len(events)} source spikes total")

    if len(events) == 0:
        raise ValueError(f"No events found for mode='{mode}'.")

    # Resolve start position
    target_frame = start_time_s * fs
    start_idx    = int(np.searchsorted(events[:, 0], target_frame))
    start_idx    = max(0, min(start_idx, len(events) - 1))
    end_idx      = min(start_idx + n_traces, len(events))
    chosen_evt   = events[start_idx:end_idx]
    print(f"Extracting {len(chosen_evt)} events "
          f"({chosen_evt[0,0]/fs:.3f}s - {chosen_evt[-1,0]/fs:.3f}s)")

    # Extract snippets
    snips_list  = [[] for _ in axis_chs]
    valid_evts  = []
    valid_spk_t = []

    for row in chosen_evt:
        sf = int(row[0])
        s  = _snippet(sf, src_ch)
        if s is None:
            continue

        tgt_t = None
        if mode == 'src_tgt':
            lo_idx = np.searchsorted(tgt_times, sf + link_lo, side='left')
            tgt_t  = int(tgt_times[lo_idx])

        evt_spk = [sf]
        for k in range(1, len(axis_chs) - 1):
            it  = inter_times[k - 1]
            idx = np.searchsorted(it, sf, side='right')
            if mode == 'src_tgt':
                evt_spk.append(int(it[idx]) if idx < len(it) and it[idx] < tgt_t else None)
            else:
                evt_spk.append(int(it[idx]) if idx < len(it) else None)
        evt_spk.append(int(tgt_t) if tgt_t is not None else None)

        valid_evts.append(row)
        valid_spk_t.append(evt_spk)
        snips_list[0].append(s)

        for k in range(1, len(axis_chs)):
            if 0 < k < len(axis_chs) - 1 and mode == 'src_tgt':
                it = inter_times[k - 1]
                if not (np.searchsorted(it, sf, side='right') <
                        np.searchsorted(it, tgt_t, side='right')):
                    snips_list[k].append(np.full(win_f, np.nan))
                    continue
            seg = _snippet(sf, axis_chs[k])
            snips_list[k].append(seg if seg is not None else np.full(win_f, np.nan))

    print(f"Extracted waveforms for {len(snips_list[0])} events across "
          f"{len(axis_chs)} electrodes.")

    return dict(
        arr_list    = [np.array(s) for s in snips_list],
        t_ms        = t_ms,
        axis_chs    = axis_chs,
        axis_labels = axis_labels,
        axis_xy     = axis_xy,
        valid_evts  = valid_evts,
        valid_spk_t = valid_spk_t,
        n_frames    = n_frames,
        mode        = mode,
        pre_ms      = pre_ms,
        post_ms     = post_ms,
        fs          = fs,
    )


# ==========================================================================
# plot_waveforms
# ==========================================================================

def plot_waveforms(
    rec              : Recording,
    axis             : AxisSelection,
    intermediates    : list,
    waveform_data    : dict,
    overlay          : 'np.ndarray | None' = None,
    out_path         : 'str | None'        = None,
    electrode_colors : 'dict | None'       = None,
    alpha            : float               = 0.25,
    lw_trace         : float               = 0.7,
    lw_mean          : float               = 2.2,
    dpi              : int                 = 300,
) -> plt.Figure:
    """
    Plot stacked waveform traces with mean +/- SD per electrode and an
    inset electrode position map.

    Parameters
    ----------
    rec              : Recording
    axis             : AxisSelection
    intermediates    : intermediate channel indices
    waveform_data    : dict returned by extract_waveforms()
    overlay          : RGBA overlay array (or None)
    out_path         : save path for PNG (or None to skip saving)
    electrode_colors : dict mapping label -> colour (falls back to DEFAULT_COLORS)
    alpha, lw_trace, lw_mean : trace plotting aesthetics
    dpi              : output resolution

    Returns
    -------
    matplotlib Figure
    """
    arr_list    = waveform_data['arr_list']
    t_ms        = waveform_data['t_ms']
    axis_chs    = waveform_data['axis_chs']
    axis_labels = waveform_data['axis_labels']
    axis_xy     = waveform_data['axis_xy']
    n_elecs     = len(axis_chs)
    n_paired    = len(arr_list[0])
    gc          = rec.gc
    gr          = rec.gr

    colors_map = {**DEFAULT_COLORS, **(electrode_colors or {})}
    d_along    = axis.d_along
    d_lo       = axis.d_lo
    d_hi       = axis.d_hi
    axis_fracs = [
        (d_along[c] - d_lo) / (d_hi - d_lo) if d_hi != d_lo else f
        for f, c in zip(np.linspace(0, 1, n_elecs), axis_chs)
    ]
    axis_cols = [colors_map.get(lbl, plt.cm.coolwarm(f))
                 for lbl, f in zip(axis_labels, axis_fracs)]
    axis_gcu  = [ex / PITCH for ex, ey in axis_xy]
    axis_gru  = [ey / PITCH for ex, ey in axis_xy]
    radius_gu = axis.search_band_um / PITCH

    t_start_s = waveform_data['valid_evts'][0][0]  / waveform_data['fs']
    t_end_s   = waveform_data['valid_evts'][-1][0] / waveform_data['fs']

    fig = plt.figure(figsize=(12, 6))
    gs  = fig.add_gridspec(n_elecs, 2, width_ratios=[3, 1], wspace=0.25, hspace=0.0)
    axes_w = [fig.add_subplot(gs[k, 0]) for k in range(n_elecs)]
    ax_m   = fig.add_subplot(gs[:, 1])
    for ax in axes_w[1:]:
        ax.sharex(axes_w[0])

    for k, (arr, col, lbl, ch, (ex, ey)) in enumerate(
            zip(arr_list, axis_cols, axis_labels, axis_chs, axis_xy)):
        ax = axes_w[k]
        for snip in arr:
            ax.plot(t_ms, snip, color=col, lw=lw_trace, alpha=alpha)
        if arr.size > 0:
            mean_a = np.nanmean(arr, axis=0)
            sd_a   = np.nanstd(arr,  axis=0)
            ax.fill_between(t_ms, mean_a - sd_a, mean_a + sd_a, color=col, alpha=0.18)
            ax.plot(t_ms, mean_a, color=col, lw=lw_mean, zorder=5)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                lo, hi = float(finite.min()), float(finite.max())
                margin = (hi - lo) * 0.20
                ax.set_ylim(lo - margin, hi + margin)
        ax.axvline(0, color='gray', ls='--', lw=0.9, alpha=0.7)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.grid(True, alpha=0.15, axis='x')
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.text(-0.01, 0.5, f'{lbl}\n({ex:.0f},{ey:.0f})um',
                transform=ax.transAxes, color=col, fontsize=9,
                ha='right', va='center', fontweight='bold', clip_on=False)
        if k == 0:
            mode_str = waveform_data['mode'].replace('_', '-')
            ax.set_title(
                f'Principal-axis waveform stack  [{mode_str}]  '
                f'{n_paired} events  ({t_start_s:.2f}-{t_end_s:.2f} s)\n'
                f'{os.path.basename(rec.h5_path)}', fontsize=10)
        if k < n_elecs - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time relative to source spike (ms)', fontsize=10)

    # Inset electrode map
    if overlay is not None:
        ax_m.imshow(overlay, extent=[0, GRID_COLS, GRID_ROWS, 0],
                    origin='upper', aspect='auto', alpha=0.7, zorder=0)
    ax_m.plot(gc, gr, '.', color='cornflowerblue', markersize=2.5,
              alpha=0.6, zorder=1, linestyle='none')
    ax_m.plot([axis_gcu[0], axis_gcu[-1]], [axis_gru[0], axis_gru[-1]],
              color='blue', lw=1.0, ls='--', alpha=0.6, zorder=2)
    for gcu, gru, lbl, col in zip(axis_gcu, axis_gru, axis_labels, axis_cols):
        ax_m.plot(gcu, gru, 'o', color=col, markersize=7,
                  markeredgecolor='blue', markeredgewidth=1.0, zorder=4)
        ax_m.text(gcu + radius_gu * 0.9, gru, lbl, color=col, fontsize=7,
                  fontweight='bold', va='center', zorder=5)
    ax_m.set_aspect('equal')
    ax_m.set_xlabel('Grid col', fontsize=8)
    ax_m.set_ylabel('Grid row', fontsize=8)
    ax_m.set_title('Electrode positions', fontsize=9)
    ax_m.tick_params(labelsize=7)
    ax_m.grid(True, alpha=0.15)

    plt.suptitle(
        f'5-electrode principal-axis overlay  [{waveform_data["mode"]}]  '
        f'{os.path.basename(rec.h5_path)}',
        fontsize=11, fontweight='bold')
    plt.tight_layout()

    # Amplitude scalebar
    valid_arrs = [a for a in arr_list if a.size > 0 and np.any(np.isfinite(a))]
    if valid_arrs:
        glob_hi   = max(float(np.nanmax(a[np.isfinite(a)])) for a in valid_arrs)
        glob_lo   = min(float(np.nanmin(a[np.isfinite(a)])) for a in valid_arrs)
        src_ylim  = axes_w[0].get_ylim()
        src_pos   = axes_w[0].get_position()
        last_pos  = axes_w[-1].get_position()
        frac_per_uv = src_pos.height / (src_ylim[1] - src_ylim[0])
        sb_h    = (glob_hi - glob_lo) * frac_per_uv
        sb_x    = last_pos.x1 + 0.02
        sb_mid  = (last_pos.y0 + last_pos.y1) / 2
        tw      = 0.004
        fig.add_artist(_Line2D([sb_x, sb_x], [sb_mid - sb_h/2, sb_mid + sb_h/2],
                       transform=fig.transFigure, color='k', lw=2,
                       clip_on=False, zorder=10))
        for yt in (sb_mid + sb_h/2, sb_mid - sb_h/2):
            fig.add_artist(_Line2D([sb_x - tw, sb_x + tw], [yt, yt],
                           transform=fig.transFigure, color='k', lw=2,
                           clip_on=False, zorder=10))
        fig.text(sb_x + tw + 0.003, sb_mid + sb_h/2,
                 f'+{glob_hi:.0f} uV', fontsize=8, va='center',
                 ha='left', transform=fig.transFigure)
        fig.text(sb_x + tw + 0.003, sb_mid - sb_h/2,
                 f'{glob_lo:.0f} uV', fontsize=8, va='center',
                 ha='left', transform=fig.transFigure)

    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved waveform plot: {out_path}")

    return fig


# ==========================================================================
# save_speed_csv
# ==========================================================================

def save_speed_csv(waveform_data: dict, out_path: str):
    """
    Write per-event propagation latency and speed to a CSV file.

    Parameters
    ----------
    waveform_data : dict returned by extract_waveforms()
    out_path      : destination CSV path
    """
    arr_list    = waveform_data['arr_list']
    axis_labels = waveform_data['axis_labels']
    axis_chs    = waveform_data['axis_chs']
    axis_xy     = waveform_data['axis_xy']
    valid_evts  = waveform_data['valid_evts']
    valid_spk_t = waveform_data['valid_spk_t']
    fs          = waveform_data['fs']
    pre_ms      = waveform_data['pre_ms']
    post_ms     = waveform_data['post_ms']

    def _fmt(v):
        return '' if (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)

    seg_pairs = list(zip(range(len(axis_labels) - 1),
                         range(1, len(axis_labels))))
    rows = []

    for evt_i, (row_evt, spk_t) in enumerate(zip(valid_evts, valid_spk_t)):
        sf      = int(row_evt[0])
        t_src_s = sf / fs

        spk_del_ms = []
        spk_abs_s  = []
        for k in range(len(axis_labels)):
            spk = spk_t[k]
            if spk is None:
                spk_del_ms.append(np.nan); spk_abs_s.append(np.nan)
            else:
                spk_del_ms.append((spk - sf) / fs * 1000.0)
                spk_abs_s.append(spk / fs)

        # Enforce monotonic SRC < inter < TGT
        src_del = spk_del_ms[0]; tgt_del = spk_del_ms[-1]
        for k in range(1, len(axis_chs) - 1):
            d = spk_del_ms[k]
            if np.isnan(d) or np.isnan(src_del) or np.isnan(tgt_del):
                spk_del_ms[k] = np.nan; spk_abs_s[k] = np.nan
            elif not (src_del < d < tgt_del):
                spk_del_ms[k] = np.nan; spk_abs_s[k] = np.nan

        hi_uv, lo_uv = [], []
        for k in range(len(axis_labels)):
            snip = arr_list[k][evt_i] if evt_i < len(arr_list[k]) else None
            if snip is None or not np.any(np.isfinite(snip)):
                hi_uv.append(np.nan); lo_uv.append(np.nan)
            else:
                hi_uv.append(float(np.nanmax(snip)))
                lo_uv.append(float(np.nanmin(snip)))

        row = {
            'event_idx':        evt_i,
            'source_spike_t_s': round(t_src_s, 6),
            'trace_start_t_s':  round(t_src_s - pre_ms / 1000, 6),
            'trace_end_t_s':    round(t_src_s + post_ms / 1000, 6),
        }
        for k, (lbl, ch, (ex, ey)) in enumerate(
                zip(axis_labels, axis_chs, axis_xy)):
            row[f'ch_{lbl}']          = ch
            row[f'x_{lbl}_um']        = round(ex, 1)
            row[f'y_{lbl}_um']        = round(ey, 1)
            row[f'spk_del_{lbl}_ms']  = _fmt(spk_del_ms[k])
            row[f'spk_abs_{lbl}_s']   = _fmt(spk_abs_s[k])
            row[f'high_amp_{lbl}_uV'] = _fmt(hi_uv[k])
            row[f'low_amp_{lbl}_uV']  = _fmt(lo_uv[k])

        for k_fr, k_to in seg_pairs:
            ex_fr, ey_fr = axis_xy[k_fr]
            ex_to, ey_to = axis_xy[k_to]
            lbl_fr = axis_labels[k_fr]; lbl_to = axis_labels[k_to]
            seg     = f'{lbl_fr}_{lbl_to}'
            dist_um = float(np.hypot(ex_to - ex_fr, ey_to - ey_fr))
            delay   = spk_del_ms[k_to] - spk_del_ms[k_fr]
            row[f'dist_{seg}_um']          = round(dist_um, 2)
            row[f'delay_{seg}_ms']         = '' if np.isnan(delay) else round(delay, 4)
            if np.isnan(delay) or delay == 0:
                row[f'speed_{seg}_um_per_ms'] = ''
                row[f'speed_{seg}_m_per_s']   = ''
            else:
                spd = dist_um / delay
                row[f'speed_{seg}_um_per_ms'] = round(spd, 3)
                row[f'speed_{seg}_m_per_s']   = round(spd / 1000.0, 6)
        rows.append(row)

    if rows:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved speed CSV: {out_path}  ({len(rows)} rows)")
    else:
        print("[warn] No rows to write to speed CSV.")


# ==========================================================================
# concatenate_axis_csvs
# ==========================================================================

def concatenate_axis_csvs(
    base_dir : str,
    out_path : 'str | None' = None,
) -> pd.DataFrame:
    """
    Walk the SpikeCountAxisTool directory structure and concatenate all
    axis_spike_count_electrodes.csv files into one DataFrame.

    Expected structure:
        base_dir/
            {microstructure}_{channel_type}/
                {channel#}_{direction}/
                    axis_spike_count_electrodes.csv

    Parameters
    ----------
    base_dir : path to the SpikeCountAxisTool directory
    out_path : if provided, write the combined CSV to this path

    Returns
    -------
    pd.DataFrame with columns:
        microstructure, channel_type, channel, direction,
        label, electrode, x_um, y_um, d_along_um, d_along_pct,
        d_perp_um, bin, n_spikes, spike_rate_hz
    """
    rows       = []
    fieldnames = None

    for ms_ch_type_dir in sorted(os.listdir(base_dir)):
        ms_ch_type_path = os.path.join(base_dir, ms_ch_type_dir)
        if not os.path.isdir(ms_ch_type_path):
            continue

        parts = ms_ch_type_dir.rsplit('_', 1)
        if len(parts) != 2:
            print(f"  Skipping unexpected folder: {ms_ch_type_dir}")
            continue
        microstructure, channel_type = parts

        for ch_dir in sorted(os.listdir(ms_ch_type_path)):
            ch_path  = os.path.join(ms_ch_type_path, ch_dir)
            if not os.path.isdir(ch_path):
                continue
            csv_path = os.path.join(ch_path, 'axis_spike_count_electrodes.csv')
            if not os.path.exists(csv_path):
                continue

            ch_parts = ch_dir.split('_', 1)
            if len(ch_parts) != 2:
                print(f"  Skipping unexpected channel folder: {ch_dir}")
                continue
            channel_raw, direction = ch_parts
            channel = channel_raw.lstrip('ch') if channel_raw.startswith('ch') else channel_raw

            with open(csv_path, newline='') as f:
                reader     = csv.DictReader(f)
                csv_fields = ['electrode' if field == 'channel' else field
                              for field in reader.fieldnames]
                if fieldnames is None:
                    fieldnames = ['microstructure', 'channel_type', 'channel',
                                  'direction'] + csv_fields
                for row in reader:
                    renamed = {('electrode' if k == 'channel' else k): v
                               for k, v in row.items()}
                    rows.append({
                        'microstructure': microstructure,
                        'channel_type':   channel_type,
                        'channel':        channel,
                        'direction':      direction,
                        **renamed,
                    })

            print(f"  Read {ms_ch_type_dir}/{ch_dir}  ({len(rows)} rows total)")

    if not rows:
        print("No CSV files found — check base_dir.")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=fieldnames)
    df['bin']           = df['bin'].astype(int)
    df['spike_rate_hz'] = df['spike_rate_hz'].astype(float)

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"\nSaved combined CSV ({len(df)} rows): {out_path}")

    return df


# ==========================================================================
# compute_stats
# ==========================================================================

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and SEM of spike_rate_hz grouped by channel_type and bin.

    Parameters
    ----------
    df : DataFrame from concatenate_axis_csvs()

    Returns
    -------
    pd.DataFrame with columns: channel_type, bin, mean, sem
    """
    stats_df = (df.groupby(['channel_type', 'bin'])['spike_rate_hz']
                  .agg(mean='mean', sem=lambda x: x.sem())
                  .reset_index())
    return stats_df


# ==========================================================================
# plot_firing_rate
# ==========================================================================

def plot_firing_rate(
    df       : pd.DataFrame,
    out_path : 'str | None' = None,
    figsize  : tuple        = (9, 5),
) -> plt.Figure:
    """
    Plot mean +/- SEM firing rate vs bin position, coloured by channel type.

    Parameters
    ----------
    df       : DataFrame from concatenate_axis_csvs()
    out_path : save path for PNG (or None to skip saving)
    figsize  : figure dimensions

    Returns
    -------
    matplotlib Figure
    """
    stats_df      = compute_stats(df)
    channel_types = sorted(df['channel_type'].unique())
    bins          = sorted(df['bin'].unique())
    colors        = plt.cm.tab10(np.linspace(0, 1, len(channel_types)))

    fig, ax = plt.subplots(figsize=figsize)
    for ct, color in zip(channel_types, colors):
        sub = stats_df[stats_df['channel_type'] == ct].sort_values('bin')
        ax.errorbar(sub['bin'], sub['mean'], yerr=sub['sem'],
                    label=ct, color=color,
                    marker='o', markersize=6,
                    linewidth=1.8, capsize=4, capthick=1.2)

    ax.set_xlabel('Bin (position along principal axis)', fontsize=12)
    ax.set_ylabel('Mean firing rate +/- SEM (Hz)', fontsize=12)
    ax.set_title('Average firing rate vs bin position by channel type', fontsize=13)
    ax.set_xticks(bins)
    ax.legend(title='Channel type', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved firing rate plot: {out_path}")

    return fig


# ==========================================================================
# run_lme
# ==========================================================================

def run_lme(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model:

        spike_rate_hz ~ channel_type * bin  +  (1 | microstructure)

    bin and channel_type are treated as categorical so each level gets its
    own coefficient.  microstructure is the random intercept grouping.

    Parameters
    ----------
    df : DataFrame from concatenate_axis_csvs()

    Returns
    -------
    statsmodels MixedLMResults object — call .summary() to print the full table
    """
    df = df.copy()
    df['bin_cat']          = pd.Categorical(df['bin'])
    df['channel_type_cat'] = pd.Categorical(df['channel_type'])

    model  = smf.mixedlm(
        'spike_rate_hz ~ C(channel_type_cat) * C(bin_cat)',
        data   = df,
        groups = df['microstructure'],
    )
    result = model.fit(reml=True)

    print("=" * 60)
    print("Linear Mixed Effects Model")
    print("  spike_rate_hz ~ channel_type * bin  |  (1 | microstructure)")
    print("=" * 60)
    print(result.summary())

    return result
