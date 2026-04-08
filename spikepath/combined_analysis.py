"""
spikepath/combined_analysis.py

Population-level axis analysis for multi-recording SpikePath datasets.
Compares early vs late spike rates along the principal axis across channels
and microstructures using Mann-Whitney U with Benjamini-Hochberg FDR correction.

Functions
---------
detect_group_col(df)                                              -> str
load_combined_data(csv_path)                                      -> (df, group_col, list[str])
get_segment_rates(df, group_col, ms, early_pct, late_pct)        -> list[dict]
compute_segment_stats(df, group_col, ms, early_pct, late_pct)    -> pd.DataFrame
plot_segment_boxplots(df, group_col, ms, out_dir, ...)           -> Figure
plot_significant_channels_vs_distance(df, group_col, ms, ...)    -> Figure | None
plot_significance_fraction(df, group_col, microstructures, ...)  -> Figure
plot_sig_channels_scatter(df, group_col, microstructures, ...)   -> Figure | None
plot_representative_channels(df, group_col, rep_channels, ...)   -> None
run_combined_analysis(csv_path, out_dir, ...)                    -> None
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, linregress
from statsmodels.stats.multitest import multipletests


EARLY_PCT_DEFAULT = 25
LATE_PCT_DEFAULT  = 75


# ==========================================================================
# Helpers
# ==========================================================================

def detect_group_col(df):
    """Auto-detect the microstructure grouping column from the DataFrame."""
    for col in ('dir', 'ctl', 'microstructure'):
        if col in df.columns:
            return col
    raise ValueError(f"No grouping column found in DataFrame. Got: {list(df.columns)}")


def _detect_labels(df, group_col):
    """Return (channel_types, ctl_label, dir_label) from the DataFrame."""
    channel_types = sorted(df['channel_type'].unique())
    ctl_label = next((c for c in channel_types if 'ctl' in c.lower()), channel_types[0])
    dir_label = next((c for c in channel_types if 'dir' in c.lower()), channel_types[-1])
    return channel_types, ctl_label, dir_label


def _sig_stars(p):
    if np.isnan(p):  return ''
    if p < 0.001:    return '***'
    if p < 0.01:     return '**'
    if p < 0.05:     return '*'
    return 'ns'


def _fdr_correct(p_raw):
    """Apply BH FDR correction to a list of p-values (NaN entries preserved)."""
    valid_mask = [not np.isnan(p) for p in p_raw]
    p_fdr = [np.nan] * len(p_raw)
    if any(valid_mask):
        valid_p = [p for p, v in zip(p_raw, valid_mask) if v]
        _, corrected, _, _ = multipletests(valid_p, method='fdr_bh')
        j = 0
        for i, v in enumerate(valid_mask):
            if v:
                p_fdr[i] = corrected[j]; j += 1
    return p_fdr


# ==========================================================================
# Data loading
# ==========================================================================

def load_combined_data(csv_path):
    """
    Load the combined spike axis CSV produced by concatenate_axis_csvs().

    Returns
    -------
    df              : filtered DataFrame (spike_rate_hz > 0)
    group_col       : auto-detected grouping column name
    microstructures : sorted list of unique microstructure names
    """
    df = pd.read_csv(csv_path)
    df['spike_rate_hz'] = df['spike_rate_hz'].astype(float)
    df['d_along_pct']   = df['d_along_pct'].astype(float)
    df = df[df['spike_rate_hz'] > 0].reset_index(drop=True)
    group_col       = detect_group_col(df)
    microstructures = sorted(df[group_col].unique())
    return df, group_col, microstructures


# ==========================================================================
# Segment rate extraction
# ==========================================================================

def get_segment_rates(df, group_col, ms,
                      early_pct=EARLY_PCT_DEFAULT, late_pct=LATE_PCT_DEFAULT):
    """
    For a single microstructure, return early and late spike_rate_hz per channel.

    Returns a list of dicts: [{'channel': ch, 'early': array, 'late': array}, ...]
    """
    ms_df = df[df[group_col].str.contains(ms, na=False)].dropna(
                subset=['d_along_pct', 'spike_rate_hz'])
    results = []
    for ch in sorted(ms_df['channel'].unique()):
        ch_df = ms_df[ms_df['channel'] == ch]
        early = ch_df[ch_df['d_along_pct'] < early_pct]['spike_rate_hz'].values
        late  = ch_df[ch_df['d_along_pct'] >= late_pct]['spike_rate_hz'].values
        results.append({'channel': ch, 'early': early, 'late': late})
    return results


# ==========================================================================
# Statistical testing
# ==========================================================================

def compute_segment_stats(df, group_col, ms,
                          early_pct=EARLY_PCT_DEFAULT, late_pct=LATE_PCT_DEFAULT):
    """
    Run Mann-Whitney U (early vs late) for every channel in a microstructure,
    with Benjamini-Hochberg FDR correction across all channels.

    Returns
    -------
    pd.DataFrame with columns: channel, channel_type, p_raw, p_fdr, significant
    """
    ct_lookup = df[[group_col, 'channel', 'channel_type']].drop_duplicates()
    segments  = get_segment_rates(df, group_col, ms, early_pct, late_pct)
    if not segments:
        return pd.DataFrame()

    p_raw = []
    for seg in segments:
        if len(seg['early']) >= 3 and len(seg['late']) >= 3:
            _, p = mannwhitneyu(seg['early'], seg['late'], alternative='two-sided')
        else:
            p = np.nan
        p_raw.append(p)

    p_fdr = _fdr_correct(p_raw)

    rows = []
    for seg, pr, pf in zip(segments, p_raw, p_fdr):
        ch = seg['channel']
        ct = ct_lookup.loc[
            ct_lookup[group_col].str.contains(ms, na=False) &
            (ct_lookup['channel'] == ch), 'channel_type']
        ct_val = ct.iloc[0] if not ct.empty else 'unknown'
        rows.append({'channel': ch, 'channel_type': ct_val,
                     'p_raw': pr, 'p_fdr': pf,
                     'significant': (pf < 0.05) if not np.isnan(pf) else False})
    return pd.DataFrame(rows)


# ==========================================================================
# Plot 1 — Paired boxplots: early vs late per channel
# ==========================================================================

def plot_segment_boxplots(df, group_col, ms, out_dir,
                          early_pct=EARLY_PCT_DEFAULT, late_pct=LATE_PCT_DEFAULT):
    """
    Paired boxplots (early vs late) for every channel in one microstructure,
    with Mann-Whitney stars (FDR-corrected) above each pair.
    """
    segments = get_segment_rates(df, group_col, ms, early_pct, late_pct)
    n_ch = len(segments)
    if n_ch == 0:
        print(f"No data for {ms}")
        return

    p_raw = []
    for seg in segments:
        if len(seg['early']) >= 3 and len(seg['late']) >= 3:
            _, p = mannwhitneyu(seg['early'], seg['late'], alternative='two-sided')
        else:
            p = np.nan
        p_raw.append(p)
    p_fdr = _fdr_correct(p_raw)

    early_color = '#4878cf'
    late_color  = '#d62728'
    width = 0.10
    gap   = 0.05

    all_vals = np.concatenate([
        np.concatenate([seg['early'], seg['late']])
        for seg in segments if len(seg['early']) and len(seg['late'])
    ]) if any(len(seg['early']) and len(seg['late']) for seg in segments) else np.array([0, 1])

    y_max_all = all_vals.max()
    y_min_all = all_vals.min()
    y_span    = y_max_all - y_min_all

    fig, ax = plt.subplots(figsize=(n_ch * 1.4 + 2, 10))
    ax.set_ylim(y_min_all - y_span * 0.18, y_max_all + y_span * 0.18)

    x_ticks, x_labels = [], []

    for i, (seg, p) in enumerate(zip(segments, p_fdr)):
        x_early = i * (2 * width + gap)
        x_late  = x_early + width

        for vals, x, color in [(seg['early'], x_early, early_color),
                                (seg['late'],  x_late,  late_color)]:
            if len(vals) == 0:
                continue
            ax.boxplot(vals, positions=[x], widths=width * 0.85,
                       patch_artist=True, manage_ticks=False, showfliers=False,
                       boxprops=dict(facecolor=color, alpha=0.4),
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color=color),
                       capprops=dict(color=color))
            jitter = np.random.uniform(-width * 0.2, width * 0.2, size=len(vals))
            ax.scatter(x + jitter, vals, color=color, s=12, alpha=0.6,
                       linewidths=0, zorder=3)
            ax.text(x, y_min_all - y_span * 0.05, f'n={len(vals)}',
                    ha='center', va='top', fontsize=13, color='#444444')

        stars = _sig_stars(p)
        if stars:
            x_mid = x_early + width / 2
            y_ann = y_max_all + y_span * 0.06
            ax.plot([x_early, x_late], [y_ann, y_ann], color='black', linewidth=0.8)
            ax.text(x_mid, y_ann + y_span * 0.01, stars, ha='center', va='bottom',
                    fontsize=16, color='black' if stars != 'ns' else '#888888')

        x_ticks.append(x_early + width / 2)
        x_labels.append(f'ch{int(seg["channel"])}')

    ax.set_xlim(0 - width, (n_ch - 1) * (2 * width + gap) + 2 * width)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=15, rotation=45, ha='right')
    ax.set_ylabel('Spike rate (Hz)', fontsize=16)
    ax.set_title(f'Firing rate — first {early_pct}% vs last {100 - late_pct}%  |  {ms}',
                 fontsize=18)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=early_color, alpha=0.6,
                              label=f'First {early_pct}%'),
               plt.Rectangle((0, 0), 1, 1, facecolor=late_color, alpha=0.6,
                              label=f'Last {100 - late_pct}%')]
    ax.legend(handles=handles, fontsize=15, loc='lower right',
              bbox_to_anchor=(1, 1.01), borderaxespad=0, ncol=2)
    plt.tight_layout()

    safe = ms.replace(' ', '_').replace('/', '-')
    out  = os.path.join(out_dir, f'segment_boxplot_{safe}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)
    return fig


# ==========================================================================
# Plot 2 — Significant channels: spike rate vs distance along axis
# ==========================================================================

def plot_significant_channels_vs_distance(df, group_col, ms, out_dir,
                                          early_pct=EARLY_PCT_DEFAULT,
                                          late_pct=LATE_PCT_DEFAULT):
    """
    For channels that pass the early vs late Mann-Whitney test (p_fdr < 0.05),
    plot spike_rate_hz as a function of d_along_pct.
    """
    stats = compute_segment_stats(df, group_col, ms, early_pct, late_pct)
    if stats.empty:
        print(f"No stats for {ms}")
        return None

    sig_channels = stats[stats['significant']]['channel'].tolist()
    if not sig_channels:
        print(f"No significant channels in {ms}")
        return None

    ms_df = df[df[group_col].str.contains(ms, na=False)].dropna(
                subset=['d_along_pct', 'spike_rate_hz'])

    bin_edges   = np.arange(0, 100 + early_pct, early_pct)
    bin_centers = bin_edges[:-1] + early_pct / 2
    colors      = plt.cm.tab20(np.linspace(0, 1, len(sig_channels)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for ch, color in zip(sig_channels, colors):
        ch_df = ms_df[ms_df['channel'] == ch].sort_values('d_along_pct')
        ct    = stats.loc[stats['channel'] == ch, 'channel_type'].iloc[0]
        p_val = stats.loc[stats['channel'] == ch, 'p_fdr'].iloc[0]
        p_str = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'

        ax.scatter(ch_df['d_along_pct'], ch_df['spike_rate_hz'],
                   color=color, s=14, alpha=0.4, linewidths=0)

        ch_df = ch_df.copy()
        ch_df['bin'] = pd.cut(ch_df['d_along_pct'], bins=bin_edges,
                               labels=bin_centers).astype(float)
        binned = ch_df.groupby('bin')['spike_rate_hz'].mean().dropna()
        ax.plot(binned.index, binned.values,
                color=color, linewidth=1.8, alpha=0.9, marker='o', markersize=4,
                label=f'ch{int(ch)} ({ct})  {p_str}')

    ax.axvspan(0, early_pct,  alpha=0.06, color='#4878cf',
               label=f'Early (<{early_pct}%)')
    ax.axvspan(late_pct, 100, alpha=0.06, color='#d62728',
               label=f'Late (≥{late_pct}%)')

    ax.set_xlabel('Distance along principal axis (%)', fontsize=12)
    ax.set_ylabel('Spike rate (Hz)', fontsize=12)
    ax.set_title(f'Significant channels — spike rate vs distance  |  {ms}\n'
                 f'({len(sig_channels)} of {len(stats)} channels, p_fdr < 0.05)',
                 fontsize=12)
    ax.legend(fontsize=12, ncol=2, title='Channel')
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    safe = ms.replace(' ', '_').replace('/', '-')
    out  = os.path.join(out_dir, f'sig_channels_vs_distance_{safe}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)
    return fig


# ==========================================================================
# Plot 3 — Fraction of significant channels per microstructure × channel type
# ==========================================================================

def plot_significance_fraction(df, group_col, microstructures, out_dir,
                                early_pct=EARLY_PCT_DEFAULT, late_pct=LATE_PCT_DEFAULT):
    """
    Bar chart showing the fraction of channels with significant early vs late
    firing rate change (p_fdr < 0.05), grouped by microstructure and channel type.
    """
    _, ctl_label, dir_label = _detect_labels(df, group_col)

    all_rows = []
    for ms in microstructures:
        stats = compute_segment_stats(df, group_col, ms, early_pct, late_pct)
        if stats.empty:
            continue
        for ct, grp in stats.groupby('channel_type'):
            n_total = len(grp)
            n_sig   = grp['significant'].sum()
            frac    = n_sig / n_total if n_total > 0 else 0
            all_rows.append({group_col: ms, 'channel_type': ct,
                             'frac_sig': frac, 'n_sig': n_sig, 'n_total': n_total})

    if not all_rows:
        print("No data for significance fraction plot.")
        return None

    frac_df = (pd.DataFrame(all_rows)
                 .sort_values(['channel_type', group_col])
                 .reset_index(drop=True))

    ct_colors = {ctl_label: '#4878cf', dir_label: '#d62728'}
    GAP = 1.2
    positions, bar_colors = [], []
    x, prev_ct = 0, None
    for _, row in frac_df.iterrows():
        if prev_ct is not None and row['channel_type'] != prev_ct:
            x += GAP
        positions.append(x)
        bar_colors.append(ct_colors.get(row['channel_type'], '#888888'))
        prev_ct = row['channel_type']
        x += 1

    fig_f, ax_f = plt.subplots(figsize=(max(8, len(frac_df) * 0.8 + GAP), 5))
    bars = ax_f.bar(positions, frac_df['frac_sig'],
                    color=bar_colors, alpha=0.75, edgecolor='white', linewidth=0.5)

    for bar, (_, row) in zip(bars, frac_df.iterrows()):
        ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{int(row['n_sig'])}/{int(row['n_total'])}",
                  ha='center', va='bottom', fontsize=8)

    ax_f.axhline(0.05, color='black', linewidth=1, linestyle='--', alpha=0.4,
                 label='p=0.05 reference')
    ax_f.set_xticks(positions)
    ax_f.set_xticklabels(frac_df[group_col], fontsize=8, rotation=45, ha='right')
    ax_f.set_ylabel('Fraction of channels  p < 0.05  (FDR)', fontsize=12)
    ax_f.set_title('Fraction of channels with significant early vs late firing rate change',
                   fontsize=13)
    ax_f.set_ylim(0, 1.1)
    ax_f.grid(True, alpha=0.2, axis='y')
    ax_f.spines[['top', 'right']].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ct_colors.get(ct, '#888'), alpha=0.75,
                              label=ct) for ct in sorted(ct_colors)]
    ax_f.legend(handles=handles, fontsize=10)
    plt.tight_layout()

    out = os.path.join(out_dir, 'fraction_significant_early_vs_late.png')
    fig_f.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig_f)
    return fig_f


# ==========================================================================
# Plot 4 — All significant channels: scatter + linear regression
# ==========================================================================

def plot_sig_channels_scatter(df, group_col, microstructures, out_dir,
                               early_pct=EARLY_PCT_DEFAULT, late_pct=LATE_PCT_DEFAULT):
    """
    For all channels that are significant across any microstructure, plot
    spike_rate_hz vs d_along_pct with a linear regression line per channel.
    Also prints a table of endpoint ratio metrics (regression, mean, and bin-based).
    """
    sig_frames = []
    for ms in microstructures:
        stats = compute_segment_stats(df, group_col, ms, early_pct, late_pct)
        if stats.empty:
            continue
        sig_chs = stats[stats['significant']]['channel'].tolist()
        ms_df = df[df[group_col].str.contains(ms, na=False)].dropna(
                    subset=['d_along_pct', 'spike_rate_hz'])
        for ch in sig_chs:
            ch_df = ms_df[ms_df['channel'] == ch].copy()
            ch_df['ms'] = ms
            sig_frames.append(ch_df)

    if not sig_frames:
        print("No significant channels found across any microstructure.")
        return None

    sig_all  = pd.concat(sig_frames, ignore_index=True)
    BIN_SIZE = early_pct

    # Print ratio table
    print(f"\n{'='*90}")
    print(f"{'MS':<12} {'ch':>4}  {'type':<6}  {'opt1(reg)':>10}  "
          f"{'opt2(means)':>12}  {'opt3(endpoints)':>16}")
    print(f"{'-'*90}")
    for ms in sorted(sig_all['ms'].unique()):
        for ch in sorted(sig_all[sig_all['ms'] == ms]['channel'].unique()):
            ch_data = sig_all[(sig_all['ms'] == ms) & (sig_all['channel'] == ch)]
            ct = ch_data['channel_type'].iloc[0]

            if len(ch_data) >= 3:
                slope, intercept, *_ = linregress(ch_data['d_along_pct'],
                                                   ch_data['spike_rate_hz'])
                x_min, x_max = ch_data['d_along_pct'].min(), ch_data['d_along_pct'].max()
                y_start = intercept + slope * x_min
                y_end   = intercept + slope * x_max
                opt1    = y_end / y_start if y_start != 0 else float('nan')
            else:
                opt1 = float('nan')

            early = ch_data[ch_data['d_along_pct'] < early_pct]['spike_rate_hz'].mean()
            late  = ch_data[ch_data['d_along_pct'] >= late_pct]['spike_rate_hz'].mean()
            opt2  = late / early if early and early != 0 else float('nan')

            first_bin = ch_data[ch_data['d_along_pct'] < BIN_SIZE]['spike_rate_hz'].mean()
            last_bin  = ch_data[ch_data['d_along_pct'] >= (100 - BIN_SIZE)]['spike_rate_hz'].mean()
            opt3      = last_bin / first_bin if first_bin and first_bin != 0 else float('nan')

            print(f"{ms:<12} {int(ch):>4}  {ct:<6}  {opt1:>10.3f}  "
                  f"{opt2:>12.3f}  {opt3:>16.3f}")
    print(f"{'='*90}\n")

    def ms_dominant_type(ms):
        types = sig_all[sig_all['ms'] == ms]['channel_type']
        return types.mode().iloc[0] if not types.empty else 'zzz'

    ms_list = sorted(sig_all['ms'].unique(),
                     key=lambda m: (0 if 'ctl' in ms_dominant_type(m).lower() else 1, m))
    n_ms = len(ms_list)

    fig_s, axes_s = plt.subplots(1, n_ms, figsize=(5 * n_ms, 5), sharey=True)
    if n_ms == 1:
        axes_s = [axes_s]

    for ax, ms in zip(axes_s, ms_list):
        ms_data  = sig_all[sig_all['ms'] == ms]
        channels = sorted(ms_data['channel'].unique())
        ch_colors = plt.cm.tab20(np.linspace(0, 1, max(len(channels), 1)))

        for ch, color in zip(channels, ch_colors):
            ch_data = ms_data[ms_data['channel'] == ch]
            ax.scatter(ch_data['d_along_pct'], ch_data['spike_rate_hz'],
                       s=10, alpha=0.3, linewidths=0, color=color)

            if len(ch_data) >= 3:
                slope, intercept, r, p, _ = linregress(ch_data['d_along_pct'],
                                                        ch_data['spike_rate_hz'])
                x_line = np.linspace(ch_data['d_along_pct'].min(),
                                      ch_data['d_along_pct'].max(), 200)
                p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
                ax.plot(x_line, intercept + slope * x_line, color=color, linewidth=1.8,
                        label=f'ch{int(ch)}  slope={slope:+.3f}  R²={r**2:.2f}  {p_str}')

        ax.legend(fontsize=7, ncol=1, title='Channel',
                  loc='upper center', bbox_to_anchor=(0.5, -0.22),
                  borderaxespad=0, frameon=True)
        ax.set_xlabel('Distance along principal axis (%)', fontsize=11)
        ax.set_title(f'{ms}\n(n={len(channels)} electrodes)', fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

    axes_s[0].set_ylabel('Spike rate (Hz)', fontsize=12)
    fig_s.suptitle('Significant channels — spike rate vs distance', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)

    out_s = os.path.join(out_dir, 'sig_channels_scatter_bestfit_all_ms.png')
    fig_s.savefig(out_s, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_s}")
    plt.close(fig_s)
    return fig_s


# ==========================================================================
# Plot 5 — Representative single-channel plots
# ==========================================================================

def plot_representative_channels(df, group_col, rep_channels, out_dir,
                                  early_pct=EARLY_PCT_DEFAULT, late_pct=LATE_PCT_DEFAULT):
    """
    Plot mean firing rate + linear regression for a set of representative channels.

    Parameters
    ----------
    rep_channels : list of (ms, channel, channel_type, color) tuples
                   e.g. [('P002275', 8, 'dir', '#d62728'), ...]
    """
    if not rep_channels:
        return

    bin_edges   = np.arange(0, 100 + early_pct, early_pct)
    bin_centers = bin_edges[:-1] + early_pct / 2

    # Shared y-axis: compute after Tukey fence outlier removal
    _rep_ymax = 0
    for ms, ch, ct, color in rep_channels:
        _d = df[df[group_col].str.contains(ms, na=False) &
                (df['channel'] == ch)].dropna(subset=['spike_rate_hz'])
        if _d.empty:
            continue
        q1, q3 = _d['spike_rate_hz'].quantile([0.25, 0.75])
        iqr = q3 - q1
        _d = _d[(_d['spike_rate_hz'] >= q1 - 1.25 * iqr) &
                (_d['spike_rate_hz'] <= q3 + 1.25 * iqr)]
        _rep_ymax = max(_rep_ymax, _d['spike_rate_hz'].max())
    _rep_ymax *= 1.05

    for ms, ch, ct, color in rep_channels:
        ch_data = df[df[group_col].str.contains(ms, na=False) &
                     (df['channel'] == ch)].dropna(subset=['d_along_pct', 'spike_rate_hz'])
        if ch_data.empty:
            print(f"WARNING: no data for {ms} ch{ch}")
            continue

        q1, q3 = ch_data['spike_rate_hz'].quantile([0.25, 0.75])
        iqr = q3 - q1
        ch_data = ch_data[(ch_data['spike_rate_hz'] >= q1 - 1.25 * iqr) &
                          (ch_data['spike_rate_hz'] <= q3 + 1.25 * iqr)]

        mean_early = ch_data[ch_data['d_along_pct'] < early_pct]['spike_rate_hz'].mean()
        mean_late  = ch_data[ch_data['d_along_pct'] >= late_pct]['spike_rate_hz'].mean()
        print(f"{ms} ch{ch} ({ct}):  mean FR first {early_pct}% = {mean_early:.3f} Hz  |  "
              f"mean FR last {100 - late_pct}% = {mean_late:.3f} Hz")

        ch_data = ch_data.copy()
        ch_data['bin'] = pd.cut(ch_data['d_along_pct'], bins=bin_edges,
                                 labels=bin_centers).astype(float)

        slope, intercept, r, p, _ = linregress(ch_data['d_along_pct'],
                                                ch_data['spike_rate_hz'])
        x_line = np.linspace(ch_data['d_along_pct'].min(),
                              ch_data['d_along_pct'].max(), 200)

        first_bin = ch_data[ch_data['d_along_pct'] < early_pct]['spike_rate_hz'].mean()
        last_bin  = ch_data[ch_data['d_along_pct'] >= (100 - early_pct)]['spike_rate_hz'].mean()
        opt3      = last_bin / first_bin if first_bin and first_bin != 0 else float('nan')

        fig_r, ax_r = plt.subplots(figsize=(9, 5))
        ax_r.scatter(ch_data['d_along_pct'], ch_data['spike_rate_hz'],
                     s=14, alpha=0.3, linewidths=0, color=color)
        ax_r.plot(x_line, intercept + slope * x_line,
                  color='black', linewidth=2, linestyle='--')
        ax_r.set_ylim(0, _rep_ymax)
        ax_r.set_xlabel('Distance along principal axis (%)', fontsize=12)
        ax_r.set_ylabel('Spike rate (Hz)', fontsize=12)
        ax_r.set_title(f'{ms}  ch{ch}  ({ct})\nEndpoint ratio (opt3): {opt3:.3f}', fontsize=13)
        ax_r.grid(True, alpha=0.2)
        ax_r.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

        out_r = os.path.join(out_dir, f'representative_{ms}_ch{ch}.png')
        fig_r.savefig(out_r, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_r}")
        plt.close(fig_r)


# ==========================================================================
# Top-level entry point
# ==========================================================================

def run_combined_analysis(csv_path, out_dir,
                          early_pct=EARLY_PCT_DEFAULT,
                          late_pct=LATE_PCT_DEFAULT,
                          rep_channels=None):
    """
    Run the full combined axis analysis pipeline.

    Parameters
    ----------
    csv_path     : path to combined_spike_axis.csv (from concatenate_axis_csvs)
    out_dir      : directory to write all output plots
    early_pct    : axis-position percentile threshold for the 'early' segment (default 25)
    late_pct     : axis-position percentile threshold for the 'late' segment (default 75)
    rep_channels : list of (ms, channel, channel_type, color) tuples for representative
                   channel plots, or None to skip
    """
    os.makedirs(out_dir, exist_ok=True)

    df, group_col, microstructures = load_combined_data(csv_path)
    channel_types, ctl_label, dir_label = _detect_labels(df, group_col)

    print(f"Grouping column  : {group_col}")
    print(f"Channel types    : {channel_types}")
    print(f"Microstructures  : {microstructures}")
    print(f"Early cutoff     : <{early_pct}%   Late cutoff: >={late_pct}%")
    print()

    for ms in microstructures:
        plot_segment_boxplots(df, group_col, ms, out_dir, early_pct, late_pct)
        plot_significant_channels_vs_distance(df, group_col, ms, out_dir, early_pct, late_pct)

    plot_significance_fraction(df, group_col, microstructures, out_dir, early_pct, late_pct)
    plot_sig_channels_scatter(df, group_col, microstructures, out_dir, early_pct, late_pct)

    if rep_channels:
        plot_representative_channels(df, group_col, rep_channels, out_dir, early_pct, late_pct)

    print(f"\nCombined analysis complete. Results in: {out_dir}")
