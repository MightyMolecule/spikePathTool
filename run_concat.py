#!/usr/bin/env python3
"""
run_concat.py

Multi-recording aggregation and combined axis analysis pipeline:
  1. Walk --base_dir for spikepath output folders and concatenate all
     axis_spike_count_electrodes.csv files into combined_spike_axis.csv
  2. Run combined analysis: early vs late firing rate comparison per channel,
     Mann-Whitney U with FDR correction, and summary plots

Usage
-----
python run_concat.py \\
    --base_dir /path/to/SpikeCountAxisTool \\
    [--out_dir  /path/to/output] \\
    [--early_pct 25] \\
    [--late_pct  75] \\
    [--rep "P002275,8,dir,#d62728"] \\
    [--rep "P001314,1,ctl,#1f77b4"] \\
    [--skip_concat]

--rep format: "recording_substring,channel_int,channel_type[,hex_color]"
  recording_substring : string matched against the grouping column (microstructure name)
  channel_int         : integer channel index
  channel_type        : channel type label (e.g. 'dir', 'ctl')
  hex_color           : optional hex color (e.g. '#d62728'); auto-assigned if omitted

--skip_concat
  Skip step 1 and use an existing combined_spike_axis.csv in --out_dir.

Output layout
-------------
<out_dir>/
├── combined_spike_axis.csv
└── combined_analysis/
    ├── segment_boxplot_<ms>.png
    ├── sig_channels_vs_distance_<ms>.png
    ├── fraction_significant_early_vs_late.png
    ├── sig_channels_scatter_bestfit_all_ms.png
    └── representative_<ms>_ch<N>.png    (only if --rep is provided)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spikepath


# ==========================================================================
# Argument parser
# ==========================================================================

_DEFAULT_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e',
                   '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def _parse_rep_args(rep_list):
    """
    Parse a list of --rep strings into (ms, channel, channel_type, color) tuples.

    Each string must be:  "recording_substring,channel,type[,#color]"
    """
    result = []
    for i, s in enumerate(rep_list or []):
        parts = [p.strip() for p in s.split(',')]
        if len(parts) < 3:
            sys.exit(
                f"[error] --rep requires at least 3 comma-separated fields "
                f"(recording,channel,type[,color]), got: {s!r}"
            )
        ms    = parts[0]
        try:
            ch = int(parts[1])
        except ValueError:
            sys.exit(f"[error] --rep channel must be an integer, got: {parts[1]!r}")
        ct    = parts[2]
        color = parts[3] if len(parts) >= 4 else _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        result.append((ms, ch, ct, color))
    return result or None


def _build_parser():
    p = argparse.ArgumentParser(
        description="SpikePath — multi-recording concatenation and combined axis analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    req = p.add_argument_group("required")
    req.add_argument(
        "--base_dir", required=True,
        help=(
            "Directory containing spikepath output folders structured as "
            "{microstructure}_{channel_type}/{channel}_{direction}/axis_spike_count_electrodes.csv"
        ),
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    paths = p.add_argument_group("paths")
    paths.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: <base_dir>/combined_output)",
    )
    paths.add_argument(
        "--skip_concat", action="store_true",
        help="Skip CSV concatenation and use an existing combined_spike_axis.csv in --out_dir",
    )

    # ── Analysis parameters ───────────────────────────────────────────────
    ana = p.add_argument_group("combined analysis")
    ana.add_argument(
        "--early_pct", type=float, default=25.0,
        help="Axis position percentile upper bound for the 'early' segment",
    )
    ana.add_argument(
        "--late_pct", type=float, default=75.0,
        help="Axis position percentile lower bound for the 'late' segment",
    )
    ana.add_argument(
        "--rep", action="append", metavar="MS,CH,TYPE[,COLOR]",
        help=(
            "Representative channel to plot individually. "
            "Format: 'recording_substring,channel_int,channel_type[,#hex_color]'. "
            "Repeat --rep to add multiple channels."
        ),
    )

    return p


# ==========================================================================
# Main
# ==========================================================================

def main():
    args   = _build_parser().parse_args()
    out_dir = args.out_dir or os.path.join(args.base_dir, "combined_output")
    os.makedirs(out_dir, exist_ok=True)

    combined_csv = os.path.join(out_dir, "combined_spike_axis.csv")

    # ── Step 1: Concatenate CSVs ──────────────────────────────────────────
    if args.skip_concat:
        if not os.path.exists(combined_csv):
            sys.exit(
                f"[error] --skip_concat specified but no combined CSV found at:\n"
                f"        {combined_csv}"
            )
        print(f"Skipping concatenation — using existing CSV: {combined_csv}")
    else:
        print("=" * 60)
        print("Step 1 — Concatenating axis CSV files")
        print("=" * 60)
        if not os.path.isdir(args.base_dir):
            sys.exit(f"[error] base_dir not found: {args.base_dir}")
        df = spikepath.concatenate_axis_csvs(args.base_dir, out_path=combined_csv)
        if df.empty:
            sys.exit("[error] No CSV files found in base_dir — check directory structure.")

    # ── Step 2: Combined analysis ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Combined axis analysis")
    print("=" * 60)
    rep_channels = _parse_rep_args(args.rep)

    analysis_dir = os.path.join(out_dir, "combined_analysis")
    spikepath.run_combined_analysis(
        csv_path     = combined_csv,
        out_dir      = analysis_dir,
        early_pct    = args.early_pct,
        late_pct     = args.late_pct,
        rep_channels = rep_channels,
    )

    print(f"\nDone.  Results in: {out_dir}")


if __name__ == "__main__":
    main()
