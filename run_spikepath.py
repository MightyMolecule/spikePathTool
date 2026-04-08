#!/usr/bin/env python3
"""
run_spikepath.py

Full SpikePath single-recording pipeline:
  1.  Detect spikes if cache is missing
  2.  Load recording + optional overlay
  3.  Select principal axis (SRC / TGT passed as arguments or interactive click)
  4.  Confirm axis + band  [interactive — Enter to continue, Q/Esc to abort]
  5.  Branch A — spike count & firing rate along axis
  6.  Branch B — confirm/override intermediate electrodes, set n_traces,
                 waveform extraction, propagation speed CSV

Usage
-----
python run_spikepath.py \\
    --h5       /mnt/f/ephys/.../recording.raw.h5 \\
    --src      42 \\
    --tgt      87 \\
    [--overlay /mnt/f/ephys/.../impedance.png] \\
    [--out_dir /path/to/custom/results]

Run with -h to see all options.

Output layout
-------------
<out_dir>/
├── spike_count_axis/
│   ├── summary.txt
│   ├── axis_spike_count_map.png
│   ├── axis_spike_count_bars.png
│   ├── axis_spike_count_electrodes.csv
│   └── axis_spike_count_bins.csv
└── waveform_traces/
    ├── waveforms.png
    └── propagation_speed.csv
"""

import argparse
import os
import sys

# Allow running from the spikePathTool_cleanScripts directory directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spikepath
from spikepath.interactive import confirm_axis, confirm_intermediates, prompt_n_traces


# ==========================================================================
# Argument parser
# ==========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SpikePath — single-recording principal-axis analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    req = p.add_argument_group("required")
    req.add_argument("--h5", required=True,
                     help="Path to .raw.h5 recording file")
    req.add_argument("--src", type=int, default=None,
                     help="Source electrode channel index (omit for interactive click)")
    req.add_argument("--tgt", type=int, default=None,
                     help="Target electrode channel index (omit for interactive click)")
    req.add_argument("--yes", action="store_true",
                     help="Skip interactive axis confirmation (useful in headless/WSL environments)")

    # ── Recording ─────────────────────────────────────────────────────────
    rec = p.add_argument_group("recording")
    rec.add_argument("--mapping_path",
                     default="/data_store/data0000/settings/mapping",
                     help="Internal HDF5 path to the channel mapping dataset")
    rec.add_argument("--fs", type=float, default=20_000,
                     help="Sampling rate (Hz)")
    rec.add_argument("--refractory_ms", type=float, default=1.0,
                     help="Refractory period for spike de-duplication (ms)")
    rec.add_argument("--gain", type=float, default=3.14,
                     help="au -> µV conversion factor used during spike detection")

    # ── Paths ─────────────────────────────────────────────────────────────
    paths = p.add_argument_group("paths")
    paths.add_argument("--overlay", default=None,
                       help="Path to impedance overlay image (.png/.jpg/.npy)")
    paths.add_argument("--out_dir", default=None,
                       help="Output directory (default: ./<basename>_spikepath_src<N>_tgt<N>)")

    # ── Axis selection ────────────────────────────────────────────────────
    ax = p.add_argument_group("axis selection")
    ax.add_argument("--search_band_um", type=float, default=25.0,
                    help="Max perpendicular distance from axis for candidate electrodes (µm)")

    # ── Branch A: spike count ─────────────────────────────────────────────
    ba = p.add_argument_group("Branch A — spike count axis")
    ba.add_argument("--group_frac", type=float, default=0.2,
                    help="Bin width as fraction of axis length (0.2 → 5 bins)")

    # ── Branch B: waveforms ───────────────────────────────────────────────
    bb = p.add_argument_group("Branch B — waveform traces")
    bb.add_argument("--n_inter", type=int, default=3,
                    help="Number of intermediate electrodes to auto-select")
    bb.add_argument("--mode", choices=["src_tgt", "src_only"], default="src_tgt",
                    help="Trigger mode: paired SRC->TGT events or all source spikes")
    bb.add_argument("--n_traces", type=int, default=30,
                    help="Default number of traces to extract (prompted at runtime)")
    bb.add_argument("--start_time_s", type=float, default=0.0,
                    help="Start position in recording for trace extraction (s)")
    bb.add_argument("--pre_ms", type=float, default=1.0,
                    help="Window before spike (ms)")
    bb.add_argument("--post_ms", type=float, default=2.5,
                    help="Window after spike (ms)")
    bb.add_argument("--link_min_ms", type=float, default=0.5,
                    help="Min SRC->TGT delay for paired events (ms)")
    bb.add_argument("--link_max_ms", type=float, default=1.5,
                    help="Max SRC->TGT delay for paired events (ms)")

    return p


# ==========================================================================
# Main
# ==========================================================================

def main():
    args = _build_parser().parse_args()

    # ── Validate H5 path ──────────────────────────────────────────────────
    if not os.path.exists(args.h5):
        sys.exit(f"[error] H5 file not found: {args.h5}")

    h5_base = os.path.basename(args.h5).replace('.raw.h5', '')

    # ── Step 1: Spike detection (if cache missing) ────────────────────────
    spike_cache = args.h5.replace('.raw.h5', '_spikes.npy')
    if not os.path.exists(spike_cache):
        print("=" * 60)
        print("Spike cache not found — running detect_spikes()")
        print("=" * 60)
        x_arr, y_arr, n_channels, outlier_idx = spikepath.load_h5_mapping(
            args.h5, args.mapping_path
        )
        spikepath.detect_spikes(
            args.h5, n_channels, outlier_idx,
            fs=args.fs, gain=args.gain,
        )

    # ── Step 2: Load recording (raw traces needed for Branch B) ───────────
    print("=" * 60)
    print("Loading recording")
    print("=" * 60)
    rec = spikepath.load_recording(
        args.h5,
        args.mapping_path,
        fs=args.fs,
        refractory_ms=args.refractory_ms,
        load_raw=True,
    )

    # ── Step 3: Overlay ───────────────────────────────────────────────────
    overlay = spikepath.load_overlay(args.overlay) if args.overlay else None

    # ── Step 4: Principal axis selection ──────────────────────────────────
    print("=" * 60)
    src_str = str(args.src) if args.src is not None else "interactive"
    tgt_str = str(args.tgt) if args.tgt is not None else "interactive"
    print(f"Axis selection  SRC=ch{src_str}  TGT=ch{tgt_str}")
    print("=" * 60)
    axis = spikepath.select_axis(
        rec, overlay,
        manual_src=args.src,
        manual_tgt=args.tgt,
        search_band_um=args.search_band_um,
    )

    # ── Step 5: Axis confirmation ─────────────────────────────────────────
    if not args.yes:
        confirm_axis(rec, axis, overlay)
    else:
        print(f"Axis confirmed (--yes):  SRC=ch{axis.src_ch}  TGT=ch{axis.tgt_ch}  "
              f"{axis.axis_len:.0f} µm  {len(axis.cands)} candidates")

    # ── Output directory (deferred so interactive src/tgt are captured) ───
    out_dir = args.out_dir or os.path.join(
        os.getcwd(),
        f"{h5_base}_spikepath_src{axis.src_ch}_tgt{axis.tgt_ch}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    # ── Branch A: spike count along axis ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Branch A — spike count axis")
    print("=" * 60)
    branch_a_dir = os.path.join(out_dir, "spike_count_axis")
    spikepath.run_spike_count_axis(
        rec, axis, branch_a_dir,
        group_frac=args.group_frac,
        overlay=overlay,
    )

    # ── Branch B: waveforms + propagation speed ───────────────────────────
    print("\n" + "=" * 60)
    print("Branch B — waveform traces")
    print("=" * 60)

    # Auto-select intermediates, then let user confirm or override
    auto_inter = spikepath.select_intermediates(
        rec, axis,
        n_inter=args.n_inter,
        link_min_ms=args.link_min_ms,
        link_max_ms=args.link_max_ms,
    )
    intermediates = confirm_intermediates(rec, axis, auto_inter)

    # Count available events, let user choose n_traces
    n_traces = prompt_n_traces(
        rec, axis,
        mode=args.mode,
        link_min_ms=args.link_min_ms,
        link_max_ms=args.link_max_ms,
        default=args.n_traces,
    )

    waveform_data = spikepath.extract_waveforms(
        rec, axis, intermediates,
        mode=args.mode,
        n_traces=n_traces,
        start_time_s=args.start_time_s,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        link_min_ms=args.link_min_ms,
        link_max_ms=args.link_max_ms,
    )

    branch_b_dir = os.path.join(out_dir, "waveform_traces")
    os.makedirs(branch_b_dir, exist_ok=True)

    spikepath.plot_waveforms(
        rec, axis, intermediates, waveform_data,
        overlay=overlay,
        out_path=os.path.join(branch_b_dir, "waveforms.png"),
    )

    spikepath.save_speed_csv(
        waveform_data,
        out_path=os.path.join(branch_b_dir, "propagation_speed.csv"),
    )

    print(f"\nDone.  Results in: {out_dir}")


if __name__ == "__main__":
    main()
