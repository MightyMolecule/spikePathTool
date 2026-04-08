# SpikePathTool

This is a slim python spike analysis toolkit for MaxWell HD-MEA recordings that enables detection of passive action potentials through microstructure channels with interactive user features. SpikePathTool was initially built to characterize passive directional signaling of axons in asymmetric&directional PDMS microchannels. SpikePathTool enables interactive selection of Source/Target electrode pairs overlaid from impedance scanning to characterize signal propagation and variation along a principal axis. Brab
 electrodes along a 1 electrode search radius. 
 
 ## Branch A: 
 Runs a spike count analysis along the selected axis. It bins the electrodes spatially into segments (default: quintile bins) and counts how many spikes each electrode and bin detected from the spike cache. Branch A outputs a spatial heatmap, bar chart, and CSVs showing firing activity distributed along the axon path.

## Branch B: 
Extracts waveform traces and computes propagation speed. It selects a few intermediate electrodes between the source and target (default 3), finds paired spike events where the source fires and the target responds within a defined timing window (0.5–1.5 ms by default), then cuts short voltage snippets from the raw traces around each event. The result is an overlaid waveform plot showing the action potential shape at each electrode along the path, and a CSV with the estimated conduction velocity.

---

## Repository structure

```
spikePathTool_cleanScripts/
│
├── run_spikepath.py              ← single-recording CLI entry point
├── run_concat.py                 ← multi-recording aggregation & combined analysis CLI
├── neuroflow_utils.py            ← utility helpers
├── example_notebook.ipynb        ← example Jupyter notebook
├── environment.yml               ← conda environment setup
├── README.md
│
├── spikepath/                    ← importable library
│   ├── __init__.py
│   ├── filtering.py              ← chip constants, spike detection, refractory filter
│   ├── selection.py              ← Recording/AxisSelection, load_recording, select_axis
│   ├── analysis.py               ← waveform extraction, plotting, stats, CSV concatenation
│   ├── combined_analysis.py      ← population-level early vs late axis analysis
│   └── interactive.py            ← confirm_axis, confirm_intermediates, prompt_n_traces
│
└── exampleData/
    ├── P002352_voltageMapReconstructed.png
    ├── Trace_..._spikes.npy      ← cached spike detection output
    └── exampleOutput/
        └── ch1_LR/
            ├── summary.txt
            ├── axis_spike_count_map.png
            ├── axis_spike_count_bars.png
            ├── axis_spike_count_electrodes.csv
            └── axis_spike_count_bins.csv
```

---

## Setup

### Required input files

```
/path/to/recordings/
├── recording.raw.h5          ← required (MaxWell HD-MEA output)
├── recording_spikes.npy      ← auto-generated on first run, reused after
└── impedance.png             ← optional overlay (--overlay)
```

The `.raw.h5` file must contain the channel mapping at the internal HDF5 path
`/data_store/data0000/settings/mapping` (default `--mapping_path`). Files
recorded on a MaxWell system already have this.

### Create the conda environment (once)

```bash
conda env create -f /path/to/spikePathTool_cleanScripts/environment.yml
```

---

## Quick start — CLI

```bash
# 1. Activate the environment
conda activate spikepath

# 2. Move to the tool directory
cd /path/to/spikePathTool_cleanScripts

# 3. Run
python run_spikepath.py \
    --h5      /path/to/recording.raw.h5 \
    --src     1 \ #(Optional) source electrode
    --tgt     2 \ #(Optional) target electrode
    --overlay /path/to/impedance.png \
    --yes  \ #(Optional) skipping of 2ms timing window constraint
```
(Notes)
Omit `--src` and `--tgt` to select source and target interactively.

Add `--yes` to skip the interactive axis-confirmation window (required in WSL / headless environments):


### Interactive prompts

1. **Axis confirmation** *(skipped with `--yes`)* — a matplotlib window shows the axis. Press `Enter`/`Space` to confirm, `Q`/`Escape` to abort.

2. **Intermediate electrodes** — the script auto-selects `--n_inter` candidates (default 3). When prompted:
   ```
   Press Enter to accept, or type comma-separated channel IDs to override:
   ```
   Type e.g. `466, 83, 180` to use specific channels. They are automatically sorted by position along the axis.

3. **Number of traces** — shows how many events are available, then asks how many to extract. Press `Enter` to accept the default (30).

### All CLI options

| Group | Argument | Default | Description |
|---|---|---|---|
| required | `--h5` | — | Path to `.raw.h5` recording file |
| required | `--src` | interactive | Source electrode channel index |
| required | `--tgt` | interactive | Target electrode channel index |
| recording | `--mapping_path` | `/data_store/data0000/settings/mapping` | Internal HDF5 path to channel mapping |
| recording | `--fs` | `20000` | Sampling rate (Hz) |
| recording | `--refractory_ms` | `1.0` | Refractory period for spike de-duplication (ms) |
| recording | `--gain` | `3.14` | au → µV conversion factor |
| paths | `--overlay` | None | Path to impedance overlay image (`.png`/`.jpg`/`.npy`) |
| paths | `--out_dir` | auto | Output directory (default: `./<basename>_spikepath_src<N>_tgt<N>`) |
| axis | `--search_band_um` | `25.0` | Max perpendicular distance from axis for candidate electrodes (µm) |
| axis | `--yes` | False | Skip interactive axis confirmation |
| Branch A | `--group_frac` | `0.2` | Bin width as fraction of axis length (0.2 → 5 bins) |
| Branch B | `--n_inter` | `3` | Number of intermediate electrodes to auto-select |
| Branch B | `--mode` | `src_tgt` | Trigger mode: `src_tgt` (paired events) or `src_only` (all source spikes) |
| Branch B | `--n_traces` | `30` | Default number of traces to extract |
| Branch B | `--start_time_s` | `0.0` | Start position in recording for trace extraction (s) |
| Branch B | `--pre_ms` | `1.0` | Window before spike (ms) |
| Branch B | `--post_ms` | `2.5` | Window after spike (ms) |
| Branch B | `--link_min_ms` | `0.5` | Min SRC→TGT delay for paired events (ms) |
| Branch B | `--link_max_ms` | `1.5` | Max SRC→TGT delay for paired events (ms) |

### Output layout — single recording

```
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
```

---

## Quick start — combined analysis (multi-recording)

Run after collecting `spike_count_axis/` outputs from multiple recordings.

```bash
python run_concat.py \
    --base_dir /path/to/SpikeCountAxisTool \
    --early_pct 25 \
    --late_pct  75 \
    --rep "P002275,8,dir,#d62728" \
    --rep "P001314,1,ctl,#1f77b4"
```

Omit `--rep` to skip the representative channel plots. Add `--skip_concat` to reuse an existing `combined_spike_axis.csv` without re-scanning the directory.

### All CLI options — `run_concat.py`

| Group | Argument | Default | Description |
|---|---|---|---|
| required | `--base_dir` | — | Directory containing spikepath output folders |
| paths | `--out_dir` | `<base_dir>/combined_output` | Output directory |
| paths | `--skip_concat` | False | Skip CSV concatenation, use existing `combined_spike_axis.csv` |
| analysis | `--early_pct` | `25.0` | Upper axis-position percentile defining the early segment |
| analysis | `--late_pct` | `75.0` | Lower axis-position percentile defining the late segment |
| analysis | `--rep` | None | Representative channel: `"recording,channel,type[,#color]"` — repeat for multiple |

### Output layout — combined analysis

```
<out_dir>/
├── combined_spike_axis.csv
└── combined_analysis/
    ├── segment_boxplot_<ms>.png
    ├── sig_channels_vs_distance_<ms>.png
    ├── fraction_significant_early_vs_late.png
    ├── sig_channels_scatter_bestfit_all_ms.png
    └── representative_<ms>_ch<N>.png        (only if --rep is provided)
```

---

## Pipeline


### 1) Spike Detection
The raw .raw.h5 recording is read channel-by-channel in 200,000-frame chunks. Each channel is bandpass filtered (200–6,000 Hz), and negative peaks exceeding 3.6× the channel RMS are detected as spikes. Results are stored as a .npy cache file alongside the recording and reloaded on all subsequent runs, so this step only runs once per recording. A refractory filter then removes duplicate spikes on the same electrode within 1 ms of each other.

### 2) Recording & Axis Selection
Channel positions are read from the internal HDF5 mapping (/data_store/data0000/settings/mapping), and the single outlier electrode furthest from the centroid is excluded. Source and target electrodes are either passed as arguments or selected interactively by clicking on an impedance overlay image. All electrodes within a 25 µm perpendicular band of the SRC→TGT line are collected as axis candidates and sorted by their position along the axis.

### 3) Branch A — Spike Count Axis
Each candidate electrode's spike count is divided by the recording duration to give a firing rate in Hz. Electrodes are grouped into spatial bins (default 5 equal bins along the axis length) and the per-bin totals are computed. Outputs include a spatial heatmap of firing rate overlaid on the electrode map, a bar chart of spike counts per bin, and two CSVs — one at electrode resolution and one at bin resolution.

Note on binning: The bin width is set as a fraction of axis length (--group_frac, default 0.2 → 5 bins). Bin count is therefore relative to the physical distance between SRC and TGT, not an absolute spatial scale. Choosing a very small --group_frac can produce sparsely populated bins if the axis is short or few electrodes fall within the band.

### 4)Branch B — Waveform Traces & Propagation Speed
A small number of intermediate electrodes (default 3) are auto-selected along the axis, then presented to the user for confirmation or manual override. Spike events are filtered to paired SRC→TGT events — cases where the source fires and the target responds within a defined delay window (default 0.5–1.5 ms). For each such event, a raw voltage snippet (default 1 ms before to 2.5 ms after the source spike) is cut from all axis electrodes. Individual traces are plotted as a stacked waveform panel with mean ± SD, and per-event propagation latencies and speeds (µm/ms and m/s) are written to a CSV.

Note on event pairing: The 0.5–1.5 ms delay window is a constraint on what counts as a "propagating" event. Events outside this window — whether too fast (electrical crosstalk) or too slow (independent firing) — are excluded. In src_only mode this constraint is lifted and all source spikes are used as triggers, which is useful for characterizing source waveform shape independent of propagation.

### 5) Combined Analysis — Early vs Late Axis Firing Rate (multi-recording)

After running `run_spikepath.py` across multiple recordings, `run_concat.py` walks the output directory structure and concatenates all `axis_spike_count_electrodes.csv` files into a single `combined_spike_axis.csv`. It then runs a population-level analysis comparing spike rates in the proximal (early) and distal (late) portions of the principal axis for each channel in each microstructure condition. A Mann-Whitney U test is applied per channel with Benjamini-Hochberg FDR correction across all channels within each microstructure. Outputs include paired boxplots of early vs late firing rate per channel, spike rate vs axis distance plots for significant channels, a summary bar chart of the fraction of significant channels by microstructure and channel type, and a multi-panel scatter plot with linear regression lines for all significant channels across conditions. An optional set of representative single-channel plots can be generated by passing `--rep` arguments.

Note on segmentation: The early and late cutoffs (`--early_pct`, `--late_pct`) define non-overlapping axis-position windows; the middle region between them is excluded from the statistical comparison. The default (25% / 75%) captures the proximal quarter vs the distal quarter of the axis. Narrowing these windows increases statistical power per bin but reduces the number of electrodes contributing to each segment.

---

## Key Findings

