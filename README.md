# spikePathTool


Principal-axis spike analysis toolkit for MaxWell HD-MEA recordings.


---

## How to launch: 
 # Required data files:
    /path/to/recordings/
    ├── recording.raw.h5          ← required
    ├── recording_spikes.npy      ← auto-generated on first run, reused after
    └── impedance.png             ← optional overlay (--overlay)

Run the following commands: 
# 1. Create the environment (once)
conda env create -f /pathTospikePathTool/environment.yml

# 2. Activate it (every session)
conda activate spikepath

# 3. Move to the tool directory
cd /mnt/d/ephys/Analysis/Meeting/spikePathTool_cleanScripts

# 4. Run
python run_spikepath.py \
    --h5      /pathToRecording/recording.raw.h5 \
    --overlay /pathToOverlay/impedance.png

```

```
spikePathTool_cleanScripts/
│
├── run_spikepath.py              ← main CLI entry point
├── neuroflow_utils.py            ← utility helpers
├── example_notebook.ipynb        ← example Jupyter notebook
├── environment.yml               ← conda environment setup
├── README.md
│
├── spikepath/                    ← importable library
│   ├── __init__.py
│   ├── filtering.py              ← chip constants, spike detection, refractory filter
│   ├── selection.py              ← Recording/AxisSelection, load_recording, select_axis
│   ├── analysis.py               ← waveform extraction, plotting, stats
│   └── interactive.py            ← confirm_axis, confirm_intermediates, prompt_n_traces
│
├── exampleData/
│   ├── P002352_voltageMapReconstructed.png
│   ├── Trace_..._spikes.npy      ← cached spike detection output
│   └── exampleOutput/
│       └── ch1_LR/
│           ├── summary.txt
│           ├── axis_spike_count_map.png
│           ├── axis_spike_count_bars.png
│           ├── axis_spike_count_electrodes.csv
│           └── axis_spike_count_bins.csv
│
└── <recording>_spikepath_src<N>_tgt<N>/    ← auto-generated per run
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

---

## Quick start — CLI

Run from the `spikePathTool_cleanScripts/` directory so that `import spikepath` resolves.

```bash
python run_spikepath.py \
    --h5      /mnt/f/ephys/.../recording.raw.h5 \
    --src     327 \
    --tgt     85 \
    --overlay /mnt/f/ephys/.../impedance.png
```

Add `--yes` to skip the interactive axis-confirmation window to reject propogation window filtration parameters (required in WSL / headless environments):

```bash
python run_spikepath.py \
    --h5      /mnt/f/ephys/.../recording.raw.h5 \
    --src     327 \
    --tgt     85 \
    --overlay /mnt/f/ephys/.../impedance.png \
    --yes
```

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

### Output layout

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

## Using the library directly

```python
import spikepath

H5      = '/mnt/f/ephys/.../recording.raw.h5'
MAPPING = '/data_store/data0000/settings/mapping'
```

### 1. Spike detection

Spikes are cached to `<h5_path>_spikes.npy` alongside the H5 file. Run once; subsequent calls load the cache automatically.

```python
x_arr, y_arr, n_channels, outlier_idx = spikepath.load_h5_mapping(H5, MAPPING)

spike_array = spikepath.detect_spikes(
    H5, n_channels, outlier_idx,
    fs=20_000,
    gain=3.14,
)
# spike_array: ndarray (N, 3) — [frame, amplitude, channel_idx]
```

### 2. Load recording

```python
rec = spikepath.load_recording(
    H5,
    MAPPING,
    fs=20_000,
    refractory_ms=1.0,
    load_raw=True,      # required for waveform extraction
)
# rec.xy          — (n_channels, 2) positions in µm
# rec.spike_array — (N, 3) spikes
# rec.sp_times    — spike frame numbers
# rec.sp_ch       — spike channel indices
# rec.duration_s  — recording length in seconds
# rec.rec_f       — bandpass-filtered SpikeInterface recording (if load_raw=True)
```

### 3. Load overlay (optional)

```python
overlay = spikepath.load_overlay('/mnt/f/.../impedance.png')
# Returns an RGBA ndarray for use in plots, or None if path is None
```

### 4. Select axis

```python
axis = spikepath.select_axis(
    rec,
    overlay,
    manual_src=327,
    manual_tgt=85,
    search_band_um=25.0,   # µm perpendicular to axis
)
# axis.src_ch, axis.tgt_ch  — source/target channel indices
# axis.axis_len              — length in µm
# axis.cands                 — indices of on-axis candidate electrodes
# axis.d_along               — along-axis distance for every channel (µm)
# axis.d_lo, axis.d_hi       — axis distance bounds
```

### 5. Branch A — spike count along axis

```python
results = spikepath.run_spike_count_axis(
    rec, axis,
    out_dir='/tmp/spike_count_axis',
    group_frac=0.2,     # bin width = 20% of axis length
    overlay=overlay,
)
```

Saves `summary.txt`, `axis_spike_count_map.png`, `axis_spike_count_bars.png`, and two CSVs to `out_dir`.

### 6. Branch B — waveform traces & propagation speed

**Select intermediates:**

```python
# Auto-select n_inter electrodes scored by coincident firing
intermediates = spikepath.select_intermediates(
    rec, axis,
    n_inter=3,
    link_min_ms=0.5,
    link_max_ms=1.5,
)
# Or specify manually:
intermediates = [466, 83, 180]   # sorted by axis position automatically
```

**Extract waveforms:**

```python
waveform_data = spikepath.extract_waveforms(
    rec, axis, intermediates,
    mode='src_tgt',     # 'src_tgt' or 'src_only'
    n_traces=30,
    start_time_s=0.0,
    pre_ms=1.0,
    post_ms=2.5,
    link_min_ms=0.5,
    link_max_ms=1.5,
)
# Returns dict with keys:
#   arr_list    — list of (n_traces, n_frames) arrays, one per electrode
#   t_ms        — time axis relative to source spike
#   axis_chs    — [SRC, M1, M2, M3, TGT] channel indices
#   axis_labels — ['SRC', 'M1', 'M2', 'M3', 'TGT']
#   valid_evts  — spike events used
#   valid_spk_t — per-event spike times for each electrode
```

**Plot waveforms:**

```python
fig = spikepath.plot_waveforms(
    rec, axis, intermediates, waveform_data,
    overlay=overlay,
    out_path='/tmp/waveform_traces/waveforms.png',
)
```

**Save propagation speed CSV:**

```python
spikepath.save_speed_csv(
    waveform_data,
    out_path='/tmp/waveform_traces/propagation_speed.csv',
)
```

The CSV contains per-event columns for each electrode: spike delay (ms), absolute spike time (s), waveform amplitude range (µV), plus segment-by-segment distance (µm), delay (ms), and propagation speed (µm/ms and m/s).

---

## Multi-recording statistics

After running Branch A across multiple recordings, aggregate the CSVs for population analysis.

### Concatenate CSVs

Expects directory structure:
```
base_dir/
    {microstructure}_{channel_type}/
        {channel}_{direction}/
            axis_spike_count_electrodes.csv
```

```python
df = spikepath.concatenate_axis_csvs(
    base_dir='/mnt/f/ephys/results/SpikeCountAxisTool',
    out_path='/tmp/combined.csv',   # optional
)
```

### Summary statistics

```python
stats_df = spikepath.compute_stats(df)
# columns: channel_type, bin, mean, sem
```

### Firing rate plot

```python
fig = spikepath.plot_firing_rate(df, out_path='/tmp/firing_rate.png')
```

### Linear mixed-effects model

```python
result = spikepath.run_lme(df)
# spike_rate_hz ~ channel_type * bin  +  (1 | microstructure)
print(result.summary())
```

---

## Chip geometry constants

```python
spikepath.PITCH      # 17.5 µm — electrode pitch
spikepath.GRID_ROWS  # 120
spikepath.GRID_COLS  # 220
```

---

## WSL / headless notes

- Matplotlib GUI windows (axis confirmation, heatmap selection) require a display. In WSL without WSLg, use `--yes` to skip the axis confirmation window.
- Check your backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`. If it prints `Agg`, set `export MPLBACKEND=TkAgg` before running.
- The intermediates prompt and n_traces prompt are text-only and work without a display.
