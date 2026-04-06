"""
SpikePath — principal-axis spike analysis toolkit for MaxWell HD-MEA recordings.

Modules
-------
filtering  : chip constants, H5 mapping, heatmap selection, spike detection, refractory filter
selection  : Recording/AxisSelection dataclasses, load_recording, select_axis,
             run_spike_count_axis, select_intermediates
analysis   : extract_waveforms, plot_waveforms, save_speed_csv,
             concatenate_axis_csvs, compute_stats, plot_firing_rate, run_lme

All public names are importable directly from spikepath:

    import spikepath
    rec  = spikepath.load_recording(h5_path, mapping_path)
    axis = spikepath.select_axis(rec)

    from spikepath import load_recording, run_spike_count_axis, extract_waveforms
"""

from .filtering import *   # GRID_ROWS, GRID_COLS, PITCH + all filtering functions
from .selection import *   # Recording, AxisSelection + all selection functions
from .analysis  import *   # DEFAULT_COLORS + all analysis functions

from . import filtering
from . import selection
from . import analysis
