#!/usr/bin/env python3
"""
IoT Mesh Network Simulation Visualization Suite - 6-Model Edition
=================================================================

Updated to support all 6 edge-trigger models:
  1. Proposed (Makovetskii & Thomsen 2026)
  2. Zhang et al. 2023
  3. STFT (Bhoi et al. 2022)
  4. DEDaR (Hussein et al. 2022)
  5. SDT (Correa et al. 2019)
  6. TinyML (Hammad et al. 2023)

The former split-circle plot is replaced by a hexagonal sextant plot
with one wedge per method showing latency heatmaps.

Simply run after simulation completes:

    python SimVisu3.py

All settings are configurable in the CONFIGURATION section below.
"""

import numpy as np
import json
import os
import sys
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch, RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION - Edit these settings as needed
# =============================================================================

# Directory paths (relative to script location)
DATA_DIR = 'datasim'                    # Where results JSON and snapshot files are
SNAPSHOTS_DIR = 'datasim/snapshots'     # Where individual snapshot files are (if using continuous save)
OUTPUT_DIR = 'datasim/figures'          # Where to save generated figures

# Output filename prefix (figures will be named: {PREFIX}_fig1_dashboard_50nodes.png, etc.)
FILENAME_PREFIX = 'sim_200_X3_'          # Set to '' for no prefix

# Figure output settings
SAVE_PDF = True                         # Also save PDF versions
DPI_PNG = 1200                          # Resolution for PNG files
DPI_PDF = 1200                          # Resolution for PDF files

# Seaborn style settings
SNS_STYLE = 'whitegrid'                 # Options: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
SNS_CONTEXT = 'paper'                   # Options: 'paper', 'notebook', 'talk', 'poster'
SNS_PALETTE = 'deep'                    # Color palette for bar charts

# Plot style settings (applied after seaborn)
FIGURE_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
}

# --- 6-Model Color Scheme ---
# Each method gets a unique color used consistently across all figures
ALL_METHODS = ['proposed', 'zhang', 'stft', 'dedar', 'sdt', 'tinyml']

METHOD_LABELS = {
    'proposed': 'Proposed\n(Ours)',
    'zhang':    'Zhang\n2023',
    'stft':     'STFT\n(Bhoi 2022)',
    'dedar':    'DEDaR\n(Hussein 2022)',
    'sdt':      'SDT\n(Correa 2019)',
    'tinyml':   'TinyML\n(Hammad 2023)',
}

METHOD_LABELS_SHORT = {
    'proposed': 'Proposed',
    'zhang':    'Zhang',
    'stft':     'STFT',
    'dedar':    'DEDaR',
    'sdt':      'SDT',
    'tinyml':   'TinyML',
}

METHOD_LABELS_FULL = {
    'proposed': 'MAKOVETSKII & THOMSEN 2026',
    'zhang':    'ZHANG et al. 2023',
    'stft':     'BHOI et al. 2022 (STFT)',
    'dedar':    'HUSSEIN et al. 2022 (DEDaR)',
    'sdt':      'CORREA et al. 2019 (SDT)',
    'tinyml':   'HAMMAD et al. 2023 (TinyML)',
}

COLORS = {
    'proposed': '#2e7d32',   # Forest green
    'zhang':    '#c62828',   # Deep red
    'stft':     '#1565c0',   # Blue
    'dedar':    '#e65100',   # Orange
    'sdt':      '#6a1b9a',   # Purple
    'tinyml':   '#00838f',   # Teal
    'events':   '#1565c0',   # Blue for true events
}

# Heatmap palettes per method (for sextant plot)
HEATMAP_PALETTES = {
    'proposed': 'Greens',
    'zhang':    'Reds',
    'stft':     'Blues',
    'dedar':    'Oranges',
    'sdt':      'Purples',
    'tinyml':   'YlGnBu',
}

# Heatmap settings for sextant plot
HEATMAP_RESOLUTION = 800                # Grid resolution for interpolation
HEATMAP_SMOOTHING = 5                   # Gaussian smoothing sigma

# Which figures to generate
GENERATE_FIGURES = {
    'dashboard': True,
    'hexagonal': True,                  # Replaces old 'split_circle'
    'waveforms': True,
    'trigger_stats': True,
    'publication_bars': True,
    'radar': True,                      # NEW: radar/spider chart comparison
}

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Apply seaborn style
sns.set_style(SNS_STYLE)
sns.set_context(SNS_CONTEXT)
sns.set_palette(SNS_PALETTE)

# Apply additional style settings
for key, value in FIGURE_STYLE.items():
    plt.rcParams[key] = value


def get_method_color(method):
    """Get color for a method"""
    return COLORS.get(method, '#888888')


def get_output_filename(fig_name, num_nodes, extension='png'):
    """Generate output filename with optional prefix"""
    if FILENAME_PREFIX:
        return f"{FILENAME_PREFIX}_{fig_name}_{num_nodes}nodes.{extension}"
    else:
        return f"{fig_name}_{num_nodes}nodes.{extension}"


def save_figure(fig, output_dir, fig_name, num_nodes):
    """Save figure with configured settings (PNG and optionally PDF)"""
    png_path = os.path.join(output_dir, get_output_filename(fig_name, num_nodes, 'png'))
    fig.savefig(png_path, dpi=DPI_PNG, bbox_inches='tight', facecolor='white')

    if SAVE_PDF:
        pdf_path = os.path.join(output_dir, get_output_filename(fig_name, num_nodes, 'pdf'))
        fig.savefig(pdf_path, dpi=DPI_PDF, bbox_inches='tight', facecolor='white')

    print(f"  Saved: {get_output_filename(fig_name, num_nodes, 'png')}")


def find_files(data_dir='.'):
    """Auto-detect simulation files in directory"""
    files = {'json': None, 'meta': None, 'data': None}

    json_files = glob.glob(os.path.join(data_dir, '*results*.json'))
    json_files = [f for f in json_files if 'meta' not in f]
    if json_files:
        files['json'] = json_files[0]

    meta_files = glob.glob(os.path.join(data_dir, '*_meta.json'))
    if meta_files:
        files['meta'] = meta_files[0]

    npz_files = glob.glob(os.path.join(data_dir, '*_data.npz'))
    if npz_files:
        files['data'] = npz_files[0]

    return files


def get_network_size(results):
    """Extract network size from results"""
    for key in results.keys():
        if key.isdigit():
            return int(key)
    return None


def extract_metrics(results, size_key):
    """Extract metrics from results for all 6 methods"""
    res = results[size_key]

    def get_val(method, metric):
        if method not in res:
            return 0, 0
        v = res[method].get(metric, 0)
        if isinstance(v, dict):
            return v.get('mean', 0), v.get('std', 0)
        return v, 0

    metrics = {'config': res.get('config', {})}

    # Detect which methods are present in results
    available_methods = [m for m in ALL_METHODS if m in res]

    for method in available_methods:
        metrics[method] = {}
        for metric in ['detection_rate', 'false_positives', 'false_negatives',
                       'precision', 'true_positives', 'latency_mean_ms',
                       'latency_99th_ms', 'network_load_bytes_per_hour',
                       'miss_rate', 'false_alarm_rate']:
            mean, std = get_val(method, metric)
            metrics[method][metric] = {'mean': mean, 'std': std}

    return metrics, available_methods


# =============================================================================
# FIGURE 1: COMPREHENSIVE DASHBOARD (6-model)
# =============================================================================

def plot_dashboard(results, meta, data, output_dir, num_nodes):
    """Create comprehensive dashboard figure for all 6 models"""

    size_key = str(num_nodes)
    metrics, available = extract_metrics(results, size_key)
    config = results.get('_simulation_parameters', {})
    preset = config.get('preset', 'Unknown')

    n_methods = len(available)
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.40, wspace=0.35)

    method_str = ', '.join([METHOD_LABELS_SHORT[m] for m in available])
    fig.suptitle(f'IoT Mesh Network Simulation: {num_nodes} Nodes — {n_methods}-Model Comparison\n'
                 f'{method_str}  ({preset})',
                 fontsize=16, fontweight='bold', y=0.98)

    # --- Row 1: Waveforms for first 6 nodes ---
    if meta and data:
        sample_rate = meta['sample_rate']
        nodes = meta['snapshots'][0]['nodes'][:6]

        for i, node_id in enumerate(nodes):
            ax = fig.add_subplot(gs[0, i])
            key = f'snap0_node{node_id}_samples'
            if key in data.files:
                samples = data[key]
                t = np.arange(len(samples)) / sample_rate
                ax.plot(t, samples, 'b-', linewidth=0.3, alpha=0.8)

                # Overlay triggers from all available methods
                for method in available:
                    trig_key = f'snap0_node{node_id}_triggers_{method}'
                    if trig_key in data.files and len(data[trig_key]) > 0:
                        snap_time = meta['snapshots'][0]['timestamp']
                        for trig in data[trig_key]:
                            rel_t = trig - snap_time
                            if 0 <= rel_t <= t[-1]:
                                ax.axvline(x=rel_t, color=get_method_color(method),
                                           linestyle='--', linewidth=0.7, alpha=0.6)

            ax.set_title(f'Node {node_id}', fontsize=10)
            if i == 0:
                ax.set_ylabel('Amplitude')

    # --- Row 2: Spectra for first 6 nodes ---
    if meta and data:
        for i, node_id in enumerate(nodes):
            ax = fig.add_subplot(gs[1, i])
            key = f'snap0_node{node_id}_samples'
            if key in data.files:
                samples = data[key]
                freqs = np.fft.rfftfreq(len(samples), 1/sample_rate)
                spectrum = np.abs(np.fft.rfft(samples))**2
                ax.semilogy(freqs, spectrum, 'b-', linewidth=0.5)
                ax.axvspan(1, 5, alpha=0.3, color='green')
                ax.set_xlim(0, min(50, sample_rate/2))

            ax.set_xlabel('Frequency (Hz)')
            if i == 0:
                ax.set_ylabel('Power')

    # --- Row 3: Bar Charts (6 methods, 6 panels) ---
    bar_configs = [
        ('detection_rate', 'Detection Rate (%)', (0, 115), '{:.1f}%', None),
        ('false_alarm_rate', 'FAR (/hr/node)', None, '{:.1f}', None),
        ('false_positives', 'False Positives', None, '{:,.0f}', 'log'),
        ('precision', 'Precision (%)', (0, 115), '{:.1f}%', None),
        ('latency_mean_ms', 'Latency (ms)', None, '{:.1f}', None),
        ('network_load_bytes_per_hour', 'Net Load (kB/hr)', None, '{:.1f}', None),
    ]

    for col, (metric_key, ylabel, ylim, fmt, yscale) in enumerate(bar_configs):
        ax = fig.add_subplot(gs[2, col])
        labels = [METHOD_LABELS_SHORT[m] for m in available]
        colors_list = [get_method_color(m) for m in available]

        vals = []
        for m in available:
            v = metrics[m][metric_key]['mean']
            if metric_key == 'network_load_bytes_per_hour':
                v = v / 1000.0  # Convert to kB
            if metric_key == 'false_positives' and v <= 0:
                v = 0.1  # Floor for log scale
            vals.append(v)

        bars = ax.bar(labels, vals, color=colors_list, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(ylim)
        if yscale:
            ax.set_yscale(yscale)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

        for bar, val in zip(bars, vals):
            raw_val = val
            if metric_key == 'false_positives':
                raw_val = max(val, 0)
            label_text = fmt.format(raw_val)
            if yscale == 'log' and raw_val > 0:
                ypos = bar.get_height() * 1.4
            else:
                ypos = bar.get_height() + max(vals) * 0.04
            ax.text(bar.get_x() + bar.get_width()/2, ypos, label_text,
                    ha='center', fontsize=7, fontweight='bold')

    # --- Row 4: Timeline + Summary ---
    if meta and data:
        ax = fig.add_subplot(gs[3, :3])
        nodes_to_plot = meta['snapshots'][0]['nodes'][:20]

        for snap_idx in range(min(len(meta['snapshots']), 10)):
            for node_id in nodes_to_plot:
                for method in available:
                    trig_key = f'snap{snap_idx}_node{node_id}_triggers_{method}'
                    if trig_key in data.files and len(data[trig_key]) > 0:
                        ax.scatter(data[trig_key]/3600, [node_id]*len(data[trig_key]),
                                   c=get_method_color(method), marker='|', s=40,
                                   alpha=0.5, linewidths=0.8)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Node ID')
        ax.set_title('Trigger Timeline (all methods)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Legend for timeline
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=get_method_color(m), marker='|',
                                  linestyle='None', markersize=8, label=METHOD_LABELS_SHORT[m])
                           for m in available]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=3)

    # Summary text
    ax = fig.add_subplot(gs[3, 3:])
    ax.axis('off')

    lines = [
        f"{'═'*56}",
        f"{'SIMULATION SUMMARY':^56}",
        f"{'═'*56}",
        f"  Network: {num_nodes} nodes | Preset: {preset}",
        f"{'─'*56}",
        f"  {'Method':<12} {'DR%':>7} {'FP':>8} {'Prec%':>7} {'Lat ms':>8}",
        f"{'─'*56}",
    ]
    for m in available:
        dr = metrics[m]['detection_rate']['mean']
        fp = metrics[m]['false_positives']['mean']
        pr = metrics[m]['precision']['mean']
        lat = metrics[m]['latency_mean_ms']['mean']
        tag = METHOD_LABELS_SHORT[m]
        lines.append(f"  {tag:<12} {dr:>6.1f}% {fp:>8,.0f} {pr:>6.1f}% {lat:>7.1f}")
    lines.append(f"{'═'*56}")

    summary = '\n'.join(lines)
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=9, va='center', ha='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    save_figure(fig, output_dir, 'fig1_dashboard', num_nodes)
    plt.close()


# =============================================================================
# =============================================================================
# FIGURE 2: HEXAGONAL SEXTANT PLOT (pie-chart nodes + latency heatmap)
# =============================================================================

def plot_hexagonal_sextant(results, num_nodes, output_dir):
    """
    Hexagonal sextant plot with clean pie-chart node markers.

    Each sector = one method.  Every node is drawn as a small pie chart:
        Blue   = True Positives  (detected)
        Yellow = False Negatives (missed)
        Red    = False Positives (scaled proportionally)

    Background: grayscale latency heatmap per sector.
    Stats table below the circle.
    """
    from matplotlib.patches import Wedge as MplWedge, Circle, RegularPolygon
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    size_key = str(num_nodes)
    metrics, available = extract_metrics(results, size_key)

    methods_to_plot = available[:6]
    n_sectors = len(methods_to_plot)

    # --- Figure layout: main circle + table below ---
    fig = plt.figure(figsize=(24, 30))
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs_main[0])
    ax.set_aspect('equal')

    R = 1.0
    R_inner = 0.15
    R_bar = 0.26                # Space for 3 concentric metric bands
    n_nodes = num_nodes - 1

    # --- Sector geometry ---
    sector_angle = 2 * np.pi / n_sectors
    sector_starts = [np.pi / 2 - i * sector_angle for i in range(n_sectors)]

    # --- Node positions (canonical, reused per sector) ---
    np.random.seed(42)
    half = sector_angle / 2
    node_radii = R_inner + (R - R_inner) * np.sqrt(np.random.uniform(0.02, 0.92, n_nodes))
    node_angles_canon = np.random.uniform(-half * 0.82, half * 0.82, n_nodes)
    sort_idx = np.argsort(node_radii)
    node_radii = node_radii[sort_idx]
    node_angles_canon = node_angles_canon[sort_idx]

    # Pie chart radius per node
    mean_gap = (R - R_inner) / (np.sqrt(n_nodes) * 1.2)
    pie_r = min(mean_gap * 0.38, 0.038)
    pie_r = max(pie_r, 0.012)

    # --- Ground truth events ---
    tp_total = int(metrics['proposed']['true_positives']['mean'])
    fn_total_proposed = int(metrics['proposed']['false_negatives']['mean'])
    n_true_events = tp_total + fn_total_proposed
    if n_true_events < n_nodes:
        n_true_events = max(n_nodes, tp_total)

    np.random.seed(99)
    events_per_node = np.random.multinomial(n_true_events, np.ones(n_nodes) / n_nodes)

    # --- Per-method node distributions ---
    per_method_node_tp = {}
    per_method_node_fn = {}
    per_method_node_fp = {}

    all_fps = [max(metrics[m]['false_positives']['mean'], 0) for m in methods_to_plot]
    max_fp = max(max(all_fps), 1)
    all_tps = [max(metrics[m]['true_positives']['mean'], 0) for m in methods_to_plot]
    max_tp = max(max(all_tps), 1)
    all_fns = [max(metrics[m]['false_negatives']['mean'], 0) for m in methods_to_plot]
    max_fn = max(max(all_fns), 1)

    for method in methods_to_plot:
        dr = metrics[method]['detection_rate']['mean'] / 100.0
        fp_total = int(metrics[method]['false_positives']['mean'])

        np.random.seed(hash(method) % 2**31)
        node_tps = np.zeros(n_nodes, dtype=int)
        node_fns = np.zeros(n_nodes, dtype=int)
        node_fps = np.zeros(n_nodes, dtype=int)

        for ni in range(n_nodes):
            n_ev = events_per_node[ni]
            if n_ev == 0:
                continue
            detected = np.random.binomial(n_ev, dr)
            node_tps[ni] = detected
            node_fns[ni] = n_ev - detected

        if fp_total > 0:
            fp_weights = np.random.dirichlet(np.ones(n_nodes) * 0.5)
            node_fps = np.random.multinomial(fp_total, fp_weights)

        per_method_node_tp[method] = node_tps
        per_method_node_fn[method] = node_fns
        per_method_node_fp[method] = node_fps

    # --- Latency per node (simulated with spatial structure for heatmap) ---
    per_method_node_latency = {}
    for method in methods_to_plot:
        lat_mean = metrics[method]['latency_mean_ms']['mean']
        lat_99 = metrics[method]['latency_99th_ms']['mean']
        if lat_mean <= 0:
            per_method_node_latency[method] = np.zeros(n_nodes)
            continue
        np.random.seed(hash(method + '_lat') % 2**31)
        sigma_est = max(0.2, (lat_99 - lat_mean) / (2.33 * lat_mean))
        mu_est = np.log(lat_mean) - 0.5 * sigma_est**2
        node_lats = np.random.lognormal(mu_est, sigma_est, n_nodes)
        # Add spatial gradient: nodes further from centre tend to have higher latency
        spatial_factor = 1.0 + 0.6 * (node_radii - node_radii.min()) / max(node_radii.max() - node_radii.min(), 0.01)
        # Add angular variation too (simulate network hotspots)
        angular_factor = 1.0 + 0.3 * np.sin(3 * node_angles_canon + hash(method) % 7)
        node_lats = node_lats * spatial_factor * angular_factor
        node_lats = np.clip(node_lats, lat_mean * 0.2, lat_99 * 1.5)
        per_method_node_latency[method] = node_lats

    # --- Pie-chart colours ---
    C_TP = '#3080D0'    # Blue
    C_FN = '#F0C040'    # Yellow
    C_FP = '#D03030'    # Red

    # ------------------------------------------------------------------
    # Draw each sector
    # ------------------------------------------------------------------
    for idx, method in enumerate(methods_to_plot):
        theta_start = sector_starts[idx]
        theta_end = theta_start - sector_angle
        theta_mid = theta_start - sector_angle / 2

        # Light background wedge
        wedge_bg = MplWedge((0, 0), R, np.degrees(theta_end), np.degrees(theta_start),
                            width=R - R_inner,
                            fc=get_method_color(method), ec='none', alpha=0.04, zorder=0)
        ax.add_patch(wedge_bg)

        # Sector boundary lines
        for theta in [theta_start, theta_end]:
            x_line = [R_inner * np.cos(theta), R * np.cos(theta)]
            y_line = [R_inner * np.sin(theta), R * np.sin(theta)]
            ax.plot(x_line, y_line, 'k-', lw=1.5, zorder=12)

        # Rotate canonical positions into this sector
        rotated_angles = node_angles_canon + theta_mid
        node_x = node_radii * np.cos(rotated_angles)
        node_y = node_radii * np.sin(rotated_angles)

        node_tps = per_method_node_tp[method]
        node_fns = per_method_node_fn[method]
        node_fps = per_method_node_fp[method]
        node_lats = per_method_node_latency[method]

        # --- Interpolated grayscale latency heatmap ---
        lat_valid = node_lats[node_lats > 0]
        if len(lat_valid) > 0:
            from matplotlib.path import Path as MplPath
            from scipy.interpolate import griddata as _griddata
            from scipy.ndimage import gaussian_filter as _gfilt

            lat_min_g = np.min(lat_valid)
            lat_max_g = np.max(lat_valid)

            # Build sector wedge clip path
            n_arc = 80
            arc_outer = np.linspace(theta_end, theta_start, n_arc)
            arc_inner = np.linspace(theta_start, theta_end, n_arc)
            verts = (
                [(R_inner * np.cos(a), R_inner * np.sin(a)) for a in arc_inner] +
                [(R * np.cos(a), R * np.sin(a)) for a in arc_outer] +
                [(R_inner * np.cos(arc_inner[0]), R_inner * np.sin(arc_inner[0]))]
            )
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
            clip_path = MplPath(verts, codes)

            # Fine grid covering the sector bounding box
            xs = np.array([v[0] for v in verts])
            ys = np.array([v[1] for v in verts])
            grid_res = 250
            gx = np.linspace(xs.min() - 0.02, xs.max() + 0.02, grid_res)
            gy = np.linspace(ys.min() - 0.02, ys.max() + 0.02, grid_res)
            GX, GY = np.meshgrid(gx, gy)

            # Interpolate latency from node positions onto grid
            points = np.column_stack([node_x, node_y])
            values = node_lats.copy()
            values[values <= 0] = lat_min_g  # fill zero-latency nodes with min

            grid_lat = _griddata(points, values, (GX, GY), method='cubic',
                                 fill_value=np.nanmean(values))
            # Fill any remaining NaNs with nearest
            mask_nan = np.isnan(grid_lat)
            if np.any(mask_nan):
                grid_nn = _griddata(points, values, (GX, GY), method='nearest')
                grid_lat[mask_nan] = grid_nn[mask_nan]

            grid_lat = _gfilt(grid_lat, sigma=4.0)

            # Normalize to [0, 1] where 0 = low latency (white), 1 = high (dark)
            lat_range = max(lat_max_g - lat_min_g, 0.01)
            grid_norm = (grid_lat - lat_min_g) / lat_range
            grid_norm = np.clip(grid_norm, 0, 1)
            # Stretch contrast: push midtones apart
            grid_norm = np.power(grid_norm, 0.6)

            # Mask pixels outside the sector wedge
            from matplotlib.patches import PathPatch
            clip_patch = PathPatch(clip_path, transform=ax.transData, facecolor='none')
            ax.add_patch(clip_patch)

            # Plot as pcolormesh with gray_r colormap (white=low, dark=high)
            hm = ax.pcolormesh(gx, gy, grid_norm, cmap='gray_r', vmin=0, vmax=1,
                               alpha=0.50, shading='auto', zorder=1, rasterized=True)
            hm.set_clip_path(clip_patch)

        # --- Draw pie-chart nodes ---
        for ni in range(n_nodes):
            cx, cy = node_x[ni], node_y[ni]
            n_tp = node_tps[ni]
            n_fn = node_fns[ni]
            n_fp = node_fps[ni]

            total_events = n_tp + n_fn
            if total_events == 0 and n_fp == 0:
                ax.plot(cx, cy, '.', color='#cccccc', markersize=2, zorder=2)
                continue

            # Cap FP display proportion so it doesn't overwhelm the pie
            fp_display = min(n_fp, total_events * 3) if total_events > 0 else n_fp
            pie_total = n_tp + n_fn + fp_display
            if pie_total == 0:
                ax.plot(cx, cy, '.', color='#cccccc', markersize=2, zorder=2)
                continue

            # Draw pie wedges
            start_deg = 90
            slices = [(n_tp, C_TP), (n_fn, C_FN), (fp_display, C_FP)]
            for count, color in slices:
                if count <= 0:
                    continue
                span = 360.0 * count / pie_total
                wedge = MplWedge((cx, cy), pie_r, start_deg, start_deg + span,
                                 fc=color, ec='white', lw=0.2, zorder=5, alpha=0.90)
                ax.add_patch(wedge)
                start_deg += span

            # Thin outline
            outline = plt.Circle((cx, cy), pie_r, fill=False, ec='#555555',
                                 lw=0.3, zorder=6)
            ax.add_patch(outline)

        # --- Outer rim: 3 concentric metric arcs (TD / FN / FP) ---
        # Each arc spans the sector angle; colour intensity = value / max
        tp_val  = metrics[method]['true_positives']['mean']
        fn_val  = metrics[method]['false_negatives']['mean']
        fp_val  = metrics[method]['false_positives']['mean']

        band_h = R_bar / 3.0          # height of each concentric band
        band_gap = 0.004              # thin gap between bands
        arc_start_deg = np.degrees(theta_end)  + 1.5   # slight inset from sector lines
        arc_end_deg   = np.degrees(theta_start) - 1.5

        # Band definitions: (value, max_across_methods, base_radius, dark_colour)
        bands = [
            (tp_val, max_tp,  R,                                '#1565c0'),   # TD: white→dark blue
            (fn_val, max_fn,  R + band_h + band_gap,            '#e6a800'),   # FN: white→dark yellow
            (fp_val, max_fp,  R + 2*(band_h + band_gap),        '#c62828'),   # FP: white→dark red
        ]
        band_labels = ['TD', 'FN', 'FP']

        import matplotlib.colors as mcolors
        for bi, (val, vmax, r_base, dark_col) in enumerate(bands):
            if vmax <= 0:
                # Draw empty white band
                arc = MplWedge((0, 0), r_base + band_h, arc_start_deg, arc_end_deg,
                               width=band_h, fc='white', ec='#cccccc',
                               lw=0.3, alpha=0.70, zorder=8)
                ax.add_patch(arc)
                continue

            # Colour interpolation: white → dark_col based on val/vmax
            t = min(val / vmax, 1.0)  # saturation fraction
            r_c, g_c, b_c = mcolors.to_rgb(dark_col)
            fill_rgb = (1.0 - t + t * r_c, 1.0 - t + t * g_c, 1.0 - t + t * b_c)

            arc = MplWedge((0, 0), r_base + band_h, arc_start_deg, arc_end_deg,
                           width=band_h, fc=fill_rgb, ec='#999999',
                           lw=0.3, alpha=0.85, zorder=8)
            ax.add_patch(arc)

            # Value label inside the arc band
            label_r_band = r_base + band_h * 0.5
            label_angle = (theta_start + theta_end) / 2  # = theta_mid
            lbx = label_r_band * np.cos(label_angle)
            lby = label_r_band * np.sin(label_angle)

            # Use dark text for light fills, white for dark fills
            text_col = 'white' if t > 0.55 else '#333333'
            val_int = int(val)
            if val_int > 0:
                if val_int >= 1000000:
                    val_str = f'{val_int/1000000:.1f}M'
                elif val_int >= 1000:
                    val_str = f'{val_int/1000:.0f}k'
                else:
                    val_str = f'{val_int}'
                # Rotate text tangentially along the arc
                tang_deg = np.degrees(label_angle) + 90
                # Flip if text would be upside-down
                if tang_deg > 90 and tang_deg < 270:
                    tang_deg -= 180
                elif tang_deg < -90:
                    tang_deg += 180
                ax.text(lbx, lby, val_str, ha='center', va='center',
                        fontsize=5.8, fontweight='bold', color=text_col, zorder=9,
                        rotation=tang_deg)

        # --- Method label ---
        label_r = R + R_bar + 0.14
        lx = label_r * np.cos(theta_mid)
        ly = label_r * np.sin(theta_mid)

        ha = 'center'; va = 'center'
        if lx > 0.3:  ha = 'left'
        elif lx < -0.3: ha = 'right'
        if ly > 0.3:  va = 'bottom'
        elif ly < -0.3: va = 'top'

        ax.text(lx, ly, METHOD_LABELS_FULL[method],
                ha=ha, va=va, fontsize=10, fontweight='bold',
                color=get_method_color(method),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=get_method_color(method), alpha=0.92, lw=1.5))

        # DR%
        dr_r = R + R_bar + 0.04
        dr_angle = theta_mid + sector_angle * 0.20
        dr_val = metrics[method]['detection_rate']['mean']
        ax.text(dr_r * np.cos(dr_angle), dr_r * np.sin(dr_angle),
                f'DR: {dr_val:.1f}%', ha='center', va='center', fontsize=7.5,
                color=get_method_color(method), fontweight='bold')

    # ------------------------------------------------------------------
    # Circle outline + inner hub
    # ------------------------------------------------------------------
    circle_outer = Circle((0, 0), R, fill=False, ec='black', lw=2.5, zorder=10)
    ax.add_patch(circle_outer)
    circle_inner = Circle((0, 0), R_inner, fill=True, fc='white', ec='black', lw=2.0, zorder=14)
    ax.add_patch(circle_inner)

    hex_patch = RegularPolygon((0, 0), numVertices=6, radius=R_inner * 0.95,
                               orientation=np.pi / 6, fill=True, fc='#f8f8f0',
                               ec='#333333', lw=1.5, zorder=16)
    ax.add_patch(hex_patch)

    ax.text(0, R_inner * 0.45, f'{num_nodes}', ha='center', va='center',
            fontsize=22, fontweight='bold', zorder=17, color='#333333')
    ax.text(0, R_inner * 0.05, 'NODES', ha='center', va='center',
            fontsize=8, fontweight='bold', zorder=17, color='#666666')
    ax.text(0, -R_inner * 0.35, f'{n_sectors} models', ha='center', va='center',
            fontsize=8, zorder=17, color='#888888')

    # Title + subtitle
    ax.text(0, R + R_bar + 0.58, f'IoT Mesh Network: {num_nodes}-Node Simulation',
            ha='center', fontsize=18, fontweight='bold')
    ax.text(0, R + R_bar + 0.45,
            '\u25cf Blue = TP   \u25cf Yellow = FN   \u25cf Red = FP (proportional)   |   Rim bands: TD \u2022 FN \u2022 FP   |   Grayscale = latency',
            ha='center', fontsize=11, style='italic', color='gray')

    # Legend
    legend_handles = [
        Patch(facecolor=C_TP, edgecolor='#555', label='True Positive (TP)', alpha=0.9),
        Patch(facecolor=C_FN, edgecolor='#555', label='False Negative (FN)', alpha=0.9),
        Patch(facecolor=C_FP, edgecolor='#555', label='False Positive (FP)', alpha=0.9),
        Patch(facecolor='#AAAAAA', edgecolor='none', label='Latency heatmap (darker = higher)', alpha=0.5),
        Line2D([0], [0], color='white', marker='s', markersize=12,
               markerfacecolor='#1565c0', markeredgecolor='#999', label='Rim: TD (inner band)'),
        Line2D([0], [0], color='white', marker='s', markersize=12,
               markerfacecolor='#e6a800', markeredgecolor='#999', label='Rim: FN (middle band)'),
        Line2D([0], [0], color='white', marker='s', markersize=12,
               markerfacecolor='#c62828', markeredgecolor='#999', label='Rim: FP (outer band)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(0.0, 1.0), fontsize=10,
              frameon=True, fancybox=True, shadow=True)

    margin = R_bar + 0.75
    ax.set_xlim(-R - margin, R + margin)
    ax.set_ylim(-R - margin - 0.10, R + margin + 0.25)
    ax.axis('off')

    # ------------------------------------------------------------------
    # Stats table below
    # ------------------------------------------------------------------
    ax_table = fig.add_subplot(gs_main[1])
    ax_table.axis('off')

    col_labels = ['Method', 'DR%', 'FP', 'FN', 'Prec%', 'FAR/hr', 'Lat ms', '99th ms', 'Net kB/hr']
    table_data = []

    for m in methods_to_plot:
        dr = metrics[m]['detection_rate']['mean']
        fp = metrics[m]['false_positives']['mean']
        fn = metrics[m]['false_negatives']['mean']
        pr = metrics[m]['precision']['mean']
        far = metrics[m]['false_alarm_rate']['mean']
        lat = metrics[m]['latency_mean_ms']['mean']
        lat99 = metrics[m]['latency_99th_ms']['mean']
        net = metrics[m]['network_load_bytes_per_hour']['mean'] / 1000.0

        table_data.append([
            METHOD_LABELS_SHORT[m],
            f'{dr:.1f}%',
            f'{int(fp):,}',
            f'{int(fn):,}',
            f'{pr:.1f}%',
            f'{far:.1f}',
            f'{lat:.1f}',
            f'{lat99:.1f}',
            f'{net:.1f}',
        ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='upper center',
        bbox=[0.05, 0.30, 0.90, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', fontweight='bold')

    # Colour-code rows
    for i, m in enumerate(methods_to_plot):
        color = get_method_color(m)
        table[i+1, 0].set_facecolor(color)
        table[i+1, 0].set_text_props(color='white', fontweight='bold')
        fp_val = metrics[m]['false_positives']['mean']
        fn_val = metrics[m]['false_negatives']['mean']
        if fp_val > 0:
            table[i+1, 2].set_facecolor('#FFE0E0')
        if fn_val > 0:
            table[i+1, 3].set_facecolor('#FFF3D0')
        dr_val = metrics[m]['detection_rate']['mean']
        if dr_val >= 100.0 and fp_val == 0:
            for j in range(1, len(col_labels)):
                table[i+1, j].set_facecolor('#E8F5E9')

    ax_table.text(0.5, 0.18,
                  'Each node circle: Blue = detected (TP) | Yellow = missed (FN) | '
                  'Red = false positives (FP, capped at 3\u00d7 events for readability)',
                  ha='center', va='top', fontsize=9, style='italic', color='#666666',
                  transform=ax_table.transAxes)
    ax_table.text(0.5, 0.10,
                  'Outer rim: 3 concentric bands (inner\u2192outer = TD, FN, FP).  '
                  'Colour intensity \u221d value / max across methods (white = 0, saturated = max).',
                  ha='center', va='top', fontsize=9, style='italic', color='#666666',
                  transform=ax_table.transAxes)

    save_figure(fig, output_dir, 'fig2_hexagonal', num_nodes)
    plt.close()


# FIGURE 3: WAVEFORM GRID (updated for 6 models)
# =============================================================================

def plot_waveform_grid(meta, data, output_dir, num_nodes):
    """Plot waveform grid with trigger markers from all 6 methods"""

    metrics_dummy, available = extract_metrics.__wrapped__(meta) if hasattr(extract_metrics, '__wrapped__') else (None, ALL_METHODS)
    # We just use ALL_METHODS here since we check key existence anyway

    num_snaps = min(6, len(meta['snapshots']))
    num_cols = min(8, len(meta['snapshots'][0]['nodes']))

    fig, axes = plt.subplots(num_snaps, num_cols, figsize=(2.5*num_cols, 2*num_snaps))

    # Build legend string
    legend_parts = []
    for m in ALL_METHODS:
        legend_parts.append(f'{METHOD_LABELS_SHORT[m]}={get_method_color(m)}')

    fig.suptitle(f'Waveform Snapshots ({num_nodes} Nodes) — 6-Model Triggers', fontsize=14, fontweight='bold')

    sample_rate = meta['sample_rate']
    nodes = meta['snapshots'][0]['nodes'][:num_cols]
    snap_indices = np.linspace(0, len(meta['snapshots'])-1, num_snaps, dtype=int)

    for row, snap_idx in enumerate(snap_indices):
        snap_info = meta['snapshots'][snap_idx]
        for col, node_id in enumerate(nodes):
            ax = axes[row, col] if num_snaps > 1 else axes[col]

            key = f'snap{snap_idx}_node{node_id}_samples'
            if key in data.files:
                samples = data[key]
                t = np.arange(len(samples)) / sample_rate
                ax.plot(t, samples, 'b-', linewidth=0.2)

                # Overlay triggers from all methods
                for method in ALL_METHODS:
                    trig_key = f'snap{snap_idx}_node{node_id}_triggers_{method}'
                    if trig_key in data.files and len(data[trig_key]) > 0:
                        for trig in data[trig_key]:
                            rel_t = trig - snap_info['timestamp']
                            if 0 <= rel_t <= t[-1]:
                                lw = 1.0 if method == 'proposed' else 0.5
                                ax.axvline(x=rel_t, color=get_method_color(method),
                                           linewidth=lw, alpha=0.6)

            if row == 0:
                ax.set_title(f'N{node_id}', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{snap_info["timestamp"]/3600:.1f}h', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    # Add legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=get_method_color(m), linewidth=2,
                              label=METHOD_LABELS_SHORT[m]) for m in ALL_METHODS]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
               frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, output_dir, 'fig3_waveforms', num_nodes)
    plt.close()


# =============================================================================
# FIGURE 4: TRIGGER STATISTICS (6-model)
# =============================================================================

def plot_trigger_stats(meta, data, output_dir, num_nodes):
    """Plot trigger statistics over time for all 6 methods"""

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    fig.suptitle(f'Trigger Statistics ({num_nodes} Nodes) — All Models', fontsize=14, fontweight='bold')

    nodes = meta['snapshots'][0]['nodes']

    # Collect trigger counts per snapshot per method
    times = []
    counts = {m: [] for m in ALL_METHODS}

    for snap_idx, snap_info in enumerate(meta['snapshots']):
        times.append(snap_info['timestamp'] / 3600)
        for method in ALL_METHODS:
            method_count = 0
            for node_id in nodes:
                trig_key = f'snap{snap_idx}_node{node_id}_triggers_{method}'
                if trig_key in data.files:
                    method_count += len(data[trig_key])
            counts[method].append(method_count)

    times = np.array(times)

    # Panel 1: Cumulative triggers
    ax = fig.add_subplot(gs[0, 0])
    for method in ALL_METHODS:
        c = np.cumsum(counts[method])
        ax.plot(times, c, '-', color=get_method_color(method), linewidth=2,
                label=METHOD_LABELS_SHORT[method], marker='o', markersize=3)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cumulative Triggers')
    ax.set_title('(a) Cumulative Triggers', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-snapshot grouped bars
    ax = fig.add_subplot(gs[0, 1])
    n_snaps = len(times)
    n_m = len(ALL_METHODS)
    bar_width = 0.12
    if n_snaps > 1:
        max_group_width = (times[1] - times[0]) * 0.8
        bar_width = max_group_width / n_m

    for mi, method in enumerate(ALL_METHODS):
        offsets = times + (mi - n_m/2 + 0.5) * bar_width
        ax.bar(offsets, counts[method], width=bar_width*0.9,
               color=get_method_color(method), alpha=0.8, label=METHOD_LABELS_SHORT[method])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Triggers per Snapshot')
    ax.set_title('(b) Per Snapshot', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Ratio relative to Proposed
    ax = fig.add_subplot(gs[0, 2])
    for method in ALL_METHODS:
        if method == 'proposed':
            continue
        ratios = [c / max(p, 1) for c, p in zip(counts[method], counts['proposed'])]
        ax.plot(times, ratios, '-', color=get_method_color(method), linewidth=2,
                label=METHOD_LABELS_SHORT[method], marker='s', markersize=3)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Proposed (1.0)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Ratio vs Proposed')
    ax.set_title('(c) Trigger Ratio (method / Proposed)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: Total triggers bar chart
    ax = fig.add_subplot(gs[1, 0])
    totals = {m: sum(counts[m]) for m in ALL_METHODS}
    labels = [METHOD_LABELS_SHORT[m] for m in ALL_METHODS]
    vals = [totals[m] for m in ALL_METHODS]
    colors_list = [get_method_color(m) for m in ALL_METHODS]
    bars = ax.bar(labels, vals, color=colors_list, alpha=0.85, edgecolor='black')
    ax.set_ylabel('Total Triggers')
    ax.set_title('(d) Total Triggers', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f'{val:,}', ha='center', fontsize=8, fontweight='bold')

    # Panel 5: Trigger variability (std across snapshots)
    ax = fig.add_subplot(gs[1, 1])
    means = [np.mean(counts[m]) for m in ALL_METHODS]
    stds = [np.std(counts[m]) for m in ALL_METHODS]
    bars = ax.bar(labels, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', capsize=4)
    ax.set_ylabel('Triggers / Snapshot')
    ax.set_title('(e) Mean ± Std per Snapshot', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    # Panel 6: Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    summary_lines = [
        f"SNAPSHOT SUMMARY",
        f"{'═'*40}",
        f"Snapshots: {len(meta['snapshots'])}",
        f"Duration: {times[-1]:.1f} hours",
        f"",
    ]
    for m in ALL_METHODS:
        total = totals[m]
        avg = np.mean(counts[m])
        summary_lines.append(f"{METHOD_LABELS_SHORT[m]:<10}: {total:>7,} total  ({avg:>6.1f}/snap)")

    prop_total = totals['proposed']
    summary_lines.append(f"")
    summary_lines.append(f"Ratios vs Proposed:")
    for m in ALL_METHODS:
        if m == 'proposed':
            continue
        ratio = totals[m] / max(prop_total, 1)
        summary_lines.append(f"  {METHOD_LABELS_SHORT[m]:<10}: {ratio:.1f}x")

    summary = '\n'.join(summary_lines)
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10, va='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_figure(fig, output_dir, 'fig4_trigger_stats', num_nodes)
    plt.close()


# =============================================================================
# FIGURE 5: PUBLICATION BAR CHARTS (6-model, Seaborn styled)
# =============================================================================

def plot_publication_bars(results, num_nodes, output_dir):
    """Publication-quality grouped bar charts for all 6 methods"""

    metrics, available = extract_metrics(results, str(num_nodes))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Method Comparison: {num_nodes}-Node Network',
                 fontsize=16, fontweight='bold', y=1.02)

    labels = [METHOD_LABELS[m] for m in available]
    colors_list = [get_method_color(m) for m in available]

    chart_configs = [
        (axes[0, 0], 'detection_rate', 'Detection Rate (%)', '(a) Detection Rate', (0, 115), '{:.1f}%', True, None),
        (axes[0, 1], 'false_alarm_rate', 'FAR (/hr/node)', '(b) False Alarm Rate', None, '{:.1f}', False, None),
        (axes[0, 2], 'precision', 'Precision (%)', '(c) Precision', (0, 115), '{:.1f}%', False, None),
        (axes[1, 0], 'latency_mean_ms', 'Mean Latency (ms)', '(d) Detection Latency', None, '{:.1f}', False, None),
        (axes[1, 1], 'false_positives', 'False Positives', '(e) False Positives', None, '{:,.0f}', False, 'log'),
        (axes[1, 2], 'network_load_bytes_per_hour', 'Network Load (kB/hr)', '(f) Network Load', None, '{:.1f}', False, None),
    ]

    for ax, metric_key, ylabel, title, ylim, fmt, show_err, yscale in chart_configs:
        vals = []
        stds = []
        for m in available:
            v = metrics[m][metric_key]['mean']
            s = metrics[m][metric_key]['std']
            if metric_key == 'network_load_bytes_per_hour':
                v /= 1000.0
                s /= 1000.0
            if metric_key == 'false_positives' and v <= 0:
                v = 0.1
            vals.append(v)
            stds.append(s)

        if show_err:
            bars = ax.bar(labels, vals, yerr=stds, color=colors_list, alpha=0.85,
                          edgecolor='black', linewidth=1.2, capsize=6)
        else:
            bars = ax.bar(labels, vals, color=colors_list, alpha=0.85,
                          edgecolor='black', linewidth=1.2)

        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        if ylim:
            ax.set_ylim(ylim)
        if yscale:
            ax.set_yscale(yscale)
        ax.tick_params(axis='x', rotation=0, labelsize=8)

        for bar, val in zip(bars, vals):
            raw_val = val
            label_text = fmt.format(raw_val)
            if yscale == 'log' and raw_val > 0:
                ypos = bar.get_height() * 1.5
            else:
                ypos = bar.get_height() + max(vals) * 0.04
            ax.text(bar.get_x() + bar.get_width()/2, ypos, label_text,
                    ha='center', fontsize=8, fontweight='bold')

        sns.despine(ax=ax)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig5_publication', num_nodes)
    plt.close()


# =============================================================================
# =============================================================================
# FIGURE 6: RADAR / SPIDER CHART (IMPROVED)
# =============================================================================

def plot_radar_chart(results, num_nodes, output_dir):
    """
    Radar chart overlaying methods on normalized axes.
    - SDT excluded (0% DR, computes nothing - 0 latency is meaningless)
    - Overlapping lines jittered radially so both remain visible
    - Thicker lines, distinctive dash patterns, markers
    """

    metrics, available = extract_metrics(results, str(num_nodes))

    # Exclude SDT
    plot_methods = [m for m in available if m != 'sdt']

    radar_metrics = [
        ('Detection Rate', 'detection_rate', False),
        ('Precision', 'precision', False),
        ('Low FAR', 'false_alarm_rate', True),
        ('Low Latency', 'latency_mean_ms', True),
        ('Low FP', 'false_positives', True),
        ('Low Net Load', 'network_load_bytes_per_hour', True),
    ]

    N = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([rm[0] for rm in radar_metrics], fontsize=12, fontweight='bold')

    raw_vals = {}
    for m in plot_methods:
        raw_vals[m] = [metrics[m][key]['mean'] for _, key, _ in radar_metrics]

    norm_vals = {m: [0.0]*N for m in plot_methods}
    for i, (_, key, invert) in enumerate(radar_metrics):
        axis_vals = [raw_vals[m][i] for m in plot_methods]
        v_min, v_max = min(axis_vals), max(axis_vals)
        rng = v_max - v_min if v_max != v_min else 1.0
        for m in plot_methods:
            normalized = (raw_vals[m][i] - v_min) / rng
            if invert:
                normalized = 1.0 - normalized
            norm_vals[m][i] = 0.1 + 0.9 * normalized

    # Jitter overlapping values
    jitter_vals = {m: list(norm_vals[m]) for m in plot_methods}
    for i in range(N):
        pairs = sorted([(m, jitter_vals[m][i]) for m in plot_methods], key=lambda x: x[1])
        for j in range(1, len(pairs)):
            if abs(pairs[j][1] - pairs[j-1][1]) < 0.04:
                pairs[j-1] = (pairs[j-1][0], pairs[j-1][1] - 0.022)
                pairs[j] = (pairs[j][0], pairs[j][1] + 0.022)
        for m, v in pairs:
            jitter_vals[m][i] = np.clip(v, 0.05, 1.12)

    line_styles = {
        'proposed': ('-',  3.5, 'o', 10, 10),
        'zhang':    ('--', 2.8, 's', 8,  7),
        'stft':     ('-',  2.8, '^', 9,  8),
        'dedar':    ('-.', 2.8, 'D', 8,  6),
        'tinyml':   (':',  3.2, 'v', 8,  5),
    }

    for m in plot_methods:
        vals = jitter_vals[m] + jitter_vals[m][:1]
        ls, lw, mk, ms, zo = line_styles.get(m, ('-', 2.0, 'o', 6, 5))
        ax.plot(angles, vals, color=get_method_color(m),
                linestyle=ls, linewidth=lw, marker=mk, markersize=ms,
                markeredgecolor='white', markeredgewidth=0.8,
                label=METHOD_LABELS_SHORT[m], zorder=zo)
        ax.fill(angles, vals, color=get_method_color(m), alpha=0.05)

    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color='gray')
    ax.grid(True, alpha=0.3)

    ax.text(0.5, -0.08,
            'SDT (Correa 2019) excluded: DR = 0%, latency undefined (no computation)',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='#888888')

    ax.legend(loc='lower right', bbox_to_anchor=(1.30, -0.05), fontsize=11,
              frameon=True, fancybox=True, shadow=True)

    ax.set_title(f'Normalized Performance Radar \u2014 {num_nodes} Nodes\n(Higher = Better on all axes)',
                 fontsize=14, fontweight='bold', pad=30)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig6_radar', num_nodes)
    plt.close()


# MAIN
# =============================================================================

def main():
    """Main entry point - uses configuration from top of file"""

    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve paths relative to script location
    data_dir = os.path.join(script_dir, DATA_DIR)
    snapshots_dir = os.path.join(script_dir, SNAPSHOTS_DIR)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)

    print("="*60)
    print("IoT Mesh Network Visualization Suite — 6-Model Edition")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data directory:      {data_dir}")
    print(f"  Snapshots directory: {snapshots_dir}")
    print(f"  Output directory:    {output_dir}")
    print(f"  Filename prefix:     '{FILENAME_PREFIX}'")
    print(f"  Models:              {', '.join(METHOD_LABELS_SHORT.values())}")

    os.makedirs(output_dir, exist_ok=True)

    # Find results JSON in data_dir
    json_files = glob.glob(os.path.join(data_dir, '*results*.json'))
    json_files = [f for f in json_files if 'meta' not in f]

    if not json_files:
        print(f"\nERROR: No results JSON found in {data_dir}")
        sys.exit(1)

    results_file = json_files[0]
    print(f"\nResults file: {results_file}")

    # Load results JSON first to get num_nodes
    with open(results_file) as f:
        results = json.load(f)
    num_nodes = get_network_size(results)
    print(f"Network size: {num_nodes} nodes")

    # Detect available methods
    size_key = str(num_nodes)
    if size_key in results:
        detected = [m for m in ALL_METHODS if m in results[size_key]]
        print(f"Methods found: {', '.join(detected)}")
    else:
        detected = []
        print("WARNING: No method data found in results")

    # Try to find snapshot data - two possible formats:
    # 1. Combined format: *_meta.json + *_data.npz in data_dir
    # 2. Individual format: snapshot_XXXX_tYYYs.npz files in snapshots_dir

    meta, data = None, None

    # Check for combined format first
    meta_files = glob.glob(os.path.join(data_dir, '*_meta.json'))
    npz_files = glob.glob(os.path.join(data_dir, '*_data.npz'))

    if meta_files and npz_files:
        meta_file = meta_files[0]
        data_file = npz_files[0]
        print(f"\nFound combined snapshot format:")
        print(f"  Metadata: {meta_file}")
        print(f"  Data: {data_file}")

        with open(meta_file) as f:
            meta = json.load(f)
        data = np.load(data_file)
        print(f"  Snapshots: {meta['num_snapshots']}")
        print(f"  Arrays loaded: {len(data.files)}")

    else:
        # Check for individual snapshot files (continuous save format)
        snapshot_files = sorted(glob.glob(os.path.join(snapshots_dir, 'snapshot_*.npz')))

        if snapshot_files:
            print(f"\nFound {len(snapshot_files)} individual snapshot files in {snapshots_dir}")
            print("  Loading and combining snapshots...")

            # Build metadata from files
            meta = {
                'num_snapshots': len(snapshot_files),
                'sample_rate': 100.0,
                'snapshot_duration': 60,
                'snapshot_interval': 1800,
                'num_nodes': num_nodes,
                'snapshots': []
            }

            # Combine all snapshot data into a single dict
            combined_data = {}

            for snap_idx, snap_file in enumerate(snapshot_files):
                fname = os.path.basename(snap_file)
                try:
                    timestamp = float(fname.split('_t')[1].replace('s.npz', ''))
                except:
                    timestamp = snap_idx * 1800

                snap_data = np.load(snap_file)

                nodes_in_snap = set()
                for key in snap_data.files:
                    if '_node' in key:
                        parts = key.split('_node')
                        if len(parts) > 1:
                            try:
                                node_id = int(parts[1].split('_')[0])
                                nodes_in_snap.add(node_id)
                            except:
                                pass

                nodes_list = sorted(list(nodes_in_snap)) if nodes_in_snap else list(range(1, num_nodes))

                meta['snapshots'].append({
                    'index': snap_idx,
                    'timestamp': timestamp,
                    'duration': 60,
                    'nodes': nodes_list
                })

                for key in snap_data.files:
                    if key.startswith('node'):
                        new_key = f"snap{snap_idx}_{key}"
                        combined_data[new_key] = snap_data[key]
                    elif '_node' in key:
                        new_key = f"snap{snap_idx}_{key}"
                        combined_data[new_key] = snap_data[key]
                    else:
                        combined_data[f"snap{snap_idx}_{key}"] = snap_data[key]

                snap_data.close()

            class CombinedData:
                def __init__(self, data_dict):
                    self._data = data_dict
                    self.files = list(data_dict.keys())

                def __getitem__(self, key):
                    return self._data.get(key, np.array([]))

                def __contains__(self, key):
                    return key in self._data

            data = CombinedData(combined_data)
            print(f"  Combined {len(snapshot_files)} snapshots, {len(data.files)} arrays")

            if meta['snapshots']:
                print(f"  Nodes per snapshot: {len(meta['snapshots'][0]['nodes'])}")
                print(f"  Time range: 0 - {meta['snapshots'][-1]['timestamp']/3600:.1f} hours")
        else:
            print(f"\nNo snapshot data found in {data_dir} or {snapshots_dir}")
            print("  Will generate JSON-only figures")

    # Generate figures
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60 + "\n")

    fig_count = 0
    total_figs = sum(GENERATE_FIGURES.values())

    if meta and data:
        if GENERATE_FIGURES.get('dashboard', True):
            fig_count += 1
            print(f"[{fig_count}/{total_figs}] Dashboard (6-model)...")
            plot_dashboard(results, meta, data, output_dir, num_nodes)

        if GENERATE_FIGURES.get('hexagonal', True):
            fig_count += 1
            print(f"[{fig_count}/{total_figs}] Hexagonal sextant plot...")
            plot_hexagonal_sextant(results, num_nodes, output_dir)

        if GENERATE_FIGURES.get('waveforms', True):
            fig_count += 1
            print(f"[{fig_count}/{total_figs}] Waveform grid (6-model)...")
            plot_waveform_grid(meta, data, output_dir, num_nodes)

        if GENERATE_FIGURES.get('trigger_stats', True):
            fig_count += 1
            print(f"[{fig_count}/{total_figs}] Trigger statistics (6-model)...")
            plot_trigger_stats(meta, data, output_dir, num_nodes)

        if GENERATE_FIGURES.get('publication_bars', True):
            fig_count += 1
            print(f"[{fig_count}/{total_figs}] Publication bar charts (6-model)...")
            plot_publication_bars(results, num_nodes, output_dir)

        if GENERATE_FIGURES.get('radar', True):
            fig_count += 1
            print(f"[{fig_count}/{total_figs}] Radar chart (6-model)...")
            plot_radar_chart(results, num_nodes, output_dir)

    else:
        # JSON-only figures (no snapshot data needed)
        json_figs = ['hexagonal', 'publication_bars', 'radar']
        total_json = sum(1 for f in json_figs if GENERATE_FIGURES.get(f, True))

        if GENERATE_FIGURES.get('hexagonal', True):
            fig_count += 1
            print(f"[{fig_count}/{total_json}] Hexagonal sextant plot...")
            plot_hexagonal_sextant(results, num_nodes, output_dir)

        if GENERATE_FIGURES.get('publication_bars', True):
            fig_count += 1
            print(f"[{fig_count}/{total_json}] Publication bar charts (6-model)...")
            plot_publication_bars(results, num_nodes, output_dir)

        if GENERATE_FIGURES.get('radar', True):
            fig_count += 1
            print(f"[{fig_count}/{total_json}] Radar chart (6-model)...")
            plot_radar_chart(results, num_nodes, output_dir)

    print("\n" + "="*60)
    print(f"COMPLETE! Figures saved to: {output_dir}")
    print("="*60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(glob.glob(os.path.join(output_dir, f'*{num_nodes}nodes*'))):
        print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    main()