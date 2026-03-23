#!/usr/bin/env python3
"""
Discrete-Event Simulation: IoT Mesh Network Event Triggering
Compares: Temporal Spectral Noise-Floor Adaptation vs Zhang et al. 2023

Enhanced version with detailed logging and progress tracking.

Author: GNACODE INC
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import heapq
from enum import Enum
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import json
import time
import sys
import os
from datetime import datetime, timedelta

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class Logger:
    """Simple logger with levels and timestamps"""
    
    LEVELS = {'DEBUG': 0, 'INFO': 1, 'PROGRESS': 2, 'WARNING': 3, 'ERROR': 4}
    
    def __init__(self, level: str = 'INFO', show_timestamp: bool = True):
        self.level = self.LEVELS.get(level.upper(), 1)
        self.show_timestamp = show_timestamp
        self.start_time = time.time()
        
    def _format(self, level: str, msg: str) -> str:
        elapsed = time.time() - self.start_time
        if self.show_timestamp:
            return f"[{elapsed:8.1f}s] [{level:8s}] {msg}"
        return f"[{level:8s}] {msg}"
    
    def debug(self, msg: str):
        if self.level <= self.LEVELS['DEBUG']:
            print(self._format('DEBUG', msg))
    
    def info(self, msg: str):
        if self.level <= self.LEVELS['INFO']:
            print(self._format('INFO', msg))
    
    def progress(self, msg: str):
        if self.level <= self.LEVELS['PROGRESS']:
            print(self._format('PROGRESS', msg))
    
    def warning(self, msg: str):
        if self.level <= self.LEVELS['WARNING']:
            print(self._format('WARNING', msg))
    
    def error(self, msg: str):
        if self.level <= self.LEVELS['ERROR']:
            print(self._format('ERROR', msg))
    
    def section(self, title: str):
        """Print a section header"""
        if self.level <= self.LEVELS['INFO']:
            print("\n" + "="*70)
            print(f" {title}")
            print("="*70)
    
    def subsection(self, title: str):
        """Print a subsection header"""
        if self.level <= self.LEVELS['INFO']:
            print(f"\n--- {title} ---")


# Global logger instance
log = Logger(level='INFO')


# =============================================================================
# ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗ ██╗   ██╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗██║   ██║██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║██║   ██║██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
# =============================================================================
# EDIT THESE SETTINGS TO CONTROL THE SIMULATION
# =============================================================================

# --- SPEED vs ACCURACY ---
# Options: 'FAST', 'MEDIUM', 'ACCURATE', 'OVERNIGHT'
#   FAST:      ~1 min,  1 hour sim,  2 MC runs  (quick test)
#   MEDIUM:    ~5 min,  4 hour sim,  3 MC runs  (default)
#   ACCURATE:  ~30 min, 24 hour sim, 5 MC runs  (paper quality)
#   OVERNIGHT: ~1 hr,   24 hour sim, 10 MC runs (publication quality)
SIMULATION_PRESET = 'ACCURATE'

# --- NETWORK SIZES TO SIMULATE ---
# Set to True/False to include/exclude each network size
RUN_10_NODES = False
RUN_50_NODES = True
RUN_1000_NODES = False

# --- ALGORITHM PARAMETERS (from real hardware data) ---
# γ_d: Digital noise filter window [3-5]
GAMMA_D = 3

# γ_a: Long-term adaptation window [64-128]  
GAMMA_A = 64

# ζ: Threshold coefficient (threshold = ζ × noise_floor)
# From real hardware: noise/threshold ≈ 0.17, so ζ = 6
ZETA = 6.0

# --- EVENT PARAMETERS ---
# Event rate (events per hour per node)
EVENT_RATE = 1.0

# Event SNR in dB (must exceed 20*log10(ζ) ≈ 16 dB to be detected)
EVENT_SNR_DB = 18.0

# Event frequency band (Hz) - human movement detection
EVENT_FREQ_LOW = 1.0
EVENT_FREQ_HIGH = 5.0

# --- ZHANG METHOD PARAMETERS (for comparison) ---
# Zhang operates in TIME-DOMAIN (no FFT filtering), but processes per-FRAME
# Uses same 6x threshold as proposed for fair comparison
ZHANG_THRESHOLD = 6.0  # Trigger when frame_max > threshold × noise_floor
ZHANG_BETA = 0.95      # Smoothing factor for adaptive noise floor
ZHANG_DECIMATION = 15  # (legacy - not used in frame-based mode)

# --- OUTPUT ---
# Save results to JSON file (use /mnt/user-data/outputs/ for downloadable files)
SAVE_RESULTS = True
OUTPUT_DIR = 'U://PAYROLL/datasim'  # Directory for all output files
RESULTS_FILENAME = f'{OUTPUT_DIR}/simulation_results-50-A17.json'

# --- RAW DATA SNAPSHOTS ---
# For long simulations, save periodic raw data samples
ENABLE_SNAPSHOTS = True            # Save raw waveforms (set False to disable)
SNAPSHOT_DURATION_SEC = 60         # Duration of each snapshot (e.g., 1 min = 60 sec)
SNAPSHOT_INTERVAL_SEC = 1800       # Interval between snapshots (e.g., 30 min = 1800 sec)
SNAPSHOT_NODES = 'ALL'             # 'ALL' or list of node IDs e.g. [1, 2, 3]

# --- CONTINUOUS SAVING (for long simulations) ---
CONTINUOUS_SAVE = True             # Save data continuously (don't wait until end)
CHECKPOINT_INTERVAL_SEC = 3600    # Save checkpoint results every N simulated seconds (e.g., 1 hour)
SNAPSHOT_OUTPUT_DIR = f'{OUTPUT_DIR}/snapshots'  # Directory for snapshot files

# --- NOISE MODEL ---
# Fast noise (always present, high frequency)
NOISE_EMI_FREQ = 60.0              # Power line frequency (Hz) - 50 or 60
NOISE_EMI_AMPLITUDE = 0.3          # Relative to base noise (0-1)
NOISE_DIGITAL_PROB = 0.1           # Probability of digital burst per frame
NOISE_DIGITAL_FREQ_MIN = 800       # Digital noise frequency range (Hz)
NOISE_DIGITAL_FREQ_MAX = 2000

# Environmental noise (per-node varying)
NOISE_ENV_ENABLED = True           # Enable environmental noise sources
NOISE_ENV_RAIN_PROB = 0.05         # Probability of rain starting per hour
NOISE_ENV_WIND_PROB = 0.1          # Probability of wind gust per hour
NOISE_ENV_MOTOR_PROB = 0.02        # Probability of motor/machinery nearby

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class ProgressTracker:
    """Track and display simulation progress"""
    
    def __init__(self, total: float, description: str = "Progress", 
                 update_interval: float = 5.0):
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update = 0
        self.last_progress = 0
        
    def update(self, current: float, force: bool = False):
        """Update progress display"""
        now = time.time()
        progress = current / self.total
        
        if force or (now - self.last_update) >= self.update_interval:
            elapsed = now - self.start_time
            
            if progress > 0:
                eta_seconds = (elapsed / progress) * (1 - progress)
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "calculating..."
            
            # Calculate rate
            rate = current / elapsed if elapsed > 0 else 0
            
            # Progress bar
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # Clear line and print progress
            sys.stdout.write(f"\r{self.description}: [{bar}] {progress*100:5.1f}% | "
                           f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                           f"ETA: {eta_str} | "
                           f"Rate: {rate:.1f}/s    ")
            sys.stdout.flush()
            
            self.last_update = now
            self.last_progress = progress
    
    def finish(self):
        """Mark progress as complete"""
        elapsed = time.time() - self.start_time
        print(f"\r{self.description}: [{'█'*30}] 100.0% | "
              f"Completed in {timedelta(seconds=int(elapsed))}              ")


# =============================================================================
# CONFIGURATION
# =============================================================================

# =============================================================================
# TIME PRESETS - Easy configuration for simulation speed vs accuracy
# =============================================================================

class TimePreset:
    """Predefined time configurations for low-frequency sensing (1-5 Hz)
    
    For human movement detection at 1-5 Hz, we need:
    - Sample rate: 100 Hz (adequate for 5 Hz max frequency)
    - FFT size: 128 → frequency resolution = 0.78 Hz/bin
    - Frame duration: 1.28 sec (N/f_s = 128/100)
    """
    
    # FAST: Quick validation runs
    FAST = {
        'name': 'FAST',
        'duration_hours': 1,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 2,
        'description': 'Quick validation (~1 min for 10 nodes)'
    }
    
    # MEDIUM: Reasonable accuracy with moderate runtime
    MEDIUM = {
        'name': 'MEDIUM', 
        'duration_hours': 4,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 3,
        'description': 'Balanced accuracy/speed (~5 min for 10 nodes)'
    }
    
    # ACCURATE: Full simulation with realistic timing
    ACCURATE = {
        'name': 'ACCURATE',
        'duration_hours': 24,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 5,
        'description': 'Publication quality (~30 min for 10 nodes)'
    }
    
    # OVERNIGHT: Maximum accuracy for paper submission
    OVERNIGHT = {
        'name': 'OVERNIGHT',
        'duration_hours': 24,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 10,
        'description': 'Maximum statistical confidence (~1 hour total)'
    }
    
    @classmethod
    def list_presets(cls):
        """Print available presets"""
        print("\nAvailable Time Presets:")
        print("-" * 60)
        for name in ['FAST', 'MEDIUM', 'ACCURATE', 'OVERNIGHT']:
            preset = getattr(cls, name)
            print(f"  {name:12} - {preset['description']}")
            print(f"               Duration: {preset['duration_hours']}h, "
                  f"Frame: {preset['frame_duration']*1000:.1f}ms, "
                  f"MC runs: {preset['monte_carlo_runs']}")
        print("-" * 60)


@dataclass
class SimulationConfig:
    """Simulation parameters"""
    # Network
    num_nodes: int = 10
    area_size: float = 500.0  # meters (reduced for better connectivity)
    comm_radius: float = 150.0  # meters
    data_rate: float = 250e3  # bits/sec (IEEE 802.15.4)
    
    # Timing - FOR LOW-FREQUENCY SENSING (1-5 Hz human movement)
    # Need adequate frequency resolution: Δf = f_s / N
    # For 1-5 Hz band with bins 1-6: need Δf ≈ 0.78 Hz → f_s = 100 Hz, N = 128
    simulation_duration: float = 4 * 3600  # seconds (default 4 hours)
    frame_duration: float = 1.28  # 128 samples @ 100 Hz = 1.28 sec
    fft_size: int = 128  # N=128 samples
    sample_rate: float = 100.0  # 100 Hz (adequate for 1-5 Hz events)
    
    # Events - HUMAN MOVEMENT DETECTION
    # Events at 1-5 Hz are slow, sustained changes
    # Frame duration = 1.28 sec, so event must last > γ_d × 1.28 = 3.84 sec
    event_rate: float = 1.0      # events/hour/node
    event_duration: float = 5.0  # 5 seconds - human movement (persists > γ_d frames)
    event_decay_tau: float = 2.0 # 2 sec decay time constant (slow, sustained)
    event_snr: float = 18.0      # dB above noise (must exceed ζ=6 threshold, i.e., >16 dB)
    
    # Noise model
    base_noise_power: float = 1.0
    noise_variation_db: float = 6.0  # ±6 dB variation
    noise_cycle_period: float = 3600.0  # 1 hour cycle
    
    # EMI and fast noise
    emi_freq: float = 60.0             # Power line frequency (Hz)
    emi_amplitude: float = 0.3         # Relative to base noise
    digital_noise_prob: float = 0.1    # Probability per frame
    
    # Environmental noise (per-node)
    env_noise_enabled: bool = True     # Enable rain/wind/motor noise
    
    # Raw data snapshots
    enable_snapshots: bool = False
    snapshot_duration: float = 300.0   # seconds (5 min default)
    snapshot_interval: float = 3600.0  # seconds (1 hour default)
    snapshot_nodes: str = 'ALL'        # 'ALL' or list of node IDs
    
    # Continuous saving (for long simulations)
    continuous_save: bool = True       # Save snapshots immediately when completed
    checkpoint_interval: float = 3600.0  # Save checkpoint every N simulated seconds
    snapshot_output_dir: str = ''      # Directory for snapshot files (empty = same as results)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PROPOSED METHOD PARAMETERS - PAPER NOMENCLATURE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # γ_d (gamma_d) ∈ [3, 5]: Digital noise filter window
    # Removes transient spikes that don't persist > γ_d samples
    # True events (human movement) persist > γ_d samples → pass through
    gamma_d: int = 3
    
    # γ_a (gamma_a) ∈ [64, 128]: Long-term adaptation window
    # Higher γ_a = more smoothing = more stable N_k = HIGHER sensitivity
    # Equivalent α = 1 - 1/γ_a (γ_a=64 → α=0.984, γ_a=128 → α=0.992)
    gamma_a: int = 64
    
    # ζ_k (zeta_k): Threshold coefficient per frequency bin k
    # Trigger if X̄_k > ζ_k × N_k
    # From real hardware data: Threshold ≈ 6× average noise level
    # This gives 0 false positives while detecting events with SNR > 16 dB
    zeta_k: float = 6.0
    
    # Noise floor update control: only update N_k if X̄_k < ratio × Threshold
    noise_update_ratio: float = 0.8  # Update when clearly below threshold
    
    # Frequency band of interest: 1-5 Hz (human movement, very low frequency)
    # EMI (50/60 Hz) and digital noise (kHz) are outside this band → ignored
    # Events are consistent over > γ_d samples when they occur
    event_freq_low: float = 1.0    # Hz
    event_freq_high: float = 5.0   # Hz
    
    # Legacy parameter aliases (backward compatibility)
    alpha: float = 0.984           # = 1 - 1/γ_a when γ_a=64
    short_term_avg_samples: int = 3  # Alias for γ_d
    delta_margin_ratio: float = 6.0  # Alias for ζ_k
    delta_update_ratio: float = 0.8  # Alias for noise_update_ratio
    num_fft_bins: int = 8
    min_bins_trigger: int = 4
    
    # Zhang et al. parameters
    # Note: Zhang operates in TIME-DOMAIN (no FFT), but processes per-FRAME like proposed
    # Uses 6x threshold for fair comparison with proposed method
    zhang_beta: float = 0.95   # threshold smoothing (EMA for frame-based noise floor)
    zhang_margin_ratio: float = 6.0  # trigger ratio (6x threshold)
    zhang_decimation: int = 15  # (legacy - not used in frame-based mode)
    
    # Network protocol
    slot_time: float = 0.00032  # 320 µs
    cw_min: int = 8
    cw_max: int = 64
    max_retries: int = 4
    prop_delay_per_m: float = 3.33e-9  # ~speed of light
    processing_delay: float = 0.0025  # 2.5 ms per hop
    
    # Payload sizes (bits)
    proposed_payload: int = 64  # timestamp + trigger strength
    zhang_payload: int = 256  # larger payload with sample data
    
    # Random seed
    seed: int = 42
    
    @classmethod
    def from_preset(cls, preset: dict, num_nodes: int = 10, **overrides):
        """Create config from a TimePreset"""
        config = cls(
            num_nodes=num_nodes,
            simulation_duration=preset['duration_hours'] * 3600,
            frame_duration=preset['frame_duration'],
            fft_size=preset['fft_size'],
            **overrides
        )
        return config
    
    def estimate_runtime(self) -> str:
        """Estimate simulation runtime"""
        # Rough estimate: ~50,000 frames/second processing speed
        frames_per_node = self.simulation_duration / self.frame_duration
        total_frames = (self.num_nodes - 1) * frames_per_node
        estimated_seconds = total_frames / 50000  # empirical rate
        
        if estimated_seconds < 60:
            return f"~{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"~{estimated_seconds/60:.0f} minutes"
        else:
            return f"~{estimated_seconds/3600:.1f} hours"
    
    def __str__(self):
        frames_per_sec = 1.0 / self.frame_duration
        total_frames = (self.num_nodes - 1) * self.simulation_duration / self.frame_duration
        
        return (f"SimulationConfig(\n"
                f"  Network: {self.num_nodes} nodes, {self.area_size}m² area, "
                f"{self.comm_radius}m radius\n"
                f"  Timing:  {self.simulation_duration/3600:.1f}h duration, "
                f"{self.frame_duration*1000:.1f}ms frames (N={self.fft_size}), "
                f"{frames_per_sec:.1f} frames/sec/node\n"
                f"  Load:    {total_frames/1e6:.1f}M total frames, "
                f"estimated runtime: {self.estimate_runtime()}\n"
                f"  Events:  {self.event_rate}/hr/node, SNR={self.event_snr}dB, "
                f"band={self.event_freq_low}-{self.event_freq_high}Hz\n"
                f"  Proposed: γ_d={self.gamma_d}, γ_a={self.gamma_a}, ζ_k={self.zeta_k}\n"
                f"  Zhang:    β={self.zhang_beta}, trigger={self.zhang_margin_ratio}x (frame-based)\n"
                f")")


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    FRAME_READY = 1
    TRUE_EVENT_START = 2
    TRUE_EVENT_END = 3
    PACKET_TX_START = 4
    PACKET_TX_COMPLETE = 5
    PACKET_ARRIVAL = 6
    NOISE_UPDATE = 7


@dataclass(order=True)
class SimEvent:
    """Simulation event for priority queue"""
    time: float
    event_type: EventType = field(compare=False)
    node_id: int = field(compare=False)
    data: dict = field(default_factory=dict, compare=False)


# =============================================================================
# NODE MODELS
# =============================================================================

class ProposedMethod:
    """Temporal Spectral Noise-Floor Adaptation
    
    Paper Nomenclature:
    ─────────────────────────────────────────────────────────────────────────────
    γ_d ∈ [3, 5]    : Digital noise filter window (short-term averaging)
    γ_a ∈ [64, 128] : Long-term noise floor adaptation window
    ζ_k = 6         : Threshold coefficient (from real hardware data)
    N(t)            : Adaptive noise floor (smoothed average)
    X̄(t)            : Filtered band magnitude at time t
    
    REAL HARDWARE BEHAVIOR (from data analysis):
    ─────────────────────────────────────────────────────────────────────────────
    During NON-EVENTS:
        Filter/Threshold ratio: mean=0.17, max=1.0
        → Noise is ~17% of threshold, NEVER exceeds threshold
        → 0 false positives
        
    During EVENTS:
        Filter/Threshold ratio: mean=1.37, >1.0 in 96% of samples
        → Events clearly exceed threshold
        → High detection rate
        
    Algorithm:
        1. N(t) = smoothed average of X̄ (exponential smoothing with α = 1-1/γ_a)
        2. Threshold(t) = ζ × N(t) where ζ = 6
        3. Trigger when X̄(t) > Threshold(t)
    """
    
    def __init__(self, config: SimulationConfig, node_id: int):
        self.config = config
        self.node_id = node_id
        self.fft_size = config.fft_size
        
        # Frequency resolution: Δf = f_s / N
        self.freq_resolution = config.sample_rate / self.fft_size
        
        # Event frequency band (1-5 Hz for human movement)
        self.event_freq_low = config.event_freq_low
        self.event_freq_high = config.event_freq_high
        
        # Convert frequency band to bin indices
        self.bin_low = max(1, int(self.event_freq_low / self.freq_resolution))
        self.bin_high = min(self.fft_size // 2 - 1, int(self.event_freq_high / self.freq_resolution))
        self.n_monitored_bins = max(1, self.bin_high - self.bin_low + 1)
        
        # γ_d: Digital noise filter window ∈ [3, 5]
        self.gamma_d = config.gamma_d
        self.filter_buffer = []
        
        # γ_a: Long-term adaptation window ∈ [64, 128]
        # α = 1 - 1/γ_a for exponential smoothing
        self.gamma_a = config.gamma_a
        self.alpha = 1.0 - 1.0 / self.gamma_a
        
        # ζ_k: Threshold coefficient = 6 (from real hardware)
        # Threshold = ζ × N, where N is the smoothed average noise
        self.zeta_k = config.zeta_k
        
        # N(t): Adaptive noise floor (smoothed average)
        # Initialize to expected magnitude for unit noise
        expected_magnitude = np.sqrt(self.fft_size) * np.sqrt(config.base_noise_power)
        self.N = expected_magnitude
        
        # Track max in recent buffer for display (like real hardware's MaxBuffer)
        self.max_buffer = expected_magnitude
        self.max_buffer_window = []
        
        self.frames_processed = 0
        self.triggers_issued = 0
        
    def process_frame(self, samples: np.ndarray, current_noise_power: float) -> Tuple[bool, float]:
        """Process one frame
        
        Algorithm:
            1. Compute FFT, extract band magnitude X
            2. Apply γ_d averaging: X̄ = mean of last γ_d values
            3. Update noise floor: N = α·N + (1-α)·X̄  (only when not triggered)
            4. Compute threshold: Threshold = ζ × N
            5. Trigger if X̄ > Threshold
        
        Returns:
            trigger: bool - whether event was detected
            strength: float - ratio X̄ / Threshold (Filter/Threshold in real data)
        """
        self.frames_processed += 1
        
        # Compute FFT magnitude spectrum
        full_spectrum = np.abs(np.fft.fft(samples))
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Extract band magnitude (1-5 Hz for human movement)
        # ═══════════════════════════════════════════════════════════════════════
        band_spectrum = full_spectrum[self.bin_low:self.bin_high + 1]
        if len(band_spectrum) == 0:
            band_spectrum = full_spectrum[1:2]
        
        # Aggregate: max across bins (like real hardware)
        X_raw = np.max(band_spectrum)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Digital Noise Filter (γ_d averaging)
        # ═══════════════════════════════════════════════════════════════════════
        self.filter_buffer.append(X_raw)
        if len(self.filter_buffer) > self.gamma_d:
            self.filter_buffer.pop(0)
        
        X_bar = np.mean(self.filter_buffer)  # This is "Filter" in real data
        
        # Update max buffer (for display, like real hardware's MaxBuffer column)
        self.max_buffer_window.append(X_bar)
        if len(self.max_buffer_window) > self.gamma_a:
            self.max_buffer_window.pop(0)
        self.max_buffer = max(self.max_buffer_window) if self.max_buffer_window else X_bar
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Compute Threshold = ζ × N
        # ═══════════════════════════════════════════════════════════════════════
        Threshold = self.zeta_k * self.N
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Trigger decision - X̄ > Threshold ?
        # ═══════════════════════════════════════════════════════════════════════
        ratio = X_bar / Threshold if Threshold > 0 else 0  # Filter/Threshold ratio
        trigger = ratio > 1.0  # Trigger when Filter > Threshold
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Update noise floor N (only when not triggered)
        # N(t) = α·N(t-1) + (1-α)·X̄(t)
        # ═══════════════════════════════════════════════════════════════════════
        if not trigger:
            # Only update N when X̄ is clearly below threshold (no event)
            if ratio < self.config.noise_update_ratio:
                self.N = self.alpha * self.N + (1 - self.alpha) * X_bar
        
        if trigger:
            self.triggers_issued += 1
            log.debug(f"Node {self.node_id} TRIGGER: X̄={X_bar:.1f}, N={self.N:.1f}, "
                     f"Threshold={Threshold:.1f}, ratio={ratio:.2f}")
        
        return trigger, ratio
    
    def reset(self):
        """Reset state for new simulation"""
        expected_magnitude = np.sqrt(self.fft_size) * np.sqrt(self.config.base_noise_power)
        self.N = expected_magnitude
        self.max_buffer = expected_magnitude
        self.max_buffer_window = []
        self.filter_buffer = []
        self.frames_processed = 0
        self.triggers_issued = 0
    
    def get_stats(self) -> Dict:
        return {
            'frames_processed': self.frames_processed,
            'triggers_issued': self.triggers_issued,
            'gamma_d': self.gamma_d,
            'gamma_a': self.gamma_a,
            'zeta_k': self.zeta_k,
            'N': float(self.N),
            'Threshold': float(self.zeta_k * self.N),
            'max_buffer': float(self.max_buffer),
            'monitored_band_Hz': f"{self.event_freq_low}-{self.event_freq_high}",
            'monitored_bins': f"{self.bin_low}-{self.bin_high}"
        }


class ZhangMethod:
    """
    Time-domain adaptive thresholding (Zhang et al. 2023)
    
    Updated to process per-FRAME instead of per-sample for fair comparison.
    Uses the same frame boundaries as the proposed method, but operates in 
    time-domain (no FFT filtering).
    
    Frame-based algorithm:
    1. Compute frame statistic: X = max(|samples|) over the frame
    2. Update noise floor: N = β·N + (1-β)·X  (when not triggered)
    3. Trigger decision: X > 6·N
    """
    
    def __init__(self, config: SimulationConfig, node_id: int):
        self.config = config
        self.node_id = node_id
        # Initialize noise floor to expected RMS of noise
        self.noise_floor = np.sqrt(config.base_noise_power)
        self.frames_processed = 0
        self.triggers_issued = 0
        
    def process_frame(self, samples: np.ndarray, current_noise_power: float) -> Tuple[bool, float]:
        """
        Process one frame (same frame boundaries as proposed method).
        
        Zhang method operates in time-domain: no FFT, no frequency filtering.
        This means it sees ALL noise including EMI and digital noise that
        the proposed method's FFT filtering removes.
        """
        self.frames_processed += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Compute frame statistic (time-domain, no FFT)
        # Use max absolute value over the frame - captures peaks
        # ═══════════════════════════════════════════════════════════════════════
        frame_max = np.max(np.abs(samples))
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Compute threshold (6x noise floor, same as proposed)
        # ═══════════════════════════════════════════════════════════════════════
        trigger_level = self.noise_floor * self.config.zhang_margin_ratio
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Trigger decision
        # ═══════════════════════════════════════════════════════════════════════
        ratio = frame_max / self.noise_floor if self.noise_floor > 0 else 0
        trigger = frame_max > trigger_level
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Update noise floor (only when not triggered)
        # N(t) = β·N(t-1) + (1-β)·X(t)
        # ═══════════════════════════════════════════════════════════════════════
        if not trigger:
            # Only update when clearly below threshold
            if ratio < 0.8:  # Same update criterion as proposed method
                self.noise_floor = (self.config.zhang_beta * self.noise_floor + 
                                   (1 - self.config.zhang_beta) * frame_max)
        
        if trigger:
            self.triggers_issued += 1
            log.debug(f"Node {self.node_id} ZHANG trigger: X={frame_max:.2f}, "
                     f"N={self.noise_floor:.2f}, ratio={ratio:.2f}x")
        
        return trigger, ratio
    
    def reset(self):
        """Reset state for new simulation"""
        self.noise_floor = np.sqrt(self.config.base_noise_power)
        self.frames_processed = 0
        self.triggers_issued = 0
    
    def get_stats(self) -> Dict:
        return {
            'frames_processed': self.frames_processed,
            'triggers_issued': self.triggers_issued,
            'noise_floor': self.noise_floor,
            'threshold': self.noise_floor * self.config.zhang_margin_ratio
        }


# =============================================================================
# NOISE GENERATOR
# =============================================================================

class NoiseGenerator:
    """
    Realistic noise generator with per-node dynamics.
    
    Two categories of noise:
    1. FAST NOISE (always present, high frequency - OUTSIDE event band):
       - EMI: 50/60 Hz power line + harmonics
       - Sampling artifacts: Quantization noise, ADC glitches
       - Digital switching: kHz range bursts from MCU, regulators
       
    2. ENVIRONMENTAL NOISE (varying per node - may overlap event band):
       - Rain: Broadband noise, intensity varies over hours
       - Wind: Low frequency rumble (1-10 Hz), gusts
       - Motors/Propellers: Specific frequencies based on RPM
       
    The proposed method's FFT filtering removes fast noise (outside 1-5 Hz band)
    but environmental noise may partially overlap the event band.
    """
    
    def __init__(self, node_id: int, config: 'SimulationConfig', seed: int = None):
        self.node_id = node_id
        self.config = config
        
        # Per-node random state for reproducibility
        self.rng = np.random.RandomState(seed if seed else node_id * 12345)
        
        # Fast noise parameters (constant for this node)
        self.emi_phase = self.rng.uniform(0, 2 * np.pi)  # Random phase offset
        self.emi_freq = config.emi_freq if hasattr(config, 'emi_freq') else 60.0
        
        # Environmental noise state (varies over time)
        self.rain_intensity = 0.0      # 0 = no rain, 1 = heavy rain
        self.rain_duration = 0.0       # Remaining rain time
        self.wind_intensity = 0.0      # 0 = calm, 1 = strong gusts
        self.wind_gust_time = 0.0      # Time of current gust
        self.motor_active = False      # Nearby motor running
        self.motor_freq = 0.0          # Motor frequency (Hz)
        self.motor_duration = 0.0      # Remaining motor time
        
        # Propeller/fan parameters (if motor active)
        self.propeller_blades = self.rng.choice([2, 3, 4, 6])  # Number of blades
        
    def update_environmental_state(self, dt: float, current_time: float):
        """Update environmental noise sources based on time progression"""
        
        # Rain dynamics - slow changes over hours
        if self.rain_duration > 0:
            self.rain_duration -= dt
            # Rain intensity varies slowly
            self.rain_intensity += self.rng.uniform(-0.01, 0.01) * dt
            self.rain_intensity = np.clip(self.rain_intensity, 0.1, 1.0)
            if self.rain_duration <= 0:
                self.rain_intensity = 0.0
        else:
            # Chance of rain starting (per hour → per second probability)
            rain_prob_per_sec = 0.05 / 3600  # ~5% per hour
            if self.rng.random() < rain_prob_per_sec * dt:
                self.rain_intensity = self.rng.uniform(0.2, 0.8)
                self.rain_duration = self.rng.uniform(600, 3600)  # 10 min to 1 hour
        
        # Wind dynamics - gusts are shorter events
        if self.wind_intensity > 0:
            # Wind dies down
            self.wind_intensity *= np.exp(-dt / 30)  # 30 sec decay
            if self.wind_intensity < 0.05:
                self.wind_intensity = 0.0
        else:
            # Chance of wind gust
            wind_prob_per_sec = 0.1 / 3600  # ~10% per hour
            if self.rng.random() < wind_prob_per_sec * dt:
                self.wind_intensity = self.rng.uniform(0.3, 1.0)
        
        # Motor/machinery dynamics
        if self.motor_duration > 0:
            self.motor_duration -= dt
            if self.motor_duration <= 0:
                self.motor_active = False
                self.motor_freq = 0.0
        else:
            # Chance of motor starting nearby
            motor_prob_per_sec = 0.02 / 3600  # ~2% per hour
            if self.rng.random() < motor_prob_per_sec * dt:
                self.motor_active = True
                # Motor frequency: RPM / 60 → Hz, typical 1800-3600 RPM
                rpm = self.rng.uniform(1200, 3600)
                self.motor_freq = rpm / 60  # 20-60 Hz base frequency
                self.motor_duration = self.rng.uniform(60, 600)  # 1-10 minutes
    
    def generate_noise(self, n_samples: int, t: np.ndarray, base_noise_power: float) -> np.ndarray:
        """
        Generate realistic noise signal for one frame.
        
        Args:
            n_samples: Number of samples in frame
            t: Time array for this frame
            base_noise_power: Base noise power level
            
        Returns:
            Noise signal array
        """
        noise = np.zeros(n_samples)
        
        # =====================================================================
        # 1. THERMAL/SENSOR NOISE (broadband white noise)
        # =====================================================================
        noise += self.rng.randn(n_samples) * np.sqrt(base_noise_power)
        
        # =====================================================================
        # 2. EMI - Power line interference (50/60 Hz + harmonics)
        #    OUTSIDE 1-5 Hz event band → filtered by FFT
        # =====================================================================
        emi_amp = 0.3 * np.sqrt(base_noise_power)
        noise += emi_amp * np.sin(2 * np.pi * self.emi_freq * t + self.emi_phase)
        noise += emi_amp * 0.5 * np.sin(2 * np.pi * 2 * self.emi_freq * t + self.emi_phase)  # 2nd harmonic
        noise += emi_amp * 0.25 * np.sin(2 * np.pi * 3 * self.emi_freq * t + self.emi_phase)  # 3rd harmonic
        
        # =====================================================================
        # 3. DIGITAL SWITCHING NOISE (kHz range bursts)
        #    OUTSIDE 1-5 Hz event band → filtered by FFT
        # =====================================================================
        if self.rng.random() < 0.1:  # 10% chance per frame
            digital_freq = self.rng.uniform(800, 2000)
            digital_amp = self.rng.uniform(0.5, 2.0) * np.sqrt(base_noise_power)
            burst_start = self.rng.randint(0, n_samples // 2)
            burst_len = self.rng.randint(10, 30)
            burst_mask = np.zeros(n_samples)
            burst_mask[burst_start:min(burst_start + burst_len, n_samples)] = 1.0
            noise += digital_amp * np.sin(2 * np.pi * digital_freq * t) * burst_mask
        
        # =====================================================================
        # 4. RAIN NOISE (broadband, partially in event band)
        #    Rain creates broadband noise including 1-5 Hz components
        # =====================================================================
        if self.rain_intensity > 0:
            # Rain is broadband but has low-frequency rumble
            rain_noise = self.rng.randn(n_samples) * self.rain_intensity * 0.5 * np.sqrt(base_noise_power)
            # Add low-frequency component (partially in event band!)
            rain_low_freq = self.rng.uniform(0.5, 3.0)  # Hz - overlaps event band
            rain_noise += self.rain_intensity * 0.3 * np.sqrt(base_noise_power) * \
                         np.sin(2 * np.pi * rain_low_freq * t + self.rng.uniform(0, 2*np.pi))
            noise += rain_noise
        
        # =====================================================================
        # 5. WIND NOISE (low frequency rumble, IN event band)
        #    Wind creates 0.5-5 Hz pressure fluctuations
        # =====================================================================
        if self.wind_intensity > 0:
            # Wind is primarily low frequency - IN the event band
            wind_freq = self.rng.uniform(0.5, 4.0)  # Hz - IN event band!
            wind_amp = self.wind_intensity * 0.6 * np.sqrt(base_noise_power)
            # Wind has irregular pattern
            wind_mod = 1 + 0.5 * np.sin(2 * np.pi * 0.2 * t)  # Slow modulation
            noise += wind_amp * np.sin(2 * np.pi * wind_freq * t) * wind_mod
        
        # =====================================================================
        # 6. MOTOR/PROPELLER NOISE (specific frequency + harmonics)
        #    Motor base freq 20-60 Hz (outside band), but propeller blades
        #    create subharmonics that may be in 1-5 Hz band
        # =====================================================================
        if self.motor_active and self.motor_freq > 0:
            motor_amp = 0.4 * np.sqrt(base_noise_power)
            # Base motor frequency (outside event band)
            noise += motor_amp * np.sin(2 * np.pi * self.motor_freq * t)
            
            # Propeller blade-pass frequency = motor_freq * blades
            # But also creates LOW frequency vibration from imbalance
            imbalance_freq = self.motor_freq / self.propeller_blades  # Could be 5-20 Hz
            if imbalance_freq < 10:  # If low enough to matter
                noise += motor_amp * 0.3 * np.sin(2 * np.pi * imbalance_freq * t)
        
        return noise
    
    def get_state(self) -> Dict:
        """Return current environmental state for debugging/visualization"""
        return {
            'rain_intensity': self.rain_intensity,
            'rain_duration': self.rain_duration,
            'wind_intensity': self.wind_intensity,
            'motor_active': self.motor_active,
            'motor_freq': self.motor_freq,
        }


@dataclass
class RawDataSnapshot:
    """Container for raw waveform data snapshot"""
    timestamp: float              # Simulation time when snapshot started
    duration: float               # Duration of snapshot in seconds
    sample_rate: float            # Samples per second
    node_data: Dict[int, Dict]    # {node_id: {'samples': array, 'events': list, 'triggers': list}}
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'timestamp': self.timestamp,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'nodes': {
                str(node_id): {
                    'samples': data['samples'].tolist() if isinstance(data['samples'], np.ndarray) else data['samples'],
                    'events': data.get('events', []),
                    'triggers_proposed': data.get('triggers_proposed', []),
                    'triggers_zhang': data.get('triggers_zhang', []),
                    'noise_state': data.get('noise_state', {})
                }
                for node_id, data in self.node_data.items()
            }
        }


# =============================================================================
# NETWORK MODEL
# =============================================================================

@dataclass
class Node:
    """Sensor node in the mesh network"""
    node_id: int
    x: float
    y: float
    neighbors: List[int] = field(default_factory=list)
    route_to_sink: List[int] = field(default_factory=list)
    hop_count: int = 0
    
    # Detection methods
    proposed: Optional[ProposedMethod] = None
    zhang: Optional[ZhangMethod] = None
    
    # Noise generator (per-node dynamics)
    noise_gen: Optional[NoiseGenerator] = None
    
    # State
    tx_queue: List = field(default_factory=list)
    is_transmitting: bool = False
    backoff_stage: int = 0
    current_cw: int = 8


class MeshNetwork:
    """Mesh network topology and routing"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.nodes: Dict[int, Node] = {}
        self.sink_id = 0
        
        log.subsection("Building Network Topology")
        self._build_topology()
        
    def _build_topology(self):
        """Create random node placement and connectivity"""
        np.random.seed(self.config.seed)
        
        log.info(f"Placing {self.config.num_nodes} nodes in {self.config.area_size}m × "
                f"{self.config.area_size}m area")
        
        # Place sink at center
        center = self.config.area_size / 2
        self.nodes[0] = Node(0, center, center)
        log.debug(f"Sink node placed at ({center:.0f}, {center:.0f})")
        
        # Place other nodes randomly
        for i in range(1, self.config.num_nodes):
            x = np.random.uniform(0, self.config.area_size)
            y = np.random.uniform(0, self.config.area_size)
            self.nodes[i] = Node(i, x, y)
            self.nodes[i].proposed = ProposedMethod(self.config, i)
            self.nodes[i].zhang = ZhangMethod(self.config, i)
        
        log.info(f"Placed {self.config.num_nodes - 1} sensor nodes")
        
        # Build connectivity (unit-disk model)
        log.info(f"Building connectivity with radius={self.config.comm_radius}m")
        positions = np.array([[n.x, n.y] for n in self.nodes.values()])
        distances = cdist(positions, positions)
        
        total_links = 0
        for i in range(self.config.num_nodes):
            for j in range(self.config.num_nodes):
                if i != j and distances[i, j] <= self.config.comm_radius:
                    self.nodes[i].neighbors.append(j)
                    total_links += 1
        
        avg_neighbors = total_links / self.config.num_nodes
        log.info(f"Created {total_links//2} bidirectional links "
                f"(avg {avg_neighbors:.1f} neighbors/node)")
        
        # Compute routing (BFS from sink)
        self._compute_routes()
        
    def _compute_routes(self):
        """Compute minimum-hop routes to sink using BFS"""
        log.info("Computing minimum-hop routes to sink...")
        
        visited = {0}
        queue = [(0, 0)]  # (node_id, hop_count)
        self.nodes[0].hop_count = 0
        self.nodes[0].route_to_sink = []
        
        while queue:
            current, hops = queue.pop(0)
            
            for neighbor in self.nodes[current].neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.nodes[neighbor].hop_count = hops + 1
                    self.nodes[neighbor].route_to_sink = (
                        self.nodes[current].route_to_sink + [current]
                    )
                    queue.append((neighbor, hops + 1))
        
        # Handle disconnected nodes
        disconnected = 0
        for node_id, node in self.nodes.items():
            if node_id not in visited:
                node.hop_count = 999
                node.route_to_sink = []
                disconnected += 1
        
        # Compute hop statistics
        hop_counts = [n.hop_count for n in self.nodes.values() if n.hop_count < 999]
        if hop_counts:
            log.info(f"Routing complete: max_hops={max(hop_counts)}, "
                    f"avg_hops={np.mean(hop_counts):.1f}")
        
        if disconnected > 0:
            log.warning(f"{disconnected} nodes are disconnected from sink!")
    
    def get_propagation_delay(self, from_id: int, to_id: int) -> float:
        """Calculate propagation delay between two nodes"""
        n1, n2 = self.nodes[from_id], self.nodes[to_id]
        distance = np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        return distance * self.config.prop_delay_per_m
    
    def get_transmission_time(self, payload_bits: int) -> float:
        """Calculate transmission time for given payload"""
        return payload_bits / self.config.data_rate
    
    def print_topology_stats(self):
        """Print detailed topology statistics"""
        log.subsection("Network Topology Statistics")
        
        hop_counts = [n.hop_count for n in self.nodes.values() if n.hop_count < 999]
        neighbor_counts = [len(n.neighbors) for n in self.nodes.values()]
        
        log.info(f"Total nodes: {self.config.num_nodes}")
        log.info(f"Connected nodes: {len(hop_counts)}")
        log.info(f"Hop count distribution: min={min(hop_counts)}, max={max(hop_counts)}, "
                f"mean={np.mean(hop_counts):.1f}, median={np.median(hop_counts):.0f}")
        log.info(f"Neighbor count distribution: min={min(neighbor_counts)}, "
                f"max={max(neighbor_counts)}, mean={np.mean(neighbor_counts):.1f}")


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

@dataclass
class TriggerRecord:
    """Record of a trigger event"""
    time: float
    node_id: int
    method: str  # 'proposed' or 'zhang'
    is_true_positive: bool
    strength: float
    arrival_time: Optional[float] = None
    latency: Optional[float] = None


class NetworkSimulator:
    """Discrete-event simulation engine"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        log.section(f"Initializing {config.num_nodes}-Node Network Simulator")
        log.info(str(config))
        
        self.network = MeshNetwork(config)
        self.event_queue: List[SimEvent] = []
        self.current_time = 0.0
        
        # Noise state (global baseline - varies slowly)
        self.current_noise_power = config.base_noise_power
        
        # Per-node noise generators
        self._init_noise_generators()
        
        # True event tracking
        self.active_events: Dict[int, float] = {}  # node_id -> event_end_time
        self.true_event_times: Dict[int, List[float]] = defaultdict(list)
        
        # Results tracking
        self.proposed_triggers: List[TriggerRecord] = []
        self.zhang_triggers: List[TriggerRecord] = []
        
        # Network stats
        self.total_bytes_proposed = 0
        self.total_bytes_zhang = 0
        self.congestion_events = 0
        self.packet_collisions = 0
        
        # Channel state
        self.channel_busy_until = 0.0
        
        # Snapshot tracking
        self.snapshots: List[RawDataSnapshot] = []
        self.current_snapshot: Optional[Dict] = None
        self.snapshot_start_time: float = 0.0
        self.next_snapshot_time: float = 0.0 if config.enable_snapshots else float('inf')
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'true_events_started': 0,
            'true_events_ended': 0,
            'noise_updates': 0,
            'tx_attempts': 0,
            'tx_completions': 0
        }
    
    def _init_noise_generators(self):
        """Initialize per-node noise generators"""
        for node_id, node in self.network.nodes.items():
            if node_id != 0:  # Skip sink node
                node.noise_gen = NoiseGenerator(
                    node_id=node_id,
                    config=self.config,
                    seed=self.config.seed + node_id * 1000
                )
        
    def initialize(self):
        """Set up initial events"""
        log.subsection("Initializing Simulation State")
        
        self.event_queue = []
        self.current_time = 0.0
        self.proposed_triggers = []
        self.zhang_triggers = []
        self.total_bytes_proposed = 0
        self.total_bytes_zhang = 0
        self.congestion_events = 0
        self.channel_busy_until = 0.0
        self.stats = {k: 0 for k in self.stats}
        
        # Reset node states
        log.info("Resetting node states...")
        for node in self.network.nodes.values():
            if node.proposed:
                node.proposed.reset()
            if node.zhang:
                node.zhang.reset()
            node.tx_queue = []
            node.is_transmitting = False
            node.backoff_stage = 0
            node.current_cw = self.config.cw_min
        
        # Schedule initial frame processing for all nodes
        log.info(f"Scheduling initial frames for {self.config.num_nodes - 1} sensor nodes...")
        for node_id in range(1, self.config.num_nodes):
            offset = np.random.uniform(0, self.config.frame_duration)
            heapq.heappush(self.event_queue, SimEvent(
                time=offset,
                event_type=EventType.FRAME_READY,
                node_id=node_id
            ))
        
        # Schedule true events (Poisson process)
        self._schedule_true_events()
        
        # Schedule noise updates
        heapq.heappush(self.event_queue, SimEvent(
            time=60.0,
            event_type=EventType.NOISE_UPDATE,
            node_id=-1
        ))
        
        log.info(f"Event queue initialized with {len(self.event_queue)} events")
    
    def _schedule_true_events(self):
        """Generate Poisson-distributed true events for all nodes"""
        log.info(f"Scheduling true events (rate={self.config.event_rate}/hr/node)...")
        
        rate_per_second = self.config.event_rate / 3600.0
        total_events = 0
        
        for node_id in range(1, self.config.num_nodes):
            t = 0.0
            node_events = 0
            while t < self.config.simulation_duration:
                t += np.random.exponential(1.0 / rate_per_second)
                if t < self.config.simulation_duration:
                    self.true_event_times[node_id].append(t)
                    heapq.heappush(self.event_queue, SimEvent(
                        time=t,
                        event_type=EventType.TRUE_EVENT_START,
                        node_id=node_id
                    ))
                    node_events += 1
                    total_events += 1
        
        expected = (self.config.num_nodes - 1) * self.config.event_rate * \
                   (self.config.simulation_duration / 3600)
        log.info(f"Scheduled {total_events} true events (expected: {expected:.0f})")
    
    def _generate_frame_samples(self, node_id: int) -> np.ndarray:
        """Generate signal samples for one frame using per-node noise generator
        
        Noise model with frequency separation for human movement detection:
        ─────────────────────────────────────────────────────────────────────
        FAST NOISE (always present, OUTSIDE 1-5 Hz band):
        - EMI: 50/60 Hz power line + harmonics
        - Digital: kHz range switching noise
        → Filtered out by FFT band selection
        
        ENVIRONMENTAL NOISE (per-node varying, may overlap event band):
        - Rain: Broadband + low-frequency rumble
        - Wind: 0.5-5 Hz pressure fluctuations (IN event band!)
        - Motors: Base freq outside band, but imbalance harmonics may be inside
        
        TRUE EVENTS (human movement, 1-5 Hz):
        - Occur in 1-5 Hz band
        - Consistent over > γ_d samples
        - Low frequency, slow changes
        """
        n_samples = self.config.fft_size
        t = np.linspace(0, self.config.frame_duration, n_samples)
        
        # Get node's noise generator
        node = self.network.nodes.get(node_id)
        
        if node and node.noise_gen:
            # Update environmental state based on time elapsed
            node.noise_gen.update_environmental_state(
                dt=self.config.frame_duration, 
                current_time=self.current_time
            )
            
            # Generate noise using per-node generator
            signal = node.noise_gen.generate_noise(n_samples, t, self.current_noise_power)
        else:
            # Fallback: simple white noise
            signal = np.random.randn(n_samples) * np.sqrt(self.current_noise_power)
        
        # Add TRUE EVENT signal (human movement) - in 1-5 Hz band
        if node_id in self.active_events:
            # Event frequency in monitored band (1-5 Hz for human movement)
            freq_margin = 0.2  # Hz margin from band edges
            event_freq = np.random.uniform(
                self.config.event_freq_low + freq_margin,
                self.config.event_freq_high - freq_margin
            )
            
            event_amplitude = np.sqrt(self.current_noise_power * 
                                     10**(self.config.event_snr / 10))
            
            # Human movement is a slow, sustained change (not a sharp transient)
            event_signal = event_amplitude * np.sin(2 * np.pi * event_freq * t)
            envelope = np.exp(-t / self.config.event_decay_tau)
            signal += event_signal * envelope
        
        return signal
    
    def _is_true_positive(self, node_id: int, trigger_time: float) -> bool:
        """Check if trigger corresponds to a true event
        
        A trigger is a true positive if it occurs within a window around a true event.
        The window accounts for:
        1. Event duration (event_duration)
        2. γ_d averaging tail (gamma_d * frame_duration)
        3. Some margin for timing jitter
        """
        # Window extends from event start to (event_end + γ_d tail)
        margin = self.config.frame_duration  # Small margin for timing jitter
        window_before = margin  # Can trigger slightly before event starts
        window_after = self.config.event_duration + self.config.gamma_d * self.config.frame_duration + margin
        
        for event_time in self.true_event_times[node_id]:
            if (event_time - window_before) <= trigger_time <= (event_time + window_after):
                return True
        return False
    
    # =========================================================================
    # SNAPSHOT COLLECTION
    # =========================================================================
    
    def _should_collect_snapshot(self, node_id: int) -> bool:
        """Check if we should collect data for this node"""
        if self.config.snapshot_nodes == 'ALL':
            return True
        elif isinstance(self.config.snapshot_nodes, list):
            return node_id in self.config.snapshot_nodes
        return False
    
    def _start_snapshot(self):
        """Start a new snapshot collection period"""
        self.snapshot_start_time = self.current_time
        self.current_snapshot = {}
        
        # Initialize data structures for each node
        for node_id in range(1, self.config.num_nodes):
            if self._should_collect_snapshot(node_id):
                node = self.network.nodes.get(node_id)
                self.current_snapshot[node_id] = {
                    'samples': [],
                    'timestamps': [],
                    'events': [],
                    'triggers_proposed': [],
                    'triggers_zhang': [],
                    'noise_state': node.noise_gen.get_state() if node and node.noise_gen else {}
                }
        
        log.info(f"Snapshot started at t={self.current_time:.1f}s, "
                f"collecting {len(self.current_snapshot)} nodes for {self.config.snapshot_duration}s")
    
    def _collect_snapshot_data(self, node_id: int, samples: np.ndarray, node: 'Node'):
        """Collect frame data for snapshot"""
        if node_id not in self.current_snapshot:
            return
        
        # Store samples and timestamp
        self.current_snapshot[node_id]['samples'].extend(samples.tolist())
        self.current_snapshot[node_id]['timestamps'].append(self.current_time)
        
        # Track active events
        if node_id in self.active_events:
            self.current_snapshot[node_id]['events'].append({
                'time': self.current_time,
                'type': 'active'
            })
    
    def _end_snapshot(self):
        """End current snapshot and save"""
        if self.current_snapshot is None:
            return
        
        # Create snapshot object
        snapshot = RawDataSnapshot(
            timestamp=self.snapshot_start_time,
            duration=self.config.snapshot_duration,
            sample_rate=self.config.sample_rate,
            node_data={
                node_id: {
                    'samples': np.array(data['samples']),
                    'events': data['events'],
                    'triggers_proposed': data['triggers_proposed'],
                    'triggers_zhang': data['triggers_zhang'],
                    'noise_state': data['noise_state']
                }
                for node_id, data in self.current_snapshot.items()
            }
        )
        
        snapshot_idx = len(self.snapshots)
        self.snapshots.append(snapshot)
        
        # Calculate size for logging
        total_samples = sum(len(data['samples']) for data in self.current_snapshot.values())
        log.info(f"Snapshot completed at t={self.current_time:.1f}s: "
                f"{len(self.current_snapshot)} nodes, {total_samples:,} samples")
        
        # CONTINUOUS SAVE: Save snapshot immediately to disk
        if self.config.continuous_save and self.config.snapshot_output_dir:
            self._save_single_snapshot(snapshot, snapshot_idx)
        
        self.current_snapshot = None
    
    def _save_single_snapshot(self, snapshot: RawDataSnapshot, idx: int):
        """Save a single snapshot immediately to disk"""
        try:
            output_dir = self.config.snapshot_output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as individual .npz file
            filename = os.path.join(output_dir, f"snapshot_{idx:04d}_t{snapshot.timestamp:.0f}s.npz")
            
            arrays = {}
            for node_id, data in snapshot.node_data.items():
                arrays[f"node{node_id}_samples"] = data['samples']
                arrays[f"node{node_id}_triggers_proposed"] = np.array(data['triggers_proposed'])
                arrays[f"node{node_id}_triggers_zhang"] = np.array(data['triggers_zhang'])
            
            # Add metadata
            arrays['_metadata'] = np.array([snapshot.timestamp, snapshot.duration, snapshot.sample_rate])
            
            np.savez_compressed(filename, **arrays)
            log.info(f"  → Saved snapshot {idx} to {filename}")
            
        except Exception as e:
            log.info(f"  ! Warning: Could not save snapshot {idx}: {e}")
    
    def _check_snapshot_timing(self):
        """Check if we need to start/end snapshot collection"""
        if not self.config.enable_snapshots:
            return
        
        # Check if we should start a new snapshot
        if self.current_snapshot is None and self.current_time >= self.next_snapshot_time:
            self._start_snapshot()
            self.next_snapshot_time = self.current_time + self.config.snapshot_interval
        
        # Check if current snapshot should end
        if self.current_snapshot is not None:
            elapsed = self.current_time - self.snapshot_start_time
            if elapsed >= self.config.snapshot_duration:
                self._end_snapshot()
    
    def save_snapshots(self, filename: str):
        """Save collected snapshots to file
        
        Snapshots are saved as compressed numpy archives (.npz) for efficiency,
        with a companion JSON file for metadata.
        """
        if not self.snapshots:
            log.info("No snapshots to save")
            return
        
        import gzip
        
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        # Save metadata as JSON
        metadata = {
            'num_snapshots': len(self.snapshots),
            'sample_rate': self.config.sample_rate,
            'snapshot_duration': self.config.snapshot_duration,
            'snapshot_interval': self.config.snapshot_interval,
            'num_nodes': self.config.num_nodes,
            'snapshots': []
        }
        
        for i, snap in enumerate(self.snapshots):
            snap_meta = {
                'index': i,
                'timestamp': snap.timestamp,
                'duration': snap.duration,
                'nodes': list(snap.node_data.keys()),
                'samples_per_node': {
                    str(nid): len(data['samples']) 
                    for nid, data in snap.node_data.items()
                }
            }
            metadata['snapshots'].append(snap_meta)
        
        meta_filename = f"{base_name}_snapshots_meta.json"
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save raw data as compressed numpy archive
        data_filename = f"{base_name}_snapshots_data.npz"
        
        arrays_to_save = {}
        for i, snap in enumerate(self.snapshots):
            for node_id, data in snap.node_data.items():
                key = f"snap{i}_node{node_id}_samples"
                arrays_to_save[key] = np.array(data['samples'])
                
                # Save events and triggers as arrays too
                key_events = f"snap{i}_node{node_id}_events"
                arrays_to_save[key_events] = np.array([
                    e['time'] for e in data.get('events', [])
                ]) if data.get('events') else np.array([])
                
                key_trig_prop = f"snap{i}_node{node_id}_triggers_proposed"
                arrays_to_save[key_trig_prop] = np.array(
                    data.get('triggers_proposed', [])
                )
                
                key_trig_zhang = f"snap{i}_node{node_id}_triggers_zhang"
                arrays_to_save[key_trig_zhang] = np.array(
                    data.get('triggers_zhang', [])
                )
        
        np.savez_compressed(data_filename, **arrays_to_save)
        
        # Calculate file sizes
        meta_size = os.path.getsize(meta_filename)
        data_size = os.path.getsize(data_filename)
        total_samples = sum(
            len(data['samples']) 
            for snap in self.snapshots 
            for data in snap.node_data.values()
        )
        
        log.info(f"Snapshots saved:")
        log.info(f"  Metadata: {meta_filename} ({meta_size/1024:.1f} KB)")
        log.info(f"  Data: {data_filename} ({data_size/1024/1024:.2f} MB)")
        log.info(f"  Total: {len(self.snapshots)} snapshots, {total_samples:,} samples")
        
        return meta_filename, data_filename

    def _update_noise_power(self):
        """Update time-varying noise power"""
        old_power = self.current_noise_power
        phase = 2 * np.pi * self.current_time / self.config.noise_cycle_period
        variation_db = self.config.noise_variation_db * np.sin(phase)
        variation_db += np.random.uniform(-1, 1)
        self.current_noise_power = self.config.base_noise_power * 10**(variation_db / 10)
        
        self.stats['noise_updates'] += 1
        log.debug(f"Noise power updated: {old_power:.3f} -> {self.current_noise_power:.3f} "
                 f"({variation_db:+.1f} dB)")
    
    def _attempt_transmission(self, node_id: int, method: str, trigger_record: TriggerRecord):
        """Attempt to transmit a trigger packet with CSMA/CA"""
        node = self.network.nodes[node_id]
        self.stats['tx_attempts'] += 1
        
        if node.hop_count == 999:
            log.debug(f"Node {node_id} disconnected, dropping {method} packet")
            return
        
        payload = (self.config.proposed_payload if method == 'proposed' 
                  else self.config.zhang_payload)
        
        if self.current_time < self.channel_busy_until:
            # Channel busy - backoff
            cw = min(self.config.cw_max, self.config.cw_min * (2 ** node.backoff_stage))
            backoff_slots = np.random.randint(0, cw)
            backoff_time = backoff_slots * self.config.slot_time
            
            node.backoff_stage = min(node.backoff_stage + 1, self.config.max_retries)
            self.congestion_events += 1
            
            log.debug(f"Node {node_id} {method}: channel busy, backoff={backoff_time*1000:.2f}ms "
                     f"(stage {node.backoff_stage})")
            
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + backoff_time,
                event_type=EventType.PACKET_TX_START,
                node_id=node_id,
                data={'method': method, 'record': trigger_record, 'retry': True}
            ))
        else:
            # Channel free - transmit
            tx_time = self.network.get_transmission_time(payload)
            self.channel_busy_until = self.current_time + tx_time
            
            log.debug(f"Node {node_id} {method}: transmitting {payload} bits, "
                     f"tx_time={tx_time*1000:.2f}ms")
            
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + tx_time,
                event_type=EventType.PACKET_TX_COMPLETE,
                node_id=node_id,
                data={'method': method, 'record': trigger_record}
            ))
            
            if method == 'proposed':
                self.total_bytes_proposed += payload // 8
            else:
                self.total_bytes_zhang += payload // 8
            
            node.backoff_stage = 0
    
    def _propagate_to_sink(self, node_id: int, method: str, trigger_record: TriggerRecord):
        """Propagate packet through mesh to sink"""
        node = self.network.nodes[node_id]
        self.stats['tx_completions'] += 1
        
        if not node.route_to_sink:
            prop_delay = self.network.get_propagation_delay(node_id, 0)
            arrival_time = self.current_time + prop_delay + self.config.processing_delay
            hops = 1
        else:
            total_delay = 0
            current = node_id
            hops = 0
            for next_hop in node.route_to_sink[::-1] + [0]:
                prop_delay = self.network.get_propagation_delay(current, next_hop)
                total_delay += prop_delay + self.config.processing_delay
                
                payload = (self.config.proposed_payload if method == 'proposed' 
                          else self.config.zhang_payload)
                total_delay += self.network.get_transmission_time(payload)
                
                current = next_hop
                hops += 1
            
            arrival_time = self.current_time + total_delay
        
        trigger_record.arrival_time = arrival_time
        trigger_record.latency = arrival_time - trigger_record.time
        
        log.debug(f"Node {node_id} {method}: packet delivered via {hops} hops, "
                 f"latency={trigger_record.latency*1000:.2f}ms")
    
    def process_event(self, event: SimEvent):
        """Process a single simulation event"""
        self.current_time = event.time
        
        if event.event_type == EventType.FRAME_READY:
            self.stats['frames_processed'] += 1
            node = self.network.nodes[event.node_id]
            samples = self._generate_frame_samples(event.node_id)
            
            # Collect snapshot data if active
            if self.current_snapshot is not None:
                self._collect_snapshot_data(event.node_id, samples, node)
            
            # Run both detection methods
            prop_trigger, prop_strength = node.proposed.process_frame(
                samples, self.current_noise_power)
            zhang_trigger, zhang_strength = node.zhang.process_frame(
                samples, self.current_noise_power)
            
            # Record triggers for snapshot
            if self.current_snapshot is not None:
                if prop_trigger:
                    self.current_snapshot[event.node_id]['triggers_proposed'].append(self.current_time)
                if zhang_trigger:
                    self.current_snapshot[event.node_id]['triggers_zhang'].append(self.current_time)
            
            # Handle proposed method trigger
            if prop_trigger:
                is_tp = self._is_true_positive(event.node_id, self.current_time)
                record = TriggerRecord(
                    time=self.current_time,
                    node_id=event.node_id,
                    method='proposed',
                    is_true_positive=is_tp,
                    strength=prop_strength
                )
                self.proposed_triggers.append(record)
                self._attempt_transmission(event.node_id, 'proposed', record)
            
            # Handle Zhang method trigger
            if zhang_trigger:
                is_tp = self._is_true_positive(event.node_id, self.current_time)
                record = TriggerRecord(
                    time=self.current_time,
                    node_id=event.node_id,
                    method='zhang',
                    is_true_positive=is_tp,
                    strength=zhang_strength
                )
                self.zhang_triggers.append(record)
                self._attempt_transmission(event.node_id, 'zhang', record)
            
            # Schedule next frame
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + self.config.frame_duration,
                event_type=EventType.FRAME_READY,
                node_id=event.node_id
            ))
        
        elif event.event_type == EventType.TRUE_EVENT_START:
            self.stats['true_events_started'] += 1
            self.active_events[event.node_id] = (
                self.current_time + self.config.event_duration)
            
            log.debug(f"TRUE EVENT started at node {event.node_id}, t={self.current_time:.3f}s")
            
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + self.config.event_duration,
                event_type=EventType.TRUE_EVENT_END,
                node_id=event.node_id
            ))
        
        elif event.event_type == EventType.TRUE_EVENT_END:
            self.stats['true_events_ended'] += 1
            if event.node_id in self.active_events:
                del self.active_events[event.node_id]
        
        elif event.event_type == EventType.PACKET_TX_START:
            self._attempt_transmission(
                event.node_id, 
                event.data['method'],
                event.data['record']
            )
        
        elif event.event_type == EventType.PACKET_TX_COMPLETE:
            self._propagate_to_sink(
                event.node_id,
                event.data['method'],
                event.data['record']
            )
        
        elif event.event_type == EventType.NOISE_UPDATE:
            self._update_noise_power()
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + 60.0,
                event_type=EventType.NOISE_UPDATE,
                node_id=-1
            ))
    
    def run(self) -> Dict:
        """Run the simulation"""
        log.section("Running Simulation")
        
        self.initialize()
        self.network.print_topology_stats()
        
        log.subsection("Simulation Progress")
        progress = ProgressTracker(
            self.config.simulation_duration, 
            description="Simulating",
            update_interval=2.0
        )
        
        events_processed = 0
        last_stats_time = 0
        stats_interval = self.config.simulation_duration / 10  # Report every 10%
        
        # Checkpoint tracking
        last_checkpoint_time = 0
        checkpoint_count = 0
        
        while self.event_queue and self.current_time < self.config.simulation_duration:
            event = heapq.heappop(self.event_queue)
            self.process_event(event)
            events_processed += 1
            
            # Check snapshot timing (start/end snapshots)
            self._check_snapshot_timing()
            
            # Update progress bar
            progress.update(self.current_time)
            
            # Periodic detailed stats
            if self.current_time - last_stats_time >= stats_interval:
                last_stats_time = self.current_time
                self._print_interim_stats()
            
            # CHECKPOINT SAVING: Save partial results periodically
            if self.config.continuous_save and self.config.checkpoint_interval > 0:
                if self.current_time - last_checkpoint_time >= self.config.checkpoint_interval:
                    last_checkpoint_time = self.current_time
                    checkpoint_count += 1
                    self._save_checkpoint(checkpoint_count)
        
        # End any active snapshot
        if self.current_snapshot is not None:
            self._end_snapshot()
        
        progress.finish()
        
        log.subsection("Simulation Complete")
        log.info(f"Total events processed: {events_processed:,}")
        log.info(f"Simulation time: {self.current_time/3600:.2f} hours")
        if self.snapshots:
            log.info(f"Collected {len(self.snapshots)} raw data snapshots")
        
        return self._compute_results()
    
    def _save_checkpoint(self, checkpoint_num: int):
        """Save a checkpoint of current results"""
        try:
            output_dir = self.config.snapshot_output_dir or '.'
            os.makedirs(output_dir, exist_ok=True)
            
            # Compute partial results
            prop_tp = sum(1 for t in self.proposed_triggers if t.is_true_positive)
            prop_fp = len(self.proposed_triggers) - prop_tp
            zhang_tp = sum(1 for t in self.zhang_triggers if t.is_true_positive)
            zhang_fp = len(self.zhang_triggers) - zhang_tp
            
            checkpoint_data = {
                'checkpoint_num': checkpoint_num,
                'sim_time_sec': self.current_time,
                'sim_time_hours': self.current_time / 3600,
                'events_so_far': self.stats['true_events_started'],
                'proposed': {'tp': prop_tp, 'fp': prop_fp},
                'zhang': {'tp': zhang_tp, 'fp': zhang_fp},
                'snapshots_collected': len(self.snapshots),
                'congestion_events': self.congestion_events
            }
            
            filename = os.path.join(output_dir, f"checkpoint_{checkpoint_num:03d}.json")
            with open(filename, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            log.info(f"  💾 Checkpoint {checkpoint_num} saved: t={self.current_time/3600:.1f}h, "
                    f"proposed={prop_tp}TP/{prop_fp}FP, zhang={zhang_tp}TP/{zhang_fp}FP")
                    
        except Exception as e:
            log.info(f"  ! Checkpoint save failed: {e}")
        
        return self._compute_results()
    
    def _print_interim_stats(self):
        """Print interim statistics during simulation"""
        hours = self.current_time / 3600
        prop_tp = sum(1 for t in self.proposed_triggers if t.is_true_positive)
        prop_fp = len(self.proposed_triggers) - prop_tp
        zhang_tp = sum(1 for t in self.zhang_triggers if t.is_true_positive)
        zhang_fp = len(self.zhang_triggers) - zhang_tp
        
        log.info(f"  t={hours:.1f}h: frames={self.stats['frames_processed']:,}, "
                f"events={self.stats['true_events_started']}, "
                f"proposed={prop_tp}TP/{prop_fp}FP, zhang={zhang_tp}TP/{zhang_fp}FP, "
                f"congestion={self.congestion_events}")
    
    def _compute_results(self) -> Dict:
        """Compute performance metrics"""
        log.subsection("Computing Results")
        
        total_true_events = sum(len(events) for events in self.true_event_times.values())
        log.info(f"Total true events: {total_true_events}")
        
        # Proposed method metrics
        prop_tp = sum(1 for t in self.proposed_triggers if t.is_true_positive)
        prop_fp = sum(1 for t in self.proposed_triggers if not t.is_true_positive)
        prop_latencies = [t.latency for t in self.proposed_triggers 
                        if t.latency is not None and t.is_true_positive]
        
        # Zhang method metrics
        zhang_tp = sum(1 for t in self.zhang_triggers if t.is_true_positive)
        zhang_fp = sum(1 for t in self.zhang_triggers if not t.is_true_positive)
        zhang_latencies = [t.latency for t in self.zhang_triggers 
                         if t.latency is not None and t.is_true_positive]
        
        # ═══════════════════════════════════════════════════════════════════════
        # FALSE NEGATIVES: Count events that had NO detection
        # An event is "detected" if at least one TP trigger occurred during it
        # ═══════════════════════════════════════════════════════════════════════
        margin = self.config.frame_duration
        window_before = margin
        window_after = self.config.event_duration + self.config.gamma_d * self.config.frame_duration + margin
        
        # Count detected events for Proposed method
        prop_events_detected = 0
        for node_id, event_times in self.true_event_times.items():
            for event_time in event_times:
                # Check if any TP trigger falls within this event's window
                detected = False
                for trigger in self.proposed_triggers:
                    if trigger.node_id == node_id and trigger.is_true_positive:
                        if (event_time - window_before) <= trigger.time <= (event_time + window_after):
                            detected = True
                            break
                if detected:
                    prop_events_detected += 1
        
        prop_fn = total_true_events - prop_events_detected  # False negatives
        
        # Count detected events for Zhang method
        zhang_events_detected = 0
        for node_id, event_times in self.true_event_times.items():
            for event_time in event_times:
                detected = False
                for trigger in self.zhang_triggers:
                    if trigger.node_id == node_id and trigger.is_true_positive:
                        if (event_time - window_before) <= trigger.time <= (event_time + window_after):
                            detected = True
                            break
                if detected:
                    zhang_events_detected += 1
        
        zhang_fn = total_true_events - zhang_events_detected  # False negatives
        
        log.info(f"Proposed: {prop_tp} TP, {prop_fp} FP, {prop_fn} FN ({prop_events_detected}/{total_true_events} events detected)")
        log.info(f"Zhang: {zhang_tp} TP, {zhang_fp} FP, {zhang_fn} FN ({zhang_events_detected}/{total_true_events} events detected)")
        
        hours = self.config.simulation_duration / 3600.0
        num_sensor_nodes = self.config.num_nodes - 1
        
        results = {
            'config': {
                'num_nodes': self.config.num_nodes,
                'duration_hours': hours,
                'event_rate': self.config.event_rate
            },
            'true_events': {
                'total': total_true_events,
                'per_node_per_hour': total_true_events / num_sensor_nodes / hours if num_sensor_nodes > 0 else 0
            },
            'proposed': {
                'triggers': len(self.proposed_triggers),
                'true_positives': prop_tp,
                'false_positives': prop_fp,
                'false_negatives': prop_fn,
                'events_detected': prop_events_detected,
                'detection_rate': prop_events_detected / total_true_events * 100 if total_true_events > 0 else 0,  # % of events detected
                'miss_rate': prop_fn / total_true_events * 100 if total_true_events > 0 else 0,  # % of events missed
                'false_alarm_rate': prop_fp / hours / num_sensor_nodes if num_sensor_nodes > 0 else 0,
                'precision': prop_tp / (prop_tp + prop_fp) * 100 if (prop_tp + prop_fp) > 0 else 0,  # TP / (TP + FP)
                'latency_mean_ms': np.mean(prop_latencies) * 1000 if prop_latencies else 0,
                'latency_median_ms': np.median(prop_latencies) * 1000 if prop_latencies else 0,
                'latency_90th_ms': np.percentile(prop_latencies, 90) * 1000 if len(prop_latencies) >= 2 else (prop_latencies[0] * 1000 if prop_latencies else 0),
                'latency_99th_ms': np.percentile(prop_latencies, 99) * 1000 if len(prop_latencies) >= 2 else (prop_latencies[0] * 1000 if prop_latencies else 0),
                'network_load_bytes_per_hour': self.total_bytes_proposed / hours
            },
            'zhang': {
                'triggers': len(self.zhang_triggers),
                'true_positives': zhang_tp,
                'false_positives': zhang_fp,
                'false_negatives': zhang_fn,
                'events_detected': zhang_events_detected,
                'detection_rate': zhang_events_detected / total_true_events * 100 if total_true_events > 0 else 0,  # % of events detected
                'miss_rate': zhang_fn / total_true_events * 100 if total_true_events > 0 else 0,  # % of events missed
                'false_alarm_rate': zhang_fp / hours / num_sensor_nodes if num_sensor_nodes > 0 else 0,
                'precision': zhang_tp / (zhang_tp + zhang_fp) * 100 if (zhang_tp + zhang_fp) > 0 else 0,  # TP / (TP + FP)
                'latency_mean_ms': np.mean(zhang_latencies) * 1000 if zhang_latencies else 0,
                'latency_median_ms': np.median(zhang_latencies) * 1000 if zhang_latencies else 0,
                'latency_90th_ms': np.percentile(zhang_latencies, 90) * 1000 if len(zhang_latencies) >= 2 else (zhang_latencies[0] * 1000 if zhang_latencies else 0),
                'latency_99th_ms': np.percentile(zhang_latencies, 99) * 1000 if len(zhang_latencies) >= 2 else (zhang_latencies[0] * 1000 if zhang_latencies else 0),
                'network_load_bytes_per_hour': self.total_bytes_zhang / hours
            },
            'network': {
                'congestion_events': self.congestion_events,
                'congestion_per_day': self.congestion_events * 24 / hours
            },
            'simulation_stats': self.stats,
            'snapshots': {
                'count': len(self.snapshots),
                'duration_sec': self.config.snapshot_duration if self.snapshots else 0,
                'interval_sec': self.config.snapshot_interval if self.snapshots else 0
            }
        }
        
        return results


# =============================================================================
# MONTE CARLO RUNNER
# =============================================================================

def run_monte_carlo(config: SimulationConfig, num_runs: int = 10) -> Tuple[Dict, Optional[List]]:
    """Run multiple simulations and aggregate results
    
    Returns:
        Tuple of (aggregated_results, snapshots_from_first_run)
    """
    log.section(f"Monte Carlo Study: {num_runs} runs")
    
    all_results = []
    first_run_snapshots = None
    
    for run in range(num_runs):
        log.subsection(f"Monte Carlo Run {run + 1}/{num_runs}")
        config.seed = 42 + run
        sim = NetworkSimulator(config)
        results = sim.run()
        all_results.append(results)
        
        # Collect snapshots from first run only (to avoid memory bloat)
        if run == 0 and sim.snapshots:
            first_run_snapshots = sim.snapshots
            log.info(f"Collected {len(first_run_snapshots)} snapshots from first run")
        
        # Print run summary (detection_rate is already in %)
        log.info(f"Run {run + 1} complete: "
                f"Proposed DR={results['proposed']['detection_rate']:.1f}%, "
                f"Zhang DR={results['zhang']['detection_rate']:.1f}%")
    
    log.subsection("Aggregating Results")
    
    aggregated = {
        'config': all_results[0]['config'],
        'num_runs': num_runs,
        'proposed': {},
        'zhang': {},
        'network': {}
    }
    
    for method in ['proposed', 'zhang']:
        for metric in all_results[0][method].keys():
            values = [r[method][metric] for r in all_results]
            aggregated[method][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            log.debug(f"{method}.{metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    for metric in all_results[0]['network'].keys():
        values = [r['network'][metric] for r in all_results]
        aggregated['network'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return aggregated, first_run_snapshots


def format_results_table(results: Dict, title: str) -> str:
    """Format results as ASCII table"""
    lines = [
        f"\n{'='*80}",
        f"{title}",
        f"{'='*80}",
        f"Configuration: {results['config']['num_nodes']} nodes, "
        f"{results['config']['duration_hours']:.0f} hours",
        f"{'='*80}",
        "",
        f"{'Metric':<35} {'Proposed':>18} {'Zhang et al.':>18}",
        f"{'-'*80}"
    ]
    
    def get_val(r, method, metric):
        val = r[method][metric]
        return val['mean'] if isinstance(val, dict) else val
    
    # Detection rate (already in %)
    p_dr = get_val(results, 'proposed', 'detection_rate')
    z_dr = get_val(results, 'zhang', 'detection_rate')
    lines.append(f"{'Detection Rate (%)':<35} {p_dr:>17.1f}% {z_dr:>17.1f}%")
    
    # Miss rate / False negatives (already in %)
    p_mr = get_val(results, 'proposed', 'miss_rate')
    z_mr = get_val(results, 'zhang', 'miss_rate')
    lines.append(f"{'Miss Rate (% events missed)':<35} {p_mr:>17.1f}% {z_mr:>17.1f}%")
    
    # False negatives (count)
    p_fn = get_val(results, 'proposed', 'false_negatives')
    z_fn = get_val(results, 'zhang', 'false_negatives')
    lines.append(f"{'False Negatives (count)':<35} {p_fn:>18.0f} {z_fn:>18.0f}")
    
    # False positives (count)
    p_fp = get_val(results, 'proposed', 'false_positives')
    z_fp = get_val(results, 'zhang', 'false_positives')
    lines.append(f"{'False Positives (count)':<35} {p_fp:>18.0f} {z_fp:>18.0f}")
    
    # False alarm rate
    p_far = get_val(results, 'proposed', 'false_alarm_rate')
    z_far = get_val(results, 'zhang', 'false_alarm_rate')
    lines.append(f"{'False Alarm Rate (/hr/node)':<35} {p_far:>18.2f} {z_far:>18.2f}")
    
    # Precision (already in %)
    p_prec = get_val(results, 'proposed', 'precision')
    z_prec = get_val(results, 'zhang', 'precision')
    lines.append(f"{'Precision (% TP of triggers)':<35} {p_prec:>17.1f}% {z_prec:>17.1f}%")
    
    lines.append(f"{'-'*80}")
    
    # Mean latency
    p_lat = get_val(results, 'proposed', 'latency_mean_ms')
    z_lat = get_val(results, 'zhang', 'latency_mean_ms')
    lines.append(f"{'Mean Latency (ms)':<35} {p_lat:>18.1f} {z_lat:>18.1f}")
    
    # 99th percentile latency
    p_99 = get_val(results, 'proposed', 'latency_99th_ms')
    z_99 = get_val(results, 'zhang', 'latency_99th_ms')
    lines.append(f"{'99th %ile Latency (ms)':<35} {p_99:>18.1f} {z_99:>18.1f}")
    
    # Network load
    p_load = get_val(results, 'proposed', 'network_load_bytes_per_hour')
    z_load = get_val(results, 'zhang', 'network_load_bytes_per_hour')
    
    def fmt_bytes(b):
        if b > 1e6:
            return f"{b/1e6:.2f} MB"
        elif b > 1e3:
            return f"{b/1e3:.1f} kB"
        else:
            return f"{b:.0f} B"
    
    lines.append(f"{'Network Load (/hr)':<35} {fmt_bytes(p_load):>18} {fmt_bytes(z_load):>18}")
    
    if 'network' in results:
        cong = results['network'].get('congestion_per_day', {})
        if isinstance(cong, dict):
            cong = cong.get('mean', 0)
        lines.append(f"{'Congestion Events (/day)':<35} {cong:>18.0f}")
    
    lines.append(f"{'='*80}")
    
    return '\n'.join(lines)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_result(results: Dict, num_nodes: int, save_path: str = None, params: Dict = None):
    """Create visualization for a single network size result"""
    log.info(f"Generating single-result plot for {num_nodes} nodes...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    
    # Main title
    fig.suptitle(f'Simulation Results: {num_nodes}-Node Network', fontsize=16, fontweight='bold', y=0.98)
    
    # Parameter subtitle if provided
    if params:
        param_text = (f"Proposed: γ_d={params['proposed']['gamma_d']}, "
                     f"γ_a={params['proposed']['gamma_a']}, "
                     f"ζ={params['proposed']['zeta']}  |  "
                     f"Zhang: threshold={params['zhang']['threshold']}×, "
                     f"β={params['zhang']['beta']}, "
                     f"decimate=1/{params['zhang']['decimation']}  |  "
                     f"Events: {params['events']['rate']}/hr, "
                     f"SNR={params['events']['snr_db']}dB")
        fig.text(0.5, 0.94, param_text, ha='center', fontsize=10, style='italic', color='#555')
    
    def get_mean(r, method, metric):
        val = r[method][metric]
        return val['mean'] if isinstance(val, dict) else val
    
    def get_std(r, method, metric):
        val = r[method][metric]
        return val.get('std', 0) if isinstance(val, dict) else 0
    
    methods = ['Proposed', 'Zhang et al.']
    colors = ['#2ecc71', '#e74c3c']
    x = np.arange(len(methods))
    width = 0.6
    
    # (a) Detection Rate - already in %
    ax = axes[0, 0]
    vals = [get_mean(results, 'proposed', 'detection_rate'),
            get_mean(results, 'zhang', 'detection_rate')]
    errs = [get_std(results, 'proposed', 'detection_rate'),
            get_std(results, 'zhang', 'detection_rate')]
    bars = ax.bar(x, vals, width, color=colors, yerr=errs, capsize=5)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    # Auto-scale Y axis with margin
    max_val = max(vals) + max(errs) if errs else max(vals)
    ax.set_ylim([0, max_val * 1.15])
    ax.set_title('(a) Detection Rate')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # (b) Miss Rate (False Negatives %) - already in %
    ax = axes[0, 1]
    vals = [get_mean(results, 'proposed', 'miss_rate'),
            get_mean(results, 'zhang', 'miss_rate')]
    bars = ax.bar(x, vals, width, color=colors)
    ax.set_ylabel('Miss Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    max_val = max(vals) if max(vals) > 0 else 1
    ax.set_ylim([0, max_val * 1.15])
    ax.set_title('(b) Miss Rate (False Negatives)')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # (c) False Alarm Rate
    ax = axes[0, 2]
    vals = [get_mean(results, 'proposed', 'false_alarm_rate'),
            get_mean(results, 'zhang', 'false_alarm_rate')]
    bars = ax.bar(x, vals, width, color=colors)
    ax.set_ylabel('False Alarm Rate (/hr/node)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    max_val = max(vals) if max(vals) > 0 else 1
    ax.set_ylim([0, max_val * 1.15])
    ax.set_title('(c) False Alarm Rate')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # (d) Precision - already in %
    ax = axes[1, 0]
    vals = [get_mean(results, 'proposed', 'precision'),
            get_mean(results, 'zhang', 'precision')]
    bars = ax.bar(x, vals, width, color=colors)
    ax.set_ylabel('Precision (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    max_val = max(vals) if max(vals) > 0 else 1
    ax.set_ylim([0, max_val * 1.15])
    ax.set_title('(d) Precision (TP / all triggers)')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # (e) Mean Latency
    ax = axes[1, 1]
    vals = [get_mean(results, 'proposed', 'latency_mean_ms'),
            get_mean(results, 'zhang', 'latency_mean_ms')]
    # Filter out zeros for proper display
    bars = ax.bar(x, vals, width, color=colors)
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    max_val = max(vals) if max(vals) > 0 else 1
    ax.set_ylim([0, max_val * 1.15])
    ax.set_title('(e) Mean Latency')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # (f) Network Load
    ax = axes[1, 2]
    vals = [get_mean(results, 'proposed', 'network_load_bytes_per_hour') / 1000,
            get_mean(results, 'zhang', 'network_load_bytes_per_hour') / 1000]
    bars = ax.bar(x, vals, width, color=colors)
    ax.set_ylabel('Network Load (kB/hr)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title('(f) Network Load')
    ax.grid(axis='y', alpha=0.3)
    # Use log scale if values span multiple orders of magnitude
    if max(vals) > 0 and min([v for v in vals if v > 0], default=1) > 0:
        if max(vals) / min([v for v in vals if v > 0], default=1) > 100:
            ax.set_yscale('log')
        else:
            ax.set_ylim([0, max(vals) * 1.15])
    for bar, val in zip(bars, vals):
        if val > 0:
            ypos = bar.get_height() * 1.1 if ax.get_yscale() == 'log' else bar.get_height() + max(vals) * 0.03
            ax.text(bar.get_x() + bar.get_width()/2, ypos, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Figure saved to {save_path}")
    
    plt.close(fig)
    return fig


def plot_comparison(results_10: Dict, results_100: Dict, results_1000: Dict, 
                   save_path: str = None, params: Dict = None):
    """Create comparison visualization"""
    log.info("Generating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    
    # Main title
    fig.suptitle('Simulation Comparison: Network Scalability', fontsize=16, fontweight='bold', y=0.98)
    
    # Parameter subtitle if provided
    if params:
        param_text = (f"Proposed: γ_d={params['proposed']['gamma_d']}, "
                     f"γ_a={params['proposed']['gamma_a']}, "
                     f"ζ={params['proposed']['zeta']}  |  "
                     f"Zhang: threshold={params['zhang']['threshold']}×, "
                     f"β={params['zhang']['beta']}, "
                     f"decimate=1/{params['zhang']['decimation']}  |  "
                     f"Events: {params['events']['rate']}/hr, "
                     f"SNR={params['events']['snr_db']}dB")
        fig.text(0.5, 0.94, param_text, ha='center', fontsize=10, style='italic', color='#555')
    
    nodes = [10, 100, 1000]
    all_results = [results_10, results_100, results_1000]
    
    def get_mean(r, method, metric):
        val = r[method][metric]
        return val['mean'] if isinstance(val, dict) else val
    
    x = np.arange(len(nodes))
    width = 0.35
    
    # (a) Detection Rate - already in %
    ax = axes[0, 0]
    prop_dr = [get_mean(r, 'proposed', 'detection_rate') for r in all_results]
    zhang_dr = [get_mean(r, 'zhang', 'detection_rate') for r in all_results]
    
    ax.bar(x - width/2, prop_dr, width, label='Proposed', color='#2ecc71')
    ax.bar(x + width/2, zhang_dr, width, label='Zhang et al.', color='#e74c3c')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlabel('Number of Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    # Auto-scale Y axis
    all_vals = prop_dr + zhang_dr
    max_val = max(all_vals) if all_vals else 100
    min_val = min(all_vals) if all_vals else 0
    margin = (max_val - min_val) * 0.1 if max_val > min_val else 5
    ax.set_ylim([max(0, min_val - margin), min(100, max_val + margin)])
    ax.legend()
    ax.set_title('(a) Detection Rate vs Network Scale')
    ax.grid(axis='y', alpha=0.3)
    
    # (b) Miss Rate (False Negatives) - already in %
    ax = axes[0, 1]
    prop_mr = [get_mean(r, 'proposed', 'miss_rate') for r in all_results]
    zhang_mr = [get_mean(r, 'zhang', 'miss_rate') for r in all_results]
    
    ax.bar(x - width/2, prop_mr, width, label='Proposed', color='#2ecc71')
    ax.bar(x + width/2, zhang_mr, width, label='Zhang et al.', color='#e74c3c')
    ax.set_ylabel('Miss Rate (%)')
    ax.set_xlabel('Number of Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    all_vals = prop_mr + zhang_mr
    max_val = max(all_vals) if max(all_vals) > 0 else 1
    ax.set_ylim([0, max_val * 1.15])
    ax.legend()
    ax.set_title('(b) Miss Rate vs Network Scale')
    ax.grid(axis='y', alpha=0.3)
    
    # (c) False Alarm Rate
    ax = axes[0, 2]
    prop_far = [get_mean(r, 'proposed', 'false_alarm_rate') for r in all_results]
    zhang_far = [get_mean(r, 'zhang', 'false_alarm_rate') for r in all_results]
    
    ax.bar(x - width/2, prop_far, width, label='Proposed', color='#2ecc71')
    ax.bar(x + width/2, zhang_far, width, label='Zhang et al.', color='#e74c3c')
    ax.set_ylabel('False Alarm Rate (/hr/node)')
    ax.set_xlabel('Number of Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    all_vals = prop_far + zhang_far
    max_val = max(all_vals) if max(all_vals) > 0 else 1
    ax.set_ylim([0, max_val * 1.15])
    ax.legend()
    ax.set_title('(c) False Alarm Rate vs Network Scale')
    ax.grid(axis='y', alpha=0.3)
    
    # (d) Precision - already in %
    ax = axes[1, 0]
    prop_prec = [get_mean(r, 'proposed', 'precision') for r in all_results]
    zhang_prec = [get_mean(r, 'zhang', 'precision') for r in all_results]
    
    ax.bar(x - width/2, prop_prec, width, label='Proposed', color='#2ecc71')
    ax.bar(x + width/2, zhang_prec, width, label='Zhang et al.', color='#e74c3c')
    ax.set_ylabel('Precision (%)')
    ax.set_xlabel('Number of Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    all_vals = prop_prec + zhang_prec
    max_val = max(all_vals) if max(all_vals) > 0 else 1
    ax.set_ylim([0, max_val * 1.15])
    ax.legend()
    ax.set_title('(d) Precision vs Network Scale')
    ax.grid(axis='y', alpha=0.3)
    
    # (e) 99th Percentile Latency
    ax = axes[1, 1]
    prop_lat = [get_mean(r, 'proposed', 'latency_99th_ms') for r in all_results]
    zhang_lat = [get_mean(r, 'zhang', 'latency_99th_ms') for r in all_results]
    
    ax.bar(x - width/2, prop_lat, width, label='Proposed', color='#2ecc71')
    ax.bar(x + width/2, zhang_lat, width, label='Zhang et al.', color='#e74c3c')
    ax.set_ylabel('99th Percentile Latency (ms)')
    ax.set_xlabel('Number of Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    all_vals = [v for v in prop_lat + zhang_lat if v > 0]
    if all_vals and max(all_vals) / min(all_vals) > 10:
        ax.set_yscale('log')
    else:
        max_val = max(all_vals) if all_vals else 1
        ax.set_ylim([0, max_val * 1.15])
    ax.legend()
    ax.set_title('(e) 99th Percentile Latency vs Network Scale')
    ax.grid(axis='y', alpha=0.3)
    
    # (f) Network Load
    ax = axes[1, 2]
    prop_load = [get_mean(r, 'proposed', 'network_load_bytes_per_hour') / 1000 for r in all_results]
    zhang_load = [get_mean(r, 'zhang', 'network_load_bytes_per_hour') / 1000 for r in all_results]
    
    ax.bar(x - width/2, prop_load, width, label='Proposed', color='#2ecc71')
    ax.bar(x + width/2, zhang_load, width, label='Zhang et al.', color='#e74c3c')
    ax.set_ylabel('Network Load (kB/hr)')
    ax.set_xlabel('Number of Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    all_vals = [v for v in prop_load + zhang_load if v > 0]
    if all_vals and max(all_vals) / min(all_vals) > 10:
        ax.set_yscale('log')
    else:
        max_val = max(all_vals) if all_vals else 1
        ax.set_ylim([0, max_val * 1.15])
    ax.legend()
    ax.set_title('(f) Network Load vs Network Scale')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Figure saved to {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete simulation study using configuration from top of file"""
    global log
    
    # Set log level (DEBUG for verbose, INFO for normal, PROGRESS for minimal)
    log = Logger(level='INFO')
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log.section("IoT Mesh Network Simulation Study")
    log.info("Comparing: Temporal Spectral Noise-Floor Adaptation vs Zhang et al. 2023")
    log.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==========================================================================
    # READ CONFIGURATION FROM TOP OF FILE
    # ==========================================================================
    
    # Map preset name to TimePreset
    preset_map = {
        'FAST': TimePreset.FAST,
        'MEDIUM': TimePreset.MEDIUM,
        'ACCURATE': TimePreset.ACCURATE,
        'OVERNIGHT': TimePreset.OVERNIGHT
    }
    SELECTED_PRESET = preset_map.get(SIMULATION_PRESET.upper(), TimePreset.FAST)
    
    # Show configuration
    log.subsection("Configuration (from top of file)")
    log.info(f"  Preset: {SIMULATION_PRESET}")
    log.info(f"  Network sizes: " + ", ".join([
        "10" if RUN_10_NODES else "",
        "100" if RUN_50_NODES else "",
        "1000" if RUN_1000_NODES else ""
    ]).replace(", ,", ",").strip(", "))
    log.info(f"  Proposed: γ_d={GAMMA_D}, γ_a={GAMMA_A}, ζ={ZETA}")
    log.info(f"  Zhang: threshold={ZHANG_THRESHOLD}, β={ZHANG_BETA}, decimate=1/{ZHANG_DECIMATION}")
    log.info(f"  Events: {EVENT_RATE}/hr/node, SNR={EVENT_SNR_DB}dB, band={EVENT_FREQ_LOW}-{EVENT_FREQ_HIGH}Hz")
    
    # Show available presets
    TimePreset.list_presets()
    log.info(f"\nUsing preset: {SELECTED_PRESET['name']} - {SELECTED_PRESET['description']}")
    
    # ==========================================================================
    # BUILD CONFIGS FOR SELECTED NETWORK SIZES
    # ==========================================================================
    
    configs = {}
    
    # Custom parameters from top of file
    custom_params = {
        'gamma_d': GAMMA_D,
        'gamma_a': GAMMA_A,
        'zeta_k': ZETA,
        'event_rate': EVENT_RATE,
        'event_snr': EVENT_SNR_DB,
        'event_freq_low': EVENT_FREQ_LOW,
        'event_freq_high': EVENT_FREQ_HIGH,
        # Zhang method parameters
        'zhang_margin_ratio': ZHANG_THRESHOLD,
        'zhang_beta': ZHANG_BETA,
        'zhang_decimation': ZHANG_DECIMATION,
        # Snapshot parameters
        'enable_snapshots': ENABLE_SNAPSHOTS,
        'snapshot_duration': SNAPSHOT_DURATION_SEC,
        'snapshot_interval': SNAPSHOT_INTERVAL_SEC,
        'snapshot_nodes': SNAPSHOT_NODES,
        # Continuous saving parameters
        'continuous_save': CONTINUOUS_SAVE,
        'checkpoint_interval': CHECKPOINT_INTERVAL_SEC,
        'snapshot_output_dir': SNAPSHOT_OUTPUT_DIR,
        # Noise model parameters
        'emi_freq': NOISE_EMI_FREQ,
        'env_noise_enabled': NOISE_ENV_ENABLED,
    }
    
    if RUN_10_NODES:
        configs[10] = SimulationConfig.from_preset(SELECTED_PRESET, num_nodes=10, area_size=300.0, **custom_params)
    if RUN_50_NODES:
        configs[200] = SimulationConfig.from_preset(SELECTED_PRESET, num_nodes=200, area_size=1500.0, **custom_params)
    if RUN_1000_NODES:
        configs[1000] = SimulationConfig.from_preset(SELECTED_PRESET, num_nodes=1000, area_size=2500.0, **custom_params)
    
    if not configs:
        log.error("No network sizes selected! Enable at least one: RUN_10_NODES, RUN_100_NODES, or RUN_1000_NODES")
        return
    
    # Adjust Monte Carlo runs based on preset
    num_runs = SELECTED_PRESET['monte_carlo_runs']
    
    results = {}
    
    # Print estimated total runtime
    log.subsection("Estimated Runtimes")
    for num_nodes, config in configs.items():
        estimate = config.estimate_runtime()
        log.info(f"  {num_nodes:4d} nodes × {num_runs} runs: {estimate} per run")
    
    all_snapshots = {}  # Collect snapshots by network size
    
    for num_nodes, config in configs.items():
        log.section(f"Simulating {num_nodes}-Node Network")
        log.info(str(config))
        
        mc_results, snapshots = run_monte_carlo(config, num_runs=num_runs)
        results[num_nodes] = mc_results
        
        if snapshots:
            all_snapshots[num_nodes] = snapshots
        
        # Print results table
        print(format_results_table(results[num_nodes], 
                                  f"{num_nodes}-Node Network Results"))
    
    # Save snapshots if any were collected
    if all_snapshots and ENABLE_SNAPSHOTS:
        results_base = RESULTS_FILENAME.rsplit('.', 1)[0]
        
        # Determine output directory (same as results file)
        output_dir = os.path.dirname(RESULTS_FILENAME) or '.'
        
        for num_nodes, snapshots in all_snapshots.items():
            snap_filename = os.path.join(output_dir, f"{results_base}_{num_nodes}nodes")
            
            # Save metadata
            metadata = {
                'num_snapshots': len(snapshots),
                'sample_rate': configs[num_nodes].sample_rate,
                'snapshot_duration': configs[num_nodes].snapshot_duration,
                'snapshot_interval': configs[num_nodes].snapshot_interval,
                'num_nodes': num_nodes,
                'snapshots': []
            }
            
            arrays_to_save = {}
            for i, snap in enumerate(snapshots):
                snap_meta = {
                    'index': i,
                    'timestamp': snap.timestamp,
                    'duration': snap.duration,
                    'nodes': list(snap.node_data.keys()),
                }
                metadata['snapshots'].append(snap_meta)
                
                for node_id, data in snap.node_data.items():
                    arrays_to_save[f"snap{i}_node{node_id}_samples"] = np.array(data['samples'])
                    arrays_to_save[f"snap{i}_node{node_id}_triggers_proposed"] = np.array(
                        data.get('triggers_proposed', []))
                    arrays_to_save[f"snap{i}_node{node_id}_triggers_zhang"] = np.array(
                        data.get('triggers_zhang', []))
            
            meta_file = f"{snap_filename}_snapshots_meta.json"
            data_file = f"{snap_filename}_snapshots_data.npz"
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            np.savez_compressed(data_file, **arrays_to_save)
            
            meta_size = os.path.getsize(meta_file) / 1024
            data_size = os.path.getsize(data_file) / 1024 / 1024
            total_samples = sum(len(d['samples']) for s in snapshots for d in s.node_data.values())
            
            log.info(f"Saved {num_nodes}-node snapshots: {len(snapshots)} snapshots, "
                    f"{total_samples:,} samples ({data_size:.2f} MB)")
            log.info(f"  Metadata: {meta_file}")
            log.info(f"  Data: {data_file}")
    
    # Save results to JSON
    if SAVE_RESULTS:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        # Build output with configuration at top
        output_data = {
            '_simulation_parameters': {
                'preset': SIMULATION_PRESET,
                'proposed_method': {
                    'gamma_d': GAMMA_D,
                    'gamma_a': GAMMA_A,
                    'zeta': ZETA,
                },
                'zhang_method': {
                    'threshold': ZHANG_THRESHOLD,
                    'beta': ZHANG_BETA,
                    'decimation': ZHANG_DECIMATION,
                },
                'events': {
                    'rate_per_hour_per_node': EVENT_RATE,
                    'snr_db': EVENT_SNR_DB,
                    'freq_band_hz': [EVENT_FREQ_LOW, EVENT_FREQ_HIGH],
                },
                'network_sizes': {
                    '10_nodes': RUN_10_NODES,
                    '100_nodes': RUN_50_NODES,
                    '1000_nodes': RUN_1000_NODES,
                }
            },
            **results  # Add all results after config
        }
        
        with open(RESULTS_FILENAME, 'w') as f:
            json.dump(convert_numpy(output_data), f, indent=2)
        log.info(f"Results saved to {RESULTS_FILENAME}")
    
    # Generate plot filenames from RESULTS_FILENAME
    results_base = RESULTS_FILENAME.rsplit('.', 1)[0]  # Remove extension
    comparison_plot_path = f"{results_base}_comparison.png"
    single_plot_path = f"{results_base}_results.png"
    
    # Build params dict for plot titles
    plot_params = {
        'proposed': {
            'gamma_d': GAMMA_D,
            'gamma_a': GAMMA_A,
            'zeta': ZETA,
        },
        'zhang': {
            'threshold': ZHANG_THRESHOLD,
            'beta': ZHANG_BETA,
            'decimation': ZHANG_DECIMATION,
        },
        'events': {
            'rate': EVENT_RATE,
            'snr_db': EVENT_SNR_DB,
        }
    }
    
    # Generate comparison plot if we have multiple network sizes
    if len(results) >= 3 and all(k in results for k in [10, 100, 1000]):
        plot_comparison(results[10], results[100], results[1000], 
                       save_path=comparison_plot_path, params=plot_params)
    elif len(results) >= 1:
        # Generate a simple bar chart for single network size
        plot_single_result(list(results.values())[0], list(results.keys())[0],
                          save_path=single_plot_path, params=plot_params)
    
    # Print final summary
    log.section("FINAL SUMMARY")
    log.info(f"Preset: {SIMULATION_PRESET}")
    log.info(f"Parameters: γ_d={GAMMA_D}, γ_a={GAMMA_A}, ζ={ZETA}")
    
    for num_nodes in sorted(results.keys()):
        r = results[num_nodes]
        log.info(f"\n{num_nodes}-Node Network:")
        log.info(f"  Proposed: DR={r['proposed']['detection_rate']['mean']:.1f}% "
                f"(±{r['proposed']['detection_rate']['std']:.1f}%), "
                f"MissRate={r['proposed']['miss_rate']['mean']:.1f}%, "
                f"FAR={r['proposed']['false_alarm_rate']['mean']:.2f}/hr/node")
        log.info(f"  Zhang:    DR={r['zhang']['detection_rate']['mean']:.1f}% "
                f"(±{r['zhang']['detection_rate']['std']:.1f}%), "
                f"MissRate={r['zhang']['miss_rate']['mean']:.1f}%, "
                f"FAR={r['zhang']['false_alarm_rate']['mean']:.2f}/hr/node")
    
    log.info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()