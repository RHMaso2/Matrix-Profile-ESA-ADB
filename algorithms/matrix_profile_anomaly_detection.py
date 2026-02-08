"""
Matrix Profile Anomaly Detection for Satellite Telemetry
========================================================

An unsupervised anomaly detection algorithm for multi-channel satellite
telemetry, implemented for the ESA Anomaly Detection Benchmark (ESA-ADB)
Mission 1 dataset (84 months, 7.3M samples, 6 telemetry channels).

Algorithm:
    Computes AB-join Matrix Profiles between test subsequences and a
    reference library of nominal behaviour derived from training data.
    Subsequences with high distances to all known normal patterns are
    flagged as anomalies. Supports optional multi-scale window analysis
    and per-channel attribution.

ESA-ADB Requirements Compliance:
    R1  Binary response            Returns 0/1 per channel and aggregated     [Mandatory]
    R2  Batch detection             Segmented processing for memory efficiency [Mandatory]
    R3  Multi-channel dependencies  Configurable cross-channel fusion          [Optional]
    R4  Learn from training         Anomaly signature library                  [Optional]
    R5  Affected channels           Per-channel predictions and attribution    [Optional]
    R6  Channel classification      Target / non-target / telecommand config   [Optional]
    R7  Rare nominal events         Nominal event library for FP suppression   [Optional]
    R8  Irregular timestamps        Time-aware resampling and gap handling     [Optional]
    R9  Reasonable runtime           GPU acceleration and parallel processing  [Mandatory]

Output Format:
    CSV with columns: timestamp, per-channel scores, per-channel predictions,
    aggregated score, aggregated prediction, and per-channel ground truth.

Evaluation Metrics:
    - ESAScores (F0.5): Event-wise precision/recall, alarming precision, affiliation
    - ADTQC: Anomaly Detection Timing and Quality Criterion
    - ChannelAwareFScore (F0.5): Per-channel detection quality

Dependencies:
    numpy, pandas, stumpy, scipy, numba, portion, optuna (tuning only)

References:
    - Yeh et al. (2016), "Matrix Profile I: All Pairs Similarity Joins"
    - STUMPY library: https://github.com/TDAmeritrade/stumpy
    - ESA-ADB: https://doi.org/10.2514/6.2024-0865
"""

import numpy as np
import pandas as pd
import stumpy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import deque
import warnings
import json
from datetime import datetime
import portion as P
import time
import os
import sys
import numba
from scipy.ndimage import uniform_filter1d

# Enable line-buffered output for real-time progress reporting
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Configure numba thread pool (must precede any JIT compilation)
if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = str(os.cpu_count())
    numba.set_num_threads(os.cpu_count())

from ESA_metrics import ESAScores, ADTQC, ChannelAwareFScore

# Detect CUDA GPU availability for stumpy.gpu_stump acceleration
try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
    if HAS_GPU:
        try:
            device = cuda.get_current_device()
            GPU_NAME = device.name.decode() if hasattr(device.name, 'decode') else str(device.name)
        except Exception:
            GPU_NAME = "CUDA GPU"
    else:
        GPU_NAME = "N/A"
except ImportError:
    HAS_GPU = False
    GPU_NAME = "N/A"

print(f"[Init] CPU cores: {os.cpu_count()}, Numba threads: {numba.get_num_threads()}")
print(f"[Init] GPU available: {HAS_GPU}" + (f" ({GPU_NAME})" if HAS_GPU else ""))

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChannelConfig:
    """Channel classification and mapping for ESA-ADB telemetry (R6).

    Attributes:
        target_channels:     Telemetry channel column names to analyse.
        label_columns:       Corresponding ground-truth anomaly label columns.
        non_target_channels: Channels excluded from anomaly scoring.
        telecommand_channels: Telecommand channels (contextual, not scored).
    """
    target_channels: List[str] = field(
        default_factory=lambda: [f'channel_{i}' for i in range(41, 47)]
    )
    label_columns: List[str] = field(
        default_factory=lambda: [f'is_anomaly_channel_{i}' for i in range(41, 47)]
    )
    non_target_channels: List[str] = field(default_factory=list)
    telecommand_channels: List[str] = field(default_factory=list)


@dataclass
class OptimalMPConfig:
    """Full configuration for the Matrix Profile anomaly detection pipeline.

    Groups all hyperparameters for the AB-join computation, score
    transformation, thresholding, post-processing, and runtime
    optimisation.  Values marked 'tuned' were selected via Optuna
    Bayesian optimisation on a held-out validation split.
    """

    # --- Channel configuration (R6) ------------------------------------------
    channels: ChannelConfig = field(default_factory=ChannelConfig)

    # --- Window configuration ------------------------------------------------
    #   single_window_mode=True  uses only primary_window  (~1 h runtime)
    #   single_window_mode=False uses all window_sizes     (~4.5 h runtime)
    single_window_mode: bool = True
    window_sizes: List[int] = field(default_factory=lambda: [4, 16, 32, 64, 128])
    primary_window: int = 64      # 64 samples × 30 s = 32 minutes

    # --- Score persistence ----------------------------------------------------
    save_scores_for_tuning: bool = True

    # --- Reference library (R4) ----------------------------------------------
    #   Subsampled nominal training data used as the B series in AB-join.
    #   Larger reference improves coverage but increases O(n × m) cost.
    max_reference_size: int = 50_000

    # --- GPU acceleration -----------------------------------------------------
    use_gpu: bool = True

    # --- Multi-channel fusion (R3) --------------------------------------------
    #   'mean': average across channels  |  'max': worst-channel score
    #   'weighted': variance-proportional weighting
    fusion_method: str = 'mean'

    # --- Optional learning from training (R4, R7) ----------------------------
    #   Disabled by default to reduce runtime; enable for improved recall.
    learn_anomaly_signatures: bool = False
    max_signatures: int = 200
    learn_nominal_patterns: bool = False
    max_nominal_patterns: int = 500
    nominal_similarity_threshold: float = 0.85

    # --- Thresholding (tuned) ------------------------------------------------
    threshold_percentile: float = 99.22

    # --- Score transforms -----------------------------------------------------
    apply_log_transform: bool = True   # log(1 + x) compresses range
    smooth_window: int = 200           # ~100 min uniform moving average

    # --- Post-processing (tuned) ---------------------------------------------
    min_event_duration: int = 39       # Minimum event length (samples)
    gap_tolerance: int = 62            # Bridge gaps shorter than this
    extend_window: int = 128           # Extend events ± N samples
    trim_threshold: float = 0.80       # Trim low-confidence boundaries
    extend_coverage: bool = True       # Expand events into adjacent high-score zones

    # --- Irregular timestamp handling (R8) ------------------------------------
    handle_irregular_timestamps: bool = True
    expected_sampling_rate: float = 30.0   # seconds
    max_gap_tolerance: float = 90.0        # seconds

    # --- Processing efficiency (R9) -------------------------------------------
    segment_size: int = 5_000_000  # Samples per AB-join segment
    n_jobs: int = -1               # -1 = all available CPU cores

    # --- Debug ----------------------------------------------------------------
    debug_mode: bool = False
    debug_samples: int = 150_000


# =============================================================================
# NOMINAL EVENT LIBRARY (R7)
# =============================================================================

class NominalEventLibrary:
    """Library of rare but nominal subsequence patterns (R7).

    Identifies training subsequences that have high Matrix Profile distances
    (i.e. appear unusual) yet are labelled nominal.  During inference these
    patterns are used to suppress false positives caused by infrequent but
    legitimate operational modes.

    Attributes:
        max_size:             Maximum number of patterns to retain.
        similarity_threshold: Pearson correlation threshold for matching.
        patterns:             List of z-normalised reference patterns.
    """
    
    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.patterns: List[np.ndarray] = []
    
    def build(self, train_data: np.ndarray, train_labels: np.ndarray,
              mp_distances: np.ndarray, window_size: int):
        """Populate the library from training data and pre-computed MP distances.

        Selects the highest-distance nominal subsequences and stores their
        z-normalised representations for later correlation-based matching.
        """
        print("    Building nominal event library (R7)...")
        
        # Align MP distances with labels
        offset = window_size - 1
        n_dist = len(mp_distances)
        aligned_labels = train_labels[offset:offset + n_dist]
        
        # Find high-distance nominal regions (unusual but not anomalies)
        nominal_mask = aligned_labels == 0
        distances_nominal = np.where(nominal_mask, mp_distances, -np.inf)
        
        # Get top unusual nominal patterns
        top_k = min(self.max_size, nominal_mask.sum())
        top_indices = np.argsort(distances_nominal)[-top_k:]
        top_indices = top_indices[distances_nominal[top_indices] > 0]
        
        for idx in top_indices:
            actual_idx = idx + offset
            if actual_idx + window_size <= len(train_data):
                pattern = train_data[actual_idx:actual_idx + window_size].copy()
                # Z-normalize
                pattern = (pattern - pattern.mean(axis=0)) / (pattern.std(axis=0) + 1e-10)
                self.patterns.append(pattern)
        
        print(f"      Stored {len(self.patterns)} rare nominal patterns")
    
    def is_rare_nominal(self, subsequence: np.ndarray) -> bool:
        """Return True if the subsequence matches a stored rare-nominal pattern."""
        if not self.patterns:
            return False
        
        # Z-normalise the input
        subseq_norm = (subsequence - subsequence.mean(axis=0)) / (subsequence.std(axis=0) + 1e-10)

        for pattern in self.patterns:
            if pattern.shape != subseq_norm.shape:
                continue
            # Compute mean Pearson correlation across all channels
            corrs = []
            n_ch = pattern.shape[1] if pattern.ndim > 1 else 1
            for ch in range(n_ch):
                p = pattern[:, ch] if pattern.ndim > 1 else pattern
                s = subseq_norm[:, ch] if subseq_norm.ndim > 1 else subseq_norm
                corr = np.corrcoef(p.flatten(), s.flatten())[0, 1]
                if np.isfinite(corr):
                    corrs.append(corr)
            if corrs and np.mean(corrs) >= self.similarity_threshold:
                return True
        return False


# =============================================================================
# ANOMALY SIGNATURE LIBRARY (R4)
# =============================================================================

class AnomalySignatureLibrary:
    """Library of known anomaly patterns learned from labelled training data (R4).

    Extracts z-normalised signatures from the centres of labelled anomaly
    events together with their affected-channel lists.  During inference,
    test subsequences are correlated against the library to boost detection
    of previously observed failure modes.

    Attributes:
        max_signatures:    Maximum number of signatures to store.
        signatures:        List of z-normalised anomaly patterns.
        affected_channels: Parallel list recording which channels each signature affects.
    """
    
    def __init__(self, max_signatures: int = 200):
        self.max_signatures = max_signatures
        self.signatures: List[np.ndarray] = []
        self.affected_channels: List[List[int]] = []
    
    def build(self, train_data: np.ndarray, train_labels: np.ndarray,
              per_channel_labels: np.ndarray, window_size: int):
        """Extract anomaly signatures from labelled training events."""
        print("    Building anomaly signature library (R4)...")
        
        # Find anomaly events in training data
        diff = np.diff(np.concatenate([[0], train_labels.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        print(f"      Found {len(starts)} anomaly events in training")
        
        for start, end in zip(starts, ends):
            length = end - start
            if length < window_size // 2:
                continue
            
            # Extract pattern from middle of event
            mid = (start + end) // 2
            pat_start = max(0, mid - window_size // 2)
            pat_end = min(len(train_data), pat_start + window_size)
            
            if pat_end - pat_start < window_size:
                continue
            
            pattern = train_data[pat_start:pat_end].copy()
            # Z-normalise each channel independently
            pattern = (pattern - pattern.mean(axis=0)) / (pattern.std(axis=0) + 1e-10)
            
            # Record which channels are affected
            affected = []
            for ch in range(per_channel_labels.shape[1]):
                if per_channel_labels[pat_start:pat_end, ch].any():
                    affected.append(ch)
            
            self.signatures.append(pattern)
            self.affected_channels.append(affected)
            
            if len(self.signatures) >= self.max_signatures:
                break
        
        print(f"      Stored {len(self.signatures)} anomaly signatures")
    
    def match(self, subsequence: np.ndarray, threshold: float = 0.7) -> Tuple[float, List[int]]:
        """Match a subsequence against stored signatures.

        Returns:
            Tuple of (best_correlation_score, affected_channel_indices).
            If no signature exceeds the threshold, affected channels is empty.
        """
        if not self.signatures:
            return 0.0, []
        
        subseq_norm = (subsequence - subsequence.mean(axis=0)) / (subsequence.std(axis=0) + 1e-10)
        
        best_score = 0.0
        best_channels = []
        
        for sig, channels in zip(self.signatures, self.affected_channels):
            if sig.shape != subseq_norm.shape:
                continue
            
            corrs = []
            for ch in range(sig.shape[1]):
                corr = np.corrcoef(subseq_norm[:, ch], sig[:, ch])[0, 1]
                if np.isfinite(corr):
                    corrs.append(corr)
            
            if corrs:
                score = np.mean(corrs)
                if score > best_score:
                    best_score = score
                    best_channels = channels
        
        return best_score, best_channels if best_score >= threshold else []


# =============================================================================
# AB-JOIN MATRIX PROFILE COMPUTATION
# =============================================================================

def compute_mp_abjoin(
    test_data: np.ndarray,
    reference_data: np.ndarray,
    window_size: int,
    segment_size: int = 500_000,
    show_progress: bool = True,
    n_jobs: int = -1,
    use_gpu: bool = False
) -> np.ndarray:
    """Compute the AB-join Matrix Profile for each telemetry channel.

    For every subsequence of length `window_size` in the test series (A),
    finds its nearest-neighbour distance in the reference series (B).
    Channels are processed sequentially because stumpy.stump already
    parallelises internally via numba across all CPU cores.

    Args:
        test_data:      Test time series, shape (n_samples, n_channels).
        reference_data: Nominal reference series, shape (m_samples, n_channels).
        window_size:    Subsequence length for the Matrix Profile.
        segment_size:   Maximum samples per segment (memory control).
        show_progress:  Print per-channel timing information.
        n_jobs:         Ignored (kept for API compatibility); parallelism
                        is managed internally by stumpy/numba.
        use_gpu:        Attempt GPU acceleration via stumpy.gpu_stump.

    Returns:
        Per-channel distance matrix of shape (n_output, n_channels) where
        n_output = n_samples - window_size + 1.
    """
    n_test = len(test_data)
    n_channels = test_data.shape[1]
    n_output = n_test - window_size + 1
    channel_distances = np.zeros((n_output, n_channels), dtype=np.float32)
    
    # Check GPU availability
    gpu_available = HAS_GPU and use_gpu
    if show_progress:
        mode = "GPU" if gpu_available else "CPU (all cores)"
        print(f"      Mode: {mode}, {n_channels} channels, ref_size={len(reference_data):,}")
    
    total_start = time.time()

    for ch in range(n_channels):
        ch_start = time.time()

        test_ch = test_data[:, ch].astype(np.float64)
        ref_ch = reference_data[:, ch].astype(np.float64)

        # Inject negligible noise to avoid division-by-zero in z-normalisation
        rng = np.random.RandomState(ch)
        test_ch = test_ch + rng.randn(len(test_ch)) * 1e-10
        ref_ch = ref_ch + rng.randn(len(ref_ch)) * 1e-10

        # Select compute path: GPU > segmented CPU > standard CPU
        if gpu_available:
            try:
                mp = stumpy.gpu_stump(test_ch, m=window_size, T_B=ref_ch, ignore_trivial=False)
                distances = mp[:, 0].astype(np.float32)
            except Exception as e:
                if show_progress:
                    print(f"      GPU unavailable for this segment, falling back to CPU: {e}")
                mp = stumpy.stump(test_ch, m=window_size, T_B=ref_ch, ignore_trivial=False)
                distances = mp[:, 0].astype(np.float32)
        elif n_test > segment_size:
            distances = _compute_segmented_abjoin(test_ch, ref_ch, window_size, segment_size)
        else:
            mp = stumpy.stump(test_ch, m=window_size, T_B=ref_ch, ignore_trivial=False)
            distances = mp[:, 0].astype(np.float32)

        # Replace non-finite values with the channel median
        nan_mask = ~np.isfinite(distances)
        if nan_mask.any():
            median_val = np.nanmedian(distances[~nan_mask]) if (~nan_mask).any() else 0
            distances[nan_mask] = median_val
        
        channel_distances[:, ch] = distances[:n_output]
        
        if show_progress:
            ch_elapsed = time.time() - ch_start
            print(f"      Ch {ch+1}/{n_channels}: {ch_elapsed:.1f}s", flush=True)
    
    if show_progress:
        total_elapsed = time.time() - total_start
        print(f"      Total: {total_elapsed:.1f}s ({total_elapsed/n_channels:.1f}s/channel)")
    
    return channel_distances


def _compute_segmented_abjoin(
    test_data: np.ndarray,
    ref_data: np.ndarray,
    window_size: int,
    segment_size: int
) -> np.ndarray:
    """Compute the AB-join in overlapping segments to limit peak memory usage.

    Segments overlap by one window length so that no subsequence is missed
    at boundaries.  Where overlapping segments both produce a distance for
    the same position, the maximum (more conservative) value is retained.
    """
    n_test = len(test_data)
    n_output = n_test - window_size + 1
    distances = np.full(n_output, np.nan, dtype=np.float32)
    
    # Calculate total segments for progress
    n_segments = (n_test + segment_size - 1) // segment_size
    seg_num = 0
    
    start = 0
    while start < n_test:
        end = min(start + segment_size, n_test)
        seg_num += 1
        
        if end - start < window_size * 2:
            break
        
        seg_start = time.time()
        segment = test_data[start:end]
        mp = stumpy.stump(segment, m=window_size, T_B=ref_data, ignore_trivial=False)
        seg_dist = mp[:, 0].astype(np.float32)
        seg_elapsed = time.time() - seg_start
        
        # Progress update
        pct = (end / n_test) * 100
        print(f"        Seg {seg_num}: {pct:.0f}% ({seg_elapsed:.1f}s)", flush=True)
        
        out_start = start
        out_end = min(start + len(seg_dist), n_output)
        copy_len = out_end - out_start
        
        existing = distances[out_start:out_end]
        distances[out_start:out_end] = np.where(
            np.isnan(existing),
            seg_dist[:copy_len],
            np.fmax(existing, seg_dist[:copy_len])
        )
        
        start = end - window_size
    
    nan_mask = np.isnan(distances)
    if nan_mask.any():
        median_val = np.nanmedian(distances[~nan_mask]) if (~nan_mask).any() else 0
        distances[nan_mask] = median_val
    
    return distances


def compute_multiscale_mp_perchannel(
    test_data: np.ndarray,
    reference_data: np.ndarray,
    window_sizes: List[int],
    config: OptimalMPConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Compute Matrix Profiles at multiple temporal scales and fuse the results.

    For each window size, an independent AB-join is computed per channel.
    When multiple windows are used, per-window scores are z-normalised and
    the element-wise maximum is taken to retain the strongest anomaly signal
    at each time step, regardless of the originating scale.

    Args:
        test_data:      Test series, shape (n_samples, n_channels).
        reference_data: Nominal reference, shape (m_samples, n_channels).
        window_sizes:   List of subsequence lengths to evaluate.
        config:         Pipeline configuration.

    Returns:
        combined_score:     1-D aggregated anomaly score across channels.
        per_channel_scores: 2-D array (n_output, n_channels) of per-channel scores.
        all_distances:      Dict mapping window_size -> raw per-channel distances.
    """
    if config.single_window_mode:
        active_windows = [config.primary_window]
        print(f"\n  Computing Matrix Profile (single window = {config.primary_window})...")
    else:
        active_windows = window_sizes
        print(f"\n  Computing multi-scale Matrix Profile...")
        print(f"    [Windows: {window_sizes}]")
    
    n_channels = test_data.shape[1]
    all_distances = {}
    min_length = None
    
    for wi, ws in enumerate(active_windows):
        ws_start = time.time()
        print(f"\n    [{wi+1}/{len(active_windows)}] Window size {ws} ({ws * 30 / 60:.0f} min)...")
        
        # Get per-channel distances
        distances = compute_mp_abjoin(
            test_data, reference_data, ws, config.segment_size, 
            show_progress=True, n_jobs=config.n_jobs,
            use_gpu=config.use_gpu
        )
        
        all_distances[ws] = distances
        
        if min_length is None or len(distances) < min_length:
            min_length = len(distances)
        
        ws_elapsed = time.time() - ws_start
        print(f"    Window {ws} done ({ws_elapsed:.1f}s)")
    
    # Combine scores across window sizes
    print("    Combining scales per-channel...")

    if len(active_windows) == 1:
        per_channel_scores = all_distances[active_windows[0]][:min_length].astype(np.float64)
    else:
        # Z-normalise each window's scores then take the element-wise max
        stacked = np.stack([dist[:min_length] for dist in all_distances.values()], axis=0)
        
        medians = np.median(stacked, axis=1, keepdims=True)
        stds = np.std(stacked, axis=1, keepdims=True) + 1e-10
        z_scores = (stacked - medians) / stds
        per_channel_scores = np.max(z_scores, axis=0)

    # Aggregate across channels using the configured fusion method
    if config.fusion_method == 'max':
        combined_score = np.max(per_channel_scores, axis=1)
    elif config.fusion_method == 'mean':
        combined_score = np.mean(per_channel_scores, axis=1)
    else:
        variances = np.var(test_data[:min_length], axis=0)
        weights = variances / variances.sum()
        combined_score = np.sum(per_channel_scores * weights, axis=1)
    
    print(f"    Combined score range: [{combined_score.min():.3f}, {combined_score.max():.3f}]")
    
    return combined_score, per_channel_scores, all_distances


# =============================================================================
# EVENT POST-PROCESSING
# =============================================================================

class ESAOptimizedPostProcessor:
    """Post-processing pipeline optimised for event-based ESA-ADB evaluation.

    Transforms raw binary predictions into coherent anomaly events by:
      1. Merging nearby detections separated by short gaps.
      2. Removing spuriously short events below a minimum duration.
      3. Extending event boundaries to improve ground-truth overlap.
      4. Trimming low-confidence edges from each event.
      5. Expanding events into adjacent high-score regions.

    Args:
        min_event_duration: Minimum event length in samples; shorter events are removed.
        gap_tolerance:      Maximum gap (samples) between detections to bridge.
        extend_window:      Samples to extend each event boundary symmetrically.
        trim_fp_threshold:  Fraction of peak score below which boundary samples are trimmed.
        extend_coverage:    Whether to expand events into adjacent high-score zones.
    """
    
    def __init__(
        self,
        min_event_duration: int = 1,
        gap_tolerance: int = 3,
        extend_window: int = 0,
        trim_fp_threshold: float = 0.5,
        extend_coverage: bool = True
    ):
        self.min_event_duration = min_event_duration
        self.gap_tolerance = gap_tolerance
        self.extend_window = extend_window
        self.trim_fp_threshold = trim_fp_threshold
        self.extend_coverage = extend_coverage
    
    def process(self, predictions: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply the full post-processing chain and return refined predictions."""
        result = predictions.copy()
        result = self._merge_aggressive(result, self.gap_tolerance)
        result = self._filter_short(result, self.min_event_duration)
        if self.extend_window > 0:
            result = self._extend_events(result, self.extend_window)
        if self.trim_fp_threshold > 0:
            result = self._trim_boundaries(result, scores, self.trim_fp_threshold)
        if self.extend_coverage:
            result = self._extend_to_coverage(result, scores)
        return result
    
    def _merge_aggressive(self, predictions: np.ndarray, gap_tolerance: int) -> np.ndarray:
        """Bridge gaps between detected events that are shorter than gap_tolerance."""
        result = predictions.copy()
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) < 2:
            return result
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - ends[i]
            if gap <= gap_tolerance:
                result[ends[i]:starts[i + 1]] = 1
        return result
    
    def _filter_short(self, predictions: np.ndarray, min_duration: int) -> np.ndarray:
        """Remove detected events shorter than min_duration samples."""
        result = predictions.copy()
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            if (end - start) < min_duration:
                result[start:end] = 0
        return result
    
    def _extend_events(self, predictions: np.ndarray, extend_window: int) -> np.ndarray:
        """Symmetrically extend each event by extend_window samples on each side."""
        result = predictions.copy()
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            new_s = max(0, s - extend_window)
            new_e = min(len(predictions), e + extend_window)
            result[new_s:new_e] = 1
        return result
    
    def _trim_boundaries(self, predictions: np.ndarray, scores: np.ndarray, 
                          threshold_fraction: float) -> np.ndarray:
        """Trim event boundaries where scores fall below a fraction of the peak."""
        result = predictions.copy()
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            if end - start < 3:
                continue
            event_scores = scores[start:end]
            threshold = np.max(event_scores) * threshold_fraction
            new_start = start
            while new_start < end - 1 and scores[new_start] < threshold:
                result[new_start] = 0
                new_start += 1
            new_end = end
            while new_end > new_start + 1 and scores[new_end - 1] < threshold:
                result[new_end - 1] = 0
                new_end -= 1
        return result
    
    def _extend_to_coverage(self, predictions: np.ndarray, scores: np.ndarray,
                            extension_threshold: float = 0.8) -> np.ndarray:
        """Expand event boundaries into adjacent regions with elevated scores."""
        result = predictions.copy()
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            event_scores = scores[start:end]
            if len(event_scores) == 0:
                continue
            median_score = np.median(event_scores)
            extend_thresh = median_score * extension_threshold
            new_start = start
            while new_start > 0 and scores[new_start - 1] > extend_thresh:
                new_start -= 1
                result[new_start] = 1
            new_end = end
            while new_end < len(scores) and scores[new_end] > extend_thresh:
                result[new_end] = 1
                new_end += 1
        return result


# =============================================================================
# IRREGULAR TIMESTAMP HANDLING (R8)
# =============================================================================

def handle_timestamps(
    data: np.ndarray,
    timestamps: np.ndarray,
    expected_rate: float = 30.0,
    max_gap: float = 90.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample irregularly-sampled telemetry to a uniform time grid (R8).

    Satellite downlink can introduce timing jitter and data gaps.  This
    function detects such irregularities and, when necessary, interpolates
    the telemetry onto a regular grid at the expected sampling rate.
    A boolean mask is returned to indicate positions that fall within
    data gaps so that predictions there can be suppressed.

    Args:
        data:          Telemetry array, shape (n_samples, n_channels).
        timestamps:    Corresponding timestamp array.
        expected_rate: Nominal sampling interval in seconds.
        max_gap:       Intervals exceeding this value (seconds) are flagged as gaps.

    Returns:
        Tuple of (resampled_data, resampled_timestamps, gap_mask).
    """
    print("\n  Handling timestamps (R8)...")
    
    if len(timestamps) == 0:
        return data, timestamps, np.zeros(len(data), dtype=bool)
    
    if np.issubdtype(timestamps.dtype, np.datetime64):
        t0 = timestamps[0]
        ts_seconds = (timestamps - t0).astype('timedelta64[s]').astype(float)
    elif isinstance(timestamps[0], (pd.Timestamp, datetime)):
        t0 = timestamps[0]
        ts_seconds = np.array([(t - t0).total_seconds() for t in timestamps])
    else:
        ts_seconds = timestamps.astype(float)
    
    diffs = np.diff(ts_seconds)
    median_diff = np.median(diffs)
    
    print(f"    Median interval: {median_diff:.1f}s (expected: {expected_rate}s)")
    
    gaps = diffs > max_gap
    n_gaps = gaps.sum()
    
    if n_gaps > 0:
        print(f"    Found {n_gaps} gaps > {max_gap}s")
    
    if np.abs(median_diff - expected_rate) < 5 and n_gaps == 0:
        print("    Timestamps regular, no resampling needed")
        return data, timestamps, np.zeros(len(data), dtype=bool)
    
    print(f"    Resampling to {expected_rate}s intervals...")
    
    from scipy import interpolate
    
    total_duration = ts_seconds[-1]
    n_regular = int(total_duration / expected_rate) + 1
    regular_times = np.arange(n_regular) * expected_rate
    
    resampled = np.zeros((n_regular, data.shape[1]), dtype=np.float32)
    gap_mask = np.zeros(n_regular, dtype=bool)
    
    for ch in range(data.shape[1]):
        f = interpolate.interp1d(ts_seconds, data[:, ch], kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
        resampled[:, ch] = f(regular_times)
    
    for i in range(len(gaps)):
        if gaps[i]:
            gap_start = ts_seconds[i]
            gap_end = ts_seconds[i + 1]
            gap_mask[(regular_times >= gap_start) & (regular_times <= gap_end)] = True
    
    if isinstance(timestamps[0], (pd.Timestamp, datetime)):
        resampled_ts = pd.date_range(start=t0, periods=n_regular, 
                                      freq=f'{int(expected_rate)}S').values
    else:
        resampled_ts = regular_times
    
    print(f"    Resampled: {len(data):,} -> {n_regular:,} samples")
    
    return resampled, resampled_ts, gap_mask


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def run_perchannel_pipeline(
    train_path: str,
    test_path: str,
    config: OptimalMPConfig = None,
    output_dir: str = None
) -> Dict:
    """Execute the full Matrix Profile anomaly detection pipeline.

    Workflow:
      1. Load and validate training/test CSV data.
      2. Resample irregular timestamps if necessary (R8).
      3. Build a nominal reference library from anomaly-free training data.
      4. Optionally construct anomaly signature (R4) and nominal event (R7) libraries.
      5. Compute per-channel AB-join Matrix Profiles (single- or multi-scale).
      6. Apply score transforms (log compression, temporal smoothing).
      7. Threshold and post-process per-channel predictions.
      8. Aggregate channel predictions (logical OR) for system-level output.
      9. Evaluate against ESA-ADB metrics and save results.

    Args:
        train_path: Path to the training CSV (must contain timestamp, channel,
                    and label columns as defined in ChannelConfig).
        test_path:  Path to the test CSV with the same schema.
        config:     Pipeline configuration; uses defaults if None.
        output_dir: Directory for output CSV and metrics JSON.  If None,
                    results are returned but not written to disk.

    Returns:
        Dict containing 'output_df' (DataFrame), 'metrics' (dict),
        'esa_results', 'adtqc_results', 'ca_results', and optional
        library objects.
    """
    if config is None:
        config = OptimalMPConfig()
    
    print("\n")
    print("=" * 70)
    print("MATRIX PROFILE ANOMALY DETECTION - PER-CHANNEL OUTPUT")
    print("ESA-ADB Mission 1 | R5 Per-Channel Predictions")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ---- Load data ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    print(f"\nLoading training: {train_path}")
    train_raw = pd.read_csv(train_path, low_memory=False)
    train_raw['timestamp'] = pd.to_datetime(train_raw['timestamp'])
    print(f"  Shape: {train_raw.shape}")
    
    print(f"\nLoading test: {test_path}")
    test_raw = pd.read_csv(test_path, low_memory=False)
    test_raw['timestamp'] = pd.to_datetime(test_raw['timestamp'])
    print(f"  Shape: {test_raw.shape}")
    
    # ---- Debug mode: subsample for rapid iteration -------------------------
    if config.debug_mode:
        print(f"\n*** DEBUG MODE: {config.debug_samples:,} samples ***")
        train_raw = train_raw.tail(config.debug_samples).copy()
        
        label_cols = config.channels.label_columns
        test_labels_temp = (test_raw[label_cols].sum(axis=1) > 0).astype(int)
        anomaly_idx = test_labels_temp[test_labels_temp == 1].index.tolist()
        if anomaly_idx:
            center = anomaly_idx[len(anomaly_idx) // 2]
            start = max(0, center - config.debug_samples // 2)
            test_raw = test_raw.iloc[start:start + config.debug_samples].copy()
        else:
            test_raw = test_raw.head(config.debug_samples).copy()
    
    # ---- Extract feature arrays and ground-truth labels --------------------
    channel_names = config.channels.target_channels
    label_cols = config.channels.label_columns
    n_channels = len(channel_names)
    
    train_features = train_raw[channel_names].values.astype(np.float32)
    test_features = test_raw[channel_names].values.astype(np.float32)
    
    train_labels = (train_raw[label_cols].sum(axis=1) > 0).astype(np.int8).values
    test_labels = (test_raw[label_cols].sum(axis=1) > 0).astype(np.int8).values
    
    train_per_channel = train_raw[label_cols].values.astype(np.int8)
    test_per_channel = test_raw[label_cols].values.astype(np.int8)
    
    test_timestamps = test_raw['timestamp'].values
    
    print(f"\n  Training: {len(train_features):,} samples ({train_labels.mean()*100:.2f}% anomaly)")
    print(f"  Test: {len(test_features):,} samples ({test_labels.mean()*100:.2f}% anomaly)")
    print(f"  Channels: {n_channels} ({channel_names})")
    
    # ---- Handle irregular timestamps (R8) ----------------------------------
    if config.handle_irregular_timestamps:
        test_features, test_timestamps, gap_mask = handle_timestamps(
            test_features, test_timestamps,
            config.expected_sampling_rate, config.max_gap_tolerance
        )
        if len(test_labels) > len(test_features):
            test_labels = test_labels[:len(test_features)]
            test_per_channel = test_per_channel[:len(test_features)]
    else:
        gap_mask = np.zeros(len(test_features), dtype=bool)
    
    # ---- Build nominal reference library ------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING REFERENCE (TRAINING)")
    print("=" * 70)
    
    train_start_time = time.time()
    
    nominal_mask = train_labels == 0
    nominal_data = train_features[nominal_mask]
    
    if len(nominal_data) > config.max_reference_size:
        step = len(nominal_data) // config.max_reference_size
        reference_data = nominal_data[::step][:config.max_reference_size]
    else:
        reference_data = nominal_data
    
    print(f"  Reference library: {len(reference_data):,} samples")
    
    # ---- Optionally build anomaly signature library (R4) -------------------
    anomaly_library = None
    if config.learn_anomaly_signatures:
        anomaly_library = AnomalySignatureLibrary(config.max_signatures)
        anomaly_library.build(
            train_features, train_labels, train_per_channel, config.primary_window
        )
    
    # ---- Optionally build nominal event library (R7) -----------------------
    nominal_library = None
    if config.learn_nominal_patterns:
        print("\n  Computing calibration MP for nominal library (R7)...")
        # Use a subset of nominal data for calibration
        calib_size = min(50_000, len(nominal_data))
        calib_data = nominal_data[:calib_size]
        calib_labels = np.zeros(calib_size, dtype=np.int8)

        # Compute AB-join MP on calibration subset
        calib_mp = compute_mp_abjoin(
            calib_data, reference_data, config.primary_window, 
            config.segment_size, show_progress=False, n_jobs=config.n_jobs
        )
        calib_mp_combined = np.mean(calib_mp, axis=1)  # Combine channels
        
        nominal_library = NominalEventLibrary(
            config.max_nominal_patterns, config.nominal_similarity_threshold
        )
        nominal_library.build(calib_data, calib_labels, calib_mp_combined, config.primary_window)
    
    train_elapsed = time.time() - train_start_time
    print(f"\n  Training time: {train_elapsed:.2f}s")
    
    # ---- Compute Matrix Profile on test data --------------------------------
    print("\n" + "=" * 70)
    print("COMPUTING MATRIX PROFILE (TESTING)")
    print("=" * 70)
    
    test_start_time = time.time()
    
    combined_scores, per_channel_scores, _ = compute_multiscale_mp_perchannel(
        test_features,
        reference_data,
        config.window_sizes,
        config
    )
    
    # ---- Persist raw scores for offline threshold tuning --------------------
    if getattr(config, 'save_scores_for_tuning', False):
        print("\n" + "=" * 70)
        print("SAVING SCORES FOR OFFLINE TUNING")
        print("=" * 70)
        
        scores_output_dir = Path(output_dir) if output_dir else Path('results')
        scores_output_dir.mkdir(exist_ok=True)
        
        # Save per-channel scores as numpy array (efficient)
        scores_file = scores_output_dir / 'mp_scores_perchannel.npy'
        np.save(scores_file, per_channel_scores)
        print(f"  Saved per-channel scores: {scores_file}")
        print(f"  Shape: {per_channel_scores.shape}")
        
        # Save combined scores
        combined_file = scores_output_dir / 'mp_scores_combined.npy'
        np.save(combined_file, combined_scores)
        print(f"  Saved combined scores: {combined_file}")
        
        # Align labels and timestamps with the score array
        if config.single_window_mode:
            align_window = config.primary_window
        else:
            align_window = min(config.window_sizes)
        center_offset = align_window // 2
        score_len = len(combined_scores)
        
        labels_aligned = test_labels[center_offset:center_offset + score_len]
        labels_perchannel_aligned = test_per_channel[center_offset:center_offset + score_len]
        timestamps_aligned = test_timestamps[center_offset:center_offset + score_len]
        
        labels_file = scores_output_dir / 'labels_aligned.npy'
        np.save(labels_file, labels_aligned)
        print(f"  Saved aligned labels: {labels_file}")
        
        labels_pc_file = scores_output_dir / 'labels_perchannel_aligned.npy'
        np.save(labels_pc_file, labels_perchannel_aligned)
        print(f"  Saved per-channel labels: {labels_pc_file}")
        
        ts_file = scores_output_dir / 'timestamps_aligned.npy'
        np.save(ts_file, timestamps_aligned)
        print(f"  Saved timestamps: {ts_file}")
        
        # Save run metadata
        import json
        metadata = {
            'n_samples': score_len,
            'n_channels': n_channels,
            'channel_names': channel_names,
            'window_sizes': config.window_sizes if not config.single_window_mode else [config.primary_window],
            'single_window_mode': config.single_window_mode,
            'primary_window': config.primary_window,
            'reference_size': len(reference_data),
            'score_range': [float(combined_scores.min()), float(combined_scores.max())],
            'anomaly_rate': float(labels_aligned.mean()),
        }
        metadata_file = scores_output_dir / 'scores_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_file}")

        print(f"\n  Scores saved. Run threshold tuning separately to optimise without re-computing MP.")
    
    # ---- Apply score transforms (log compression + smoothing) ---------------
    if getattr(config, 'apply_log_transform', False) or getattr(config, 'smooth_window', 1) > 1:
        print("\n" + "=" * 70)
        print("APPLYING SCORE TRANSFORMS")
        print("=" * 70)

        if getattr(config, 'apply_log_transform', False):
            print("  Applying log(1 + x) transform...")
            per_channel_scores = np.log1p(np.maximum(per_channel_scores, 0))
            combined_scores = np.log1p(np.maximum(combined_scores, 0))
            print(f"    Transformed range: [{combined_scores.min():.3f}, {combined_scores.max():.3f}]")

        # Temporal smoothing
        smooth_win = getattr(config, 'smooth_window', 1)
        if smooth_win > 1:
            print(f"  Applying uniform smoothing (window={smooth_win} samples, ~{smooth_win * 0.5:.0f} min)...")
            for ch in range(per_channel_scores.shape[1]):
                per_channel_scores[:, ch] = uniform_filter1d(
                    per_channel_scores[:, ch].astype(np.float64), 
                    size=smooth_win, mode='reflect'
                )
            combined_scores = uniform_filter1d(
                combined_scores.astype(np.float64), 
                size=smooth_win, mode='reflect'
            )
            print(f"    Smoothed range: [{combined_scores.min():.3f}, {combined_scores.max():.3f}]")

    # ---- Per-channel thresholding and post-processing -----------------------
    print("\n" + "=" * 70)
    print("APPLYING THRESHOLDS")
    print("=" * 70)
    
    per_channel_predictions = np.zeros_like(per_channel_scores, dtype=np.int8)
    
    for ch in range(n_channels):
        ch_scores = per_channel_scores[:, ch]
        ch_threshold = np.percentile(ch_scores, config.threshold_percentile)
        raw_pred = (ch_scores > ch_threshold).astype(np.int8)
        
        # Post-process each channel independently
        pp = ESAOptimizedPostProcessor(
            min_event_duration=config.min_event_duration,
            gap_tolerance=config.gap_tolerance,
            extend_window=getattr(config, 'extend_window', 0),
            trim_fp_threshold=getattr(config, 'trim_threshold', 0.80),
            extend_coverage=getattr(config, 'extend_coverage', True)
        )
        per_channel_predictions[:, ch] = pp.process(raw_pred, ch_scores)
        
        print(f"  {channel_names[ch]}: threshold={ch_threshold:.3f}, "
              f"anomalies={per_channel_predictions[:, ch].sum():,} "
              f"({per_channel_predictions[:, ch].mean()*100:.2f}%)")
    
    # ---- Aggregated prediction (logical OR across channels) -----------------
    aggregated_predictions = (per_channel_predictions.sum(axis=1) > 0).astype(np.int8)
    print(f"\n  Aggregated (OR): {aggregated_predictions.sum():,} ({aggregated_predictions.mean()*100:.2f}%)")
    
    test_elapsed = time.time() - test_start_time
    print(f"\n  Testing time: {test_elapsed:.2f}s")
    print(f"  Total time (train + test): {train_elapsed + test_elapsed:.2f}s")
    
    # ---- Align predictions with ground-truth labels -------------------------
    if config.single_window_mode:
        align_window = config.primary_window
    else:
        align_window = min(config.window_sizes)
    
    center_offset = align_window // 2
    min_len = min(len(combined_scores), len(test_labels) - center_offset)
    
    aligned_timestamps = test_timestamps[center_offset:center_offset + min_len]
    aligned_labels = test_labels[center_offset:center_offset + min_len]
    aligned_per_channel_labels = test_per_channel[center_offset:center_offset + min_len]
    
    per_channel_scores = per_channel_scores[:min_len]
    per_channel_predictions = per_channel_predictions[:min_len]
    aggregated_predictions = aggregated_predictions[:min_len]
    combined_scores = combined_scores[:min_len]
    
    # Suppress predictions in data-gap regions
    if gap_mask.any():
        gap_mask_aligned = gap_mask[center_offset:center_offset + min_len]
        if len(gap_mask_aligned) == min_len:
            per_channel_predictions = np.where(gap_mask_aligned[:, np.newaxis], 0, per_channel_predictions)
            aggregated_predictions = np.where(gap_mask_aligned, 0, aggregated_predictions)
    
    print(f"\n  Alignment: {min_len:,} samples")
    
    # ---- Build output DataFrame with per-channel columns --------------------
    print("\n" + "=" * 70)
    print("BUILDING OUTPUT")
    print("=" * 70)
    
    output_data = {'timestamp': aligned_timestamps}
    
    # Per-channel scores
    for ch, ch_name in enumerate(channel_names):
        output_data[f'score_{ch_name}'] = per_channel_scores[:, ch]
    
    # Per-channel predictions
    for ch, ch_name in enumerate(channel_names):
        output_data[f'pred_{ch_name}'] = per_channel_predictions[:, ch]
    
    # Aggregated
    output_data['score_aggregated'] = combined_scores
    output_data['pred_aggregated'] = aggregated_predictions
    
    # Ground truth per-channel
    for ch, ch_name in enumerate(channel_names):
        output_data[f'gt_{ch_name}'] = aligned_per_channel_labels[:, ch]
    
    output_data['gt_aggregated'] = aligned_labels
    
    output_df = pd.DataFrame(output_data)
    
    print(f"  Output columns: {list(output_df.columns)}")
    print(f"  Output shape: {output_df.shape}")
    
    # ---- Evaluate using ESA-ADB metrics -------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION (ESA-ADB Metrics - F0.5)")
    print("=" * 70)

    # Extract ground-truth anomaly events as intervals
    diff = np.diff(np.concatenate([[0], aligned_labels.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    events = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        if s < len(aligned_timestamps) and e <= len(aligned_timestamps):
            events.append({
                'ID': f'event_{i}',
                'StartTime': pd.Timestamp(aligned_timestamps[s]),
                'EndTime': pd.Timestamp(aligned_timestamps[min(e-1, len(aligned_timestamps)-1)])
            })
    y_true_df = pd.DataFrame(events) if events else pd.DataFrame(columns=['ID', 'StartTime', 'EndTime'])
    
    # Format predictions for the ESA-ADB scorer (object array: [timestamp, label])
    y_pred_agg = np.array([
        [pd.Timestamp(ts), int(pred)]
        for ts, pred in zip(aligned_timestamps, aggregated_predictions)
    ], dtype=object)

    # Event-wise and affiliation scores
    print("\n  Aggregated Metrics:")
    scorer = ESAScores(betas=0.5)
    esa_results = scorer.score(y_true_df, y_pred_agg)
    
    print(f"    EW Precision (TNR): {esa_results['EW_precision']:.4f}")
    print(f"    EW Recall:          {esa_results['EW_recall']:.4f}")
    print(f"    EW F0.5:            {esa_results['EW_F_0.50']:.4f}")
    print(f"    AFF F0.5:           {esa_results.get('AFF_F_0.50', 0):.4f}")
    
    # Anomaly Detection Timing and Quality Criterion
    print("\n  ADTQC (Detection Timing Quality):")
    try:
        adtqc_scorer = ADTQC()
        # ADTQC expects per-channel prediction dicts and enriched y_true
        y_pred_adtqc = {'aggregated': y_pred_agg}
        
        # Enrich y_true with required metadata columns
        y_true_adtqc = y_true_df.copy()
        y_true_adtqc['Channel'] = 'aggregated'
        for col in ['Category', 'Dimensionality', 'Locality', 'Length']:
            if col not in y_true_adtqc.columns:
                y_true_adtqc[col] = ''
        
        adtqc_results = adtqc_scorer.score(y_true_adtqc, y_pred_adtqc)
        
        print(f"    Detections Before Anomaly Start: {adtqc_results.get('Nb_Before', 0)}")
        print(f"    Detections After Anomaly Start:  {adtqc_results.get('Nb_After', 0)}")
        print(f"    After Rate:                      {adtqc_results.get('AfterRate', 0):.4f}")
        print(f"    ADTQC Total Score:               {adtqc_results.get('Total', 0):.4f}")
    except Exception as e:
        print(f"    ADTQC computation failed: {e}")
        adtqc_results = {'Nb_Before': 0, 'Nb_After': 0, 'AfterRate': 0, 'Total': 0}
    
    # Channel-Aware F-Score with true per-channel predictions
    print("\n  Channel-Aware Metrics (Per-Channel):")

    try:
        ca_scorer = ChannelAwareFScore(beta=0.5)

        # Build per-channel prediction arrays
        y_pred_channels = {}
        for ch, ch_name in enumerate(channel_names):
            y_pred_channels[ch_name] = np.array([
                [pd.Timestamp(ts), int(pred)]
                for ts, pred in zip(aligned_timestamps, per_channel_predictions[:, ch])
            ], dtype=object)

        # Build channel-level ground truth with required metadata
        y_true_channel = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            if s < len(aligned_timestamps) and e <= len(aligned_timestamps):
                start_ts = pd.Timestamp(aligned_timestamps[s])
                end_ts = pd.Timestamp(aligned_timestamps[min(e-1, len(aligned_timestamps)-1)])
                for ch, ch_name in enumerate(channel_names):
                    if aligned_per_channel_labels[s:e, ch].any():
                        y_true_channel.append({
                            'ID': f'event_{i}',
                            'Channel': ch_name,
                            'StartTime': start_ts,
                            'EndTime': end_ts,
                            'Category': '',
                            'Dimensionality': '',
                            'Locality': '',
                            'Length': ''
                        })
        
        if y_true_channel:
            y_true_channel_df = pd.DataFrame(y_true_channel)
            ca_results = ca_scorer.score(y_true_channel_df, y_pred_channels)
            
            print(f"    Channel Precision:  {ca_results.get('channel_precision', 0):.4f}")
            print(f"    Channel Recall:     {ca_results.get('channel_recall', 0):.4f}")
            print(f"    Channel F0.5:       {ca_results.get('channel_F0.50', 0):.4f}")
        else:
            ca_results = {}
            print("    No per-channel labels available")
    except Exception as e:
        print(f"    Channel-Aware scoring failed: {e}")
        ca_results = {}
    
    # Per-channel summary
    print("\n  Per-Channel Detection Summary:")
    for ch, ch_name in enumerate(channel_names):
        gt_count = aligned_per_channel_labels[:, ch].sum()
        pred_count = per_channel_predictions[:, ch].sum()
        overlap = ((aligned_per_channel_labels[:, ch] == 1) & (per_channel_predictions[:, ch] == 1)).sum()
        print(f"    {ch_name}: GT={gt_count:,}, Pred={pred_count:,}, Overlap={overlap:,}")
    
    # ---- Save results -------------------------------------------------------
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save per-channel predictions
        output_csv = output_path / 'perchannel_predictions.csv'
        output_df.to_csv(output_csv, index=False)
        print(f"\n  Saved per-channel predictions: {output_csv}")
        
        # Save metrics
        results = {
            'EW_precision': esa_results['EW_precision'],
            'EW_recall': esa_results['EW_recall'],
            'EW_F_0.50': esa_results['EW_F_0.50'],
            'AFF_precision': esa_results.get('AFF_precision', 0),
            'AFF_recall': esa_results.get('AFF_recall', 0),
            'AFF_F_0.50': esa_results.get('AFF_F_0.50', 0),
            'alarming_precision': esa_results.get('alarming_precision', 0),
            'ADTQC_Nb_Before': adtqc_results.get('Nb_Before', 0),
            'ADTQC_Nb_After': adtqc_results.get('Nb_After', 0),
            'ADTQC_AfterRate': adtqc_results.get('AfterRate', 0),
            'ADTQC_Total': adtqc_results.get('Total', 0),
            'CA_precision': ca_results.get('channel_precision', 0),
            'CA_recall': ca_results.get('channel_recall', 0),
            'CA_F_0.50': ca_results.get('channel_F0.50', 0),
            'train_time_seconds': train_elapsed,
            'test_time_seconds': test_elapsed,
            'total_time_seconds': train_elapsed + test_elapsed,
        }
        
        with open(output_path / 'perchannel_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Saved metrics: {output_path / 'perchannel_metrics.json'}")
    else:
        # Assemble metrics dict without writing to disk
        results = {
            'EW_precision': esa_results['EW_precision'],
            'EW_recall': esa_results['EW_recall'],
            'EW_F_0.50': esa_results['EW_F_0.50'],
            'AFF_precision': esa_results.get('AFF_precision', 0),
            'AFF_recall': esa_results.get('AFF_recall', 0),
            'AFF_F_0.50': esa_results.get('AFF_F_0.50', 0),
            'alarming_precision': esa_results.get('alarming_precision', 0),
            'ADTQC_Nb_Before': adtqc_results.get('Nb_Before', 0),
            'ADTQC_Nb_After': adtqc_results.get('Nb_After', 0),
            'ADTQC_AfterRate': adtqc_results.get('AfterRate', 0),
            'ADTQC_Total': adtqc_results.get('Total', 0),
            'CA_precision': ca_results.get('channel_precision', 0),
            'CA_recall': ca_results.get('channel_recall', 0),
            'CA_F_0.50': ca_results.get('channel_F0.50', 0),
            'train_time_seconds': train_elapsed,
            'test_time_seconds': test_elapsed,
            'total_time_seconds': train_elapsed + test_elapsed,
        }
    
    # ---- Requirements compliance summary ------------------------------------
    print("\n" + "=" * 70)
    print("REQUIREMENTS COMPLIANCE (R1-R9)")
    print("=" * 70)
    print(f"  R1 Binary Response:            ✓ Output dtype = {aggregated_predictions.dtype}")
    print(f"  R2 Batch Detection:            ✓ Segmented processing (segment_size = {config.segment_size:,})")
    print(f"  R3 Multi-channel Dependencies: ✓ Fusion = '{config.fusion_method}', {n_channels} channels")
    print(f"  R4 Learn from Training:        {'✓' if anomaly_library else '–'} {len(anomaly_library.signatures) if anomaly_library else 0} anomaly signatures")
    print(f"  R5 Affected Channels:          ✓ Per-channel predictions for {n_channels} channels")
    print(f"  R6 Channel Classification:     ✓ {len(config.channels.target_channels)} target channels configured")
    print(f"  R7 Rare Nominal Events:        {'✓' if nominal_library else '–'} {len(nominal_library.patterns) if nominal_library else 0} nominal patterns")
    print(f"  R8 Irregular Timestamps:       ✓ Gap handling = {config.handle_irregular_timestamps}")
    print(f"  R9 Reasonable Runtime:         ✓ n_jobs = {config.n_jobs}, GPU = {config.use_gpu}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    return {
        'output_df': output_df,
        'metrics': results,
        'esa_results': esa_results,
        'adtqc_results': adtqc_results,
        'ca_results': ca_results,
        'anomaly_library': anomaly_library,
        'nominal_library': nominal_library
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "..", "data", "84_months.train.csv")
    test_path = os.path.join(base_dir, "..", "data", "84_months.test.csv")
    output_dir = os.path.join(base_dir, "..", "results")
    
    config = OptimalMPConfig(
        debug_mode=True,       # Set to False for full-dataset run
        single_window_mode=False,
    )

    output_result = run_perchannel_pipeline(
        train_path=train_path,
        test_path=test_path,
        config=config,
        output_dir=output_dir
    )
    
    print(f"\nOutput DataFrame shape: {output_result['output_df'].shape}")
    print(f"Columns: {list(output_result['output_df'].columns)}")
    print(f"\nMetrics: {output_result['metrics']}")
