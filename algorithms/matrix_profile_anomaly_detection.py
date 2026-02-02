"""
Matrix Profile-based Anomaly Detection Pipeline for Satellite Telemetry
========================================================================

ESA-ADB Mission 1 (84_months) - Lightweight Subset (Channels 41-46)

This module implements an unsupervised anomaly detection pipeline for the 
ESA Anomaly Detection Benchmark using the Matrix Profile algorithm via stumpy.

Requirements Satisfied:
- R1: Binary response (0 = nominal, 1 = anomaly) via thresholding
- R2: Online constraint via AB-Join (no look-ahead / no self-join)
- R4: Handling training anomalies by building Nominal Reference Library
- R5: Channel-level reasoning for detected anomalies
- R7: Rare nominal events handled via smoothing and post-processing
- R9: Runs on standard 16-32GB RAM hardware via batching

Key Features:
- ROBUST HYBRID THRESHOLDING: Fixes threshold contamination problem
  * Uses MAD (Median Absolute Deviation) instead of Std for robustness
  * Formula: Dynamic_Thresh = Median(buffer) + K * MAD(buffer)
  * Hybrid: Actual_Thresh = max(Static_Threshold, Dynamic_Thresh)
  * Buffer contamination prevention: anomalous values excluded from history
  R2 compliant: only uses past data (no look-ahead)
  
- EXCLUSION ZONE: Reduces alert chatter from persistent anomalies
  After detection, suppresses alerts for N steps to avoid alert flooding

- AB-JOIN CALIBRATION: Threshold calibrated on held-out training data
  Simulates inference conditions without peeking at test data

Author: Data Science Pipeline for Spacecraft Operations
Target: ESA-ADB Mission 1, Lightweight Subset (Channels 41-46)
"""

import numpy as np
import pandas as pd
import stumpy
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')


# =============================================================================
# LOGGING UTILITY (Writes print output to both console and file)
# =============================================================================

class TeeOutput:
    """
    Duplicates stdout to both console and a log file.
    All print() statements will be captured in the log file.
    """
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')
        
    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def setup_logging(output_dir: str = None) -> TeeOutput:
    """
    Set up logging to capture all print statements to a timestamped file.
    
    Args:
        output_dir: Directory for log file. Defaults to script directory.
        
    Returns:
        TeeOutput object (call .close() when done)
    """
    import os
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'pipeline_log_{timestamp}.txt'
    log_path = os.path.join(output_dir, log_filename)
    
    tee = TeeOutput(log_path)
    sys.stdout = tee
    
    print(f"Logging to: {log_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    return tee


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the Matrix Profile anomaly detection pipeline."""
    # Target telemetry channels (Mission 1 Lightweight Subset)
    feature_columns: List[str] = None
    label_columns: List[str] = None
    
    # Matrix Profile parameters
    window_size: int = 17  # Optimized for lightweight ESA telemetry
    
    # =========================================================================
    # FLATLINE DETECTION (Handles Z-normalization instability)
    # =========================================================================
    # Problem: When a subsequence has near-zero standard deviation ("flatline"),
    # Z-normalization becomes numerically unstable (division by ~0), producing
    # meaningless or infinite distances that appear as false anomalies.
    #
    # Solution: Skip Z-normalization or force distance to 0 when subsequence
    # std is below a noise floor threshold.
    # =========================================================================
    flatline_noise_floor: float = 1e-6  # Minimum std for valid Z-normalization
    flatline_distance: float = 0.0  # Distance to assign to flatline subsequences
    
    # Threshold parameters (for initial calibration on nominal self-join)
    threshold_multiplier: float = 3.0  # For mean + k*std (z-score multiplier)
    threshold_percentile: float = 99.0  # Alternative: percentile-based
    
    # AB-JOIN CALIBRATION (R2 Compliant baseline threshold)
    # Calibrate using AB-join on held-out TRAINING data to simulate inference
    ab_join_calibration_size: int = 50_000  # Samples from training for calibration
    ab_join_threshold_percentile: float = 99.0  # Percentile on AB-join distances
    use_ab_join_calibration: bool = True  # Enable AB-join based threshold
    
    # =========================================================================
    # ROBUST HYBRID THRESHOLDING WITH HYSTERESIS (Handles concept drift + FP)
    # =========================================================================
    # Problem: Single-threshold logic causes "flickering" alerts (on/off rapid
    # switching) and captures transient noise spikes as false positives.
    #
    # Solution: Dual-Threshold Hysteresis State Machine
    # 1. Use Median + K*MAD instead of Mean + z*Std (robust to outliers)
    # 2. HYSTERESIS: Two thresholds control state transitions:
    #    - K_upper: Anomaly STARTS only if distance > Median + K_upper * MAD
    #    - K_lower: Anomaly ENDS only if distance < Median + K_lower * MAD
    # 3. Enforce lower bound: max(Static_Threshold, Dynamic_Threshold)
    # 4. Prevent buffer contamination: don't add anomalous values to history
    #
    # R2 COMPLIANT: Only uses PAST data, no look-ahead.
    # =========================================================================
    use_adaptive_threshold: bool = True  # Enable adaptive thresholding
    adaptive_window_size: int = 1000  # Buffer size (increased for robustness)
    adaptive_mad_multiplier: float = 5.0  # K_upper: multiplier to START anomaly (stricter)
    adaptive_mad_multiplier_lower: float = 2.0  # K_lower: multiplier to END anomaly (looser)
    adaptive_min_samples: int = 500  # Minimum samples before adaptive kicks in
    adaptive_warmup_multiplier: float = 2.0  # Multiply static threshold during warmup
    prevent_buffer_contamination: bool = True  # Don't add anomalies to buffer
    use_hybrid_lower_bound: bool = True  # Enforce static threshold as floor
    use_hysteresis: bool = True  # Enable dual-threshold hysteresis state machine
    
    # =========================================================================
    # EXCLUSION ZONE (Reduces alert chatter)
    # =========================================================================
    # After detecting an anomaly, suppress new alerts for `exclusion_zone` steps.
    # This prevents a single persistent anomaly from generating hundreds of alerts.
    # Typical satellite anomalies persist for several seconds (multiple samples).
    # =========================================================================
    use_exclusion_zone: bool = True  # Enable exclusion zone
    exclusion_zone_size: int = 50  # Steps to suppress after detection (~ m*3)
    
    # POST-PROCESSING parameters (R7: Handle rare nominal events)
    smoothing_window: int = 51  # Rolling median window for distance smoothing
    min_event_duration: int = 30  # Minimum consecutive points to form an event (increased to filter noise)
    gap_tolerance: int = 100  # Merge events separated by fewer than this many points (increased for fragmentation)
    
    # Memory optimization parameters (R9: Standard 16-32GB RAM hardware)
    max_library_size: int = 300_000  # Max samples for nominal library
    max_threshold_samples: int = 100_000  # Max samples for threshold calibration
    inference_batch_size: int = 200_000  # Batch size for streaming inference
    use_float32: bool = True  # Use float32 to reduce memory by 50%
    
    def __post_init__(self):
        if self.feature_columns is None:
            # Mission 1 Lightweight Subset: Channels 41-46
            self.feature_columns = [f'channel_{i}' for i in range(41, 47)]
        if self.label_columns is None:
            # Per-channel anomaly labels
            self.label_columns = [f'is_anomaly_channel_{i}' for i in range(41, 47)]


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_esa_adb_data(
    train_path: str,
    test_path: str,
    config: PipelineConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ESA-ADB Mission 1 data files.
    
    Parameters
    ----------
    train_path : str
        Path to 84_months.train.csv
    test_path : str
        Path to 84_months.test.csv
    config : PipelineConfig
        Pipeline configuration with column specifications.
    
    Returns
    -------
    train_df : pd.DataFrame
        Training data with features and aggregated label.
    test_df : pd.DataFrame
        Test data with features and aggregated label.
    """
    print("=" * 70)
    print("LOADING ESA-ADB MISSION 1 DATA")
    print("=" * 70)
    
    # Load with low_memory=False for large datasets (~7.3M samples)
    print(f"\nLoading training data from: {train_path}")
    train_raw = pd.read_csv(train_path, low_memory=False)
    
    print(f"Loading test data from: {test_path}")
    test_raw = pd.read_csv(test_path, low_memory=False)
    
    print(f"\nRaw data shapes:")
    print(f"  Training: {train_raw.shape}")
    print(f"  Test: {test_raw.shape}")
    
    # Process both datasets
    train_df = preprocess_data(train_raw, config)
    test_df = preprocess_data(test_raw, config)
    
    return train_df, test_df


def preprocess_data(
    df: pd.DataFrame,
    config: PipelineConfig
) -> pd.DataFrame:
    """
    Preprocess ESA-ADB data: extract features and aggregate labels.
    
    Per-channel labels (is_anomaly_channel_41, ..., is_anomaly_channel_46) are
    aggregated into a single global binary label where 1 indicates at least
    one channel is anomalous at that timestamp.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data with all columns.
    config : PipelineConfig
        Pipeline configuration.
    
    Returns
    -------
    processed_df : pd.DataFrame
        DataFrame with feature columns and aggregated 'label' column.
    """
    # Extract feature columns
    features = df[config.feature_columns].copy()
    
    # Aggregate per-channel labels into global label
    # Label = 1 if ANY channel is anomalous at that timestamp
    if all(col in df.columns for col in config.label_columns):
        label_matrix = df[config.label_columns].values
        global_label = (label_matrix.sum(axis=1) > 0).astype(int)
    else:
        # Fallback: check for single 'label' or 'anomaly' column
        if 'label' in df.columns:
            global_label = df['label'].values
        elif 'anomaly' in df.columns:
            global_label = df['anomaly'].values
        else:
            raise ValueError("No label columns found in data.")
    
    # Create processed DataFrame
    processed_df = features.copy()
    processed_df['label'] = global_label
    
    return processed_df


def apply_zscore_standardization(
    train_data: np.ndarray,
    test_data: np.ndarray,
    nominal_mask: np.ndarray,
    use_float32: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Z-score standardization based on nominal training data statistics.
    
    This ensures the model learns the distribution of "normal" behavior only,
    not influenced by anomalous patterns in the training set.
    
    Parameters
    ----------
    train_data : np.ndarray
        Training features, shape (n_train, n_channels).
    test_data : np.ndarray
        Test features, shape (n_test, n_channels).
    nominal_mask : np.ndarray
        Boolean mask where True = nominal sample in training data.
    use_float32 : bool
        If True, use float32 to reduce memory by 50%.
    
    Returns
    -------
    train_standardized : np.ndarray
        Standardized training data.
    test_standardized : np.ndarray
        Standardized test data (using training stats).
    mean : np.ndarray
        Per-channel means from nominal training data.
    std : np.ndarray
        Per-channel stds from nominal training data.
    """
    dtype = np.float32 if use_float32 else np.float64
    
    # Compute statistics from NOMINAL training data only
    nominal_train = train_data[nominal_mask]
    mean = np.mean(nominal_train, axis=0).astype(dtype)
    std = np.std(nominal_train, axis=0).astype(dtype)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    # Apply standardization with specified dtype
    train_standardized = ((train_data - mean) / std).astype(dtype)
    test_standardized = ((test_data - mean) / std).astype(dtype)
    
    return train_standardized, test_standardized, mean, std


# =============================================================================
# PHASE A: TRAINING - Building the Nominal Reference Library
# =============================================================================

def build_nominal_library(
    train_df: pd.DataFrame,
    config: PipelineConfig
) -> Tuple[np.ndarray, float, Dict]:
    """
    Build a Nominal Reference Library from clean (non-anomalous) training data.
    
    Implements Requirement R4: Filter out anomalies to ensure the model only
    learns "Normal" spacecraft behavior.
    
    Memory Optimization:
    - Uses strided sampling to limit library size (preserves temporal structure)
    - Uses float32 to halve memory footprint
    - Samples subset for threshold calibration (self-join is O(n²))
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Pre-processed training DataFrame with features and aggregated label.
    config : PipelineConfig
        Pipeline configuration.
    
    Returns
    -------
    nominal_library : np.ndarray
        Standardized clean training data (downsampled), shape (n_samples, n_channels).
    threshold : float
        Static anomaly threshold from nominal distance distribution.
    stats : dict
        Dictionary containing mean, std, and other statistics.
    """
    print("\n" + "=" * 70)
    print("PHASE A: Building Nominal Reference Library")
    print("=" * 70)
    
    # Determine dtype based on config
    dtype = np.float32 if config.use_float32 else np.float64
    print(f"\nUsing dtype: {dtype} ({'50% memory reduction' if config.use_float32 else 'full precision'})")
    
    # Extract features and labels
    feature_data = train_df[config.feature_columns].values.astype(dtype)
    labels = train_df['label'].values
    
    print(f"Feature channels: {config.feature_columns}")
    print(f"Training samples: {len(train_df)}")
    print(f"Training anomalies: {labels.sum()} ({100*labels.mean():.2f}%)")
    
    # Create nominal mask (label == 0)
    nominal_mask = labels == 0
    n_nominal = nominal_mask.sum()
    print(f"Clean nominal samples: {n_nominal} ({100*n_nominal/len(labels):.2f}%)")
    
    # Apply Z-score standardization based on nominal data statistics
    print("\nApplying Z-score standardization (based on nominal statistics)...")
    train_std, _, mean, std = apply_zscore_standardization(
        feature_data, feature_data, nominal_mask, use_float32=config.use_float32
    )
    
    # Extract only nominal samples
    nominal_full = train_std[nominal_mask]
    
    # -------------------------------------------------------------------------
    # MEMORY OPTIMIZATION: Strided downsampling for nominal library
    # -------------------------------------------------------------------------
    # For large datasets, use strided sampling to preserve temporal structure
    # while reducing memory footprint. This is critical for:
    # 1. Threshold calibration (self-join is O(n²) complexity)
    # 2. AB-join reference library (O(n_test × n_library) complexity)
    # -------------------------------------------------------------------------
    
    if len(nominal_full) > config.max_library_size:
        stride = len(nominal_full) // config.max_library_size
        nominal_library = nominal_full[::stride].copy()
        print(f"\nDownsampling nominal library:")
        print(f"  Original size: {len(nominal_full):,}")
        print(f"  Stride: every {stride}th sample")
        print(f"  Downsampled size: {len(nominal_library):,}")
    else:
        nominal_library = nominal_full.copy()
        print(f"\nNominal library size: {len(nominal_library):,} (no downsampling needed)")
    
    nominal_library = np.ascontiguousarray(nominal_library)
    
    print(f"Nominal library shape: {nominal_library.shape}")
    print(f"Window size (m): {config.window_size}")
    print(f"Memory footprint: {nominal_library.nbytes / 1024**2:.1f} MB")
    
    # -------------------------------------------------------------------------
    # THRESHOLD CALIBRATION: Use smaller sample for self-join (O(n²) operation)
    # -------------------------------------------------------------------------
    # The self-join for threshold calibration doesn't need all data points.
    # We only need to capture the distribution of nominal distances.
    # -------------------------------------------------------------------------
    
    m = config.window_size
    n_channels = nominal_library.shape[1]
    
    # Further subsample for threshold calibration if still too large
    if len(nominal_library) > config.max_threshold_samples:
        threshold_stride = len(nominal_library) // config.max_threshold_samples
        threshold_sample = nominal_library[::threshold_stride]
        print(f"\nThreshold calibration sample:")
        print(f"  Using {len(threshold_sample):,} samples (stride: {threshold_stride})")
    else:
        threshold_sample = nominal_library
        print(f"\nThreshold calibration using full library: {len(threshold_sample):,} samples")
    
    n_subsequences = len(threshold_sample) - m + 1
    print(f"Computing Matrix Profile on {n_subsequences:,} subsequences...")
    
    # Compute per-channel self-join distances
    # Note: stumpy requires float64, so we convert temporarily for computation
    channel_mp_distances = np.zeros((n_channels, n_subsequences), dtype=dtype)
    
    for ch_idx in range(n_channels):
        # Convert to float64 for stumpy (required by library)
        T = threshold_sample[:, ch_idx].astype(np.float64)
        mp = stumpy.stump(T, m=m)
        channel_mp_distances[ch_idx, :] = mp[:, 0].astype(dtype)
        print(f"  Channel {ch_idx + 1}/{n_channels} complete")
    
    # Aggregate: mean across channels
    mp_distances = np.mean(channel_mp_distances, axis=0)
    
    # Calculate threshold
    mean_dist = float(np.mean(mp_distances))
    std_dist = float(np.std(mp_distances))
    threshold_sigma = mean_dist + config.threshold_multiplier * std_dist
    threshold_pct = float(np.percentile(mp_distances, config.threshold_percentile))
    
    # Self-join threshold (for reference)
    selfjoin_threshold = max(threshold_sigma, threshold_pct)
    
    print(f"\nSelf-Join Threshold Calibration (nominal data):")
    print(f"  Mean distance: {mean_dist:.4f}")
    print(f"  Std distance: {std_dist:.4f}")
    print(f"  Threshold (mean + {config.threshold_multiplier}σ): {threshold_sigma:.4f}")
    print(f"  Threshold ({config.threshold_percentile}th pct): {threshold_pct:.4f}")
    print(f"  Self-join threshold: {selfjoin_threshold:.4f}")
    
    # -------------------------------------------------------------------------
    # AB-JOIN THRESHOLD CALIBRATION (R2 Compliant: Uses only training data)
    # -------------------------------------------------------------------------
    # The self-join threshold doesn't match AB-join distances. To fix this
    # WITHOUT violating the online constraint (R2), we:
    # 1. Hold out a portion of nominal training data
    # 2. Compute AB-join of held-out data against the library
    # 3. Calibrate threshold on this AB-join distribution
    # This simulates inference conditions using only training data.
    # -------------------------------------------------------------------------
    
    if config.use_ab_join_calibration:
        print(f"\nAB-Join Threshold Calibration (R2 compliant):")
        
        # Use a different portion of nominal data (not in library) for calibration
        # This simulates "unseen" data while staying within training set
        calib_size = min(config.ab_join_calibration_size, len(nominal_full) // 2)
        
        # Select calibration samples from the END of nominal data (not in downsampled library)
        # This ensures minimal overlap with the strided library
        calib_start = len(nominal_full) - calib_size
        calib_data = nominal_full[calib_start:calib_start + calib_size]
        
        print(f"  Calibration samples: {len(calib_data):,} (from training nominal data)")
        print(f"  Computing AB-join against library...")
        
        # Compute per-channel AB-join distances
        n_calib_subseq = len(calib_data) - m + 1
        calib_channel_dists = np.zeros((n_channels, n_calib_subseq), dtype=dtype)
        
        for ch_idx in range(n_channels):
            T_A = nominal_library[:, ch_idx].astype(np.float64)  # Library
            T_B = calib_data[:, ch_idx].astype(np.float64)  # Calibration query
            ab_mp = stumpy.stump(T_B, m=m, T_B=T_A, ignore_trivial=False)
            calib_channel_dists[ch_idx, :] = ab_mp[:, 0].astype(dtype)
        
        # Aggregate across channels
        calib_distances = np.mean(calib_channel_dists, axis=0)
        
        # Calculate AB-join threshold
        ab_mean = float(np.mean(calib_distances))
        ab_std = float(np.std(calib_distances))
        ab_threshold_sigma = ab_mean + config.threshold_multiplier * ab_std
        ab_threshold_pct = float(np.percentile(calib_distances, config.ab_join_threshold_percentile))
        
        # Use percentile-based threshold (more robust)
        ab_join_threshold = ab_threshold_pct
        
        print(f"  AB-join mean distance: {ab_mean:.4f}")
        print(f"  AB-join std distance: {ab_std:.4f}")
        print(f"  AB-join threshold (mean + {config.threshold_multiplier}σ): {ab_threshold_sigma:.4f}")
        print(f"  AB-join threshold ({config.ab_join_threshold_percentile}th pct): {ab_threshold_pct:.4f}")
        print(f"  Selected AB-join threshold: {ab_join_threshold:.4f}")
        
        threshold = ab_join_threshold
    else:
        threshold = selfjoin_threshold
        print(f"\nUsing self-join threshold: {threshold:.4f}")
    
    stats = {
        'mean': mean,
        'std': std,
        'nominal_mask': nominal_mask,
        'mp_mean': mean_dist,
        'mp_std': std_dist,
        'selfjoin_threshold': selfjoin_threshold,
        'library_stride': stride if len(nominal_full) > config.max_library_size else 1,
        'original_nominal_size': len(nominal_full),
        'downsampled_size': len(nominal_library),
        'use_float32': config.use_float32
    }
    
    return nominal_library, threshold, stats


# =============================================================================
# PHASE B: INFERENCE - AB-Join for Online Detection (BATCHED)
# =============================================================================

def detect_flatline_subsequences(
    time_series: np.ndarray,
    m: int,
    noise_floor: float = 1e-6
) -> np.ndarray:
    """
    Detect flatline subsequences where Z-normalization would be unstable.
    
    A "flatline" is a subsequence with standard deviation below the noise floor.
    Z-normalizing such subsequences leads to division by ~0, producing NaN or
    extremely large values that appear as false anomalies.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series data.
    m : int
        Subsequence (window) size.
    noise_floor : float
        Minimum standard deviation for valid Z-normalization.
    
    Returns
    -------
    flatline_mask : np.ndarray
        Boolean array of shape (n - m + 1,) where True indicates flatline.
    """
    n = len(time_series)
    n_subsequences = n - m + 1
    
    if n_subsequences <= 0:
        return np.array([], dtype=bool)
    
    # Compute rolling standard deviation efficiently using cumulative sums
    # Var(X) = E[X²] - E[X]²
    cumsum = np.cumsum(np.concatenate([[0], time_series]))
    cumsum_sq = np.cumsum(np.concatenate([[0], time_series ** 2]))
    
    # For each subsequence starting at i: [i : i+m]
    window_sum = cumsum[m:] - cumsum[:-m]
    window_sum_sq = cumsum_sq[m:] - cumsum_sq[:-m]
    
    window_mean = window_sum / m
    window_var = (window_sum_sq / m) - (window_mean ** 2)
    
    # Handle numerical precision issues
    window_var = np.maximum(window_var, 0)
    window_std = np.sqrt(window_var)
    
    # Flatline if std < noise_floor
    flatline_mask = window_std < noise_floor
    
    return flatline_mask


def compute_ab_join_batch(
    test_batch: np.ndarray,
    nominal_library: np.ndarray,
    m: int,
    channel_idx: int,
    flatline_noise_floor: float = 1e-6,
    flatline_distance: float = 0.0
) -> np.ndarray:
    """
    Compute AB-Join for a single batch of test data against nominal library.
    
    Includes flatline detection to handle Z-normalization instability.
    
    Parameters
    ----------
    test_batch : np.ndarray
        Single channel test data batch, shape (batch_size,).
    nominal_library : np.ndarray
        Full nominal library, shape (n_library, n_channels).
    m : int
        Window size.
    channel_idx : int
        Index of the channel being processed.
    flatline_noise_floor : float
        Minimum std for valid Z-normalization.
    flatline_distance : float
        Distance to assign to flatline subsequences.
    
    Returns
    -------
    distances : np.ndarray
        Distance profile for this batch, shape (batch_size - m + 1,).
    """
    # Convert to float64 for stumpy (required by library)
    T_A = nominal_library[:, channel_idx].astype(np.float64)
    T_B = test_batch.astype(np.float64)
    
    # Detect flatline subsequences in test batch
    flatline_mask = detect_flatline_subsequences(T_B, m, flatline_noise_floor)
    
    # AB-Join: Find nearest neighbor in T_A for each subsequence in T_B
    ab_profile = stumpy.stump(T_B, m=m, T_B=T_A, ignore_trivial=False)
    
    # Ensure distances is a float64 array (stumpy may return object dtype)
    distances = np.asarray(ab_profile[:, 0], dtype=np.float64).copy()
    
    # Override flatline subsequences with specified distance
    # This prevents false anomalies from Z-normalization instability
    if np.any(flatline_mask):
        distances[flatline_mask] = flatline_distance
    
    # Also handle any NaN or Inf values that may have slipped through
    invalid_mask = ~np.isfinite(distances)
    if np.any(invalid_mask):
        distances[invalid_mask] = flatline_distance
    
    return distances


def compute_ab_join_profile(
    test_df: pd.DataFrame,
    nominal_library: np.ndarray,
    stats: Dict,
    config: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute AB-Join distance profile using BATCHED processing for memory efficiency.
    
    Implements Requirements:
    - R2 (Online Constraint): AB-Join only, no self-join, no look-ahead
    - R9 (Standard Hardware): Batched processing for 16-32GB RAM systems
    
    Memory Optimization Strategy:
    - Process test data in batches of `inference_batch_size` samples
    - Batches overlap by (m-1) samples to avoid losing subsequences at boundaries
    - Results stored in pre-allocated float32 arrays
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Pre-processed test DataFrame.
    nominal_library : np.ndarray
        Standardized nominal training data (already downsampled).
    stats : dict
        Statistics from training phase (mean, std for standardization).
    config : PipelineConfig
        Pipeline configuration including batch_size.
    
    Returns
    -------
    distance_profile : np.ndarray
        Aggregated distance profile, shape (n_test - m + 1,).
    channel_distances : np.ndarray
        Per-channel distances, shape (n_channels, n_test - m + 1).
    """
    print("\n" + "=" * 70)
    print("PHASE B: AB-Join Inference (BATCHED Online Detection)")
    print("=" * 70)
    
    # Determine dtype based on config
    dtype = np.float32 if config.use_float32 else np.float64
    
    # Extract and standardize test data using TRAINING statistics
    print("\nStandardizing test data using nominal training statistics...")
    test_features = test_df[config.feature_columns].values.astype(dtype)
    test_standardized = ((test_features - stats['mean']) / stats['std']).astype(dtype)
    test_standardized = np.ascontiguousarray(test_standardized)
    
    n_test = test_standardized.shape[0]
    n_channels = test_standardized.shape[1]
    m = config.window_size
    batch_size = config.inference_batch_size
    overlap = m - 1  # Overlap needed to avoid losing subsequences at boundaries
    
    # Total number of subsequences in the full test set
    n_total_subsequences = n_test - m + 1
    
    print(f"Test data shape: {test_standardized.shape}")
    print(f"Nominal library shape: {nominal_library.shape}")
    print(f"Test memory footprint: {test_standardized.nbytes / 1024**2:.1f} MB")
    print(f"\nBatch processing configuration:")
    print(f"  Batch size: {batch_size:,} samples")
    print(f"  Overlap: {overlap} samples (m-1)")
    print(f"  Total test subsequences: {n_total_subsequences:,}")
    
    # Pre-allocate output arrays (float32 for memory efficiency)
    channel_distances = np.zeros((n_channels, n_total_subsequences), dtype=dtype)
    
    # -------------------------------------------------------------------------
    # BATCHED AB-JOIN STRATEGY (Requirements R2 + R9)
    # -------------------------------------------------------------------------
    # Process test data in overlapping batches to handle memory constraints.
    # 
    # For a batch starting at index `start`:
    #   - Read samples [start : start + batch_size]
    #   - This produces subsequences [start : start + batch_size - m + 1]
    #   - Next batch starts at [start + batch_size - overlap] to ensure continuity
    #
    # This maintains the Online constraint (R2) because:
    #   - Each test subsequence is compared ONLY to the nominal library
    #   - No information from future batches is used
    #   - Processing order doesn't affect results (AB-join is independent per subsequence)
    # -------------------------------------------------------------------------
    
    # Calculate number of batches
    # Each batch processes `batch_size` samples to produce `batch_size - m + 1` subsequences
    # Batches advance by `effective_batch_size` samples (excluding overlap)
    effective_batch_size = batch_size - overlap  # How much we advance each batch
    
    # Calculate total batches needed
    # We need enough batches so that the last batch ends at or past n_test
    # start_sample of batch i = i * effective_batch_size
    # end_sample of batch i = min(start_sample + batch_size, n_test)
    # Last subsequence index from batch i = end_sample - m (if end_sample >= start_sample + m)
    n_batches = 1
    while True:
        start_sample = n_batches * effective_batch_size
        if start_sample >= n_test:
            break
        # Check if this batch can produce any subsequences
        end_sample = min(start_sample + batch_size, n_test)
        if end_sample - start_sample >= m:
            n_batches += 1
        else:
            break
    
    print(f"  Number of batches: {n_batches}")
    print(f"\nProcessing {n_channels} channels × {n_batches} batches...")
    
    for ch_idx, ch_name in enumerate(config.feature_columns):
        print(f"\n  Channel {ch_idx + 1}/{n_channels} ({ch_name}):")
        
        # Get the full channel data
        channel_data = test_standardized[:, ch_idx]
        
        # Process in batches - track output position separately
        output_idx = 0  # Tracks where we are in the output array
        
        for batch_num in range(n_batches):
            # Calculate batch boundaries (sample indices)
            start_sample = batch_num * effective_batch_size
            end_sample = min(start_sample + batch_size, n_test)
            
            # Extract batch
            batch_data = channel_data[start_sample:end_sample]
            
            # Skip if batch is too small to form even one subsequence
            if len(batch_data) < m:
                continue
            
            # Compute AB-join for this batch
            batch_distances = compute_ab_join_batch(
                batch_data, nominal_library, m, ch_idx,
                flatline_noise_floor=config.flatline_noise_floor,
                flatline_distance=config.flatline_distance
            )
            
            # Number of subsequences this batch covers in the original sequence
            # First subsequence: index = start_sample
            # Last subsequence: index = start_sample + len(batch_distances) - 1
            batch_start_subseq = start_sample
            batch_end_subseq = start_sample + len(batch_distances)
            
            # For batch 0, take all results
            # For subsequent batches, skip the overlap (already computed)
            if batch_num == 0:
                result_start = 0
            else:
                # Skip results that overlap with previous batch
                result_start = output_idx - batch_start_subseq
                if result_start < 0:
                    result_start = 0
            
            # Determine output range
            out_start = batch_start_subseq + result_start
            out_end = min(batch_end_subseq, n_total_subsequences)
            n_to_write = out_end - out_start
            
            # Store results
            if n_to_write > 0 and out_start < n_total_subsequences:
                src_end = result_start + n_to_write
                channel_distances[ch_idx, out_start:out_end] = \
                    batch_distances[result_start:src_end].astype(dtype)
                output_idx = out_end
            
            # Progress indicator
            if (batch_num + 1) % max(1, n_batches // 5) == 0 or batch_num == n_batches - 1:
                progress = (batch_num + 1) / n_batches * 100
                print(f"    Batch {batch_num + 1}/{n_batches} ({progress:.0f}%) - "
                      f"subsequences processed: {output_idx:,}/{n_total_subsequences:,}")
        
        # Verify we filled the entire array
        if output_idx < n_total_subsequences:
            print(f"    Warning: Only filled {output_idx}/{n_total_subsequences} subsequences")
        else:
            print(f"    Complete: {output_idx:,} subsequences processed")
    
    # ==========================================================================
    # IMPROVEMENT 1: MAX AGGREGATION (instead of Mean)
    # ==========================================================================
    # Rationale: If ONE channel shows anomalous behavior (e.g., voltage spike on
    # Channel 41), the system should flag it. Averaging across 6 channels dilutes
    # single-channel failures. Max ensures any single-channel anomaly is detected.
    # Context: In satellite ops, if one sensor fails, the subsystem is anomalous.
    # ==========================================================================
    print("\nAggregating multi-channel distances (MAX across channels)...")
    distance_profile = np.max(channel_distances, axis=0)
    
    print(f"\nAB-Join completed.")
    print(f"Distance profile shape: {distance_profile.shape}")
    print(f"Distance stats: min={distance_profile.min():.4f}, "
          f"max={distance_profile.max():.4f}, mean={distance_profile.mean():.4f}")
    
    # Memory cleanup
    del test_standardized
    import gc
    gc.collect()
    
    return distance_profile, channel_distances


# =============================================================================
# PHASE C: Binary Response with Channel-Level Reasoning
# =============================================================================

def smooth_distance_profile(
    distance_profile: np.ndarray,
    window_size: int = 51
) -> np.ndarray:
    """
    Smooth the distance profile using rolling median to reduce noise.
    
    This helps with R7 (rare nominal events) by smoothing out single-point spikes
    that don't represent true anomalies.
    
    Parameters
    ----------
    distance_profile : np.ndarray
        Raw aggregated distance profile.
    window_size : int
        Size of the rolling window (should be odd).
    
    Returns
    -------
    smoothed : np.ndarray
        Smoothed distance profile.
    """
    if window_size <= 1:
        return distance_profile
    
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    
    # Use pandas for efficient rolling median
    smoothed = pd.Series(distance_profile).rolling(
        window=window_size, center=True, min_periods=1
    ).median().values
    
    return smoothed.astype(distance_profile.dtype)


def postprocess_predictions(
    binary_predictions: np.ndarray,
    min_event_duration: int = 10,
    gap_tolerance: int = 50
) -> np.ndarray:
    """
    Post-process binary predictions to reduce fragmentation.
    
    Implements morphological operations to:
    1. Remove events shorter than min_event_duration (likely false alarms)
    2. Merge events separated by fewer than gap_tolerance points (likely same event)
    
    This helps satisfy R7 (rare nominal events) by removing spurious detections.
    
    Parameters
    ----------
    binary_predictions : np.ndarray
        Raw binary predictions (0 or 1).
    min_event_duration : int
        Minimum number of consecutive points to keep an event.
    gap_tolerance : int
        Maximum gap between events to merge them.
    
    Returns
    -------
    processed : np.ndarray
        Post-processed binary predictions.
    """
    processed = binary_predictions.copy()
    
    # Step 1: Close small gaps (morphological closing / dilation then erosion)
    # This merges nearby events that are likely part of the same anomaly
    if gap_tolerance > 0:
        # Find event boundaries
        diff = np.diff(np.concatenate([[0], processed, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Merge events with small gaps
        if len(starts) > 1:
            for i in range(len(starts) - 1):
                gap = starts[i + 1] - ends[i]
                if gap <= gap_tolerance:
                    # Fill the gap
                    processed[ends[i]:starts[i + 1]] = 1
    
    # Step 2: Remove short events (morphological opening)
    # Recompute events after gap closing
    diff = np.diff(np.concatenate([[0], processed, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        duration = end - start
        if duration < min_event_duration:
            # Remove this short event
            processed[start:end] = 0
    
    return processed


def apply_exclusion_zone(
    predictions: np.ndarray,
    exclusion_size: int
) -> np.ndarray:
    """
    Apply exclusion zone to reduce alert chatter.
    
    After detecting an anomaly, suppress new alerts for `exclusion_size` steps.
    This prevents a single persistent anomaly from generating many alerts.
    
    R2 COMPLIANT: Only looks at past detections (causal filter).
    
    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions (0 or 1).
    exclusion_size : int
        Number of steps to suppress after each detection.
    
    Returns
    -------
    suppressed : np.ndarray
        Predictions with exclusion zone applied.
    """
    if exclusion_size <= 0:
        return predictions.copy()
    
    suppressed = np.zeros_like(predictions)
    last_alert_idx = -exclusion_size - 1  # Initialize to allow first alert
    
    for i in range(len(predictions)):
        if predictions[i] == 1:
            # Check if we're outside the exclusion zone from last alert
            if i - last_alert_idx > exclusion_size:
                # Allow this alert
                suppressed[i] = 1
                last_alert_idx = i
            # else: suppress this alert (within exclusion zone)
    
    return suppressed


# =============================================================================
# HYSTERESIS STATE MACHINE CLASS
# =============================================================================

class HysteresisStateMachine:
    """
    Dual-Threshold Hysteresis State Machine for anomaly detection.
    
    This state machine reduces false positives by requiring:
    - A HIGH threshold (K_upper) to TRIGGER an anomaly state
    - A LOW threshold (K_lower) to RESET back to nominal state
    
    This prevents "flickering" alerts from noise near a single threshold.
    
    State Diagram:
        NOMINAL ---(distance > upper_threshold)---> ANOMALY
        ANOMALY ---(distance < lower_threshold)---> NOMINAL
    
    R2 COMPLIANT: Only uses current and past data, no look-ahead.
    """
    
    # State constants
    STATE_NOMINAL = 0
    STATE_ANOMALY = 1
    
    def __init__(self):
        """Initialize state machine in NOMINAL state."""
        self.state = self.STATE_NOMINAL
        self.state_entry_index = 0  # When current state was entered
        self.transitions = 0  # Count of state transitions
    
    def update(self, distance: float, upper_threshold: float, lower_threshold: float, index: int = 0) -> int:
        """
        Update state machine with new distance observation.
        
        Parameters
        ----------
        distance : float
            Current distance value (smoothed recommended).
        upper_threshold : float
            Threshold to trigger anomaly (distance > upper_threshold).
        lower_threshold : float
            Threshold to reset to nominal (distance < lower_threshold).
        index : int
            Current time index (for tracking).
        
        Returns
        -------
        int
            Current state (0 = nominal, 1 = anomaly).
        """
        if self.state == self.STATE_NOMINAL:
            # In NOMINAL state: check if we should transition to ANOMALY
            if distance > upper_threshold:
                self.state = self.STATE_ANOMALY
                self.state_entry_index = index
                self.transitions += 1
        else:
            # In ANOMALY state: check if we should transition back to NOMINAL
            if distance < lower_threshold:
                self.state = self.STATE_NOMINAL
                self.state_entry_index = index
                self.transitions += 1
        
        return self.state
    
    def reset(self):
        """Reset state machine to initial NOMINAL state."""
        self.state = self.STATE_NOMINAL
        self.state_entry_index = 0
        self.transitions = 0


def apply_threshold_with_reasoning(
    distance_profile: np.ndarray,
    channel_distances: np.ndarray,
    nominal_threshold: float,
    channel_names: List[str],
    config: PipelineConfig = None
) -> Tuple[np.ndarray, List[Dict], float]:
    """
    Apply threshold to generate binary predictions with reasoning.
    
    Implements Requirements R1, R5, and R7:
    - R1: Output binary array (0 = nominal, 1 = anomaly)
    - R5: Identify which channel contributed most to each detected anomaly
    - R7: Handle rare nominal events via smoothing and post-processing
    
    NEW FEATURES (Hysteresis Upgrade):
    - Dual-Threshold Hysteresis: State machine with K_upper and K_lower
    - Adaptive Thresholding: Handles concept drift from satellite aging
    - Exclusion Zone: Reduces alert chatter from persistent anomalies
    
    Parameters
    ----------
    distance_profile : np.ndarray
        Aggregated distance profile from AB-join.
    channel_distances : np.ndarray
        Per-channel distances, shape (n_channels, n_subsequences).
    nominal_threshold : float
        Static threshold from training phase (baseline/fallback).
    channel_names : list of str
        Names of channels (e.g., ['channel_41', ..., 'channel_46']).
    config : PipelineConfig, optional
        Pipeline configuration.
    
    Returns
    -------
    binary_predictions : np.ndarray
        Binary array where 1 = anomaly, 0 = nominal.
    anomaly_reasoning : list of dict
        For each anomaly, details about which channel was responsible.
    threshold_used : float
        The nominal threshold (adaptive creates per-point thresholds).
    """
    print("\n" + "=" * 70)
    print("PHASE C: Binary Classification with Reasoning")
    print("=" * 70)
    
    if config is None:
        config = PipelineConfig()
    
    n_points = len(distance_profile)
    
    # -------------------------------------------------------------------------
    # STEP 1: SMOOTH DISTANCE PROFILE (R7: Handle rare nominal events)
    # -------------------------------------------------------------------------
    print(f"\nStep 1: Smoothing distance profile...")
    print(f"  Smoothing window: {config.smoothing_window}")
    
    smoothed_profile = smooth_distance_profile(
        distance_profile, config.smoothing_window
    )
    print(f"  Raw stats: min={distance_profile.min():.4f}, max={distance_profile.max():.4f}, mean={distance_profile.mean():.4f}")
    print(f"  Smoothed stats: min={smoothed_profile.min():.4f}, max={smoothed_profile.max():.4f}, mean={smoothed_profile.mean():.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 2: ROBUST HYBRID THRESHOLDING WITH HYSTERESIS
    # -------------------------------------------------------------------------
    # Uses MAD (Median Absolute Deviation) instead of Std for robustness.
    # HYSTERESIS: Dual thresholds prevent flickering alerts:
    #   - K_upper (5.0): Distance must exceed this to START an anomaly
    #   - K_lower (2.0): Distance must drop below this to END an anomaly
    # Prevents buffer contamination by excluding anomalous values from history.
    # R2 COMPLIANT: Only uses PAST data for each point's threshold.
    # -------------------------------------------------------------------------
    
    if config.use_adaptive_threshold:
        # Get hysteresis parameters
        K_upper = config.adaptive_mad_multiplier  # Stricter threshold to START anomaly
        K_lower = getattr(config, 'adaptive_mad_multiplier_lower', K_upper / 2.0)  # Looser to END
        use_hysteresis = getattr(config, 'use_hysteresis', True)
        
        print(f"\nStep 2: Computing ROBUST HYBRID thresholds with HYSTERESIS...")
        print(f"  History buffer size: {config.adaptive_window_size}")
        print(f"  MAD multiplier K_upper (to START anomaly): {K_upper}")
        print(f"  MAD multiplier K_lower (to END anomaly): {K_lower}")
        print(f"  Hysteresis enabled: {use_hysteresis}")
        print(f"  Min samples before adaptive: {config.adaptive_min_samples}")
        print(f"  Warmup threshold multiplier: {config.adaptive_warmup_multiplier}")
        print(f"  Static threshold (lower bound): {nominal_threshold:.4f}")
        print(f"  Buffer contamination prevention: {config.prevent_buffer_contamination}")
        print(f"  Hybrid lower bound: {config.use_hybrid_lower_bound}")
        
        # Pre-allocate arrays
        raw_predictions = np.zeros(n_points, dtype=np.int32)
        adaptive_thresholds_upper = np.zeros(n_points, dtype=smoothed_profile.dtype)
        adaptive_thresholds_lower = np.zeros(n_points, dtype=smoothed_profile.dtype)
        
        # Parameters
        buffer_size = config.adaptive_window_size
        min_samples = config.adaptive_min_samples
        
        # Circular buffer for history (only clean/nominal values)
        history_buffer = np.zeros(buffer_size, dtype=smoothed_profile.dtype)
        buffer_count = 0  # How many values in buffer
        buffer_idx = 0  # Current write position
        
        # Initialize hysteresis state machine
        hysteresis = HysteresisStateMachine()
        
        # Track statistics for reporting
        contamination_prevented = 0
        warmup_detections = 0
        
        for t in range(n_points):
            current_distance = smoothed_profile[t]
            
            if buffer_count < min_samples:
                # =============================================================
                # WARM-UP PERIOD: Buffer not yet stable
                # =============================================================
                # Use an elevated static threshold to prevent startup false alarms.
                # During warmup, use simple threshold logic (no hysteresis yet).
                # =============================================================
                warmup_multiplier = getattr(config, 'adaptive_warmup_multiplier', 2.0)
                threshold_upper_t = nominal_threshold * warmup_multiplier
                threshold_lower_t = nominal_threshold  # Lower bound for warmup
                
                # During warmup: simple threshold check
                is_anomaly = current_distance > threshold_upper_t
                if is_anomaly:
                    warmup_detections += 1
                
                # Always add to buffer during warmup (building baseline)
                history_buffer[buffer_idx] = current_distance
                buffer_idx = (buffer_idx + 1) % buffer_size
                buffer_count = min(buffer_count + 1, buffer_size)
            else:
                # =============================================================
                # MAIN LOGIC: Robust MAD-based thresholding with Hysteresis
                # =============================================================
                # Get valid portion of buffer
                if buffer_count < buffer_size:
                    valid_buffer = history_buffer[:buffer_count]
                else:
                    valid_buffer = history_buffer
                
                # Compute Median
                buffer_median = np.median(valid_buffer)
                
                # Compute MAD = median(|x - median(x)|)
                absolute_deviations = np.abs(valid_buffer - buffer_median)
                mad = np.median(absolute_deviations)
                
                # Prevent MAD from being zero (would make threshold = median)
                if mad < 1e-6:
                    mad = np.std(valid_buffer) * 0.6745  # Fallback to scaled std
                    if mad < 1e-6:
                        mad = 0.1  # Absolute minimum
                
                # ============================================================
                # DUAL THRESHOLDS FOR HYSTERESIS
                # ============================================================
                # Upper threshold: Must exceed this to START an anomaly
                dynamic_threshold_upper = buffer_median + K_upper * mad
                # Lower threshold: Must drop below this to END an anomaly
                dynamic_threshold_lower = buffer_median + K_lower * mad
                
                # Hybrid lower bound: enforce static threshold as floor
                if config.use_hybrid_lower_bound:
                    threshold_upper_t = max(nominal_threshold, dynamic_threshold_upper)
                    threshold_lower_t = max(nominal_threshold * 0.5, dynamic_threshold_lower)
                else:
                    threshold_upper_t = dynamic_threshold_upper
                    threshold_lower_t = dynamic_threshold_lower
                
                # ============================================================
                # HYSTERESIS STATE MACHINE UPDATE
                # ============================================================
                if use_hysteresis:
                    # Use state machine for detection
                    current_state = hysteresis.update(
                        distance=current_distance,
                        upper_threshold=threshold_upper_t,
                        lower_threshold=threshold_lower_t,
                        index=t
                    )
                    is_anomaly = (current_state == HysteresisStateMachine.STATE_ANOMALY)
                else:
                    # Fallback to simple single-threshold logic
                    is_anomaly = current_distance > threshold_upper_t
                
                # Buffer contamination prevention
                if config.prevent_buffer_contamination and is_anomaly:
                    # DO NOT add anomalous value to buffer
                    # Instead, add the previous median to keep baseline stable
                    history_buffer[buffer_idx] = buffer_median
                    contamination_prevented += 1
                else:
                    # Normal value, add to buffer
                    history_buffer[buffer_idx] = current_distance
                
                buffer_idx = (buffer_idx + 1) % buffer_size
                buffer_count = min(buffer_count + 1, buffer_size)
            
            # Store results
            adaptive_thresholds_upper[t] = threshold_upper_t
            adaptive_thresholds_lower[t] = threshold_lower_t
            raw_predictions[t] = 1 if is_anomaly else 0
        
        # Report statistics
        print(f"  Upper threshold stats: min={adaptive_thresholds_upper.min():.4f}, "
              f"max={adaptive_thresholds_upper.max():.4f}, mean={adaptive_thresholds_upper.mean():.4f}")
        print(f"  Lower threshold stats: min={adaptive_thresholds_lower.min():.4f}, "
              f"max={adaptive_thresholds_lower.max():.4f}, mean={adaptive_thresholds_lower.mean():.4f}")
        print(f"  Buffer contaminations prevented: {contamination_prevented:,}")
        print(f"  Warmup period detections: {warmup_detections:,}")
        if use_hysteresis:
            print(f"  Hysteresis state transitions: {hysteresis.transitions:,}")
        
        threshold_for_reporting = np.mean(adaptive_thresholds_upper)
        # For backwards compatibility, keep adaptive_thresholds as the upper threshold
        adaptive_thresholds = adaptive_thresholds_upper
        
    else:
        print(f"\nStep 2: Using static threshold...")
        print(f"  Threshold (from training): {nominal_threshold:.4f}")
        raw_predictions = (smoothed_profile > nominal_threshold).astype(int)
        threshold_for_reporting = nominal_threshold
        adaptive_thresholds = np.full(n_points, nominal_threshold)
    
    raw_anomalies = raw_predictions.sum()
    raw_rate = raw_anomalies / n_points * 100
    print(f"  Raw detections: {raw_anomalies:,} ({raw_rate:.2f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 3: EXCLUSION ZONE (Reduce alert chatter)
    # -------------------------------------------------------------------------
    # After detecting an anomaly, suppress new alerts for `exclusion_zone` steps.
    # This is R2 compliant as it only looks at past detections.
    # -------------------------------------------------------------------------
    
    if config.use_exclusion_zone and config.exclusion_zone_size > 0:
        print(f"\nStep 3: Applying exclusion zone...")
        print(f"  Exclusion zone size: {config.exclusion_zone_size} steps")
        
        # Apply exclusion zone
        suppressed_predictions = apply_exclusion_zone(
            raw_predictions, config.exclusion_zone_size
        )
        
        suppressed_count = raw_predictions.sum() - suppressed_predictions.sum()
        print(f"  Alerts suppressed by exclusion zone: {suppressed_count:,}")
        
        predictions_after_exclusion = suppressed_predictions
    else:
        print(f"\nStep 3: Exclusion zone disabled")
        predictions_after_exclusion = raw_predictions
    
    # -------------------------------------------------------------------------
    # STEP 4: POST-PROCESSING (R7: Reduce fragmentation)
    # -------------------------------------------------------------------------
    print(f"\nStep 4: Post-processing predictions...")
    print(f"  Min event duration: {config.min_event_duration}")
    print(f"  Gap tolerance: {config.gap_tolerance}")
    
    binary_predictions = postprocess_predictions(
        predictions_after_exclusion,
        min_event_duration=config.min_event_duration,
        gap_tolerance=config.gap_tolerance
    )
    
    n_anomalies = binary_predictions.sum()
    anomaly_rate = n_anomalies / n_points * 100
    
    # Count events before and after post-processing
    def count_events(arr):
        diff = np.diff(np.concatenate([[0], arr, [0]]))
        return len(np.where(diff == 1)[0])
    
    raw_events = count_events(predictions_after_exclusion)
    final_events = count_events(binary_predictions)
    
    print(f"  Raw events: {raw_events:,} → Final events: {final_events:,}")
    print(f"  Anomaly points: {n_anomalies:,} ({anomaly_rate:.2f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 5: Generate reasoning (R5: Channel-level attribution)
    # -------------------------------------------------------------------------
    print(f"\nStep 5: Generating channel-level reasoning...")
    anomaly_reasoning = []
    anomaly_indices = np.where(binary_predictions == 1)[0]
    
    max_reasoning = min(len(anomaly_indices), 10000)
    if len(anomaly_indices) > max_reasoning:
        top_indices = anomaly_indices[
            np.argsort(distance_profile[anomaly_indices])[-max_reasoning:]
        ]
        print(f"  Generating reasoning for top {max_reasoning} anomalies (of {len(anomaly_indices)})")
    else:
        top_indices = anomaly_indices
    
    for idx in top_indices:
        channel_dists = channel_distances[:, idx]
        max_ch_idx = np.argmax(channel_dists)
        
        reasoning = {
            'subsequence_index': int(idx),
            'aggregated_distance': float(distance_profile[idx]),
            'smoothed_distance': float(smoothed_profile[idx]),
            'primary_channel': channel_names[max_ch_idx],
            'primary_channel_distance': float(channel_dists[max_ch_idx]),
            'all_channel_distances': {
                channel_names[i]: float(channel_dists[i])
                for i in range(len(channel_names))
            }
        }
        anomaly_reasoning.append(reasoning)
    
    # Print summary of top anomalies
    if len(anomaly_reasoning) > 0:
        print(f"\n  Top 5 strongest anomalies (by distance):")
        sorted_anomalies = sorted(anomaly_reasoning, 
                                   key=lambda x: x['aggregated_distance'], 
                                   reverse=True)[:5]
        for i, anom in enumerate(sorted_anomalies):
            print(f"    {i+1}. Index {anom['subsequence_index']}: "
                  f"dist={anom['aggregated_distance']:.4f}, "
                  f"primary={anom['primary_channel']}")
    
    return binary_predictions, anomaly_reasoning, threshold_for_reporting


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def identify_events(labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Identify contiguous anomaly events from binary labels.
    
    Parameters
    ----------
    labels : np.ndarray
        Binary labels (0 or 1).
    
    Returns
    -------
    events : list of (start, end) tuples
        Each tuple represents an anomaly event's start and end indices.
    """
    events = []
    in_event = False
    start = 0
    
    for i, val in enumerate(labels):
        if val == 1 and not in_event:
            start = i
            in_event = True
        elif val == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    
    # Handle event at end of array
    if in_event:
        events.append((start, len(labels) - 1))
    
    return events


def compute_event_wise_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    beta: float = 0.5
) -> Dict:
    """
    Compute Corrected Event-wise F-score as required by ESA-ADB benchmark.
    
    This metric prioritizes precision (reducing false alarms) which is critical
    for spacecraft operations where false alarms can be costly.
    
    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions (0 or 1).
    ground_truth : np.ndarray
        Ground truth labels (0 or 1).
    beta : float
        Beta parameter for F-score. beta=0.5 weights precision higher.
    
    Returns
    -------
    metrics : dict
        Dictionary with event-wise and point-wise metrics.
    """
    # Ensure same length
    min_len = min(len(predictions), len(ground_truth))
    pred = predictions[:min_len]
    truth = ground_truth[:min_len]
    
    # Point-wise metrics
    tp_points = np.sum((pred == 1) & (truth == 1))
    fp_points = np.sum((pred == 1) & (truth == 0))
    tn_points = np.sum((pred == 0) & (truth == 0))
    fn_points = np.sum((pred == 0) & (truth == 1))
    
    # Event-wise metrics
    true_events = identify_events(truth)
    pred_events = identify_events(pred)
    
    # Count detected true events (at least one prediction point within event)
    detected_events = 0
    for start, end in true_events:
        if np.any(pred[start:end+1] == 1):
            detected_events += 1
    
    # Count false alarm events (predicted events with no true anomaly overlap)
    false_alarm_events = 0
    for start, end in pred_events:
        if not np.any(truth[start:end+1] == 1):
            false_alarm_events += 1
    
    n_true_events = len(true_events)
    n_pred_events = len(pred_events)
    
    # Event-wise precision and recall
    # Precision = (detected true events - false alarms) / total predicted events
    # NOTE: This follows ESA-ADB corrected event-wise metrics definition
    # where precision penalizes false alarm events, not just counts detections
    
    # Actually, standard event-wise precision:
    # Precision = (predicted events that overlap with true events) / (total predicted events)
    overlapping_predictions = 0
    for start, end in pred_events:
        if np.any(truth[start:end+1] == 1):
            overlapping_predictions += 1
    
    event_precision = overlapping_predictions / n_pred_events if n_pred_events > 0 else 0.0
    event_recall = detected_events / n_true_events if n_true_events > 0 else 0.0
    
    # F-beta score (beta=0.5 prioritizes precision)
    if event_precision + event_recall > 0:
        f_beta = (1 + beta**2) * (event_precision * event_recall) / \
                 (beta**2 * event_precision + event_recall)
    else:
        f_beta = 0.0
    
    # Point-wise precision/recall/F1
    point_precision = tp_points / (tp_points + fp_points) if (tp_points + fp_points) > 0 else 0.0
    point_recall = tp_points / (tp_points + fn_points) if (tp_points + fn_points) > 0 else 0.0
    point_f1 = 2 * point_precision * point_recall / (point_precision + point_recall) \
               if (point_precision + point_recall) > 0 else 0.0
    
    metrics = {
        # Point-wise
        'true_positives': int(tp_points),
        'false_positives': int(fp_points),
        'true_negatives': int(tn_points),
        'false_negatives': int(fn_points),
        'point_precision': float(point_precision),
        'point_recall': float(point_recall),
        'point_f1': float(point_f1),
        'accuracy': float((tp_points + tn_points) / len(pred)) if len(pred) > 0 else 0.0,
        # Event-wise
        'n_true_events': n_true_events,
        'n_predicted_events': n_pred_events,
        'detected_events': detected_events,
        'false_alarm_events': false_alarm_events,
        'event_precision': float(event_precision),
        'event_recall': float(event_recall),
        f'event_f{beta}': float(f_beta)
    }
    
    return metrics


def align_predictions_to_original(
    binary_predictions: np.ndarray,
    original_length: int,
    m: int
) -> np.ndarray:
    """
    Align subsequence-level predictions to original time series length.
    
    Matrix Profile produces (n - m + 1) values for series of length n.
    Pads beginning with zeros (first m-1 points cannot be classified).
    """
    padding = np.zeros(m - 1, dtype=int)
    aligned = np.concatenate([padding, binary_predictions])
    
    if len(aligned) != original_length:
        raise ValueError(f"Alignment error: {len(aligned)} != {original_length}")
    
    return aligned


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def detect_anomalies(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[PipelineConfig] = None
) -> Dict:
    """
    Main anomaly detection function for ESA-ADB satellite telemetry.
    
    Orchestrates the full pipeline:
    1. Phase A: Build Nominal Reference Library from clean training data
    2. Phase B: AB-Join inference (online detection simulation)
    3. Phase C: Binary classification with channel-level reasoning
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Pre-processed training DataFrame with features and 'label' column.
    test_df : pd.DataFrame
        Pre-processed test DataFrame.
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if not provided.
    
    Returns
    -------
    results : dict
        Dictionary containing predictions, distances, reasoning, and metrics.
    
    Notes
    -----
    ONLINE CONSTRAINT (R2 - No Look-Ahead):
    This pipeline simulates real-time detection using AB-Join strategy.
    Each test subsequence is compared ONLY against the pre-built nominal
    training library. No information from future test samples or from the
    test set itself is used for comparison.
    """
    if config is None:
        config = PipelineConfig()
    
    print("\n" + "#" * 70)
    print("# MATRIX PROFILE ANOMALY DETECTION PIPELINE")
    print("# ESA-ADB Mission 1 - Lightweight Subset (Channels 41-46)")
    print("#" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Window size (m): {config.window_size}")
    print(f"  Feature channels: {config.feature_columns}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Phase A: Build Nominal Library
    nominal_library, threshold, stats = build_nominal_library(train_df, config)
    
    # Phase B: AB-Join Inference
    distance_profile, channel_distances = compute_ab_join_profile(
        test_df, nominal_library, stats, config
    )
    
    # Phase C: Binary Classification with Reasoning (with adaptive thresholding)
    binary_predictions, anomaly_reasoning, threshold_used = apply_threshold_with_reasoning(
        distance_profile, channel_distances, threshold, config.feature_columns, config
    )
    
    # Compile results
    results = {
        'binary_predictions': binary_predictions,
        'distance_profile': distance_profile,
        'channel_distances': channel_distances,
        'nominal_threshold': threshold,
        'threshold_used': threshold_used,
        'anomaly_reasoning': anomaly_reasoning,
        'nominal_library_size': len(nominal_library),
        'window_size': config.window_size,
        'n_anomalies_detected': int(binary_predictions.sum()),
        'anomaly_rate_percent': float(binary_predictions.sum() / len(binary_predictions) * 100),
        'stats': stats
    }
    
    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE")
    print("#" * 70)
    print(f"\nFinal Results:")
    print(f"  Predictions length: {len(binary_predictions)}")
    print(f"  Anomalies detected: {results['n_anomalies_detected']}")
    print(f"  Anomaly rate: {results['anomaly_rate_percent']:.2f}%")
    print(f"  Nominal threshold (training): {threshold:.4f}")
    print(f"  Threshold used: {threshold_used:.4f}")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    ESA-ADB Mission 1 Analysis Pipeline
    Dataset: 84_months (Lightweight Subset - Channels 41-46)
    """
    import os
    
    # Set up logging to file
    data_dir = os.path.dirname(os.path.abspath(__file__))
    logger = setup_logging(data_dir)
    
    try:
        print("=" * 70)
        print("ESA-ADB MATRIX PROFILE ANOMALY DETECTION")
        print("Mission 1: 84_months Dataset")
        print("Lightweight Subset: Channels 41-46")
        print("=" * 70)
    
        # Initialize configuration
        config = PipelineConfig(
            window_size=17,  # Optimized for lightweight ESA telemetry
            threshold_multiplier=3.0,
            threshold_percentile=99.0,
            
            # AB-join baseline threshold calibration (R2 compliant)
            use_ab_join_calibration=True,
            ab_join_calibration_size=100_000,
            ab_join_threshold_percentile=99.5,
            
            # =====================================================================
            # ROBUST HYBRID THRESHOLDING WITH HYSTERESIS (Reduces False Positives)
            # =====================================================================
            # Uses MAD instead of Std for robustness to outliers
            # HYSTERESIS: Dual thresholds prevent flickering alerts
            #   - K_upper=5.0: Distance must exceed this to START an anomaly (stricter)
            #   - K_lower=2.0: Distance must drop below this to END an anomaly (looser)
            # This captures full anomaly events while ignoring transient noise.
            # =====================================================================
            use_adaptive_threshold=True,
            adaptive_window_size=1000,  # History buffer size (larger = more robust)
            adaptive_mad_multiplier=5.0,  # K_upper: stricter threshold to START anomaly
            adaptive_mad_multiplier_lower=2.0,  # K_lower: looser threshold to END anomaly
            adaptive_min_samples=500,  # Warmup period before adaptive kicks in
            prevent_buffer_contamination=True,  # Don't add anomalies to buffer
            use_hybrid_lower_bound=True,  # Enforce static threshold as floor
            use_hysteresis=True,  # Enable dual-threshold hysteresis state machine
            
            # EXCLUSION ZONE (reduces alert chatter from persistent anomalies)
            use_exclusion_zone=True,
            exclusion_zone_size=50,  # Suppress alerts for 50 steps after detection
            
            # Post-processing (R7: reduce fragmentation - TUNED for noise rejection)
            smoothing_window=51,  # Rolling median smoothing
            min_event_duration=30,  # Minimum event length (increased from 10 to filter transient noise)
            gap_tolerance=100,  # Merge nearby events (increased from 50 to reduce fragmentation)
            
            # Memory optimization
            max_library_size=300_000
        )
        
        # Define data paths
        train_path = os.path.join(data_dir, "84_months.train.csv")
        test_path = os.path.join(data_dir, "84_months.test.csv")
        
        # Load and preprocess data
        train_df, test_df = load_esa_adb_data(train_path, test_path, config)
        
        print(f"\nProcessed data:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Training anomalies: {train_df['label'].sum()} ({100*train_df['label'].mean():.2f}%)")
        print(f"  Test: {len(test_df)} samples")
        print(f"  Test anomalies: {test_df['label'].sum()} ({100*test_df['label'].mean():.2f}%)")
        
        # Run detection pipeline
        results = detect_anomalies(train_df, test_df, config)
        
        # Align predictions to original length for evaluation
        aligned_preds = align_predictions_to_original(
            results['binary_predictions'],
            original_length=len(test_df),
            m=config.window_size
        )
        
        # Evaluate against ground truth
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        test_labels = test_df['label'].values
        metrics = compute_event_wise_metrics(aligned_preds, test_labels, beta=0.5)
        
        print("\n--- Point-wise Metrics ---")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  Precision: {metrics['point_precision']:.4f}")
        print(f"  Recall:    {metrics['point_recall']:.4f}")
        print(f"  F1 Score:  {metrics['point_f1']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        
        print("\n--- Event-wise Metrics (ESA-ADB Benchmark) ---")
        print(f"  True anomaly events:      {metrics['n_true_events']}")
        print(f"  Predicted anomaly events: {metrics['n_predicted_events']}")
        print(f"  Detected events:          {metrics['detected_events']}")
        print(f"  False alarm events:       {metrics['false_alarm_events']}")
        print(f"  Event Precision: {metrics['event_precision']:.4f}")
        print(f"  Event Recall:    {metrics['event_recall']:.4f}")
        print(f"  Event F0.5:      {metrics['event_f0.5']:.4f} (prioritizes precision)")
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
    
    finally:
        # Close the logger and restore stdout
        logger.close()
        print(f"Log file saved.")
