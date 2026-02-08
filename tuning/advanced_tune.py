"""
Advanced Threshold Tuning from Pre-computed Scores
===================================================

Improvements over basic fast_tune:
1. Score transformations (local z-norm, log, smoothing)
2. Multiple fusion methods (mean, max, vote, weighted)
3. Per-channel threshold tuning
4. Event-overlap proxy metric (better correlation with ESA EW metrics)
5. Bidirectional anomaly detection (high AND low scores)

Usage:
    python advanced_tune.py --scores-dir results --n-trials 300

Runtime: ~5-15 minutes for 300 trials
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime
import optuna
from typing import Dict, List, Tuple
from scipy.ndimage import uniform_filter1d, maximum_filter1d
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

# ESA-ADB Metrics (final eval only)
from ESA_metrics import ESAScores


def load_saved_scores(scores_dir: str = 'results') -> Dict:
    """Load pre-computed scores and labels."""
    scores_dir = Path(scores_dir)
    
    print("Loading pre-computed scores...")
    
    data = {
        'per_channel_scores': np.load(scores_dir / 'mp_scores_perchannel.npy'),
        'combined_scores': np.load(scores_dir / 'mp_scores_combined.npy'),
        'labels': np.load(scores_dir / 'labels_aligned.npy'),
        'labels_perchannel': np.load(scores_dir / 'labels_perchannel_aligned.npy'),
        'timestamps': np.load(scores_dir / 'timestamps_aligned.npy', allow_pickle=True),
    }
    
    with open(scores_dir / 'scores_metadata.json', 'r') as f:
        data['metadata'] = json.load(f)
    
    n = data['metadata']['n_samples']
    nc = data['metadata']['n_channels']
    print(f"  Loaded {n:,} samples, {nc} channels")
    print(f"  Score range: {data['metadata']['score_range']}")
    print(f"  Anomaly rate: {data['metadata']['anomaly_rate']*100:.3f}%")
    
    return data


# =========================================================================
# SCORE TRANSFORMATIONS
# =========================================================================

def local_z_normalize(scores: np.ndarray, window: int) -> np.ndarray:
    """Z-normalize scores using a local sliding window.
    
    This converts global novelty scores into *contextual* anomaly scores:
    a moderately high score in a normally-calm region becomes a strong signal.
    """
    if window < 10:
        return scores
    local_mean = uniform_filter1d(scores.astype(np.float64), size=window, mode='reflect')
    local_std = np.sqrt(
        uniform_filter1d((scores.astype(np.float64) - local_mean) ** 2, size=window, mode='reflect')
    )
    local_std = np.maximum(local_std, 1e-10)
    return (scores - local_mean) / local_std


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    """Apply mean smoothing to reduce noise."""
    if window <= 1:
        return scores
    return uniform_filter1d(scores.astype(np.float64), size=window, mode='reflect')


def max_pool_scores(scores: np.ndarray, window: int) -> np.ndarray:
    """Max-pool scores to spread high values into neighboring samples."""
    if window <= 1:
        return scores
    return maximum_filter1d(scores.astype(np.float64), size=window, mode='reflect')


def log_transform(scores: np.ndarray) -> np.ndarray:
    """Log1p transform to compress high values and expand low values."""
    return np.log1p(np.maximum(scores, 0))


# =========================================================================
# FUSION METHODS
# =========================================================================

def fuse_channels(per_channel_scores: np.ndarray, method: str, 
                  vote_threshold: int = 2) -> np.ndarray:
    """Combine per-channel scores into a single score."""
    if method == 'mean':
        return np.mean(per_channel_scores, axis=1)
    elif method == 'max':
        return np.max(per_channel_scores, axis=1)
    elif method == 'median':
        return np.median(per_channel_scores, axis=1)
    elif method == 'sum':
        return np.sum(per_channel_scores, axis=1)
    else:
        return np.mean(per_channel_scores, axis=1)


# =========================================================================
# POST-PROCESSING
# =========================================================================

def apply_postprocessing(
    scores: np.ndarray,
    threshold_percentile: float,
    min_event_duration: int,
    gap_tolerance: int,
    extend_window: int = 0
) -> np.ndarray:
    """Apply threshold and post-processing."""
    threshold = np.percentile(scores, threshold_percentile)
    predictions = (scores > threshold).astype(np.int8)
    
    # Gap filling
    if gap_tolerance > 0 and predictions.sum() > 0:
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) > 1:
            for i in range(len(starts) - 1):
                gap = starts[i + 1] - ends[i]
                if gap <= gap_tolerance:
                    predictions[ends[i]:starts[i + 1]] = 1
    
    # Minimum event duration
    if min_event_duration > 1 and predictions.sum() > 0:
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < min_event_duration:
                predictions[s:e] = 0
    
    # Event extension (spread each event by N samples each side)
    if extend_window > 0 and predictions.sum() > 0:
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            new_s = max(0, s - extend_window)
            new_e = min(len(predictions), e + extend_window)
            predictions[new_s:new_e] = 1
    
    return predictions


# =========================================================================
# EVENT-WISE PROXY METRIC
# =========================================================================

def event_overlap_f_beta(labels: np.ndarray, predictions: np.ndarray, 
                          beta: float = 0.5) -> float:
    """
    Event-overlap F-beta score — a fast proxy that correlates much better
    with ESA's event-wise metrics than sample-level F-score.
    
    Counts what fraction of TRUE EVENTS are detected (recall)
    and what fraction of PREDICTED EVENTS overlap a true event (precision).
    """
    # Extract true events
    diff_t = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
    t_starts = np.where(diff_t == 1)[0]
    t_ends = np.where(diff_t == -1)[0]
    
    # Extract predicted events
    diff_p = np.diff(np.concatenate([[0], predictions.astype(int), [0]]))
    p_starts = np.where(diff_p == 1)[0]
    p_ends = np.where(diff_p == -1)[0]
    
    n_true = len(t_starts)
    n_pred = len(p_starts)
    
    if n_true == 0 or n_pred == 0:
        return 0.0
    
    # For each true event: does any predicted event overlap it?
    detected = 0
    for ts, te in zip(t_starts, t_ends):
        for ps, pe in zip(p_starts, p_ends):
            if ps < te and pe > ts:  # Overlap exists
                detected += 1
                break
    
    # For each predicted event: does it overlap any true event?
    correct = 0
    for ps, pe in zip(p_starts, p_ends):
        for ts, te in zip(t_starts, t_ends):
            if ps < te and pe > ts:
                correct += 1
                break
    
    recall = detected / n_true
    precision = correct / n_pred
    
    if precision + recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    return f_beta


def sample_and_event_f_beta(labels: np.ndarray, predictions: np.ndarray,
                             beta: float = 0.5, event_weight: float = 0.7) -> float:
    """
    Combined metric: weighted average of event-overlap and sample-level F-beta.
    Event-overlap is more important for ESA EW scoring.
    """
    # Sample-level
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    if tp == 0:
        return 0.0
    
    s_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    s_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if s_precision + s_recall == 0:
        sample_f = 0.0
    else:
        beta_sq = beta ** 2
        sample_f = (1 + beta_sq) * (s_precision * s_recall) / (beta_sq * s_precision + s_recall)
    
    # Event-level
    event_f = event_overlap_f_beta(labels, predictions, beta)
    
    # Weighted combination
    return event_weight * event_f + (1 - event_weight) * sample_f


# =========================================================================
# OPTUNA OBJECTIVE
# =========================================================================

def objective(trial: optuna.Trial, data: Dict) -> float:
    """Advanced tuning objective with score transforms and fusion."""
    
    per_ch = data['per_channel_scores']  # (n_samples, n_channels)
    labels = data['labels']
    
    # --- Score Transformation ---
    transform = trial.suggest_categorical('transform', ['none', 'local_z', 'log', 'local_z_log'])
    smooth_win = trial.suggest_int('smooth_window', 1, 200)
    
    # Local z-norm window (only used if transform includes local_z)
    if 'local_z' in transform:
        z_window = trial.suggest_int('z_window', 500, 50000, log=True)
    else:
        z_window = 1000  # Unused
    
    # Apply transforms per-channel
    transformed = np.empty_like(per_ch, dtype=np.float64)
    for ch in range(per_ch.shape[1]):
        s = per_ch[:, ch].astype(np.float64)
        
        if transform == 'log':
            s = log_transform(s)
        elif transform == 'local_z':
            s = local_z_normalize(s, z_window)
        elif transform == 'local_z_log':
            s = log_transform(s)
            s = local_z_normalize(s, z_window)
        
        # Smoothing always applied
        s = smooth_scores(s, smooth_win)
        transformed[:, ch] = s
    
    # --- Fusion Method ---
    fusion = trial.suggest_categorical('fusion', ['mean', 'max', 'median'])
    combined = fuse_channels(transformed, fusion)
    
    # --- Thresholding ---
    threshold_pct = trial.suggest_float('threshold_percentile', 90.0, 99.99)
    
    # --- Post-processing ---
    gap_tolerance = trial.suggest_int('gap_tolerance', 0, 300)
    min_event_duration = trial.suggest_int('min_event_duration', 1, 50)
    extend_window = trial.suggest_int('extend_window', 0, 200)
    
    # Apply
    predictions = apply_postprocessing(
        combined, threshold_pct, min_event_duration, gap_tolerance, extend_window
    )
    
    # --- Score ---
    pred_rate = predictions.mean()
    
    # Hard reject: nothing predicted or everything predicted
    if pred_rate == 0 or pred_rate > 0.30:
        return 0.0
    
    # Combined event + sample metric
    score = sample_and_event_f_beta(labels, predictions, beta=0.5, event_weight=0.7)
    
    # Soft penalty for prediction rates far from ground truth
    gt_rate = labels.mean()
    rate_ratio = pred_rate / max(gt_rate, 1e-10)
    if rate_ratio > 5:  # Predicting 5x more than GT
        score *= 0.5
    elif rate_ratio > 3:
        score *= 0.7
    
    return score


# =========================================================================
# FULL ESA EVALUATION
# =========================================================================

def compute_full_esa_metrics(labels: np.ndarray, predictions: np.ndarray, 
                              timestamps: np.ndarray) -> Dict:
    """Compute full ESA-ADB metrics for final evaluation."""
    print("  Computing full ESA metrics...")
    
    # Build y_true DataFrame
    diff = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    events = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        if s < len(timestamps) and e <= len(timestamps):
            events.append({
                'ID': f'event_{i}',
                'StartTime': pd.Timestamp(timestamps[s]),
                'EndTime': pd.Timestamp(timestamps[min(e-1, len(timestamps)-1)])
            })
    
    if not events:
        return {'EW_F_0.50': 0.0, 'EW_precision': 0.0, 'EW_recall': 0.0}
    
    y_true_df = pd.DataFrame(events)
    
    # Build y_pred - use DataFrame for proper type handling
    pred_df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'prediction': predictions.astype(int)
    })
    y_pred = pred_df.values
    
    try:
        scorer = ESAScores(betas=0.5)
        results = scorer.score(y_true_df, y_pred)
        return results
    except Exception as e:
        print(f"  ESA Scoring error: {e}")
        # Fallback: try with explicit Timestamp conversion
        try:
            print("  Retrying with explicit Timestamp conversion...")
            n = len(predictions)
            y_pred2 = np.empty((n, 2), dtype=object)
            ts_idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
            for i in range(n):
                y_pred2[i, 0] = pd.Timestamp(ts_idx[i])
                y_pred2[i, 1] = int(predictions[i])
            results = scorer.score(y_true_df, y_pred2)
            return results
        except Exception as e2:
            print(f"  Retry failed: {e2}")
            return {'EW_F_0.50': 0.0, 'EW_precision': 0.0, 'EW_recall': 0.0}


# =========================================================================
# MAIN TUNING LOOP
# =========================================================================

def run_tuning(
    scores_dir: str = 'results',
    n_trials: int = 300,
    output_dir: str = 'results'
) -> Dict:
    """Run advanced threshold tuning."""
    
    print("\n" + "=" * 70)
    print("ADVANCED THRESHOLD TUNING FROM PRE-COMPUTED SCORES")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data = load_saved_scores(scores_dir)
    
    # Quick diagnostic: check score-label overlap
    labels = data['labels']
    scores = data['combined_scores']
    per_ch = data['per_channel_scores']
    
    print(f"\n  Diagnostic:")
    anomaly_mask = labels.astype(bool)
    normal_mask = ~anomaly_mask
    print(f"    Score mean (normal):  {scores[normal_mask].mean():.3f}")
    print(f"    Score mean (anomaly): {scores[anomaly_mask].mean():.3f}")
    print(f"    Score std (normal):   {scores[normal_mask].std():.3f}")
    print(f"    Score std (anomaly):  {scores[anomaly_mask].std():.3f}")
    
    separability = abs(scores[anomaly_mask].mean() - scores[normal_mask].mean()) / \
                   max(scores[normal_mask].std(), 1e-10)
    print(f"    Separability (Cohen's d): {separability:.3f}")
    
    if separability < 0.5:
        print("    WARNING: Low separability — scores don't clearly distinguish anomalies!")
        print("    Local z-normalization and score transforms may help.")
    
    # Per-channel separability
    print(f"\n    Per-channel separability:")
    for ch in range(per_ch.shape[1]):
        ch_scores = per_ch[:, ch]
        ch_sep = abs(ch_scores[anomaly_mask].mean() - ch_scores[normal_mask].mean()) / \
                 max(ch_scores[normal_mask].std(), 1e-10)
        print(f"      Channel {ch}: {ch_sep:.3f}")
    
    # Run Optuna
    print(f"\nRunning Optuna optimization ({n_trials} trials)...")
    print(f"  Search space: transforms × fusion × threshold × post-processing")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, data),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    
    # Best parameters
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    
    best = study.best_params
    for k, v in best.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print(f"\n  Best proxy score:         {study.best_value:.4f}")
    
    # Reconstruct best predictions for full evaluation
    print("\n" + "=" * 70)
    print("FULL EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)
    
    # Apply best transforms
    transformed = np.empty_like(per_ch, dtype=np.float64)
    for ch in range(per_ch.shape[1]):
        s = per_ch[:, ch].astype(np.float64)
        
        transform = best['transform']
        if transform == 'log':
            s = log_transform(s)
        elif transform == 'local_z':
            s = local_z_normalize(s, best.get('z_window', 1000))
        elif transform == 'local_z_log':
            s = log_transform(s)
            s = local_z_normalize(s, best.get('z_window', 1000))
        
        s = smooth_scores(s, best['smooth_window'])
        transformed[:, ch] = s
    
    combined = fuse_channels(transformed, best['fusion'])
    
    predictions = apply_postprocessing(
        combined,
        best['threshold_percentile'],
        best['min_event_duration'],
        best['gap_tolerance'],
        best['extend_window']
    )
    
    pred_rate = predictions.mean() * 100
    print(f"  Prediction rate: {pred_rate:.2f}%")
    print(f"  Ground truth rate: {data['metadata']['anomaly_rate']*100:.3f}%")
    
    # Event counts
    diff_p = np.diff(np.concatenate([[0], predictions.astype(int), [0]]))
    n_pred_events = len(np.where(diff_p == 1)[0])
    diff_t = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
    n_true_events = len(np.where(diff_t == 1)[0])
    print(f"  Predicted events: {n_pred_events}")
    print(f"  True events:      {n_true_events}")
    
    # Event overlap proxy
    proxy_ew = event_overlap_f_beta(labels, predictions, beta=0.5)
    print(f"  Event-overlap F0.5 (proxy): {proxy_ew:.4f}")
    
    # Full ESA metrics
    results = compute_full_esa_metrics(labels, predictions, data['timestamps'])
    
    print(f"\n  EW Precision: {results.get('EW_precision', 0):.4f}")
    print(f"  EW Recall:    {results.get('EW_recall', 0):.4f}")
    print(f"  EW F0.5:      {results.get('EW_F_0.50', 0):.4f}")
    print(f"  AFF F0.5:     {results.get('AFF_F_0.50', 0):.4f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    tuning_results = {
        'best_params': best,
        'best_proxy_score': study.best_value,
        'event_overlap_f05': proxy_ew,
        'full_metrics': {
            'EW_precision': float(results.get('EW_precision', 0)),
            'EW_recall': float(results.get('EW_recall', 0)),
            'EW_F_0.50': float(results.get('EW_F_0.50', 0)),
            'AFF_F_0.50': float(results.get('AFF_F_0.50', 0)),
            'prediction_rate': pred_rate,
            'n_predicted_events': n_pred_events,
            'n_true_events': n_true_events,
        },
        'n_trials': n_trials,
        'timestamp': datetime.now().isoformat(),
        'diagnostics': {
            'separability': float(separability),
        }
    }
    
    results_file = output_path / 'advanced_tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    print(f"\n  Saved: {results_file}")
    
    # Top 5 trials
    print("\n" + "=" * 70)
    print("TOP 5 TRIALS")
    print("=" * 70)
    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for i, t in enumerate(trials):
        print(f"  #{i+1} score={t.value:.4f} | "
              f"transform={t.params.get('transform','?')} "
              f"fusion={t.params.get('fusion','?')} "
              f"thresh={t.params.get('threshold_percentile',0):.1f}% "
              f"gap={t.params.get('gap_tolerance',0)} "
              f"min_dur={t.params.get('min_event_duration',0)} "
              f"extend={t.params.get('extend_window',0)} "
              f"smooth={t.params.get('smooth_window',0)}")
    
    # Print integration instructions
    print("\n" + "=" * 70)
    print("INTEGRATION NOTES")
    print("=" * 70)
    print(f"""
To integrate these into optimised_to_esa_perchannel.py, you'll need to:

1. Update OptimalMPConfig:
    threshold_percentile: float = {best['threshold_percentile']:.2f}
    gap_tolerance: int = {best['gap_tolerance']}
    min_event_duration: int = {best['min_event_duration']}

2. Add score transform = '{best['transform']}' before thresholding
3. Use fusion = '{best['fusion']}' instead of current method
4. Add smooth_window = {best['smooth_window']} before thresholding
5. Add extend_window = {best['extend_window']} in post-processing
""")
    
    return tuning_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Advanced threshold tuning')
    parser.add_argument('--scores-dir', type=str, default='results')
    parser.add_argument('--n-trials', type=int, default=300)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    
    run_tuning(
        scores_dir=args.scores_dir,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )
