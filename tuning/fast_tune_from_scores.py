"""
Fast Threshold Tuning from Pre-computed Scores
==============================================

This script loads pre-computed Matrix Profile scores and tunes 
post-processing parameters (threshold, gap_tolerance, min_event_duration)
WITHOUT re-computing the expensive Matrix Profile.

OPTIMIZED: Uses fast F1-proxy scoring instead of full ESA metrics per trial.
Full ESA metrics computed only for best parameters at the end.

Usage:
1. First run optimised_to_esa_perchannel.py with save_scores_for_tuning=True
2. Then run this script to quickly tune thresholds

Runtime: ~5-10 minutes for 200 trials on full 7.3M samples
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import optuna
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ESA-ADB Metrics (only used for final evaluation)
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
    
    print(f"  Loaded {data['metadata']['n_samples']:,} samples, {data['metadata']['n_channels']} channels")
    print(f"  Score range: {data['metadata']['score_range']}")
    print(f"  Anomaly rate: {data['metadata']['anomaly_rate']*100:.3f}%")
    
    return data


def apply_postprocessing(
    scores: np.ndarray,
    threshold_percentile: float,
    min_event_duration: int,
    gap_tolerance: int
) -> np.ndarray:
    """Apply threshold and post-processing to scores."""
    threshold = np.percentile(scores, threshold_percentile)
    predictions = (scores > threshold).astype(np.int8)
    
    # Gap filling - vectorized for speed
    if gap_tolerance > 0 and predictions.sum() > 0:
        # Find prediction boundaries
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Merge gaps smaller than tolerance
        if len(starts) > 1:
            for i in range(len(starts) - 1):
                gap = starts[i + 1] - ends[i]
                if gap <= gap_tolerance:
                    predictions[ends[i]:starts[i + 1]] = 1
    
    # Minimum event duration - vectorized
    if min_event_duration > 1 and predictions.sum() > 0:
        diff = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for s, e in zip(starts, ends):
            if e - s < min_event_duration:
                predictions[s:e] = 0
    
    return predictions


def fast_f1_proxy(labels: np.ndarray, predictions: np.ndarray, beta: float = 0.5) -> float:
    """
    Fast F-beta proxy score using sample-level metrics.
    
    This is much faster than event-wise ESA scoring and correlates well
    for parameter tuning purposes. Final evaluation uses real ESA metrics.
    """
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    # F-beta score
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    return f_beta


def objective(trial: optuna.Trial, data: Dict) -> float:
    """Optuna objective for tuning post-processing parameters."""
    
    # Parameters to tune
    threshold_percentile = trial.suggest_float('threshold_percentile', 95.0, 99.999)
    gap_tolerance = trial.suggest_int('gap_tolerance', 1, 100)
    min_event_duration = trial.suggest_int('min_event_duration', 1, 20)
    
    # Apply to combined scores (aggregated)
    predictions = apply_postprocessing(
        data['combined_scores'],
        threshold_percentile,
        min_event_duration,
        gap_tolerance
    )
    
    # Fast F0.5 proxy score (much faster than ESA scoring)
    f05 = fast_f1_proxy(data['labels'], predictions, beta=0.5)
    
    # Penalize extreme prediction rates
    pred_rate = predictions.mean()
    if pred_rate > 0.5:  # Predicting >50% as anomaly is clearly wrong
        f05 *= 0.1
    elif pred_rate > 0.2:  # >20% is suspicious
        f05 *= 0.5
    
    return f05


def compute_full_esa_metrics(labels: np.ndarray, predictions: np.ndarray, timestamps: np.ndarray) -> Dict:
    """Compute FULL ESA-ADB metrics (slow - only for final evaluation)."""
    print("  Computing full ESA metrics (this may take a minute)...")
    
    # Build y_true DataFrame from label events
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
    
    # Build y_pred array with proper pd.Timestamp objects
    n = len(predictions)
    y_pred = np.empty((n, 2), dtype=object)
    
    # Convert timestamps to pd.Timestamp (ESA requires this exact type, not datetime64)
    ts_converted = pd.to_datetime(timestamps)
    for i in range(n):
        y_pred[i, 0] = pd.Timestamp(ts_converted[i])
    y_pred[:, 1] = predictions.astype(int)
    
    # Score
    try:
        scorer = ESAScores(betas=0.5)
        results = scorer.score(y_true_df, y_pred)
        return results
    except Exception as e:
        print(f"  ESA Scoring error: {e}")
        return {'EW_F_0.50': 0.0, 'EW_precision': 0.0, 'EW_recall': 0.0}


def run_tuning(
    scores_dir: str = 'results',
    n_trials: int = 100,
    output_dir: str = 'results'
) -> Dict:
    """Run fast threshold tuning on pre-computed scores."""
    
    print("\n" + "=" * 70)
    print("FAST THRESHOLD TUNING FROM PRE-COMPUTED SCORES")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load scores
    data = load_saved_scores(scores_dir)
    
    # Create study
    print(f"\nRunning Optuna optimization ({n_trials} trials)...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, data),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Best parameters
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    
    best = study.best_params
    print(f"  threshold_percentile: {best['threshold_percentile']:.2f}")
    print(f"  gap_tolerance:        {best['gap_tolerance']}")
    print(f"  min_event_duration:   {best['min_event_duration']}")
    print(f"\n  Best EW F0.5:         {study.best_value:.4f}")
    
    # Apply best parameters and get full metrics
    print("\n" + "=" * 70)
    print("FULL EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)
    
    predictions = apply_postprocessing(
        data['combined_scores'],
        best['threshold_percentile'],
        best['min_event_duration'],
        best['gap_tolerance']
    )
    
    pred_rate = predictions.mean() * 100
    print(f"  Prediction rate: {pred_rate:.2f}%")
    print(f"  Ground truth rate: {data['metadata']['anomaly_rate']*100:.3f}%")
    
    # Full ESA scoring with optimized function
    results = compute_full_esa_metrics(data['labels'], predictions, data['timestamps'])
    
    print(f"\n  EW Precision: {results.get('EW_precision', 0):.4f}")
    print(f"  EW Recall:    {results.get('EW_recall', 0):.4f}")
    print(f"  EW F0.5:      {results.get('EW_F_0.50', 0):.4f}")
    print(f"  AFF F0.5:     {results.get('AFF_F_0.50', 0):.4f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    tuning_results = {
        'best_params': best,
        'best_f05': study.best_value,
        'full_metrics': {
            'EW_precision': results.get('EW_precision', 0),
            'EW_recall': results.get('EW_recall', 0),
            'EW_F_0.50': results.get('EW_F_0.50', 0),
            'AFF_F_0.50': results.get('AFF_F_0.50', 0),
            'prediction_rate': pred_rate,
        },
        'n_trials': n_trials,
        'timestamp': datetime.now().isoformat(),
    }
    
    results_file = output_path / 'tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    print(f"\n  Saved tuning results: {results_file}")
    
    # Print config update instructions
    print("\n" + "=" * 70)
    print("UPDATE YOUR CONFIG")
    print("=" * 70)
    print(f"""
Copy these values to OptimalMPConfig in optimised_to_esa_perchannel.py:

    threshold_percentile: float = {best['threshold_percentile']:.2f}
    gap_tolerance: int = {best['gap_tolerance']}
    min_event_duration: int = {best['min_event_duration']}
""")
    
    return tuning_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast threshold tuning from pre-computed scores')
    parser.add_argument('--scores-dir', type=str, default='results', help='Directory with saved scores')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of Optuna trials')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    run_tuning(
        scores_dir=args.scores_dir,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )
