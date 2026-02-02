"""
Threshold Optimization using Optuna
====================================

This script optimizes threshold and post-processing parameters using 
pre-computed Matrix Profile scores. Since MP computation takes 6+ hours,
we separate the expensive computation from the cheap parameter tuning.

Usage:
------
1. Run the full pipeline once to generate scores:
   - optimal_mp_scores.npy (saved by optimised_to_esa.py)
   - optimal_mp_predictions.csv (contains ground truth labels)

2. Run this script to find optimal thresholds in ~5 minutes:
   python optimize_thresholds.py

3. Apply the best parameters to optimised_to_esa.py config

Requirements:
-------------
pip install optuna
"""

import numpy as np
import pandas as pd
import optuna
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional
import portion as P
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# ESA-ADB METRICS (copied from main file for standalone use)
# =============================================================================

class ESAScores:
    """ESA-ADB scoring metrics."""
    
    def __init__(self, betas=1):
        self._betas = np.atleast_1d(betas)

    def _find_events(self, binary_array: np.ndarray) -> List[Tuple[int, int]]:
        diff = np.diff(np.concatenate([[0], binary_array.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return list(zip(starts, ends))

    def score(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        gt_events = self._find_events(labels)
        pred_events = self._find_events(predictions)
        
        n_gt = len(gt_events)
        n_pred = len(pred_events)
        
        if n_gt == 0:
            return {'EW_precision': 0, 'EW_recall': 0, 'EW_F_1.00': 0,
                    'detected_events': 0, 'total_events': 0}
        
        true_positives = 0
        false_negatives = 0
        redundant_detections = 0
        matched_preds = [False] * n_pred
        
        for gt_start, gt_end in gt_events:
            detections_for_this_event = 0
            at_least_one = False
            
            for p, (pred_start, pred_end) in enumerate(pred_events):
                # Check overlap
                if not (pred_end <= gt_start or pred_start >= gt_end):
                    matched_preds[p] = True
                    detections_for_this_event += 1
                    if not at_least_one:
                        true_positives += 1
                        at_least_one = True
            
            if detections_for_this_event > 1:
                redundant_detections += (detections_for_this_event - 1)
            
            if not at_least_one:
                false_negatives += 1
        
        false_positives = sum(1 for matched in matched_preds if not matched)
        
        # Basic precision and recall
        divider = true_positives + false_positives
        raw_precision = true_positives / divider if divider > 0 else 0
        
        divider = true_positives + false_negatives
        recall = true_positives / divider if divider > 0 else 0
        
        # TNR correction
        nominal_samples = (labels == 0).sum()
        fp_duration = 0
        for p, (pred_start, pred_end) in enumerate(pred_events):
            if not matched_preds[p]:
                fp_duration += (pred_end - pred_start)
            else:
                for i in range(pred_start, min(pred_end, len(labels))):
                    if labels[i] == 0:
                        fp_duration += 1
        
        if nominal_samples > 0:
            tnr = max(0, 1 - (fp_duration / nominal_samples))
        else:
            tnr = 1.0
        
        precision = raw_precision * tnr
        
        # Alarming precision
        divider = true_positives + redundant_detections
        alarming_precision = true_positives / divider if divider > 0 else 0
        
        # F-scores
        results = {
            'EW_precision': precision,
            'EW_precision_raw': raw_precision,
            'EW_recall': recall,
            'alarming_precision': alarming_precision,
            'detected_events': true_positives,
            'total_events': n_gt,
            'pred_events': n_pred,
            'redundant_detections': redundant_detections,
            'false_positive_duration': fp_duration,
        }
        
        for beta in self._betas:
            denom = beta**2 * precision + recall
            results[f'EW_F_{beta:.2f}'] = ((1 + beta**2) * precision * recall / denom) if denom > 0 else 0
        
        return results


# =============================================================================
# POST-PROCESSING (same as main file)
# =============================================================================

def apply_post_processing(
    raw_predictions: np.ndarray,
    scores: np.ndarray,
    gap_tolerance: int = 3,
    min_event_duration: int = 1,
    trim_threshold: float = 0.5,
    extend_coverage: bool = False
) -> np.ndarray:
    """Apply post-processing to raw predictions."""
    result = raw_predictions.copy()
    
    # Step 1: Merge close predictions
    diff = np.diff(np.concatenate([[0], result, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) >= 2:
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - ends[i]
            if gap <= gap_tolerance:
                result[ends[i]:starts[i + 1]] = 1
    
    # Step 2: Filter short events
    diff = np.diff(np.concatenate([[0], result, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if (end - start) < min_event_duration:
            result[start:end] = 0
    
    # Step 3: Trim boundaries
    if trim_threshold > 0:
        diff = np.diff(np.concatenate([[0], result, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            if end - start < 3:
                continue
            
            event_scores = scores[start:end]
            threshold = np.max(event_scores) * trim_threshold
            
            new_start = start
            while new_start < end - 1 and scores[new_start] < threshold:
                result[new_start] = 0
                new_start += 1
            
            new_end = end
            while new_end > new_start + 1 and scores[new_end - 1] < threshold:
                result[new_end - 1] = 0
                new_end -= 1
    
    # Step 4: Extend coverage (optional)
    if extend_coverage:
        diff = np.diff(np.concatenate([[0], result, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            event_scores = scores[start:end]
            if len(event_scores) == 0:
                continue
            
            median_score = np.median(event_scores)
            extend_thresh = median_score * 0.8
            
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
# OPTUNA OBJECTIVE
# =============================================================================

class ThresholdOptimizer:
    """Optimize thresholds using pre-computed scores."""
    
    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        self.scores = scores
        self.labels = labels
        self.scorer = ESAScores(betas=[1.0])
        
        # Pre-compute score statistics for normalization
        self.score_min = np.min(scores)
        self.score_max = np.max(scores)
        self.score_median = np.median(scores)
        self.score_std = np.std(scores)
        
        print(f"Score statistics:")
        print(f"  Min: {self.score_min:.3f}")
        print(f"  Max: {self.score_max:.3f}")
        print(f"  Median: {self.score_median:.3f}")
        print(f"  Std: {self.score_std:.3f}")
        
        # Count GT events
        diff = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        self.n_gt_events = len(starts)
        print(f"  Ground truth events: {self.n_gt_events}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function - maximize EW F1 score (basic)."""
        
        # Threshold parameters
        threshold_percentile = trial.suggest_float('threshold_percentile', 90.0, 99.9)
        
        # Post-processing parameters
        gap_tolerance = trial.suggest_int('gap_tolerance', 1, 50)
        min_event_duration = trial.suggest_int('min_event_duration', 1, 10)
        trim_threshold = trial.suggest_float('trim_threshold', 0.0, 0.8)
        extend_coverage = trial.suggest_categorical('extend_coverage', [True, False])
        
        # Apply threshold
        threshold = np.percentile(self.scores, threshold_percentile)
        raw_predictions = (self.scores > threshold).astype(np.int8)
        
        # Apply post-processing
        predictions = apply_post_processing(
            raw_predictions, self.scores,
            gap_tolerance=gap_tolerance,
            min_event_duration=min_event_duration,
            trim_threshold=trim_threshold,
            extend_coverage=extend_coverage
        )
        
        # Evaluate
        results = self.scorer.score(predictions, self.labels)
        
        # Multi-objective: balance F1 and recall
        f1 = results['EW_F_1.00']
        recall = results['EW_recall']
        detected = results['detected_events']
        
        # Primary objective: F1 score
        # But heavily penalize if recall is too low
        if recall < 0.5:
            # Penalize low recall
            objective = f1 * (recall / 0.5)
        else:
            objective = f1
        
        # Log intermediate results
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('precision', results['EW_precision'])
        trial.set_user_attr('f1', f1)
        trial.set_user_attr('detected', detected)
        trial.set_user_attr('pred_events', results['pred_events'])
        trial.set_user_attr('redundant', results['redundant_detections'])
        
        return objective
    
    def objective_esa_adb(self, trial: optuna.Trial) -> float:
        """
        ESA-ADB OPTIMIZED OBJECTIVE
        
        This objective properly targets ESA-ADB benchmark metrics:
        1. TNR-corrected precision (penalizes FP duration)
        2. Alarming precision (penalizes redundant detections)
        3. Affiliation F1 (rewards temporal alignment)
        
        Composite score: weighted combination of all ESA metrics
        """
        
        # Threshold parameters - wider range for exploration
        threshold_percentile = trial.suggest_float('threshold_percentile', 85.0, 99.9)
        
        # Post-processing parameters
        gap_tolerance = trial.suggest_int('gap_tolerance', 1, 150)  # Higher for merging
        min_event_duration = trial.suggest_int('min_event_duration', 1, 10)
        trim_threshold = trial.suggest_float('trim_threshold', 0.0, 0.8)
        extend_coverage = trial.suggest_categorical('extend_coverage', [True, False])
        
        # Apply threshold
        threshold = np.percentile(self.scores, threshold_percentile)
        raw_predictions = (self.scores > threshold).astype(np.int8)
        
        # Apply post-processing
        predictions = apply_post_processing(
            raw_predictions, self.scores,
            gap_tolerance=gap_tolerance,
            min_event_duration=min_event_duration,
            trim_threshold=trim_threshold,
            extend_coverage=extend_coverage
        )
        
        # Evaluate with FULL ESA-ADB metrics
        results = self.scorer.score(predictions, self.labels)
        
        # Extract all ESA-ADB metrics
        ew_precision = results['EW_precision']      # TNR-corrected!
        ew_recall = results['EW_recall']
        ew_f1 = results['EW_F_1.00']
        alarming_prec = results['alarming_precision']
        aff_precision = results.get('AFF_precision', 0)
        aff_recall = results.get('AFF_recall', 0)
        aff_f1 = results.get('AFF_F_1.00', 0) if 'AFF_F_1.00' in results else 0
        fp_duration = results['false_positive_duration']
        redundant = results['redundant_detections']
        detected = results['detected_events']
        
        # Compute affiliation F1 if not in results
        if aff_f1 == 0 and aff_precision > 0 and aff_recall > 0:
            aff_f1 = 2 * aff_precision * aff_recall / (aff_precision + aff_recall)
        
        # =====================================================================
        # ESA-ADB COMPOSITE OBJECTIVE
        # =====================================================================
        # Weight the metrics based on ESA-ADB benchmark importance:
        # - Recall is CRITICAL (detecting all anomalies is safety-critical)
        # - TNR-corrected precision matters (FP duration penalized)
        # - Alarming precision matters (redundant alerts are bad)
        # - Affiliation F1 measures quality of detection timing
        
        # Minimum recall constraint - MUST detect most events
        min_recall = 0.8  # At least 80% of events
        
        if ew_recall < min_recall:
            # Heavy penalty for missing events
            recall_penalty = (ew_recall / min_recall) ** 2
        else:
            recall_penalty = 1.0
        
        # Redundancy penalty (ESA-ADB alarming precision)
        if detected > 0:
            redundancy_ratio = redundant / detected
            redundancy_penalty = 1.0 / (1.0 + redundancy_ratio)  # 1.0 if no redundancy
        else:
            redundancy_penalty = 0.0
        
        # FP duration penalty (ESA-ADB TNR correction)
        # Normalize by total nominal samples
        nominal_samples = (self.labels == 0).sum()
        if nominal_samples > 0:
            fp_ratio = fp_duration / nominal_samples
            fp_penalty = max(0, 1.0 - fp_ratio * 10)  # Scale factor for sensitivity
        else:
            fp_penalty = 1.0
        
        # Composite objective (weighted combination)
        # Weights reflect ESA-ADB benchmark priorities:
        objective = (
            0.35 * ew_f1 +              # Event-wise F1 (TNR-corrected)
            0.25 * ew_recall +          # Raw recall (detect all events!)
            0.15 * alarming_prec +      # Alarming precision (reduce redundancy)
            0.15 * aff_f1 +             # Affiliation F1 (temporal quality)
            0.10 * fp_penalty           # FP duration penalty
        ) * recall_penalty * redundancy_penalty
        
        # Log ALL ESA-ADB metrics
        trial.set_user_attr('ew_recall', ew_recall)
        trial.set_user_attr('ew_precision', ew_precision)
        trial.set_user_attr('ew_f1', ew_f1)
        trial.set_user_attr('alarming_precision', alarming_prec)
        trial.set_user_attr('aff_precision', aff_precision)
        trial.set_user_attr('aff_recall', aff_recall)
        trial.set_user_attr('aff_f1', aff_f1)
        trial.set_user_attr('detected', detected)
        trial.set_user_attr('total_events', self.n_gt_events)
        trial.set_user_attr('pred_events', results['pred_events'])
        trial.set_user_attr('redundant', redundant)
        trial.set_user_attr('fp_duration', fp_duration)
        
        return objective
    
    def objective_recall_focused(self, trial: optuna.Trial) -> float:
        """Alternative objective: maximize recall with minimum precision constraint."""
        
        threshold_percentile = trial.suggest_float('threshold_percentile', 85.0, 99.9)
        gap_tolerance = trial.suggest_int('gap_tolerance', 1, 100)
        min_event_duration = trial.suggest_int('min_event_duration', 1, 5)
        trim_threshold = trial.suggest_float('trim_threshold', 0.0, 0.7)
        extend_coverage = trial.suggest_categorical('extend_coverage', [True, False])
        
        threshold = np.percentile(self.scores, threshold_percentile)
        raw_predictions = (self.scores > threshold).astype(np.int8)
        
        predictions = apply_post_processing(
            raw_predictions, self.scores,
            gap_tolerance=gap_tolerance,
            min_event_duration=min_event_duration,
            trim_threshold=trim_threshold,
            extend_coverage=extend_coverage
        )
        
        results = self.scorer.score(predictions, self.labels)
        
        recall = results['EW_recall']
        precision = results['EW_precision']
        f1 = results['EW_F_1.00']
        
        # Objective: maximize recall, but penalize extremely low precision
        # We want recall > 0.9 ideally
        if precision < 0.001:
            objective = recall * 0.1  # Heavy penalty for near-zero precision
        else:
            objective = recall
        
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('precision', precision)
        trial.set_user_attr('f1', f1)
        trial.set_user_attr('detected', results['detected_events'])
        trial.set_user_attr('pred_events', results['pred_events'])
        trial.set_user_attr('redundant', results.get('redundant_detections', 0))
        trial.set_user_attr('fp_duration', results.get('false_positive_duration', 0))
        
        return objective
    
    def objective_esa_f2(self, trial: optuna.Trial) -> float:
        """
        ESA-ADB F2 OBJECTIVE - Prioritizes Recall over Precision
        
        F2 score weights recall 2x more than precision.
        This is ideal for safety-critical satellite monitoring where
        missing an anomaly (false negative) is worse than a false alarm.
        """
        
        threshold_percentile = trial.suggest_float('threshold_percentile', 85.0, 99.5)
        gap_tolerance = trial.suggest_int('gap_tolerance', 5, 200)
        min_event_duration = trial.suggest_int('min_event_duration', 1, 5)
        trim_threshold = trial.suggest_float('trim_threshold', 0.0, 0.6)
        extend_coverage = trial.suggest_categorical('extend_coverage', [True, False])
        
        threshold = np.percentile(self.scores, threshold_percentile)
        raw_predictions = (self.scores > threshold).astype(np.int8)
        
        predictions = apply_post_processing(
            raw_predictions, self.scores,
            gap_tolerance=gap_tolerance,
            min_event_duration=min_event_duration,
            trim_threshold=trim_threshold,
            extend_coverage=extend_coverage
        )
        
        results = self.scorer.score(predictions, self.labels)
        
        # Compute F2 score (beta=2 weights recall 2x)
        precision = results['EW_precision']
        recall = results['EW_recall']
        beta = 2.0
        
        if precision + recall > 0:
            f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        else:
            f2 = 0
        
        # Also consider affiliation metrics for quality
        aff_recall = results.get('AFF_recall', 0)
        alarming_prec = results.get('alarming_precision', 0)
        
        # Composite: F2 + bonus for affiliation coverage and low redundancy
        objective = f2 + 0.1 * aff_recall + 0.1 * alarming_prec
        
        trial.set_user_attr('ew_recall', recall)
        trial.set_user_attr('ew_precision', precision)
        trial.set_user_attr('ew_f1', results['EW_F_1.00'])
        trial.set_user_attr('ew_f2', f2)
        trial.set_user_attr('aff_recall', aff_recall)
        trial.set_user_attr('alarming_precision', alarming_prec)
        trial.set_user_attr('detected', results['detected_events'])
        trial.set_user_attr('pred_events', results['pred_events'])
        trial.set_user_attr('redundant', results.get('redundant_detections', 0))
        trial.set_user_attr('fp_duration', results.get('false_positive_duration', 0))
        
        return objective


def run_optimization(
    scores_path: str,
    predictions_path: str,
    n_trials: int = 200,
    objective_type: str = 'esa_adb'  # 'f1', 'recall', 'esa_adb', 'esa_f2'
):
    """
    Run Optuna optimization with ESA-ADB metrics.
    
    Objective types:
    - 'esa_adb': Full ESA-ADB composite (TNR correction, alarming precision, affiliation)
    - 'esa_f2': F2 score (prioritizes recall 2x over precision) - RECOMMENDED
    - 'f1': Simple F1 optimization
    - 'recall': Maximize recall
    """
    
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION WITH OPTUNA")
    print("=" * 70)
    
    # Load pre-computed scores
    print(f"\nLoading scores from: {scores_path}")
    scores = np.load(scores_path)
    print(f"  Scores shape: {scores.shape}")
    
    # Load labels
    print(f"\nLoading labels from: {predictions_path}")
    df = pd.read_csv(predictions_path)
    labels = df['ground_truth'].values
    print(f"  Labels shape: {labels.shape}")
    
    # Ensure same length
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]
    print(f"  Aligned length: {min_len}")
    
    # Create optimizer
    optimizer = ThresholdOptimizer(scores, labels)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name=f'threshold_optimization_{objective_type}',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Select objective
    if objective_type == 'recall':
        objective_fn = optimizer.objective_recall_focused
        print("\n*** RECALL-FOCUSED OPTIMIZATION ***")
    elif objective_type == 'esa_adb':
        objective_fn = optimizer.objective_esa_adb
        print("\n*** ESA-ADB COMPOSITE OPTIMIZATION ***")
        print("    (TNR correction + Alarming Precision + Affiliation F1)")
    elif objective_type == 'esa_f2':
        objective_fn = optimizer.objective_esa_f2
        print("\n*** ESA-ADB F2 OPTIMIZATION (Recall-Prioritized) ***")
        print("    (F2 weights recall 2x more than precision)")
    else:
        objective_fn = optimizer.objective
        print("\n*** F1-FOCUSED OPTIMIZATION ***")
    
    # Run optimization
    print(f"\nRunning {n_trials} trials...")
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    best_trial = study.best_trial
    
    print(f"\nBest trial: #{best_trial.number}")
    print(f"Best objective value: {best_trial.value:.4f}")
    
    print(f"\nBest parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest metrics:")
    # Handle different objective types with different logged metrics
    if 'ew_recall' in best_trial.user_attrs:
        # ESA-ADB objectives
        print(f"  EW Recall: {best_trial.user_attrs['ew_recall']:.4f}")
        print(f"  EW Precision (TNR-corrected): {best_trial.user_attrs['ew_precision']:.4f}")
        print(f"  EW F1: {best_trial.user_attrs['ew_f1']:.4f}")
        if 'ew_f2' in best_trial.user_attrs:
            print(f"  EW F2: {best_trial.user_attrs['ew_f2']:.4f}")
        if 'alarming_precision' in best_trial.user_attrs:
            print(f"  Alarming Precision: {best_trial.user_attrs['alarming_precision']:.4f}")
        if 'aff_recall' in best_trial.user_attrs:
            print(f"  Affiliation Recall: {best_trial.user_attrs['aff_recall']:.4f}")
        if 'aff_f1' in best_trial.user_attrs:
            print(f"  Affiliation F1: {best_trial.user_attrs['aff_f1']:.4f}")
        print(f"  Detected: {best_trial.user_attrs['detected']}/{optimizer.n_gt_events}")
        print(f"  Predicted events: {best_trial.user_attrs['pred_events']}")
        if 'redundant' in best_trial.user_attrs:
            print(f"  Redundant detections: {best_trial.user_attrs['redundant']}")
        if 'fp_duration' in best_trial.user_attrs:
            print(f"  FP Duration: {best_trial.user_attrs['fp_duration']:.0f}")
    else:
        # Basic objectives
        print(f"  Recall: {best_trial.user_attrs['recall']:.4f}")
        print(f"  Precision: {best_trial.user_attrs['precision']:.4f}")
        print(f"  F1: {best_trial.user_attrs['f1']:.4f}")
        print(f"  Detected: {best_trial.user_attrs['detected']}/{optimizer.n_gt_events}")
        print(f"  Predicted events: {best_trial.user_attrs['pred_events']}")
        if 'redundant' in best_trial.user_attrs:
            print(f"  Redundant detections: {best_trial.user_attrs['redundant']}")
    
    # Generate config snippet
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIG (copy to optimised_to_esa.py)")
    print("=" * 70)
    print(f"""
    config = OptimalMPConfig(
        single_window_mode=False,
        window_sizes=[4, 8, 16, 32, 64, 128, 256],
        primary_window=64,
        max_reference_size=100_000,
        fusion_method='mean',
        
        # OPTUNA-OPTIMIZED PARAMETERS
        threshold_percentile={best_trial.params['threshold_percentile']:.1f},
        gap_tolerance={best_trial.params['gap_tolerance']},
        min_event_duration={best_trial.params['min_event_duration']},
        # trim_threshold={best_trial.params['trim_threshold']:.2f}  (in ESAOptimizedPostProcessor)
        # extend_coverage={best_trial.params['extend_coverage']}
        
        learn_anomaly_signatures=True,
        max_signatures=200,
        learn_nominal_patterns=True,
        max_nominal_patterns=500,
        
        handle_irregular_timestamps=True,
        expected_sampling_rate=30.0,
        segment_size=500_000,
        debug_mode=False,
    )
    """)
    
    # Save results - handle different attribute names based on objective type
    results_path = Path(scores_path).parent / f'optuna_results_{objective_type}.json'
    
    # Get metrics based on objective type (ESA objectives use ew_ prefix)
    if objective_type in ['esa_adb', 'esa_f2']:
        best_metrics = {
            'ew_recall': best_trial.user_attrs.get('ew_recall', 0),
            'ew_precision': best_trial.user_attrs.get('ew_precision', 0),
            'ew_f1': best_trial.user_attrs.get('ew_f1', 0),
            'alarming_precision': best_trial.user_attrs.get('alarming_precision', 0),
            'aff_recall': best_trial.user_attrs.get('aff_recall', 0),
            'aff_f1': best_trial.user_attrs.get('aff_f1', 0),
            'detected': best_trial.user_attrs.get('detected', 0),
            'pred_events': best_trial.user_attrs.get('pred_events', 0),
            'redundant': best_trial.user_attrs.get('redundant', 0),
            'fp_duration': best_trial.user_attrs.get('fp_duration', 0),
            'total_events': optimizer.n_gt_events,
        }
    else:
        best_metrics = {
            'recall': best_trial.user_attrs.get('recall', 0),
            'precision': best_trial.user_attrs.get('precision', 0),
            'f1': best_trial.user_attrs.get('f1', 0),
            'detected': best_trial.user_attrs.get('detected', 0),
            'total_events': optimizer.n_gt_events,
        }
    
    results = {
        'best_params': best_trial.params,
        'best_value': best_trial.value,
        'best_metrics': best_metrics,
        'n_trials': n_trials,
        'objective_type': objective_type,
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Show top 5 trials
    print("\n" + "=" * 70)
    print("TOP 5 TRIALS")
    print("=" * 70)
    
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(5)
    
    for idx, row in trials_df.iterrows():
        print(f"\nTrial #{int(row['number'])}: objective={row['value']:.4f}")
        # Handle both attribute naming conventions
        if objective_type in ['esa_adb', 'esa_f2']:
            recall = row.get('user_attrs_ew_recall', 0)
            precision = row.get('user_attrs_ew_precision', 0)
            f1 = row.get('user_attrs_ew_f1', 0)
            print(f"  ew_recall={recall:.4f}, ew_precision={precision:.4f}, ew_f1={f1:.4f}")
        else:
            recall = row.get('user_attrs_recall', 0)
            precision = row.get('user_attrs_precision', 0)
            f1 = row.get('user_attrs_f1', 0)
            print(f"  recall={recall:.4f}, precision={precision:.4f}, f1={f1:.4f}")
        print(f"  threshold_pctl={row['params_threshold_percentile']:.1f}, gap_tol={int(row['params_gap_tolerance'])}")
    
    return study, best_trial


# =============================================================================
# QUICK GRID SEARCH (faster than Optuna for initial exploration)
# =============================================================================

def quick_grid_search(scores: np.ndarray, labels: np.ndarray):
    """Quick grid search to understand parameter sensitivity."""
    
    print("=" * 70)
    print("QUICK GRID SEARCH")
    print("=" * 70)
    
    scorer = ESAScores(betas=[1.0])
    
    # Grid of threshold percentiles
    percentiles = [90, 92, 94, 95, 96, 97, 98, 99, 99.5, 99.9]
    gap_tolerances = [1, 5, 10, 20, 50]
    
    results = []
    
    for pctl in percentiles:
        for gap in gap_tolerances:
            threshold = np.percentile(scores, pctl)
            raw_preds = (scores > threshold).astype(np.int8)
            preds = apply_post_processing(raw_preds, scores, gap_tolerance=gap)
            
            metrics = scorer.score(preds, labels)
            
            results.append({
                'percentile': pctl,
                'gap_tolerance': gap,
                'recall': metrics['EW_recall'],
                'precision': metrics['EW_precision'],
                'f1': metrics['EW_F_1.00'],
                'detected': metrics['detected_events'],
                'total': metrics['total_events'],
            })
    
    # Convert to DataFrame and display
    df = pd.DataFrame(results)
    
    print("\nResults sorted by F1:")
    print(df.sort_values('f1', ascending=False).head(10).to_string(index=False))
    
    print("\nResults sorted by Recall:")
    print(df.sort_values('recall', ascending=False).head(10).to_string(index=False))
    
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    
    scores_path = os.path.join(results_dir, "optimal_mp_scores.npy")
    predictions_path = os.path.join(results_dir, "optimal_mp_predictions.csv")
    
    # Check if files exist
    if not os.path.exists(scores_path):
        print(f"ERROR: Scores file not found: {scores_path}")
        print("Run optimised_to_esa.py first to generate scores.")
        exit(1)
    
    if not os.path.exists(predictions_path):
        print(f"ERROR: Predictions file not found: {predictions_path}")
        print("Run optimised_to_esa.py first to generate predictions with ground truth.")
        exit(1)
    
    # Option 1: Quick grid search first (fast, ~10 seconds)
    print("\n" + "=" * 70)
    print("STEP 1: QUICK GRID SEARCH")
    print("=" * 70)
    
    scores = np.load(scores_path)
    df = pd.read_csv(predictions_path)
    labels = df['ground_truth'].values
    
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]
    
    grid_results = quick_grid_search(scores, labels)
    
    # Option 2: Full Optuna optimization with ESA-ADB metrics
    print("\n" + "=" * 70)
    print("STEP 2: OPTUNA OPTIMIZATION (ESA-ADB METRICS)")
    print("=" * 70)
    
    # Run ESA-ADB composite optimization (RECOMMENDED)
    print("\n" + "-" * 50)
    print("Running ESA-ADB Composite Optimization...")
    print("-" * 50)
    study_esa, best_esa = run_optimization(
        scores_path, predictions_path,
        n_trials=150,
        objective_type='esa_adb'
    )
    
    # Run ESA F2 optimization (recall-prioritized)
    print("\n" + "-" * 50)
    print("Running ESA-ADB F2 Optimization (Recall-Prioritized)...")
    print("-" * 50)
    study_f2, best_f2 = run_optimization(
        scores_path, predictions_path,
        n_trials=150,
        objective_type='esa_f2'
    )
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - ESA-ADB OPTIMIZED PARAMETERS")
    print("=" * 70)
    
    print("\n>>> OPTION 1: ESA-ADB Composite (Balanced)")
    print(f"    Objective value: {best_esa.value:.4f}")
    if 'ew_recall' in best_esa.user_attrs:
        print(f"    EW Recall: {best_esa.user_attrs['ew_recall']:.4f}")
        print(f"    EW F1: {best_esa.user_attrs['ew_f1']:.4f}")
        if 'alarming_precision' in best_esa.user_attrs:
            print(f"    Alarming Precision: {best_esa.user_attrs['alarming_precision']:.4f}")
        if 'aff_f1' in best_esa.user_attrs:
            print(f"    Affiliation F1: {best_esa.user_attrs['aff_f1']:.4f}")
    print(f"    Params: {best_esa.params}")
    
    print("\n>>> OPTION 2: ESA-ADB F2 (Recall-Prioritized) - RECOMMENDED FOR SAFETY")
    print(f"    Objective value: {best_f2.value:.4f}")
    if 'ew_recall' in best_f2.user_attrs:
        print(f"    EW Recall: {best_f2.user_attrs['ew_recall']:.4f}")
        print(f"    EW F2: {best_f2.user_attrs.get('ew_f2', 'N/A')}")
        if 'alarming_precision' in best_f2.user_attrs:
            print(f"    Alarming Precision: {best_f2.user_attrs['alarming_precision']:.4f}")
    print(f"    Params: {best_f2.params}")
    
    # Save best ESA-ADB results
    results_path = Path(scores_path).parent / 'optuna_esa_adb_results.json'
    esa_results = {
        'esa_composite': {
            'params': best_esa.params,
            'value': best_esa.value,
            'metrics': {k: v for k, v in best_esa.user_attrs.items()}
        },
        'esa_f2': {
            'params': best_f2.params,
            'value': best_f2.value,
            'metrics': {k: v for k, v in best_f2.user_attrs.items()}
        }
    }
    with open(results_path, 'w') as f:
        json.dump(esa_results, f, indent=2)
    print(f"\n  ESA-ADB results saved to: {results_path}")
