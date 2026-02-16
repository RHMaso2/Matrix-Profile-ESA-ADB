"""
Matrix Profile-based Anomaly Detection Pipeline for 21_months Dataset
========================================================================

This module is a variant of optimised_to_esa_perchannel.py, specifically for the 21_months dataset (channels 18-28).

Usage:
    python optimised_to_esa_perchannel_21months.py

"""

import os
import argparse
from optimised_to_esa_perchannel import OptimalMPConfig, run_perchannel_pipeline

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "results")

    train_path = os.path.join(base_dir, "21_months.train.csv")
    test_path = os.path.join(base_dir, "21_months.test.csv")
    channel_range = range(18, 29)

    target_channels = [f'channel_{i}' for i in channel_range]
    label_columns = [f'is_anomaly_channel_{i}' for i in channel_range]

    config = OptimalMPConfig(
        debug_mode=False,
        single_window_mode=False,
    )
    config.channels.target_channels = target_channels
    config.channels.label_columns = label_columns
    config.expected_sampling_rate = 18.0

    output_result = run_perchannel_pipeline(
        train_path=train_path,
        test_path=test_path,
        config=config,
        output_dir=output_dir
    )

    print(f"\nOutput DataFrame shape: {output_result['output_df'].shape}")
    print(f"Columns: {list(output_result['output_df'].columns)}")
    print(f"\nMetrics: {output_result['metrics']}")
