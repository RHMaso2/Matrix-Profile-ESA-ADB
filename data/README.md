# matrix_profile_esa_dataset

# Data Directory

## Dataset: ESA-ADB Mission 1 (84 months)

### Download Instructions

1. Download the ESA Anomaly Detection Benchmark dataset from:
	https://zenodo.org/records/12528696

2. Place the following files in this directory, following subset pre-processing:
   - `84_months.train.csv`
   - `84_months.test.csv`

### Expected Format

The CSV files should contain:
- `timestamp`: DateTime column
- `channel_41` to `channel_46`: Telemetry channels (float)
- `is_anomaly_channel_41` to `is_anomaly_channel_46`: Ground truth labels (0/1)

### File Sizes
- Training: ~7.3M samples (~500MB)
- Test: ~7.3M samples (~500MB)
