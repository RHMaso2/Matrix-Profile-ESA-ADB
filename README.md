# Matrix Profile for Satellite Telemetry Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Unsupervised anomaly detection for satellite telemetry using Matrix Profile algorithm, optimized for the ESA Anomaly Detection Benchmark (ESA-ADB).

## Features

- **Matrix Profile-Based Detection**: Distance-based anomaly scoring using STUMPY
- **Multi-Channel Support**: Per-channel and aggregated predictions for 6 telemetry channels
- **ESA-ADB Compliant**: Meets all ESA-ADB requirements (R1-R9)
- **GPU Acceleration**: Optional CUDA support for 5-10× speedup
- **Hyperparameter Tuning**: Optuna-based threshold optimization
- **Fast Tuning Mode**: Tune post-processing without re-computing Matrix Profile

## Performance

| Metric | Score |
|--------|-------|
| Affiliation F0.5 | 49.0% |
| Alarming Precision | 46.6% |
| Channel-Aware F0.5 | 14.4% |
| Runtime | ~86 minutes (CPU), ~15 minutes (GPU) |

## Installation

### Requirements
- Python 3.8+
- 10GB RAM (100GB+ recommended for full speed)
- Optional: CUDA-capable GPU

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/matrix-profile-satellite-anomaly.git
cd matrix-profile-satellite-anomaly

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install cupy-cuda11x  # Replace '11x' with your CUDA version