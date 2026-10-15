# Anomaly Detector — thesis project

Offline anomaly detection on conveyor-belt vibration readings using a **Kalman filter**, **seasonal decomposition**, and an **LSTM autoencoder**. This repository contains the original thesis PDF [`Tesi (1).pdf`](Tesi%20(1).pdf) and a refactored, reproducible Python pipeline.

> **Research prototype, not a real-time safety or maintenance alarm.** The original analysis and dataset assumptions are discussed below. The private/raw sensor CSVs are **not included** in this repository.

## Why this refactor?

The historical notebook-export script mixed data loading, visualizations, filtering, feature extraction, model training and scoring in one file. It contained notebook-only syntax, empty CSV paths and references to intermediate CSV files that were never created. Its original form is retained, unmodified, at [`legacy/Anomaly detector.py`](legacy/Anomaly%20detector.py) for academic provenance.

The new implementation has:
- A runnable CLI with explicit CSV paths and parameters.
- Isolated, testable preprocessing, autoencoder and scoring modules.
- No train/test overlap in the normal-data prefix; scaling is fit **only on train**.
- A threshold derived **only from train reconstruction loss** (by default the maximum, as in the original study).
- Saved model, scaler, window-level scores, configuration and a non-interactive loss chart.
- Synthetic unit tests and GitHub Actions checks.

## Installation

Python 3.10+ is required. A TensorFlow-compatible Python environment is required for training (CPU is sufficient for small runs).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,test]"
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\activate`.

## Input data and running

Provide **two CSVs**: one recorded under normal operation, one from an anomalous-source recording. Each file must have at least three columns (x, y, z); the third column is read as the z-axis signal. By default the first row is treated as a header; pass `--no-header` if both CSVs have no column headers. Values must be numeric and finite.

```bash
python -m anomaly_detector \
  --normal-csv data/normal.csv \
  --anomalous-csv data/anomalous.csv \
  --output-dir outputs/run1
```

To reproduce the approximate **original training segment selection**, assuming you have the original 9.6-million-row normal signal, pass:

```bash
python -m anomaly_detector \
  --normal-csv data/normal.csv \
  --anomalous-csv data/anomalous.csv \
  --train-start 5900000 \
  --train-end 9000000 \
  --healthy-test-samples 1000 \
  --seasonal-period 3600 \
  --window-size 100 \
  --window-stride 1 \
  --epochs 20 \
  --batch-size 50
```

**Important:** The example mirrors intended parameters, **not verified numerical reproducibility**. The legacy script had conflicting dataset construction and intermediate-file assumptions; the original sensor CSVs, trained weights and exact environment are not available in this repository. For large datasets, `--window-stride 1` is very RAM-intensive; use the default `100` for a more practical non-overlapping analysis.

## Pipeline

1. Select a **train-only normal segment**. Its next `--healthy-test-samples` normal observations form a held-out test prefix. Append the anomalous-source recording to that prefix. The selected healthy samples are excluded from training.
2. Apply a scalar Kalman filter independently to train/test (process variance 0.01; measurement variance 0.1).
3. Extract additive seasonal structure with `statsmodels.seasonal_decompose`. The two model features are **filtered z** and **seasonality**.
4. Fit `MinMaxScaler` only on train and transform test with the same scaler.
5. Create fixed-length windows (100 samples by default). `--window-stride 100` saves memory; `1` recreates the original dense overlapping-window strategy.
6. Train an LSTM autoencoder (16→8→RepeatVector→8→16 units, MAE loss, Adam) with chronologically ordered 20% validation windows.
7. Calculate mean absolute reconstruction error per window. Flag windows with error greater than the train-only threshold (`--threshold-quantile 1.0` corresponds to the original training maximum).

**Sampling-period caveat:** the original script used `period=60*60` and described it as one hour. `seasonal_decompose` interprets this as **3,600 samples**, equivalent to **9 seconds at 400 Hz**, not one hour. The refactor keeps 3,600 samples as a configurable historical value. Establish and justify the correct physical period before interpreting findings.

**Offline caveat:** `seasonal_decompose` uses complete train/test recordings independently. In particular, test features can depend on later test observations; this is **not a causal streaming detector**. A deployment-grade design should learn a train-only seasonal representation or switch to a causal feature extractor. Source overlap (`overlaps_anomalous_source`) is provenance, not verified per-sample ground truth.

## Outputs

The output directory contains:

| File | Contents |
| --- | --- |
| `model.keras` | Trained Keras autoencoder |
| `scaler.joblib` | Train-fitted feature scaler |
| `train_scores.csv` | Train windows: start sample, MAE, threshold, anomaly flag |
| `test_scores.csv` | Test windows plus anomalous-source overlap indicator |
| `loss.png` | Train/test loss plot with threshold |
| `run.json` | Parameters, threshold and aggregate counts |

Intermediate CSVs and model artifacts are generated on demand and excluded by `.gitignore`. Keep raw sensor files in `data/` (also ignored), not in Git.

## Repository layout

```text
src/anomaly_detector/
  preprocessing.py  # CSV loading, splitting, Kalman, seasonality, windows
  model.py          # Keras architecture
  scoring.py        # reconstruction MAE and train-only threshold
  cli.py            # orchestration, outputs and CLI
  __main__.py       # python -m anomaly_detector
tests/              # synthetic tests without TensorFlow
legacy/             # preserved original notebook-export script
.github/workflows/  # Python unit test workflow
pyproject.toml       # package and dependency metadata
Tesi (1).pdf         # original thesis document
```

## Limitations and research considerations

- No raw sensor data, trained artifacts or measured scores are shipped. Results **have not been reproduced** against the thesis data.
- The original study selected z-axis vibrations out of three measured axes (400 Hz sampling), used a Kalman filter and LSTM reconstruction MAE. Default parameters and dataset boundaries are tunable; changed segmentation/stride will change numerical results.
- Anomalous-source recordings may include normal intervals. Therefore `test_scores.csv` does **not** claim verified precision/recall or a per-sample ground-truth label.
- Test-time seasonal decomposition is offline, and the training-maximum threshold is sensitive to outliers. Validate a threshold on representative held-out normal data before any practical use.
- The legacy script is archived **without editing its behavior** to preserve the original academic record.
