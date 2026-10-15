# Vibration-Based Anomaly Detection with an LSTM Autoencoder

A research project investigating **anomaly detection in industrial conveyor-belt vibration signals**. It uses the z-axis readings from a vibration sensor and an LSTM autoencoder to identify unusual behavior through reconstruction error.

📄 **Research document:** [Full thesis — `Tesi (1).pdf`](Tesi%20(1).pdf)

> **About this README.** The research description and historical dataset details below are taken from the **original project README and the archived implementation**, alongside which the thesis PDF was published. The PDF itself could not be text-extracted in this editing session; therefore, this page does **not** claim to reproduce unverified numerical results or quotations from the PDF. For definitive scientific wording and findings, consult the document above. The current Python implementation is a *refactoring*, not an exact reproduction of the original experiment.

## 1. Research question and motivation

The central question is whether an autoencoder trained to reconstruct vibration patterns associated with ordinary conveyor-belt operation can **distinguish unusual signals from typical operation** without requiring a manually annotated anomaly for every sample.

A real-time alert for every deviation could produce false alarms and operational noise; a purely retrospective analysis offers less immediate support to operators. The study describes a **hybrid detection rationale**: characterize normal behavior, calculate reconstruction errors and use an error threshold to highlight potentially anomalous intervals for further investigation. An anomaly score is **not proof of a mechanical fault**.

The research combines signal processing with unsupervised/semi-supervised learning:

1. Select the most informative sensor axis.
2. Filter measurement noise.
3. Extract recurring patterns in the vibration signal.
4. Normalize model inputs.
5. Train a sequence autoencoder on normal-operation measurements.
6. Flag windows that have an unusually high reconstruction error.

## 2. Experimental data

According to the original repository description, measurements were collected using a **VTV-122** vibration sensor on a conveyor belt.

| Property | Recorded in the original project description |
| --- | --- |
| Measurements | Three axes: `x`, `y` and `z` |
| Sampling frequency | **400 samples per second** (400 Hz) |
| Normal-operation recording | **9,632,614 rows** |
| Anomalous-source recording | **531,370 rows** |
| Selected axis | **z**, identified as the most affected by vibration |
| Described data cleaning | Removal of peaks attributed to sensor setup issues |
| Described normal-data volume after peak removal | **3,100,000 rows** |

These figures describe the **historical research material**, not data files distributed here: the original measurements are **not committed to GitHub**. No script in the refactored package automatically reproduces the historical sensor-peak cleaning; users must provide appropriately prepared data.

### Historical train/test construction

The original README discusses mixing a limited number of observations from the two source recordings when creating train/test sets (including **1,000 observations**). The archived script, however, uses a different sequence of index slices and concatenations. Consequently, the textual account and historical implementation **cannot be assumed to define one identical split**.

The current program deliberately uses a clearly defined, non-overlapping split: it trains on a chosen normal-only segment and tests on held-out normal readings followed by the anomalous-source recording. **This differs from parts of the original written description.** It is intended to make data provenance and leakage risks explicit, not to retroactively claim the exact thesis setup.

## 3. Methodology

### 3.1 Signal filtering — Kalman filter

Vibration measurements include sensor noise. The original implementation applies a scalar Kalman filter to estimate a smoother signal. The historical code uses a process-noise variance of **`Q = 0.01`** and measurement-noise variance of **`R = 0.1`**. The refactored pipeline retains those defaults.

### 3.2 Seasonal decomposition

The study investigates the recurring component of the vibration time series by separating it into **trend, seasonality and residuals** through additive seasonal decomposition (`statsmodels.tsa.seasonal.seasonal_decompose`).

The original README emphasizes seasonality as particularly useful for separating normal and anomalous patterns. The archived model-building script passes **two columns**, filtered `z` and `Seasonality`, to the scaler and network; the current implementation follows that **two-feature implementation** rather than claiming it uses seasonality alone.

**Important unit distinction:** the historical code sets `period=3600`. The decomposition API interprets this value in **samples**, not seconds. At 400 Hz, **3,600 samples = 9 seconds**, not one hour. The actual physical vibration period should be checked against the experimental setup before drawing conclusions.

### 3.3 Normalization

`MinMaxScaler` scales each feature using **training data only**, then applies the same transformation to the test recording. Keeping the test recording out of scaler fitting avoids leaking its amplitude distribution into training.

### 3.4 Sequence model — LSTM autoencoder

An autoencoder learns to reproduce its own input; reconstruction error is therefore a measure of how well an input window matches patterns the model has learned.

The architecture retained from the historical Python script is:

```text
Input sequence (window_size × 2 features)
    ↓
LSTM(16, return_sequences=True)
    ↓
LSTM(8)
    ↓
RepeatVector(window_size)
    ↓
LSTM(8, return_sequences=True)
    ↓
LSTM(16, return_sequences=True)
    ↓
TimeDistributed(Dense(2))
    ↓
Reconstructed input sequence
```

The model uses the **Adam optimizer** and **mean absolute error (MAE)** reconstruction loss. Historical script settings include **100-sample windows**, **20 epochs** and **batch size 50**. They are configurable in the refactored version.

### 3.5 Anomaly score and decision threshold

For a window with `T` time steps and `F` features, reconstruction error is:

```text
MAE = (1 / (T × F)) × Σ_t Σ_f |x[t,f] − x_reconstructed[t,f]|
```

A window is flagged when its MAE **exceeds a threshold calibrated from the training errors**. The archived script uses the **maximum training MAE**. The refactored implementation preserves this as the default and allows an alternative training-loss quantile.

Threshold exceedance signals **unusual reconstruction behavior**, not a confirmed failure mode; meaningful evaluation requires independently established anomaly labels and a suitable test protocol.

## 4. Current repository: reproducible research code

The historical code is preserved unmodified at [`legacy/Anomaly detector.py`](legacy/Anomaly%20detector.py). A reorganized Python package makes the same broad research workflow easier to inspect, run and test.

```text
.
├── Tesi (1).pdf                 # Original thesis document
├── legacy/
│   └── Anomaly detector.py     # Preserved historical implementation
├── src/anomaly_detector/
│   ├── preprocessing.py        # CSV, train/test split, Kalman, seasonality, windows
│   ├── model.py                # LSTM autoencoder
│   ├── scoring.py              # Reconstruction MAE and threshold
│   ├── cli.py                  # End-to-end offline training and scoring
│   └── __main__.py
├── tests/                       # Unit tests with synthetic signals
├── .github/workflows/test.yml   # Automated checks
└── pyproject.toml               # Package configuration and dependencies
```

### Requirements and installation

Use **Python 3.10+** and a Python/TensorFlow-compatible environment.

```bash
git clone https://github.com/Michelox/Anomaly-Detector.git
cd Anomaly-Detector
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,test]"
python -m pytest -q
```

On Windows, activate the virtual environment with `.venv\Scripts\activate`.

### Input requirements

Provide two CSVs with at least three numeric columns (`x`, `y`, `z`), one representing normal operation and the other an anomalous-source recording. By default each CSV has a header; use `--no-header` when both files have none. **Only the third column (z-axis)** is loaded.

Place input files in `data/`, which is ignored by Git. The original study's raw CSVs are not distributed in this repository.

### Run the pipeline

```bash
python -m anomaly_detector \
  --normal-csv data/normal.csv \
  --anomalous-csv data/anomalous.csv \
  --output-dir outputs/run1
```

To explore parameters close to the historical script's **index range and dense-window settings**:

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

These indices are appropriate **only for a compatible, sufficiently long input CSV**. In particular, they do not automatically apply to an already cleaned 3.1-million-row file. Dense `--window-stride 1` windows can consume substantial memory; the current **default stride of 100** reduces that cost but changes the experimental sampling of windows.

### Saved outputs

| Artifact | Purpose |
| --- | --- |
| `model.keras` | Trained Keras model |
| `scaler.joblib` | Training-fitted feature scaler |
| `train_scores.csv` | Window offsets and training reconstruction loss |
| `test_scores.csv` | Test-window loss, threshold and flagged windows |
| `loss.png` | Training versus test reconstruction-error chart |
| `run.json` | Run parameters, threshold and basic counts |

Files are written beneath `--output-dir` and excluded from version control.

## 5. Interpretations, differences and limits

**What the research proposes:** sensor-data denoising and seasonal analysis, followed by an LSTM autoencoder whose reconstruction error can surface unusual conveyor-belt behavior.

**What the current code supports:** configurable *offline* model training and window-level scoring with train-only scaling and threshold selection, including lightweight synthetic tests. The new pipeline is **not guaranteed to reproduce the original thesis graphs or results** because the source datasets, environment and historical preprocessing details are not fully available.

Other important constraints:

- **No measured thesis performance is claimed here.** The PDF is the reference for experimental findings. Do not infer accuracy, precision, recall or fault-detection capability from the code or unit tests alone.
- **Not a causal, real-time detector.** Seasonal decomposition looks at a full supplied signal, including information from later test samples.
- **Source recording is not ground truth.** A window overlapping the anomalous-source recording need not correspond to a verified fault. `overlaps_anomalous_source` in exported scores is only a provenance indicator.
- **Historical preprocessing is incomplete in the repository.** Original spike-removal decisions and intermediate CSV artifacts are not supplied.
- **Methodological differences matter.** The historic account of train/test swapping, feature selection and the original script's data slicing should be reconciled with the full thesis before claiming scientific reproducibility.
- **The max-training-error threshold is sensitive to outliers.** A deployable alarm would require independent calibration and validation.

## Thesis and provenance

The original academic document is kept in this repository:

**[Open `Tesi (1).pdf`](Tesi%20(1).pdf)**

The research overview above follows the **original repository description and archived code**, while installation and implementation sections describe the modernized package. This explicit separation prevents engineering changes introduced during refactoring from being mistaken for conclusions established by the thesis.
