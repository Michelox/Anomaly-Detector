# Anomaly Detector
### From industrial vibrations to anomaly detection with an LSTM Autoencoder

This project explores a question: **can we identify abnormal operating conditions in an industrial plant by learning to reconstruct its vibration signals?**

The workflow starts with signals from a laboratory testbed, reduces measurement noise, and extracts seasonality. An LSTM Autoencoder learns to reconstruct temporal sequences; reconstruction error then becomes the basis for detecting anomalies.

This README walks through the thesis **“Using Long Short-Term Memory Models for Anomaly Detection in Industrial Plants”** (original title: *Uso di modelli di Long Short-Term Memory per l’Anomaly Detection negli impianti industriali*), by **Michele Ciaramidaro**, Management Engineering degree programme, University of Messina, academic year **2023/2024**. Supervisor: **Prof. Dario Bruneo**.

**[Read the full thesis (Italian)](Tesi%20%281%29.pdf)** · [Walkthrough](#thesis-walkthrough) · [Results](#6-results-and-conclusions) · [Run the project](#running-the-project)

## Thesis walkthrough

### 1. The problem: recognizing a change in operation
*Chapter 1; §2.1–2.2*

In industrial monitoring, a change in a signal may indicate a change in operating conditions. Detecting it can help guide equipment inspections, although an anomaly does not necessarily imply a fault.

The study examines a **scaled replica of a plant for transporting and assembling automotive components**, consisting of two motors and six conveyor belts that move a trolley. The testbed supports several types of mechanical disturbance. In the experiment described, the conditions compared are **loaded and unloaded operation**.

Among the testbed's sensors, the analysis uses the vibration measurements attributed in the thesis to the **VTV-122**. The goal is to distinguish the behavior of the two recordings through their temporal characteristics.

### 2. From raw measurements to useful sequences
*Chapter 2*

#### Signal selection and dataset construction

The recordings contain three columns, corresponding to the `x`, `y`, and `z` axes, sampled at **400 Hz**.

| Experimental data | Value reported in the thesis |
| --- | ---: |
| Observations in the normal recording | 9,632,614 |
| Observations in the anomalous recording | 531,370 |
| Normal observations after excluding spikes | 3,100,000 |
| Selected axis | z |

The **z-axis** is retained because it shows the strongest response to vibration. Spikes in the normal recording, attributed to sensor setup errors, are excluded before constructing the datasets.

Section 2.3 describes the following exchange of observations:

- **Training set:** the cleaned normal series, with its last 1,000 observations replaced by the first 1,000 observations from the anomalous series.
- **Test set:** the anomalous series, with its first 1,000 observations replaced by the last 1,000 observations from the normal series.

The training set described in the thesis therefore also contains a small portion of the anomalous recording. Figures 1 and 2 show the initial signals and the resulting datasets, respectively.

#### Noise reduction with the Kalman filter

The Kalman filter recursively updates an estimate of the signal by combining prediction and measurement. It reduces acquisition noise before sequence analysis.

The code in §2.4.3 uses a scalar model with `A = H = 1`, process noise covariance `Q = 0.01`, and measurement noise covariance `R = 0.1`. Figure 3 shows the filtered signals.

#### Extracting seasonality

Decomposition separates the series into **trend, seasonality, and residuals**. Since the trends in the two operating conditions are similar, the study adds the seasonal component to the dataset, looking for recurring patterns that help distinguish their behavior.

The model therefore receives **two features: the filtered z-axis signal and seasonality**, consistent with the input shape reported in Chapter 6. Figure 4 illustrates the decomposition.

#### Normalization

Features are normalized with `MinMaxScaler` before entering the network. In the Python package, the scaler is fitted exclusively on the training set and then applied to the test set.

### 3. Why combine LSTMs and autoencoders?
*Chapters 3, 4, and 5*

Vibration measurements form **time series**: their order carries information. After introducing machine learning, the thesis moves from recurrent neural networks to **Long Short-Term Memory** networks, which use a memory state and information gates to model dependencies over time.

An **autoencoder** introduces a reconstruction task:

- The **encoder** compresses the sequence into an internal representation.
- The **decoder** uses that representation to reconstruct the original sequence.
- The difference between the input and its reconstruction measures how well the model represents that behavior.

Combining these approaches allows the network to learn a representation of vibration sequences. The detector's hypothesis is that sequences differing from the learned patterns will produce larger reconstruction errors.

### 4. Building and training the model
*§6.2–6.5*

The signal is divided into **100-sample windows**, equivalent to **0.25 seconds at 400 Hz**. The thesis code advances one sample at a time, producing overlapping windows. Each input has shape `(100, 2)`: 100 time steps and two features.

| Stage | Layer | Output shape per window |
| --- | --- | --- |
| Input | Vibration and seasonality sequence | `100 × 2` |
| Encoder | LSTM, 16 units, returns sequences | `100 × 16` |
| Compressed representation | LSTM, 8 units | `8` |
| Temporal expansion | RepeatVector, 100 repetitions | `100 × 8` |
| Decoder | LSTM, 8 units, returns sequences | `100 × 8` |
| Decoder | LSTM, 16 units, returns sequences | `100 × 16` |
| Reconstruction | TimeDistributed(Dense(2)) | `100 × 2` |

The LSTM layers use `tanh` activation. The output has the same shape as the input, allowing each sequence to be compared directly with its reconstruction.

| Parameter | Experiment described in the thesis |
| --- | ---: |
| Window length | 100 samples |
| Epochs | 20 |
| Batch size | 50 |
| Validation split | 25% |
| Reconstruction loss | Mean Absolute Error (MAE) |

The package compiles the model with **Adam** and **MAE** loss. Figure 11 in the thesis shows training and validation loss, helping track learning and any divergence between the two curves.

### 5. Turning reconstruction error into a decision
*§6.1, §6.6–6.7*

After training, the model reconstructs the training and test sequences. The detector measures the mean absolute error between the observed and reconstructed values.

For a window with `T` time steps and `F` features, the score used in the package is:

```math
\mathrm{MAE}(X,\hat X)=\frac{1}{TF}\sum_{t=1}^{T}\sum_{f=1}^{F}|X_{t,f}-\hat X_{t,f}|
```

The thesis sets the **threshold to the maximum reconstruction error on the training set**. A sequence is flagged as anomalous when its error exceeds this threshold.

**Figure 12** shows the training loss distribution; **Figure 13** compares the training and test distributions against the threshold. This step connects the network's learned representation to the final detection decision.

### 6. Results and conclusions
*Chapter 7, Figure 14*

The thesis reports the following results on the experimental test set:

| Metric | Reported result |
| --- | ---: |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-score | 1.00 |

The conclusions also report **100% accuracy**. The evaluation treats the test series as anomalous except for the first 1,000 observations, which originate from the normal series.

Under the described protocol, these results indicate complete separation of the evaluated classes. The study shows how **signal cleaning, seasonality, and LSTM reconstruction** can work together to distinguish operating conditions.

These results apply to the experimental conditions and test set described in the thesis. Assessing generalization requires new recordings and an independent test protocol.

## Repository guide

| Resource | Contents |
| --- | --- |
| [Full thesis](Tesi%20%281%29.pdf) | Scientific background, methodology, figures, and results |
| [Preprocessing](src/anomaly_detector/preprocessing.py) | CSV loading, splitting, Kalman filtering, seasonality, and windowing |
| [Model](src/anomaly_detector/model.py) | LSTM Autoencoder architecture |
| [Scoring](src/anomaly_detector/scoring.py) | MAE, threshold, and window classification |
| [CLI](src/anomaly_detector/cli.py) | Training, scoring, and output saving |
| [Tests](tests/test_preprocessing.py) | Checks using synthetic data |
| [Configuration](pyproject.toml) | Dependencies and installation |
| [Experiment script](legacy/Anomaly%20detector.py) | Implementation preserved for reference |

## Running the project

### Installation

You need **Python 3.10+** and a TensorFlow-compatible environment.

```bash
git clone https://github.com/Michelox/Anomaly-Detector.git
cd Anomaly-Detector
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,test]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Preparing the inputs

The experimental CSV files are not included in the repository. Supply your own data, with acquisition errors already cleaned.

Provide two CSV files: one containing normal measurements and one containing the anomalous recording. The program reads the **third column**, corresponding to the z-axis, which must contain finite numeric values. A header is expected by default; add `--no-header` if both files lack one.

For example, place the files at `data/normal.csv` and `data/anomalous.csv`. With the default seasonal period, each series produced by the split must contain at least **7,200 samples** for seasonal decomposition. The program also reserves 1,000 normal samples for testing. The period is measured in samples: 3,600 samples correspond to 9 seconds at 400 Hz. The pipeline operates offline because seasonal decomposition uses the entire supplied series.

### Training and detection

```bash
python -m anomaly_detector \
  --normal-csv data/normal.csv \
  --anomalous-csv data/anomalous.csv \
  --output-dir outputs/run1
```

Defaults include 100-sample windows, a stride of 100, 20 epochs, and a batch size of 50. To use overlapping windows as in the thesis code, add `--window-stride 1`. With millions of observations, this substantially increases memory use and computation time.

To inspect all options and run the tests:

```bash
python -m anomaly_detector --help
python -m pytest -q
```

### Outputs

| File | Contents |
| --- | --- |
| `model.keras` | Trained model |
| `scaler.joblib` | Scaler fitted on the training data |
| `train_scores.csv` | Window positions and training loss |
| `test_scores.csv` | Loss, threshold, and anomaly flags |
| `loss.png` | Reconstruction errors and threshold plot |
| `run.json` | Run parameters, threshold, and counts |

In `test_scores.csv`, `is_anomaly` represents the model's decision. `overlaps_anomalous_source` only indicates whether a window intersects the anomalous recording: it describes data provenance, not a verified fault label.

---

**Reference:** Michele Ciaramidaro, *Uso di modelli di Long Short-Term Memory per l’Anomaly Detection negli impianti industriali*, University of Messina, academic year 2023/2024. See the [full thesis (Italian)](Tesi%20%281%29.pdf) for the theoretical development and bibliography.
