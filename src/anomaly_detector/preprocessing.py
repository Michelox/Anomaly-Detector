"""CSV loading, time-series preprocessing and non-overlapping dataset splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def read_z_axis(path: str | Path, *, header: bool = True) -> np.ndarray:
    """Read the third CSV column (sensor z axis) as a finite 1D signal."""
    frame = pd.read_csv(path, header=0 if header else None, usecols=[2])
    if frame.empty:
        raise ValueError(f"{path}: CSV contains no observations")
    values = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: z axis contains missing or infinite values")
    return values


def split_signals(
    normal: np.ndarray,
    anomalous: np.ndarray,
    *,
    train_start: int = 0,
    train_end: int | None = None,
    healthy_test_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep training normal-only; prepend held-out normal data to the test signal.

    The held-out portion starts at train_end, avoiding train/test overlap.
    Returned prefix length identifies the known-normal test prefix.
    """
    normal = np.asarray(normal, dtype=np.float64)
    anomalous = np.asarray(anomalous, dtype=np.float64)
    if normal.ndim != 1 or anomalous.ndim != 1 or not len(anomalous):
        raise ValueError("Expected non-empty 1D anomalous and normal signals")
    if healthy_test_samples < 0:
        raise ValueError("healthy_test_samples must be non-negative")
    if train_end is None:
        train_end = len(normal) - healthy_test_samples
    if not (0 <= train_start < train_end <= len(normal) - healthy_test_samples):
        raise ValueError("Training range must be non-empty and exclude held-out samples")
    train = normal[train_start:train_end]
    test = np.concatenate(
        (normal[train_end:train_end + healthy_test_samples], anomalous)
    )
    return train, test, healthy_test_samples


def kalman_smooth(
    values: np.ndarray, *, process_variance: float = 0.01,
    measurement_variance: float = 0.1,
) -> np.ndarray:
    """Apply the original thesis' scalar random-walk Kalman filter."""
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or len(observations) == 0:
        raise ValueError("Kalman input must be a non-empty 1D array")
    if process_variance < 0 or measurement_variance <= 0:
        raise ValueError("Kalman variances must satisfy Q >= 0 and R > 0")
    if not np.isfinite(observations).all():
        raise ValueError("Kalman input must contain finite values")

    output = np.empty_like(observations)
    estimate, covariance = float(observations[0]), 1.0
    for index, observation in enumerate(observations):
        predicted_covariance = covariance + process_variance
        gain = predicted_covariance / (predicted_covariance + measurement_variance)
        estimate += gain * (float(observation) - estimate)
        covariance = (1.0 - gain) * predicted_covariance
        output[index] = estimate
    return output


def extract_features(
    values: np.ndarray, *, period: int = 3600,
    process_variance: float = 0.01, measurement_variance: float = 0.1,
) -> np.ndarray:
    """Return filtered z values and offline additive seasonality as two features.

    'period' is expressed in samples, not seconds. Decomposition estimates
    seasonality from the entire supplied signal: this is an OFFLINE pipeline.
    """
    if period < 2 or len(values) < 2 * period:
        raise ValueError("Seasonal decomposition needs period >= 2 and >= 2 periods")
    smoothed = kalman_smooth(
        values,
        process_variance=process_variance,
        measurement_variance=measurement_variance,
    )
    seasonal = seasonal_decompose(
        smoothed, model="additive", period=period
    ).seasonal
    features = np.column_stack((smoothed, seasonal)).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Preprocessing produced non-finite features")
    return features


def make_windows(
    features: np.ndarray, *, size: int = 100, stride: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Return windows and their starting sample offsets.

    A non-overlapping stride is the default to avoid enormous copies when
    processing millions of samples. Use stride=1 for legacy dense windows.
    """
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("Window input must have shape (samples, features)")
    if size < 1 or stride < 1 or len(data) < size:
        raise ValueError("size/stride must be positive and input >= window size")
    starts = np.arange(0, len(data) - size + 1, stride, dtype=np.int64)
    windows = np.stack([data[start:start + size] for start in starts])
    return windows, starts
