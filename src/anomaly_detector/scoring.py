"""Pure scoring functions so threshold behavior is independently testable."""

import numpy as np
import pandas as pd


def reconstruction_mae(actual: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """One mean absolute reconstruction error per time window."""
    left = np.asarray(actual)
    right = np.asarray(reconstructed)
    if left.ndim != 3 or left.shape != right.shape:
        raise ValueError("Actual and reconstructed windows must have identical 3D shapes")
    return np.mean(np.abs(left - right), axis=(1, 2))


def training_threshold(train_errors: np.ndarray, *, quantile: float = 1.0) -> float:
    """Calibrate only on training reconstruction errors (default: original maximum)."""
    errors = np.asarray(train_errors, dtype=float)
    if errors.ndim != 1 or len(errors) == 0 or not np.isfinite(errors).all():
        raise ValueError("Training errors must be a finite, non-empty 1D array")
    if not (0.0 < quantile <= 1.0):
        raise ValueError("Threshold quantile must be in (0, 1]")
    return float(np.quantile(errors, quantile))


def score_table(
    starts: np.ndarray, errors: np.ndarray, *, threshold: float,
    window_size: int, known_normal_prefix: int | None = None,
) -> pd.DataFrame:
    """Score windows; source overlap is provenance, not verified anomaly labels."""
    if len(starts) != len(errors) or window_size < 1:
        raise ValueError("Starts, errors and window size are inconsistent")
    frame = pd.DataFrame({
        "start_sample": starts,
        "loss_mae": errors,
        "threshold": threshold,
        "is_anomaly": np.asarray(errors) > threshold,
    })
    if known_normal_prefix is not None:
        frame["overlaps_anomalous_source"] = (
            np.asarray(starts) + window_size > known_normal_prefix
        )
    return frame
