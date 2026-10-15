"""Small synthetic regression tests; no proprietary dataset or TensorFlow needed."""

import numpy as np
import pytest

from anomaly_detector.preprocessing import (
    extract_features,
    kalman_smooth,
    make_windows,
    read_z_axis,
    split_signals,
)
from anomaly_detector.scoring import reconstruction_mae, score_table, training_threshold


def test_read_z_axis_with_and_without_header(tmp_path):
    with_header = tmp_path / "header.csv"
    with_header.write_text("x,y,z\n1,2,3\n4,5,6\n", encoding="utf-8")
    no_header = tmp_path / "no-header.csv"
    no_header.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
    np.testing.assert_array_equal(read_z_axis(with_header), [3, 6])
    np.testing.assert_array_equal(read_z_axis(no_header, header=False), [3, 6])


def test_split_has_disjoint_healthy_prefix():
    train, test, prefix = split_signals(
        np.arange(20), np.array([99, 100]), train_start=4,
        train_end=15, healthy_test_samples=3,
    )
    np.testing.assert_array_equal(train, np.arange(4, 15))
    np.testing.assert_array_equal(test, [15, 16, 17, 99, 100])
    assert prefix == 3


def test_split_rejects_overlapping_normal_segments():
    with pytest.raises(ValueError, match="exclude held-out"):
        split_signals(np.arange(10), np.array([12]), train_end=9, healthy_test_samples=2)


def test_kalman_constant_input_is_constant():
    actual = kalman_smooth(np.full(30, 7.0))
    np.testing.assert_allclose(actual, 7.0)


def test_seasonality_and_window_offsets():
    signal = np.tile(np.array([1., 2., 4., 2.]), 15)
    features = extract_features(signal, period=4)
    assert features.shape == (60, 2)
    assert np.isfinite(features).all()
    windows, offsets = make_windows(features, size=6, stride=4)
    assert windows.shape == (14, 6, 2)
    np.testing.assert_array_equal(offsets[:3], [0, 4, 8])
    np.testing.assert_array_equal(windows[1, 0], features[4])


def test_too_short_for_decomposition_and_windows():
    with pytest.raises(ValueError, match="decomposition"):
        extract_features(np.arange(5), period=3)
    with pytest.raises(ValueError, match="window size"):
        make_windows(np.ones((5, 2)), size=6)


def test_scoring_uses_training_threshold_only():
    actual = np.array([[[1.0], [2.0]], [[1.0], [1.0]]])
    reconstruction = np.array([[[0.0], [2.0]], [[0.0], [0.0]]])
    errors = reconstruction_mae(actual, reconstruction)
    np.testing.assert_allclose(errors, [0.5, 1.0])
    threshold = training_threshold(errors)
    assert threshold == 1.0
    assert training_threshold(errors, quantile=0.5) == 0.75
    table = score_table(
        np.array([0, 3]), errors, threshold=0.75, window_size=2,
        known_normal_prefix=4,
    )
    assert table["is_anomaly"].tolist() == [False, True]
    assert table["overlaps_anomalous_source"].tolist() == [False, True]
