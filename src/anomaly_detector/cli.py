"""Command-line workflow for reproducible OFFLINE anomaly scoring."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .model import build_autoencoder
from .preprocessing import extract_features, make_windows, read_z_axis, split_signals
from .scoring import reconstruction_mae, score_table, training_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline LSTM-autoencoder anomaly detection on z-axis vibration CSVs"
    )
    parser.add_argument("--normal-csv", type=Path, required=True)
    parser.add_argument("--anomalous-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-header", action="store_true", help="CSV has no header row")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=None, help="Exclusive row index")
    parser.add_argument("--healthy-test-samples", type=int, default=1000)
    parser.add_argument(
        "--seasonal-period", type=int, default=3600,
        help="Seasonal period IN SAMPLES (3600 samples = 9 seconds at 400 Hz)",
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument(
        "--window-stride", type=int, default=100,
        help="100 gives non-overlapping windows; 1 reproduces dense legacy windows",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--threshold-quantile", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10)
    return parser


def predict_errors(model, windows: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Bound inference memory by scoring one batch of windows at a time."""
    errors = []
    for offset in range(0, len(windows), batch_size):
        actual = windows[offset:offset + batch_size]
        predicted = model.predict(actual, batch_size=batch_size, verbose=0)
        errors.append(reconstruction_mae(actual, predicted))
    return np.concatenate(errors)


def save_plot(train_errors: np.ndarray, test_errors: np.ndarray,
              threshold: float, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, values in (("Train", train_errors), ("Test", test_errors)):
        step = max(1, len(values) // 5000)
        ax.plot(np.arange(0, len(values), step), values[::step], label=label, alpha=0.8)
    ax.axhline(threshold, linestyle="--", color="black", label="Threshold")
    ax.set(xlabel="Window index within each split", ylabel="Reconstruction MAE")
    ax.set_title("Offline LSTM autoencoder reconstruction loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    normal = read_z_axis(args.normal_csv, header=not args.no_header)
    anomalous = read_z_axis(args.anomalous_csv, header=not args.no_header)
    train, test, healthy_prefix = split_signals(
        normal, anomalous, train_start=args.train_start, train_end=args.train_end,
        healthy_test_samples=args.healthy_test_samples,
    )
    train_features = extract_features(train, period=args.seasonal_period)
    test_features = extract_features(test, period=args.seasonal_period)

    # Only training data may influence feature normalization.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = scaler.transform(test_features).astype(np.float32)
    train_windows, train_starts = make_windows(
        train_scaled, size=args.window_size, stride=args.window_stride
    )
    test_windows, test_starts = make_windows(
        test_scaled, size=args.window_size, stride=args.window_stride
    )
    if len(train_windows) < 5:
        raise ValueError("Need at least five training windows for validation")

    model = build_autoencoder(args.window_size, train_windows.shape[-1])
    # Training windows retain temporal ordering; validation uses the latest 20%.
    model.fit(
        train_windows, train_windows, epochs=args.epochs, batch_size=args.batch_size,
        validation_split=0.2, shuffle=False, verbose=2,
    )
    train_errors = predict_errors(model, train_windows, batch_size=args.batch_size)
    test_errors = predict_errors(model, test_windows, batch_size=args.batch_size)
    threshold = training_threshold(train_errors, quantile=args.threshold_quantile)
    train_scores = score_table(
        train_starts, train_errors, threshold=threshold, window_size=args.window_size
    )
    test_scores = score_table(
        test_starts, test_errors, threshold=threshold, window_size=args.window_size,
        known_normal_prefix=healthy_prefix,
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_scores.to_csv(output / "train_scores.csv", index=False)
    test_scores.to_csv(output / "test_scores.csv", index=False)
    model.save(output / "model.keras")
    joblib.dump(scaler, output / "scaler.joblib")
    save_plot(train_errors, test_errors, threshold, output / "loss.png")
    settings = {
        "normal_csv": str(args.normal_csv), "anomalous_csv": str(args.anomalous_csv),
        "train_start": args.train_start, "train_end": args.train_end,
        "healthy_test_samples": healthy_prefix, "seasonal_period_samples": args.seasonal_period,
        "window_size": args.window_size, "window_stride": args.window_stride,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "threshold_quantile": args.threshold_quantile, "threshold_mae": threshold,
        "seed": args.seed, "train_windows": len(train_windows),
        "test_windows": len(test_windows),
        "test_flagged_windows": int(test_scores["is_anomaly"].sum()),
    }
    (output / "run.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {output}; threshold={threshold:.6f}; "
          f"test anomalies={settings['test_flagged_windows']}/{len(test_windows)}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
