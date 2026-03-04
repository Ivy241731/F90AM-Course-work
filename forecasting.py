"""
Task 3/4 entry point.

Task 3: build 10->5 forecasting pairs and train forecasting models.
Task 4: prepare flattened 105-dim vectors for VAE training.

Usage:
    python forecasting.py
    python forecasting.py <csv_path>
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np

from forecasting_models import run_forecasting
from preprocessing import (
    create_forecasting_pairs,
    impute_missing,
    load_data,
    scale_features,
    split_forecasting_data,
)


def build_vae_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Flatten input window (10x10 -> 100) and append 5-step GDP targets.
    return np.concatenate([X.reshape(X.shape[0], -1), y], axis=1).astype(np.float32)


def maybe_run_vae(vae_train: np.ndarray) -> Optional[object]:
    try:
        from vae_model import run_vae  # type: ignore
    except (ImportError, AttributeError):
        print(
            "Task 4 placeholder: VAE training function `run_vae(...)` not found in vae_model.py."
        )
        print(f"Prepared VAE training matrix shape: {vae_train.shape}")
        return None
    return run_vae(vae_train)


def main(
    csv_path: str = "data/world_bank_data_dev80-23++.csv",
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    skip_task3: bool = False,
    skip_task4: bool = False,
) -> None:
    df = load_data(csv_path)
    print(f"Loaded shape: {df.shape}")

    df_imputed, missing_before, missing_after = impute_missing(df)
    print(f"Missing values: {missing_before} -> {missing_after} after imputation")

    df_scaled, _ = scale_features(df_imputed)
    X_fc, y_fc, countries_fc, starts_fc = create_forecasting_pairs(
        df_scaled, input_window=10, output_window=5
    )
    splits = split_forecasting_data(
        X_fc, y_fc, countries_fc, starts_fc, train_ratio=0.7, val_ratio=0.15
    )
    print(
        "Forecast splits: "
        f"train {splits['X_train'].shape}, "
        f"val {splits['X_val'].shape}, "
        f"test {splits['X_test'].shape}"
    )

    if not skip_task3:
        metrics = run_forecasting(
            train_data=(splits["X_train"], splits["y_train"]),
            val_data=(splits["X_val"], splits["y_val"]),
            test_data=(splits["X_test"], splits["y_test"]),
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        print("Task 3 metrics:")
        for model_name, model_metrics in metrics.items():
            print(
                f"- {model_name}: "
                f"MSE={model_metrics['mse']:.6f}, "
                f"MAE={model_metrics['mae']:.6f}, "
                f"MAPE={model_metrics['mape']:.2f}%"
            )

    if not skip_task4:
        # Task 4 (data preparation + optional model training)
        vae_train = build_vae_matrix(splits["X_train"], splits["y_train"])
        print(
            "Task 4 VAE matrix shape: "
            f"{vae_train.shape} (expected second dimension: 10*10 + 5 = 105)"
        )
        maybe_run_vae(vae_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 3/4 forecasting pipeline.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/world_bank_data_dev80-23++.csv",
        help="Path to CSV data file",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for Task 3 models")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Task 3 models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Task 3 models")
    parser.add_argument("--skip-task3", action="store_true", help="Skip Task 3 model training")
    parser.add_argument("--skip-task4", action="store_true", help="Skip Task 4 VAE step")
    args = parser.parse_args(sys.argv[1:])

    main(
        csv_path=args.csv_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        skip_task3=args.skip_task3,
        skip_task4=args.skip_task4,
    )
