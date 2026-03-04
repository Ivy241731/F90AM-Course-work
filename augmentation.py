"""
Task 5: Compare forecasting with and without VAE-based augmentation.

Usage:
    python augmentation.py
    python augmentation.py --epochs 20 --n-synth 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from forecasting_models import run_forecasting
from preprocessing import (
    create_forecasting_pairs,
    impute_missing,
    load_data,
    scale_features,
    split_forecasting_data,
)
from vae_model import VAE, generate_synthetic


def _plot_dir() -> Path:
    out = Path("plots")
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_samples(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Input is flattened [10*10 + 5] = 105 dimensions.
    X = samples[:, :100].reshape(-1, 10, 10).astype(np.float32)
    y = samples[:, 100:105].astype(np.float32)
    return X, y


def load_vae_from_checkpoint(ckpt_path: str) -> tuple[VAE, int]:
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found: {p}\n"
            "Run Task 4 first, e.g. `python vae_model.py --epochs 80`."
        )

    ckpt = torch.load(p, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    latent_dim = int(ckpt["latent_dim"])
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, latent_dim


def augment_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    vae: VAE,
    latent_dim: int,
    n_synth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    synth_flat = generate_synthetic(vae, n_samples=n_synth, latent_dim=latent_dim)
    X_fake, y_fake = split_samples(synth_flat)
    X_aug = np.concatenate([X_train, X_fake], axis=0).astype(np.float32)
    y_aug = np.concatenate([y_train, y_fake], axis=0).astype(np.float32)
    return X_aug, y_aug


def save_metric_table(
    base_metrics: Dict[str, Dict[str, float]],
    aug_metrics: Dict[str, Dict[str, float]],
    out_csv: Path,
) -> None:
    lines = [
        "model,scenario,mse,mae,mape",
    ]
    for model_name in sorted(base_metrics.keys()):
        b = base_metrics[model_name]
        a = aug_metrics[model_name]
        lines.append(f"{model_name},original,{b['mse']:.8f},{b['mae']:.8f},{b['mape']:.8f}")
        lines.append(f"{model_name},augmented,{a['mse']:.8f},{a['mae']:.8f},{a['mape']:.8f}")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_metric_comparison(
    base_metrics: Dict[str, Dict[str, float]],
    aug_metrics: Dict[str, Dict[str, float]],
) -> None:
    print("\nTask 5 comparison (lower is better):")
    header = f"{'Model':<12}{'Metric':<8}{'Original':>12}{'Augmented':>12}{'Delta':>12}"
    print(header)
    print("-" * len(header))
    for model_name in sorted(base_metrics.keys()):
        for metric in ("mse", "mae", "mape"):
            b = base_metrics[model_name][metric]
            a = aug_metrics[model_name][metric]
            d = a - b
            print(f"{model_name:<12}{metric.upper():<8}{b:>12.6f}{a:>12.6f}{d:>12.6f}")


def plot_combined_forecast(
    model_name: str,
    true: np.ndarray,
    pred_original: np.ndarray,
    pred_augmented: np.ndarray,
    path: Path,
) -> None:
    # Plot one sample to match assignment requirement for combined comparison plots.
    horizon = np.arange(true.shape[1])
    plt.figure()
    plt.plot(horizon, true[0], marker="o", label="True")
    plt.plot(horizon, pred_original[0], marker="o", label="Original Train")
    plt.plot(horizon, pred_augmented[0], marker="o", label="Augmented Train")
    plt.xlabel("Forecast Horizon (Years Ahead)")
    plt.ylabel("Scaled GDP")
    plt.title(f"TASK5 {model_name.upper()} | Original vs Augmented")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def run_augmentation(
    csv_path: str = "data/world_bank_data_dev80-23++.csv",
    vae_checkpoint: str = "models/task4_vae.pt",
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    n_synth: int | None = None,
    seed: int = 42,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = load_data(csv_path)
    df_imputed, missing_before, missing_after = impute_missing(df)
    print(f"Loaded shape: {df.shape}")
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

    if n_synth is None:
        n_synth = int(splits["X_train"].shape[0])

    vae, latent_dim = load_vae_from_checkpoint(vae_checkpoint)
    print(f"Loaded VAE checkpoint: {vae_checkpoint} (latent_dim={latent_dim})")

    print("\nTraining on original data...")
    base_metrics, base_artifacts = run_forecasting(
        train_data=(splits["X_train"], splits["y_train"]),
        val_data=(splits["X_val"], splits["y_val"]),
        test_data=(splits["X_test"], splits["y_test"]),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        plot_prefix="task5_original",
        collect_predictions=True,
    )

    print("\nGenerating synthetic samples and training on augmented data...")
    X_aug, y_aug = augment_training_set(
        splits["X_train"], splits["y_train"], vae=vae, latent_dim=latent_dim, n_synth=n_synth
    )
    print(f"Augmented train shape: X {X_aug.shape}, y {y_aug.shape}")

    aug_metrics, aug_artifacts = run_forecasting(
        train_data=(X_aug, y_aug),
        val_data=(splits["X_val"], splits["y_val"]),
        test_data=(splits["X_test"], splits["y_test"]),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        plot_prefix="task5_augmented",
        collect_predictions=True,
    )

    print_metric_comparison(base_metrics, aug_metrics)

    out_plot_dir = _plot_dir()
    for model_name in sorted(base_metrics.keys()):
        true = base_artifacts[model_name]["trues"]
        pred_original = base_artifacts[model_name]["preds"]
        pred_augmented = aug_artifacts[model_name]["preds"]
        plot_combined_forecast(
            model_name=model_name,
            true=true,
            pred_original=pred_original,
            pred_augmented=pred_augmented,
            path=out_plot_dir / f"task5_{model_name}_comparison.png",
        )

    out_csv = out_plot_dir / "task5_metrics_comparison.csv"
    save_metric_table(base_metrics, aug_metrics, out_csv)
    print(f"\nSaved comparison table: {out_csv}")
    print("Saved combined plots: plots/task5_*_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 5 augmentation experiment.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/world_bank_data_dev80-23++.csv",
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--vae-checkpoint",
        default="models/task4_vae.pt",
        help="Path to trained VAE checkpoint from Task 4",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for forecasting models")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for forecasting models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for forecasting models")
    parser.add_argument(
        "--n-synth",
        type=int,
        default=None,
        help="Number of synthetic samples to generate (default: same as train size)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_augmentation(
        csv_path=args.csv_path,
        vae_checkpoint=args.vae_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_synth=args.n_synth,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
