from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import (
    create_forecasting_pairs,
    impute_missing,
    load_data,
    scale_features,
    split_forecasting_data,
)


def build_vae_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Flatten 10x10 inputs (100 values) and append 5-step GDP targets.
    return np.concatenate([X.reshape(X.shape[0], -1), y], axis=1).astype(np.float32)


class VAE(nn.Module):
    def __init__(self, input_dim: int = 105, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = recon_loss + beta * kl
    return elbo, recon_loss, kl


def _plot_dir() -> Path:
    out = Path("plots")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _model_dir() -> Path:
    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    return out


def train_vae(
    vae: VAE,
    X_train: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str | None = None,
) -> Dict[str, list[float]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history: Dict[str, list[float]] = {
        "elbo": [],
        "recon": [],
        "kl": [],
    }

    for epoch in range(1, epochs + 1):
        vae.train()
        total_elbo = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(xb)
            elbo, recon_loss, kl = vae_loss(xb, recon, mu, logvar, beta=beta)
            elbo.backward()
            optimizer.step()
            total_elbo += float(elbo.item())
            total_recon += float(recon_loss.item())
            total_kl += float(kl.item())

        n_batches = max(1, len(loader))
        epoch_elbo = total_elbo / n_batches
        epoch_recon = total_recon / n_batches
        epoch_kl = total_kl / n_batches

        history["elbo"].append(epoch_elbo)
        history["recon"].append(epoch_recon)
        history["kl"].append(epoch_kl)

        print(
            f"VAE | Epoch {epoch:03d}/{epochs} | "
            f"ELBO {epoch_elbo:.6f} | Recon {epoch_recon:.6f} | KL {epoch_kl:.6f}"
        )

    return history


def encode_latent(vae: VAE, X: np.ndarray, device: str | None = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(device)
    vae.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        mu, _ = vae.encode(x_t)
    return mu.cpu().numpy()


def plot_vae_losses(history: Dict[str, list[float]]) -> None:
    out = _plot_dir()
    plt.figure()
    plt.plot(history["elbo"], label="ELBO")
    plt.plot(history["recon"], label="Reconstruction")
    plt.plot(history["kl"], label="KL divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task 4 VAE Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "task4_vae_losses.png", dpi=200)
    plt.close()


def plot_tsne_3d(
    latents: np.ndarray,
    path: Path,
    max_samples: int = 2000,
    random_state: int = 42,
) -> None:
    n = latents.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        data = latents[idx]
    else:
        data = latents

    perplexity = min(30, max(5, (data.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    reduced = tsne.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], s=8, alpha=0.75)
    ax.set_title("Task 4 Latent Space (t-SNE 3D)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def generate_synthetic(vae: VAE, n_samples: int, latent_dim: int, device: str | None = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(device)
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        x_hat = vae.decode(z)
    return x_hat.cpu().numpy().astype(np.float32)


def run_vae(
    vae_train: np.ndarray,
    latent_dim: int = 16,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 1.0,
) -> Dict[str, object]:
    input_dim = int(vae_train.shape[1])
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    history = train_vae(
        vae=vae,
        X_train=vae_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
    )

    plot_vae_losses(history)

    latents = encode_latent(vae, vae_train)
    plot_tsne_3d(latents, _plot_dir() / "task4_vae_latent_tsne3d.png")

    model_path = _model_dir() / "task4_vae.pt"
    torch.save(
        {
            "model_state_dict": vae.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
        },
        model_path,
    )
    print(f"Saved VAE model: {model_path}")
    print("Saved plots: plots/task4_vae_losses.png, plots/task4_vae_latent_tsne3d.png")

    return {
        "history": history,
        "model_path": str(model_path),
        "latent_shape": latents.shape,
    }


def main(
    csv_path: str = "data/world_bank_data_dev80-23++.csv",
    latent_dim: int = 16,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 1.0,
) -> None:
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

    vae_train = build_vae_matrix(splits["X_train"], splits["y_train"])
    print(
        "Task 4 VAE matrix shape: "
        f"{vae_train.shape} (expected second dimension: 10*10 + 5 = 105)"
    )

    run_vae(
        vae_train=vae_train,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 4 VAE training.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/world_bank_data_dev80-23++.csv",
        help="Path to CSV data file",
    )
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent vector dimension")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight in ELBO")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
    )
