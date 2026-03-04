from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _plot_dir() -> Path:
    out = Path("plots")
    out.mkdir(parents=True, exist_ok=True)
    return out


#########################################
# LSTM Forecasting Model
#########################################


class LSTMForecast(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


#########################################
# CNN-LSTM Forecasting Model
#########################################


class CNNLSTMForecast(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        conv_channels: int = 32,
        hidden_size: int = 64,
        output_size: int = 5,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(conv_channels, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, time, features) -> (batch, features, time) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        # Back to (batch, time, channels) for LSTM
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


#########################################
# Transformer Forecasting Model
#########################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerForecast(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


#########################################
# Training / Evaluation / Metrics
#########################################


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str | None = None,
) -> Tuple[list[float], list[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train += float(loss.item())

        train_loss = total_train / max(1, len(train_loader))
        train_losses.append(train_loss)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val += float(loss.item())

        val_loss = total_val / max(1, len(val_loader))
        val_losses.append(val_loss)
        print(
            f"{model.__class__.__name__} | Epoch {epoch:03d}/{epochs} "
            f"| Train {train_loss:.6f} | Val {val_loss:.6f}"
        )

    return train_losses, val_losses


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: str | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())

    return np.vstack(preds), np.vstack(trues)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    denom = np.clip(np.abs(y_true), 1e-6, None)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    return {"mse": mse, "mae": mae, "mape": mape}


#########################################
# Plotting
#########################################


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    model_name: str,
    plot_prefix: str = "task3",
) -> None:
    out = _plot_dir()
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name.upper()} Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{plot_prefix}_{model_name}_training_curve.png", dpi=200)
    plt.close()


def plot_forecast(
    true: np.ndarray,
    pred: np.ndarray,
    model_name: str,
    plot_prefix: str = "task3",
) -> None:
    out = _plot_dir()
    plt.figure()
    horizon = np.arange(true.shape[1])
    plt.plot(horizon, true[0], marker="o", label="True")
    plt.plot(horizon, pred[0], marker="o", label="Predicted")
    plt.title(f"{model_name.upper()} Forecast Example (One Test Sample)")
    plt.xlabel("Forecast Horizon (Years Ahead)")
    plt.ylabel("Scaled GDP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{plot_prefix}_{model_name}_forecast.png", dpi=200)
    plt.close()


#########################################
# Run Task 3 Forecasting Models
#########################################


def run_forecasting(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    plot_prefix: str = "task3",
    collect_predictions: bool = False,
) -> Dict[str, Dict[str, float]] | Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_size = int(X_train.shape[2])
    output_size = int(y_train.shape[1])

    models: Dict[str, nn.Module] = {
        "lstm": LSTMForecast(input_size=input_size, output_size=output_size),
        "cnn_lstm": CNNLSTMForecast(input_size=input_size, output_size=output_size),
        "transformer": TransformerForecast(input_size=input_size, output_size=output_size),
    }

    all_metrics: Dict[str, Dict[str, float]] = {}
    artifacts: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name, model in models.items():
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
        )
        plot_losses(train_losses, val_losses, model_name, plot_prefix=plot_prefix)

        preds, trues = evaluate_model(model, test_loader)
        plot_forecast(trues, preds, model_name, plot_prefix=plot_prefix)
        artifacts[model_name] = {"preds": preds, "trues": trues}

        metrics = compute_metrics(trues, preds)
        all_metrics[model_name] = metrics
        print(
            f"{model_name.upper()} test metrics | "
            f"MSE: {metrics['mse']:.6f}, "
            f"MAE: {metrics['mae']:.6f}, "
            f"MAPE: {metrics['mape']:.2f}%"
        )

    if collect_predictions:
        return all_metrics, artifacts
    return all_metrics


if __name__ == "__main__":
    import argparse

    from preprocessing import (
        create_forecasting_pairs,
        impute_missing,
        load_data,
        scale_features,
        split_forecasting_data,
    )

    parser = argparse.ArgumentParser(description="Run Task 3 models directly from forecasting_models.py")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/world_bank_data_dev80-23++.csv",
        help="Path to CSV data file",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    df = load_data(args.csv_path)
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

    metrics = run_forecasting(
        train_data=(splits["X_train"], splits["y_train"]),
        val_data=(splits["X_val"], splits["y_val"]),
        test_data=(splits["X_test"], splits["y_test"]),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print("Task 3 metrics:")
    for model_name, model_metrics in metrics.items():
        print(
            f"- {model_name}: "
            f"MSE={model_metrics['mse']:.6f}, "
            f"MAE={model_metrics['mae']:.6f}, "
            f"MAPE={model_metrics['mape']:.2f}%"
        )
    print("Saved individual plots:")
    print("- plots/task3_lstm_training_curve.png")
    print("- plots/task3_lstm_forecast.png")
    print("- plots/task3_cnn_lstm_training_curve.png")
    print("- plots/task3_cnn_lstm_forecast.png")
    print("- plots/task3_transformer_training_curve.png")
    print("- plots/task3_transformer_forecast.png")
    print("Note: Task 4 VAE step is in forecasting.py.")
