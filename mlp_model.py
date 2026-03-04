# task2_mlp.py
# Task 2 (based on Week 2 MLP lab style):
# - Build classification dataset from Task 1 sequences: X (N,45), y (N,)
# - Train an MLP (PyTorch) with train/val split
# - Select learning rate using K-Fold CV (like the lab)
# - Save: (1) training loss curve, (2) validation accuracy curve, (3) confusion matrix

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------
# Indicators used in coursework
# -----------------------------
INDICATORS = [
    "GDPpc_2017$",
    "Population_total",
    "Life_exectancy",
    "Literacy_rate",
    "Unemploymlent_rate",
    "Energy_use",
    "Fertility_rate",
    "Poverty_ratio",
    "Primary_school_enrolmet_rate",
    "Exports_2017$",
]


# -----------------------------
# Preprocessing (same ideas as Task 1)
# -----------------------------
def resolve_csv_path(csv_path: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    p = Path(csv_path)
    candidates = [p, script_dir / p, script_dir / "data" / p.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(f"Could not find CSV '{csv_path}'. Checked:\n{checked}")


def load_data(csv_path: str) -> pd.DataFrame:
    resolved = resolve_csv_path(csv_path)
    print(f"Loading data from: {resolved}")
    df = pd.read_csv(resolved)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df = df[["country", "year"] + INDICATORS]
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # interpolation -> ffill/bfill -> global median fallback (per country)
    df = df.copy()
    global_medians = df[INDICATORS].median(numeric_only=True)

    out = []
    for country, g in df.groupby("country", sort=False):
        g = g.sort_values("year").copy()
        for col in INDICATORS:
            s = g[col].astype(float)
            s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
            s = s.fillna(global_medians[col])
            g[col] = s
        out.append(g)

    return pd.concat(out, ignore_index=True)


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    # StandardScaler without importing sklearn (same effect, simpler)
    df = df.copy()
    X = df[INDICATORS].astype(float).values
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    df[INDICATORS] = (X - mu) / sd
    return df


def make_task1_sequences(df_scaled: pd.DataFrame, window: int = 5, shift: int = 1):
    # Returns dict[country] = (n_seq, 50) flattened sequences
    seqs = {}
    for country, g in df_scaled.groupby("country", sort=False):
        g = g.sort_values("year").reset_index(drop=True)
        years = g["year"].to_numpy(dtype=int)
        vals = g[INDICATORS].to_numpy(dtype=np.float32)

        windows = []
        for i in range(0, len(g) - window + 1, shift):
            y = years[i : i + window]
            if not np.all(np.diff(y) == 1):
                continue
            windows.append(vals[i : i + window].reshape(-1))  # 5*10=50

        if windows:
            seqs[country] = np.asarray(windows, dtype=np.float32)

    return seqs


# -----------------------------
# Task 2 dataset (matches your shapes: X (219,45), y (219,))
# -----------------------------
def build_mlp_dataset(df_imputed_raw: pd.DataFrame, task1_sequences: dict):
    # one sample per country = mean over its 5-year sequences -> (50,)
    countries = [c for c, arr in task1_sequences.items() if arr.size > 0]
    agg = np.vstack([task1_sequences[c].mean(axis=0) for c in countries]).astype(np.float32)  # (N,50)

    # remove GDP from inputs to avoid leakage: drop positions 0,10,20,30,40
    n_feat = len(INDICATORS)              # 10
    gdp_idx = INDICATORS.index("GDPpc_2017$")  # 0
    keep_cols = [i for i in range(agg.shape[1]) if (i % n_feat) != gdp_idx]
    X = agg[:, keep_cols].astype(np.float32)  # (N,45)

    # labels: quartiles of log1p(mean GDPpc) (4 classes: 0..3)
    avg_gdp = df_imputed_raw.groupby("country")["GDPpc_2017$"].mean()
    avg_gdp = avg_gdp.loc[countries].astype(float)
    labels = pd.qcut(np.log1p(avg_gdp), q=4, labels=False, duplicates="drop")
    y = np.asarray([int(labels.loc[c]) for c in countries], dtype=np.int64)

    return X, y


# -----------------------------
# MLP model (same style as Week 2 lab)
# -----------------------------
class PyTorchMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # logits
        return x


# -----------------------------
# Training / Evaluation helpers
# -----------------------------
def train_model(model, X_train, y_train, X_val, y_val, lr=0.01, epochs=150):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loss_hist = []
    val_acc_hist = []

    for _ in range(epochs):
        model.train()

        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_hist.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred)
            val_acc_hist.append(val_acc)

    return train_loss_hist, val_acc_hist


def kfold_select_lr(X, y, learning_rates=(0.001, 0.01, 0.1), k=5, epochs=120, hidden=64):
    # Like Week 2 lab: pick best learning rate by mean validation accuracy
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_lr = learning_rates[0]
    best_acc = -1.0

    for lr in learning_rates:
        fold_acc = []
        for tr_idx, va_idx in kf.split(X):
            model = PyTorchMLP(input_size=X.shape[1], hidden_size=hidden, output_size=4)
            _, val_acc_hist = train_model(
                model,
                X[tr_idx],
                y[tr_idx],
                X[va_idx],
                y[va_idx],
                lr=lr,
                epochs=epochs,
            )
            fold_acc.append(val_acc_hist[-1])

        mean_acc = float(np.mean(fold_acc))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lr = lr

    return best_lr, best_acc


def save_loss_curve(loss_hist, path="task2_training_loss.png"):
    plt.figure()
    plt.plot(loss_hist)
    plt.title("Training Loss (Task 2 MLP)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_val_acc_curve(acc_hist, path="task2_validation_accuracy.png"):
    plt.figure()
    plt.plot(acc_hist)
    plt.title("Validation Accuracy (Task 2 MLP)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_confusion_matrix(cm, path="task2_confusion_matrix.png"):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Task 2 MLP)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main(csv_path: str = "data/world_bank_data_dev80-23++.csv"):
    np.random.seed(42)
    torch.manual_seed(42)

    # Build Task 2 dataset from Task 1 logic
    df = load_data(csv_path)
    df_imp = impute_missing(df)            # raw for labels
    df_scaled = standardise(df_imp)        # standardised for features

    seqs = make_task1_sequences(df_scaled, window=5, shift=1)
    X, y = build_mlp_dataset(df_imp, seqs)  # X (N,45), y (N,)

    # Train/Test split (final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Select learning rate using K-Fold on training set (like lab)
    best_lr, best_cv = kfold_select_lr(
        X_train, y_train, learning_rates=(0.001, 0.01, 0.1), k=5, epochs=120, hidden=64
    )
    print("Best LR:", best_lr, "Mean CV acc:", best_cv)

    # Train/Val split to record curves for report
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Final model training (record curves)
    model = PyTorchMLP(input_size=X.shape[1], hidden_size=64, output_size=4)
    train_loss_hist, val_acc_hist = train_model(
        model, X_tr, y_tr, X_val, y_val, lr=best_lr, epochs=150
    )

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    print("Test accuracy:", test_acc)
    print("Confusion matrix:\n", cm)

    # Save plots for report in plots/
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_loss_curve(train_loss_hist, str(plot_dir / "task2_training_loss.png"))
    save_val_acc_curve(val_acc_hist, str(plot_dir / "task2_validation_accuracy.png"))
    save_confusion_matrix(cm, str(plot_dir / "task2_confusion_matrix.png"))


if __name__ == "__main__":
    main()
