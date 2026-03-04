from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

PLOT_COUNTRIES = ["United States", "China", "Russia", "Brazil"]
PLOT_LAYOUT = (2, 5)  # rows, cols (opposite of 5x2)


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
    df = pd.read_csv(resolved)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df = df[["country", "date", "year"] + INDICATORS]
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    return df


def impute_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    df = df.copy()
    missing_before = int(df[INDICATORS].isna().sum().sum())
    global_medians = df[INDICATORS].median(numeric_only=True)

    out = []
    for country, group in df.groupby("country", sort=False):
        g = group.sort_values("year").copy()
        for col in INDICATORS:
            s = g[col].astype(float)
            s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
            if s.isna().any():
                s = s.fillna(global_medians[col])
            g[col] = s
        g["country"] = country
        out.append(g)

    df_imputed = pd.concat(out, ignore_index=True)
    df_imputed = df_imputed[["country", "date", "year"] + INDICATORS]
    missing_after = int(df_imputed[INDICATORS].isna().sum().sum())
    return df_imputed, missing_before, missing_after


def save_country_indicator_plots(
    df_raw: pd.DataFrame,
    df_imputed: pd.DataFrame,
    out_dir: str = "plots",
    layout: Tuple[int, int] = PLOT_LAYOUT,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    alias_map = {"USA": "United States", "Russia": "Russian Federation"}

    for country in PLOT_COUNTRIES:
        target = alias_map.get(country, country)
        raw_cdf = df_raw[df_raw["country"] == target].sort_values("year").reset_index(drop=True)
        imp_cdf = (
            df_imputed[df_imputed["country"] == target]
            .sort_values("year")
            .reset_index(drop=True)
        )
        if raw_cdf.empty or imp_cdf.empty:
            continue

        n_rows, n_cols = layout
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.8 * n_cols, 3.8 * n_rows),
            sharex=True,
        )
        axes = axes.flatten()
        if len(axes) < len(INDICATORS):
            raise ValueError(
                f"Layout {layout} has {len(axes)} slots, but {len(INDICATORS)} indicators are required."
            )
        for i, indicator in enumerate(INDICATORS):
            ax = axes[i]
            raw_y = pd.to_numeric(raw_cdf[indicator], errors="coerce")
            imp_y = pd.to_numeric(imp_cdf[indicator], errors="coerce")
            years = imp_cdf["year"]

            raw_mask = raw_y.notna()
            missing_mask = ~raw_mask

            if raw_mask.any():
                raw_count = int(raw_mask.sum())
                if raw_count == 1:
                    ax.scatter(
                        years[raw_mask],
                        raw_y[raw_mask],
                        s=36,
                        color="tab:blue",
                        label="Reported",
                        zorder=3,
                    )
                else:
                    ax.plot(
                        years[raw_mask],
                        raw_y[raw_mask],
                        linewidth=1.6,
                        color="tab:blue",
                        label="Reported",
                    )

                if missing_mask.any():
                    missing_count = int(missing_mask.sum())
                    if missing_count == 1:
                        ax.scatter(
                            years[missing_mask],
                            imp_y[missing_mask],
                            s=28,
                            color="tab:orange",
                            label="Imputed",
                            zorder=2,
                        )
                    else:
                        ax.plot(
                            years[missing_mask],
                            imp_y[missing_mask],
                            linewidth=1.3,
                            linestyle="--",
                            color="tab:orange",
                            label="Imputed",
                        )
            else:
                ax.text(
                    0.5,
                    0.57,
                    "No reported data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax.text(
                    0.5,
                    0.45,
                    "(all years missing)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="gray",
                )

            ax.set_title(indicator)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.25)

        fig.suptitle(f"{target}", fontsize=16)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            uniq = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            fig.legend(
                uniq.values(),
                uniq.keys(),
                loc="lower center",
                ncol=len(uniq),
                frameon=False,
                bbox_to_anchor=(0.5, 0.005),
            )
        fig.tight_layout(rect=[0, 0.01, 1, 0.98])
        output_file = out_path / f"task1_{country.replace(' ', '_')}_all_indicators.png"
        fig.savefig(output_file, dpi=180)
        plt.close(fig)


def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[INDICATORS] = scaler.fit_transform(df_scaled[INDICATORS].astype(float).values)
    return df_scaled, scaler


def make_task1_sequences(
    df_scaled: pd.DataFrame, window: int = 5, shift: int = 1
) -> Dict[str, np.ndarray]:
    sequences: Dict[str, np.ndarray] = {}

    for country, group in df_scaled.groupby("country", sort=False):
        g = group.sort_values("year").reset_index(drop=True)
        years = g["year"].to_numpy(dtype=int)
        values = g[INDICATORS].to_numpy(dtype=np.float32)

        country_windows: List[np.ndarray] = []
        for i in range(0, len(g) - window + 1, shift):
            y = years[i : i + window]
            if not np.all(np.diff(y) == 1):
                continue
            country_windows.append(values[i : i + window].reshape(-1))

        if country_windows:
            sequences[country] = np.asarray(country_windows, dtype=np.float32)

    return sequences


def build_mlp_dataset(
    df_imputed: pd.DataFrame, task1_sequences: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    countries = [country for country, arr in task1_sequences.items() if arr.size > 0]
    if not countries:
        raise ValueError("No sequences available for building MLP dataset.")

    aggregated = np.vstack([task1_sequences[country].mean(axis=0) for country in countries])
    n_features = len(INDICATORS)
    gdp_idx = INDICATORS.index("GDPpc_2017$")
    keep_cols = [i for i in range(aggregated.shape[1]) if (i % n_features) != gdp_idx]
    X = aggregated[:, keep_cols].astype(np.float32)

    avg_gdp = df_imputed.groupby("country", as_index=True)["GDPpc_2017$"].mean()
    log_gdp = np.log1p(avg_gdp)
    labels = pd.qcut(log_gdp, q=4, labels=False, duplicates="drop")
    y = np.asarray([int(labels.loc[country]) for country in countries], dtype=np.int64)
    return X, y


def create_forecasting_pairs(
    df_scaled: pd.DataFrame, input_window: int = 10, output_window: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gdp_idx = INDICATORS.index("GDPpc_2017$")
    X_list = []
    y_list = []
    country_list = []
    start_year_list = []

    for country, group in df_scaled.groupby("country", sort=False):
        g = group.sort_values("year").reset_index(drop=True)
        values = g[INDICATORS].to_numpy(dtype=np.float32)
        years = g["year"].to_numpy(dtype=int)
        max_start = len(g) - input_window - output_window + 1

        for start in range(max_start):
            input_end = start + input_window
            target_end = input_end + output_window

            X_win = values[start:input_end]
            y_win = values[input_end:target_end, gdp_idx]

            if np.isnan(X_win).any() or np.isnan(y_win).any():
                continue

            X_list.append(X_win)
            y_list.append(y_win)
            country_list.append(country)
            start_year_list.append(int(years[start]))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    countries = np.asarray(country_list, dtype=object)
    starts = np.asarray(start_year_list, dtype=int)
    return X, y, countries, starts


def split_forecasting_data(
    X: np.ndarray,
    y: np.ndarray,
    countries: np.ndarray,
    starts: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    meta = pd.DataFrame(
        {
            "idx": np.arange(X.shape[0], dtype=int),
            "country": countries,
            "start": starts,
        }
    )

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, group in meta.groupby("country", sort=False):
        g = group.sort_values("start")
        idx = g["idx"].to_numpy()
        n = len(idx)

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        if n_train < 1:
            n_train = 1
        if n_val < 1:
            n_val = 1
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
    }


def main(csv_path: str = "data/world_bank_data_dev80-23++.csv") -> None:
    df = load_data(csv_path)
    print(f"Loaded shape: {df.shape}")

    df_imputed, missing_before, missing_after = impute_missing(df)
    print(f"Missing values: {missing_before} -> {missing_after} after imputation")

    save_country_indicator_plots(df, df_imputed, out_dir="plots")

    df_scaled, _ = scale_features(df_imputed)

    task1_sequences = make_task1_sequences(df_scaled, window=5, shift=1)
    us_shape = task1_sequences.get("United States", np.empty((0, 0))).shape
    print(f"US Task 1 sequence shape: {us_shape}")
    print(f"Countries with valid Task 1 sequences: {len(task1_sequences)}")


if __name__ == "__main__":
    main()
