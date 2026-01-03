"""
Train and evaluate ensemble models (Random Forest, XGBoost) for predicting the
Federal Reserve target policy rate at FOMC decision dates.

- Task: rate level prediction (regression)
- Split: time-based (train 2000–2015, test 2016–2025)
- Data: data/processed/fomc_ml_dataset.csv
- Output: results/ensemble_results.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "fomc_ml_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Config
# -----------------------------
TRAIN_START = "2000-01-01"
TRAIN_END = "2016-01-01"  # exclusive
TEST_START = "2016-01-01"  # inclusive

FEATURE_COLS: List[str] = [
    "cpi_roll3",
    "unemployment_roll3",
    "target_rate_lag1",
    "t10y3m_roll3",
    "baa10ym_roll3",
    "indpro_roll3",
    "core_pce_roll3",
]

TARGET_COL = "target_rate"


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the processed ML dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {path}\n"
            f"Run feature building first to generate data/processed/fomc_ml_dataset.csv."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split consistent with the proposal / instructor feedback."""
    train = df[(df["date"] >= TRAIN_START) & (df["date"] < TRAIN_END)].copy()
    test = df[df["date"] >= TEST_START].copy()
    return train, test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Metrics:
    """Compute RMSE, MAE, R²."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(rmse=rmse, mae=mae, r2=r2)


def print_header(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("=" * 60)
    print("Ensemble Models — Random Forest and XGBoost")
    print("=" * 60)
    print("\nDataset:")
    print(f"  Total observations : {len(df)} FOMC decisions")
    print(f"  Training period    : 2000–2015 ({len(train)} observations)")
    print(f"  Test period        : 2016–2025 ({len(test)} observations)")
    print("\nFeatures used:")
    print(f"  {FEATURE_COLS}\n")


def print_block(model_name: str, train_m: Metrics, test_m: Metrics) -> None:
    print("-" * 60)
    print(model_name)
    print("-" * 60)
    print("Train performance:")
    print(f"  RMSE : {train_m.rmse:.3f}")
    print(f"  MAE  : {train_m.mae:.3f}")
    print(f"  R²   : {train_m.r2:.3f}")
    print("\nTest performance:")
    print(f"  RMSE : {test_m.rmse:.3f}")
    print(f"  MAE  : {test_m.mae:.3f}")
    print(f"  R²   : {test_m.r2:.3f}\n")


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train a constrained RF (helps reduce overfitting in small samples)."""
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train a moderately regularized XGBoost model."""
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    xgb.fit(X_train, y_train)
    return xgb


def main() -> None:
    df = load_dataset()
    train, test = split_train_test(df)

    # Quick sanity checks
    missing = [c for c in (["date", TARGET_COL] + FEATURE_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    print_header(df, train, test)

    # -------------------
    # Random Forest
    # -------------------
    rf = train_random_forest(X_train, y_train)
    rf_train_m = compute_metrics(y_train, rf.predict(X_train))
    rf_test_m = compute_metrics(y_test, rf.predict(X_test))
    print_block("Random Forest", rf_train_m, rf_test_m)

    # -------------------
    # XGBoost
    # -------------------
    xgb = train_xgboost(X_train, y_train)
    xgb_train_m = compute_metrics(y_train, xgb.predict(X_train))
    xgb_test_m = compute_metrics(y_test, xgb.predict(X_test))
    print_block("XGBoost", xgb_train_m, xgb_test_m)

    # -------------------
    # Save results
    # -------------------
    results = pd.DataFrame(
        [
            ["RandomForest", rf_train_m.rmse, rf_train_m.mae, rf_train_m.r2, rf_test_m.rmse, rf_test_m.mae, rf_test_m.r2],
            ["XGBoost", xgb_train_m.rmse, xgb_train_m.mae, xgb_train_m.r2, xgb_test_m.rmse, xgb_test_m.mae, xgb_test_m.r2],
        ],
        columns=["model", "rmse_train", "mae_train", "r2_train", "rmse_test", "mae_test", "r2_test"],
    )

    out_path = RESULTS_DIR / "ensemble_results.csv"
    results.to_csv(out_path, index=False)

    print("End of ensemble model evaluation.")
    print("=" * 60)
    print(f"Saved results to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
