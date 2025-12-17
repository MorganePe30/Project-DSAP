from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost (pip/conda package: xgboost)
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "fomc_ml_dataset.csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def split_train_test(df: pd.DataFrame):
    train = df[(df["date"] >= "2000-01-01") & (df["date"] < "2016-01-01")].copy()
    test = df[df["date"] >= "2016-01-01"].copy()
    return train, test


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def main():
    df = load_dataset()
    train, test = split_train_test(df)

    # Same core feature set as Taylor baseline (simple + robust)
    # You can expand later if needed.
    feature_cols = [
        "cpi_roll3",
        "unemployment_roll3",
        "target_rate_lag1",
        "t10y3m_roll3",
        "baa10ym_roll3",
        "indpro_roll3",
        "core_pce_roll3",
    ]

    target_col = "target_rate"

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    print(f"Full dataset size: {len(df)} decisions")
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}\n")

    # -------------------
    # Random Forest
    # -------------------
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_pred_tr = rf.predict(X_train)
    rf_pred_te = rf.predict(X_test)

    rf_rmse_tr, rf_mae_tr, rf_r2_tr = evaluate(y_train, rf_pred_tr)
    rf_rmse_te, rf_mae_te, rf_r2_te = evaluate(y_test, rf_pred_te)

    print("=== Random Forest ===")
    print("Train:")
    print(f"  RMSE: {rf_rmse_tr:.3f}")
    print(f"  MAE : {rf_mae_tr:.3f}")
    print(f"  R²  : {rf_r2_tr:.3f}")
    print("Test:")
    print(f"  RMSE: {rf_rmse_te:.3f}")
    print(f"  MAE : {rf_mae_te:.3f}")
    print(f"  R²  : {rf_r2_te:.3f}\n")

    # -------------------
    # XGBoost
    # -------------------
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

    xgb_pred_tr = xgb.predict(X_train)
    xgb_pred_te = xgb.predict(X_test)

    xgb_rmse_tr, xgb_mae_tr, xgb_r2_tr = evaluate(y_train, xgb_pred_tr)
    xgb_rmse_te, xgb_mae_te, xgb_r2_te = evaluate(y_test, xgb_pred_te)

    print("=== XGBoost ===")
    print("Train:")
    print(f"  RMSE: {xgb_rmse_tr:.3f}")
    print(f"  MAE : {xgb_mae_tr:.3f}")
    print(f"  R²  : {xgb_r2_tr:.3f}")
    print("Test:")
    print(f"  RMSE: {xgb_rmse_te:.3f}")
    print(f"  MAE : {xgb_mae_te:.3f}")
    print(f"  R²  : {xgb_r2_te:.3f}\n")

    # Save a small summary table
    results = pd.DataFrame(
        [
            ["RandomForest", rf_rmse_tr, rf_mae_tr, rf_r2_tr, rf_rmse_te, rf_mae_te, rf_r2_te],
            ["XGBoost", xgb_rmse_tr, xgb_mae_tr, xgb_r2_tr, xgb_rmse_te, xgb_mae_te, xgb_r2_te],
        ],
        columns=["model", "rmse_train", "mae_train", "r2_train", "rmse_test", "mae_test", "r2_test"],
    )

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ensemble_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()