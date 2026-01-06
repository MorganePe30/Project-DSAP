from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_ml_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "fomc_ml_dataset.csv"
    return pd.read_csv(path, parse_dates=["date"])


def train_test_split_time(df: pd.DataFrame):
    """
    Time-based split:
    - Train: 1982–2015
    - Test : 2016–2025
    """
    df = df.sort_values("date").reset_index(drop=True)
    

    train = df[df["date"] < "2016-01-01"].copy()
    test = df[df["date"] >= "2016-01-01"].copy()

    y_train = train["target_rate"]
    y_test = test["target_rate"]

    X_train = train.drop(columns=["date", "target_rate"])
    X_test = test.drop(columns=["date", "target_rate"])

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_tr, mae_tr, r2_tr = compute_metrics(y_train, y_pred_train)
    rmse_te, mae_te, r2_te = compute_metrics(y_test, y_pred_test)

    print("-" * 60)
    print(f"{name}")
    print("-" * 60)

    print("Train performance:")
    print(f"  RMSE : {rmse_tr:.3f}")
    print(f"  MAE  : {mae_tr:.3f}")
    print(f"  R²   : {r2_tr:.3f}\n")

    print("Test performance:")
    print(f"  RMSE : {rmse_te:.3f}")
    print(f"  MAE  : {mae_te:.3f}")
    print(f"  R²   : {r2_te:.3f}\n")


def main():
    print("=" * 60)
    print("Baseline Models — Linear Regressions")
    print("=" * 60)

    df = load_ml_dataset()
    X_train, X_test, y_train, y_test = train_test_split_time(df)

    print("\nDataset:")
    print(f"  Total observations : {len(df)} FOMC decisions")
    print("  Training period    : 1982–2015 "
      f"({len(X_train)} observations)")
    print("  Test period        : 2016–2025 "
      f"({len(X_test)} observations)\n")

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=0)),
        ]
    )

    lasso = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.05, random_state=0)),
        ]
    )

    evaluate_model("Ridge Regression", ridge, X_train, y_train, X_test, y_test)
    evaluate_model("Lasso Regression", lasso, X_train, y_train, X_test, y_test)

    print("End of baseline model evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    main()