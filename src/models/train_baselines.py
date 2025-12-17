from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_ml_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "fomc_ml_dataset.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def train_test_split_time(df: pd.DataFrame):
    """
    Split dataset into train (2000–2015) and test (2016–2025)
    based on decision date.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # keep only data from 2000 onwards (as discussed with your instructor)
    df = df[df["date"] >= "2000-01-01"].reset_index(drop=True)

    train = df[df["date"] < "2016-01-01"].copy()
    test = df[df["date"] >= "2016-01-01"].copy()

    # Target and features
    y_train = train["target_rate"]
    y_test = test["target_rate"]

    # Drop date and target from features
    drop_cols = ["date", "target_rate"]
    X_train = train.drop(columns=drop_cols)
    X_test = test.drop(columns=drop_cols)

    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Fit model and print RMSE, MAE, R² on train and test.
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    rmse_tr, mae_tr, r2_tr = metrics(y_train, y_pred_train)
    rmse_te, mae_te, r2_te = metrics(y_test, y_pred_test)

    print(f"\n=== {name} ===")
    print("Train:")
    print(f"  RMSE: {rmse_tr:.3f}")
    print(f"  MAE : {mae_tr:.3f}")
    print(f"  R²  : {r2_tr:.3f}")
    print("Test:")
    print(f"  RMSE: {rmse_te:.3f}")
    print(f"  MAE : {mae_te:.3f}")
    print(f"  R²  : {r2_te:.3f}")


def main():
    df = load_ml_dataset()
    print(f"Full dataset size: {len(df)} decisions")
    X_train, X_test, y_train, y_test = train_test_split_time(df)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Ridge regression with scaling
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=0)),
        ]
    )

    # Lasso regression with scaling
    lasso = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.05, random_state=0)),
        ]
    )

    evaluate_model("Ridge", ridge, X_train, y_train, X_test, y_test)
    evaluate_model("Lasso", lasso, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
