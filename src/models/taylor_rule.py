from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_dataset():
    df = pd.read_csv(DATA_DIR / "fomc_ml_dataset.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def train_test_split_time(df):
    train = df[(df["date"] >= "2000-01-01") & (df["date"] < "2016-01-01")]
    test = df[df["date"] >= "2016-01-01"]

    return train, test


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    df = load_dataset()
    train, test = train_test_split_time(df)

    # --- Taylor rule variables ---
    # Inflation + real activity proxy + inertia
    X_cols = [
        "cpi_roll3",
        "unemployment_roll3",
        "target_rate_lag1",
    ]

    y_col = "target_rate"

    X_train = train[X_cols]
    y_train = train[y_col]

    X_test = test[X_cols]
    y_test = test[y_col]

    # --- OLS estimation ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluation
    rmse_tr, mae_tr, r2_tr = evaluate(y_train, y_pred_train)
    rmse_te, mae_te, r2_te = evaluate(y_test, y_pred_test)

    print("\n=== TAYLOR RULE BASELINE ===")
    print("Train (2000–2015):")
    print(f"  RMSE: {rmse_tr:.3f}")
    print(f"  MAE : {mae_tr:.3f}")
    print(f"  R²  : {r2_tr:.3f}")

    print("Test (2016–2025):")
    print(f"  RMSE: {rmse_te:.3f}")
    print(f"  MAE : {mae_te:.3f}")
    print(f"  R²  : {r2_te:.3f}")

    print("\nEstimated coefficients:")
    for name, coef in zip(X_cols, model.coef_):
        print(f"  {name}: {coef:.3f}")
    print(f"  intercept: {model.intercept_:.3f}")


if __name__ == "__main__":
    main()