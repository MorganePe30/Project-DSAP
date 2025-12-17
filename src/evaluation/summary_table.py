from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fomc_ml_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
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


def fit_and_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    rmse_tr, mae_tr, r2_tr = evaluate(y_train, pred_train)
    rmse_te, mae_te, r2_te = evaluate(y_test, pred_test)

    return rmse_tr, mae_tr, r2_tr, rmse_te, mae_te, r2_te


def main():
    df = load_dataset()
    train, test = split_train_test(df)

    # Feature sets
    # Taylor: minimal & interpretable
    taylor_features = ["cpi_roll3", "unemployment_roll3", "target_rate_lag1"]

    # ML: slightly richer (same as your ensemble script)
    ml_features = [
        "cpi_roll3",
        "unemployment_roll3",
        "target_rate_lag1",
        "t10y3m_roll3",
        "baa10ym_roll3",
        "indpro_roll3",
        "core_pce_roll3",
    ]

    y_col = "target_rate"

    Xtr_taylor = train[taylor_features]
    Xte_taylor = test[taylor_features]

    Xtr_ml = train[ml_features]
    Xte_ml = test[ml_features]

    y_train = train[y_col]
    y_test = test[y_col]

    models = [
        ("TaylorRule_OLS", LinearRegression(), Xtr_taylor, Xte_taylor),
        ("Ridge", Ridge(alpha=1.0, random_state=42), Xtr_ml, Xte_ml),
        ("Lasso", Lasso(alpha=0.01, random_state=42), Xtr_ml, Xte_ml),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            Xtr_ml,
            Xte_ml,
        ),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
            ),
            Xtr_ml,
            Xte_ml,
        ),
    ]

    rows = []
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    print("Building model comparison table...\n")

    for name, model, X_train, X_test in models:
        rmse_tr, mae_tr, r2_tr, rmse_te, mae_te, r2_te = fit_and_score(
            model, X_train, y_train, X_test, y_test
        )
        rows.append(
            {
                "model": name,
                "rmse_train": rmse_tr,
                "mae_train": mae_tr,
                "r2_train": r2_tr,
                "rmse_test": rmse_te,
                "mae_test": mae_te,
                "r2_test": r2_te,
            }
        )

    results = pd.DataFrame(rows).sort_values("r2_test", ascending=False).reset_index(drop=True)

    # Print nicely
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(results)

    out_path = RESULTS_DIR / "model_comparison.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()