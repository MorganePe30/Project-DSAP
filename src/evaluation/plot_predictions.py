from pathlib import Path
from sklearn.linear_model import Ridge, LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fomc_ml_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train = df[(df["date"] >= "2000-01-01") & (df["date"] < "2016-01-01")]
    test = df[df["date"] >= "2016-01-01"]

    features = [
        "cpi_roll3",
        "unemployment_roll3",
        "target_rate_lag1",
        "t10y3m_roll3",
        "baa10ym_roll3",
        "indpro_roll3",
        "core_pce_roll3",
    ]

    return train, test, features


def main():
    train, test, features = load_data()

    X_train = train[features]
    y_train = train["target_rate"]
    X_test = test[features]
    y_test = test["target_rate"]

    # -----------------------
    # Models
    # -----------------------
    models = {
    "Taylor (OLS)": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    ),
}

    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    # -----------------------
    # Plot 1: True vs Predicted
    # -----------------------
    plt.figure(figsize=(7, 7))
    for name, y_pred in predictions.items():
        plt.scatter(y_test, y_pred, label=name, alpha=0.7)

    lims = [
        min(y_test.min(), min(map(np.min, predictions.values()))),
        max(y_test.max(), max(map(np.max, predictions.values()))),
    ]
    plt.plot(lims, lims, "k--", alpha=0.5)

    plt.xlabel("Actual target rate")
    plt.ylabel("Predicted target rate")
    plt.title("True vs Predicted Fed Policy Rate (Test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "true_vs_predicted.png", dpi=300)
    plt.close()

    # -----------------------
    # Plot 2: Time series
    # -----------------------
    plt.figure(figsize=(10, 5))
    plt.plot(test["date"], y_test.values, label="Actual", linewidth=2)

    for name, y_pred in predictions.items():
        plt.plot(test["date"], y_pred, label=name, linestyle="--")

    plt.title("Fed Policy Rate: Actual vs Predicted (Test period)")
    plt.xlabel("Date")
    plt.ylabel("Target rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "time_series_predictions.png", dpi=300)
    plt.close()

    print("Plots saved in results/ directory")


if __name__ == "__main__":
    main()
