import sys
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "fomc_ml_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
MODEL_COMPARISON = RESULTS_DIR / "model_comparison.csv"


def run_step(command, description, log_file):
    """
    Run a command quietly (no terminal spam).
    All stdout/stderr are redirected into a log file.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"{description}\n")
        f.write(f"Command: {' '.join(command)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("-" * 60 + "\n\n")

        try:
            subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Step failed: {description}")
            print(f"        See log: {log_file}")
            sys.exit(1)

    print(f"[OK] {description}")


def pick_winner():
    """
    Winner = model with the lowest rmse_test in results/model_comparison.csv
    """
    if not MODEL_COMPARISON.exists():
        print("[ERROR] results/model_comparison.csv not found. Cannot pick a winner.")
        sys.exit(1)

    df = pd.read_csv(MODEL_COMPARISON)
    if "rmse_test" not in df.columns or "model" not in df.columns:
        print("[ERROR] model_comparison.csv does not have expected columns (model, rmse_test).")
        sys.exit(1)

    df["rmse_test"] = pd.to_numeric(df["rmse_test"], errors="coerce")
    df["r2_test"] = pd.to_numeric(df.get("r2_test", pd.NA), errors="coerce")
    df = df.dropna(subset=["rmse_test"]).sort_values("rmse_test", ascending=True)

    best = df.iloc[0]
    best_model = str(best["model"])
    best_rmse = float(best["rmse_test"])
    best_r2 = float(best["r2_test"]) if pd.notna(best["r2_test"]) else float("nan")
    return best_model, best_rmse, best_r2


def main():
    print("=" * 60)
    print("Fed Policy ML Project — Model Comparison")
    print("=" * 60)
    print()

    # 1) Data check
    print("1) Data check")
    if not PROCESSED_DATA.exists():
        print("[ERROR] Processed dataset not found:")
        print(f"        {PROCESSED_DATA}")
        print("        (Generate it first with your feature pipeline.)")
        sys.exit(1)
    print("[OK] Found processed dataset")
    print()

    # 2) Baselines
    print("2) Baseline models")
    run_step(
        ["python", "-m", "src.models.train_baselines"],
        "Train Ridge + Lasso",
        LOGS_DIR / "02_train_baselines.txt",
    )
    run_step(
        ["python", "-m", "src.models.taylor_rule"],
        "Train Taylor Rule (OLS)",
        LOGS_DIR / "02_train_taylor_rule.txt",
    )
    print()

    # 3) Ensembles
    print("3) Ensemble models")
    run_step(
        ["python", "-m", "src.models.train_ensemble"],
        "Train Random Forest + XGBoost",
        LOGS_DIR / "03_train_ensemble.txt",
    )
    print()

    # 4) Summary table + winner
    print("4) Evaluation and comparison")
    run_step(
        ["python", "-m", "src.evaluation.summary_table"],
        "Build model_comparison.csv",
        LOGS_DIR / "04_summary_table.txt",
    )

    best_model, best_rmse, best_r2 = pick_winner()

    print()
    print("-" * 60)
    print("Best model (lowest Test RMSE)")
    if best_r2 == best_r2:  # not NaN
        print(f"{best_model}  |  Test RMSE: {best_rmse:.3f}  |  Test R2: {best_r2:.3f}")
    else:
        print(f"{best_model}  |  Test RMSE: {best_rmse:.3f}")
    print("-" * 60)
    print()

    # 5) Plots
    print("5) Plots")
    run_step(
        ["python", "-m", "src.evaluation.plot_predictions"],
        "Generate prediction plots",
        LOGS_DIR / "05_plot_predictions.txt",
    )
    print()

    # 6) Leakage diagnostics
    print("6) Leakage diagnostics")
    run_step(
        ["python", "-m", "src.evaluation.leakage_check"],
        "Run leakage check (heuristic)",
        LOGS_DIR / "06_leakage_check.txt",
    )
    print()

    print("Done.")
    print("Outputs:")
    print(" - results/")
    print(" - results/logs/ (detailed outputs)")
    print(" - data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()