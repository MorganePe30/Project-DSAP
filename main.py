from pathlib import Path
import runpy
import sys


# -----------------------
# Paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "fomc_ml_dataset.csv"


def run(module_name: str) -> None:
    """Run a module exactly like: python -m <module_name>"""
    print(f"\n>>> Running: python -m {module_name}")
    runpy.run_module(module_name, run_name="__main__")


def main() -> None:
    print("=== Fed Policy ML Project: main.py ===")

    # 1) Build / refresh processed ML dataset
    if not PROCESSED_DATA.exists():
        print(f"\n[INFO] Processed dataset not found: {PROCESSED_DATA}")
        print("[INFO] Building dataset via feature builder...")
        run("src.features.feature_builder")
    else:
        print(f"\n[OK] Found processed dataset: {PROCESSED_DATA}")

    # 2) Train baseline models
    run("src.models.train_baselines")

    # 3) Train ensemble models
    run("src.models.train_ensemble")

    # 4) Build final comparison table
    run("src.evaluation.summary_table")

    # 5) Generate prediction plots
    run("src.evaluation.plot_predictions")

    # 6) Leakage check (optional but good)
    try:
        run("src.evaluation.leakage_check")
    except Exception as e:
        print(f"[WARN] Leakage check failed (not blocking): {e}")

    print("\n=== DONE ===")
    print("Outputs saved in:")
    print(" - results/")
    print(" - data/processed/")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] main.py failed: {err}")
        sys.exit(1)
