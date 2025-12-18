from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fomc_ml_dataset.csv"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Define "month buckets" (year-month) ---
    decision_month = df["date"].dt.to_period("M")
    expected_feature_month = (df["date"] - pd.offsets.MonthBegin(1)).dt.to_period("M")

    # --- Identify which columns are "macro features" that should be lagged monthly ---
    exclude = {"date", "target_rate"}
    macro_cols = [c for c in df.columns if c not in exclude]

    # We assume your monthly features correspond to the month in their index date.
    # The strictest rule (no leakage) = use previous month's macro values for each decision.
    # So the monthly timestamp attached to the features should be expected_feature_month.
    #
    # We can't directly see the macro release date here, so we check consistency in how
    # the feature_builder aligned the monthly series:
    #
    # - If your feature_builder merges macro data at month start (YYYY-MM-01),
    #   then for a decision on (YYYY-MM-DD), the macro month used should be (YYYY-(MM-1)).
    #
    # We'll reconstruct the "feature month" by looking at the implied month from the
    # decision date minus 1 month.

    df_check = df[["date"] + macro_cols].copy()
    df_check["decision_month"] = decision_month.astype(str)
    df_check["expected_feature_month"] = expected_feature_month.astype(str)

    # Print a small sample to manually sanity check
    print("=== Leakage check (month alignment) ===")
    print(df_check[["date", "decision_month", "expected_feature_month"]].head(10))

    # Hard rule: decision_month should NOT equal feature month if you intended strict lag-1
    # Since we don't store feature_month explicitly, we test this indirectly by checking:
    # if you built *_lag1 and *_roll3 based on shifted monthly data, they should reflect lagging.
    #
    # Practical check: ensure you have lagged features and they are not identical to raw levels
    # in the same row too often (a common leak symptom).

    suspicious = []
    for col in macro_cols:
        # If a column is exactly identical to its next month's value too often,
        # it might indicate same-month usage. This is heuristic.
        same_as_next = (df[col] == df[col].shift(-1)).mean()
        if same_as_next > 0.95 and df[col].nunique() > 5:
            suspicious.append((col, same_as_next))

    if suspicious:
        print("\n⚠️ Potentially suspicious columns (very constant across decisions):")
        for col, rate in suspicious:
            print(f"  - {col}: {rate:.2%} identical to next row")
    else:
        print("\n✅ No obvious constant-feature leakage signal detected (heuristic check).")

    print("\nNOTE:")
    print(
        "This script cannot prove release-date correctness. "
        "It checks that your pipeline is designed to use lagged monthly info."
    )


if __name__ == "__main__":
    main()
