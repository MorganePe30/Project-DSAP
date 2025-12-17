from pathlib import Path
import pandas as pd

from src.data.data_loader import (
    load_fomc_target_full,
    detect_fomc_decisions,
    load_macro_series,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def merge_macro_to_decisions(decisions: pd.DataFrame,
                             macro: dict) -> pd.DataFrame:
    """
    For each FOMC decision date, attach the latest available macro data
    observed BEFORE (or on) that date using merge_asof (no leakage).
    """
    df = decisions.sort_values("date").reset_index(drop=True).copy()

    for name, series in macro.items():
        series = series.sort_values("date").reset_index(drop=True).copy()
        df = pd.merge_asof(
            df,
            series,
            on="date",
            direction="backward",   # take last value <= decision date
        )

    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple, economically meaningful features:
    - monthly changes
    - 3-month rolling averages
    - 1-period lags
    """
    df = df.copy()

    # Monthly changes (first differences)
    df["cpi_diff"] = df["cpi"].diff()
    df["core_pce_diff"] = df["core_pce"].diff()
    df["unemployment_diff"] = df["unemployment"].diff()
    df["indpro_diff"] = df["indpro"].diff()
    df["t10y3m_diff"] = df["t10y3m"].diff()
    if "baa10ym" in df.columns:
        df["baa10ym_diff"] = df["baa10ym"].diff()

    # 3-month rolling averages (smoother signal)
    for col in ["cpi", "core_pce", "unemployment", "indpro", "t10y3m", "baa10ym"]:
        if col in df.columns:
            df[col + "_roll3"] = df[col].rolling(3).mean()

    # One-period lags of the target rate (inertia of policy)
    df["target_rate_lag1"] = df["target_rate"].shift(1)

    return df


def build_fomc_ml_dataset() -> pd.DataFrame:
    """
    Build the final ML dataset at FOMC decision dates:

    - target_rate (y) from DFEDTAR/DFEDTARU
    - macro features aligned with last available value before the decision
    - basic transforms (diffs, rolling means, lagged target)

    Returns:
        DataFrame with columns:
        ['date', 'target_rate', features ... ]
    """
    # 1) Target rate and decisions
    full_target = load_fomc_target_full()
    decisions = detect_fomc_decisions(full_target)

    # 2) Macro series (monthly)
    macro = load_macro_series()

    # 3) Align macro data to decision dates (no look-ahead)
    df = merge_macro_to_decisions(decisions, macro)

    # 4) Add engineered features
    df = add_basic_features(df)

    # 5) Drop initial rows with NaNs from diffs/rollings
    df = df.dropna().reset_index(drop=True)

    # 6) Save a processed copy for inspection
    out_path = PROCESSED_DIR / "fomc_ml_dataset.csv"
    df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    dataset = build_fomc_ml_dataset()
    print(dataset.head())
    print(dataset.tail())
    print("Nb observations (decisions used for ML):", len(dataset))
    print("Columns:", dataset.columns.tolist())