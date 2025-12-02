from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def load_fedfunds() -> pd.DataFrame:
    """
    Load the effective federal funds rate from data/raw/fedfunds.csv (FRED FEDFUNDS series).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: datetime64
        - fedfunds: float
    """
    path = RAW_DIR / "fedfunds.csv"
    df = pd.read_csv(path)

    # Clean column names: lowercase, strip spaces, remove BOM if present
    df.columns = [c.strip().lower().replace("\ufeff", "") for c in df.columns]

    # Detect date column name
    if "date" in df.columns:
        date_col = "date"
    elif "observation_date" in df.columns:
        date_col = "observation_date"
    else:
        raise ValueError(f"No date column found in FEDFUNDS file: {df.columns}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    return df.sort_values("date").reset_index(drop=True)


def load_fomc_target_full() -> pd.DataFrame:
    """
    Build a complete Fed policy rate series by combining:
    - DFEDTAR  (single target rate) up to 2008-12-15
    - DFEDTARU (target range upper bound) from 2008-12-16 onward

    Both series are daily FRED series.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: datetime64
        - target_rate: float
    """
    # ---------- 1) Load DFEDTAR (pre-2008) ----------
    path_dfedtar = RAW_DIR / "dfedtar.csv"
    df1 = pd.read_csv(path_dfedtar)
    df1.columns = [c.strip().lower().replace("\ufeff", "") for c in df1.columns]

    # Date column
    if "date" in df1.columns:
        date_col1 = "date"
    elif "observation_date" in df1.columns:
        date_col1 = "observation_date"
    else:
        raise ValueError(f"No date column found in DFEDTAR: {df1.columns}")

    # Value column
    value_col1 = None
    for c in ["dfedtar", "value", "target_rate"]:
        if c in df1.columns:
            value_col1 = c
            break
    if value_col1 is None:
        raise ValueError(f"No value column found in DFEDTAR: {df1.columns}")

    df1[date_col1] = pd.to_datetime(df1[date_col1])
    df1 = df1.rename(columns={date_col1: "date", value_col1: "target_rate"})
    df1 = df1[["date", "target_rate"]].sort_values("date")

    # Keep only dates strictly before the range era
    cutoff = pd.Timestamp("2008-12-16")
    df1 = df1[df1["date"] < cutoff]

    # ---------- 2) Load DFEDTARU (post-2008) ----------
    path_dfedtaru = RAW_DIR / "dfedtaru.csv"
    df2 = pd.read_csv(path_dfedtaru)
    df2.columns = [c.strip().lower().replace("\ufeff", "") for c in df2.columns]

    if "date" in df2.columns:
        date_col2 = "date"
    elif "observation_date" in df2.columns:
        date_col2 = "observation_date"
    else:
        raise ValueError(f"No date column found in DFEDTARU: {df2.columns}")

    value_col2 = None
    for c in ["dfedtaru", "value", "upper_target", "target"]:
        if c in df2.columns:
            value_col2 = c
            break
    if value_col2 is None:
        raise ValueError(f"No value column found in DFEDTARU: {df2.columns}")

    df2[date_col2] = pd.to_datetime(df2[date_col2])
    df2 = df2.rename(columns={date_col2: "date", value_col2: "target_rate"})
    df2 = df2[["date", "target_rate"]].sort_values("date")

    # Keep only the range era
    df2 = df2[df2["date"] >= cutoff]

    # ---------- 3) Concatenate and clean ----------
    df_full = pd.concat([df1, df2], ignore_index=True)
    df_full = (
        df_full.drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df_full


def detect_fomc_decisions(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Detect dates where the policy rate changes.
    These correspond to FOMC decisions (or emergency moves).

    Parameters
    ----------
    df_full : pd.DataFrame
        Output of load_fomc_target_full(), with columns:
        - date
        - target_rate

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per rate change:
        - date
        - target_rate
    """
    df = df_full.copy()
    df["prev"] = df["target_rate"].shift(1)
    decisions = df[df["target_rate"] != df["prev"]].dropna().copy()
    decisions = decisions.drop(columns=["prev"]).reset_index(drop=True)
    return decisions


if __name__ == "__main__":
    # Simple sanity checks when running this file directly

    print("=== FEDFUNDS ===")
    fed = load_fedfunds()
    print(fed.head())
    print("Nb obs FEDFUNDS:", len(fed))

    print("\n=== FED TARGET FULL (DFEDTAR + DFEDTARU) ===")
    full = load_fomc_target_full()
    print(full.head())
    print(full.tail())
    print("Nb obs daily:", len(full))

    print("\n=== FOMC DECISIONS (rate changes) ===")
    dec = detect_fomc_decisions(full)
    print(dec.head(15))
    print("Nb decisions detected:", len(dec))