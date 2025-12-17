from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


# ---------- FRED GENERIC LOADER ----------

def load_fred_series(filename: str, value_name: str) -> pd.DataFrame:
    """
    Generic loader for FRED CSV files stored in data/raw.

    - Detects date column ('date' or 'observation_date')
    - Detects value column (first non-date column)
    - Renames columns to: 'date' and <value_name>
    - If data is higher frequency than monthly (e.g. daily),
      aggregates to monthly averages.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")

    df = pd.read_csv(path)
    # clean column names
    df.columns = [c.strip().lower().replace("\ufeff", "") for c in df.columns]

    # detect date column
    if "date" in df.columns:
        date_col = "date"
    elif "observation_date" in df.columns:
        date_col = "observation_date"
    else:
        raise ValueError(f"No date column found in {df.columns}")

    # detect value column = first non-date column
    value_col = None
    for c in df.columns:
        if c != date_col:
            value_col = c
            break
    if value_col is None:
        raise ValueError(f"No value column found in {df.columns}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", value_col: value_name})
    df = df.sort_values("date").reset_index(drop=True)

    # if more rows than unique months -> probably daily -> aggregate to monthly
    n_months = df["date"].dt.to_period("M").nunique()
    if n_months < len(df):
        df = (
            df.set_index("date")[value_name]
              .resample("M")          # monthly
              .mean()                 # monthly average
              .reset_index()
        )

    return df


def load_fedfunds() -> pd.DataFrame:
    return load_fred_series("fedfunds.csv", "fedfunds")


# ---------- TARGET RATE (DFEDTAR + DFEDTARU) ----------

def load_fomc_target_full() -> pd.DataFrame:
    """
    Build full target-rate series:
    - DFEDTAR (single target) until 2008-12-15
    - DFEDTARU (upper bound) from 2008-12-16 onwards

    Returns:
        DataFrame with columns ['date', 'target_rate'] at daily frequency.
    """
    # 1) DFEDTAR
    path_dfedtar = RAW_DIR / "dfedtar.csv"
    df1 = pd.read_csv(path_dfedtar)
    df1.columns = [c.strip().lower().replace("\ufeff", "") for c in df1.columns]

    if "date" in df1.columns:
        date_col1 = "date"
    elif "observation_date" in df1.columns:
        date_col1 = "observation_date"
    else:
        raise ValueError(f"No date column in DFEDTAR: {df1.columns}")

    value_col1 = None
    for c in ["dfedtar", "value", "target_rate"]:
        if c in df1.columns:
            value_col1 = c
            break
    if value_col1 is None:
        raise ValueError(f"No value column in DFEDTAR: {df1.columns}")

    df1[date_col1] = pd.to_datetime(df1[date_col1])
    df1 = df1.rename(columns={date_col1: "date", value_col1: "target_rate"})
    df1 = df1[["date", "target_rate"]].sort_values("date")

    cutoff = pd.Timestamp("2008-12-16")
    df1 = df1[df1["date"] < cutoff]

    # 2) DFEDTARU
    path_dfedtaru = RAW_DIR / "dfedtaru.csv"
    df2 = pd.read_csv(path_dfedtaru)
    df2.columns = [c.strip().lower().replace("\ufeff", "") for c in df2.columns]

    if "date" in df2.columns:
        date_col2 = "date"
    elif "observation_date" in df2.columns:
        date_col2 = "observation_date"
    else:
        raise ValueError(f"No date column in DFEDTARU: {df2.columns}")

    value_col2 = None
    for c in ["dfedtaru", "value", "upper_target", "target"]:
        if c in df2.columns:
            value_col2 = c
            break
    if value_col2 is None:
        raise ValueError(f"No value column in DFEDTARU: {df2.columns}")

    df2[date_col2] = pd.to_datetime(df2[date_col2])
    df2 = df2.rename(columns={date_col2: "date", value_col2: "target_rate"})
    df2 = df2[["date", "target_rate"]].sort_values("date")
    df2 = df2[df2["date"] >= cutoff]

    # 3) concat
    df_full = pd.concat([df1, df2], ignore_index=True)
    df_full = df_full.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df_full


def detect_fomc_decisions(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Detect dates when the target rate changes.
    Each change corresponds to a policy decision (FOMC or emergency move).

    Returns:
        DataFrame with ['date', 'target_rate'] at decision dates only.
    """
    df = df_full.copy()
    df["prev"] = df["target_rate"].shift(1)
    decisions = df[df["target_rate"] != df["prev"]].dropna().copy()
    decisions = decisions.drop(columns=["prev"]).reset_index(drop=True)
    return decisions


# ---------- MACRO SERIES LOADER (CPI, UNRATE, etc.) ----------

def load_macro_series() -> dict:
    """
    Load all macro series as monthly data.
    Returns a dict of DataFrames keyed by series name.
    """
    series = {}

    series["fedfunds"] = load_fred_series("fedfunds.csv", "fedfunds")
    series["cpi"] = load_fred_series("cpiaucsl.csv", "cpi")
    series["core_pce"] = load_fred_series("pcepilfe.csv", "core_pce")
    series["unemployment"] = load_fred_series("unrate.csv", "unemployment")
    series["industrial_prod"] = load_fred_series("indpro.csv", "indpro")
    series["yield_curve"] = load_fred_series("t10y3m.csv", "t10y3m")

    # optional: corporate spread if file exists
    try:
        series["baa_spread"] = load_fred_series("baa10ym.csv", "baa10ym")
    except FileNotFoundError:
        pass

    return series


# ---------- QUICK TEST WHEN RUN AS SCRIPT ----------

if __name__ == "__main__":
    print("=== FEDFUNDS ===")
    fed = load_fedfunds()
    print(fed.head())
    print("Nb obs FEDFUNDS:", len(fed))

    print("\n=== FULL TARGET RATE (DFEDTAR + DFEDTARU) ===")
    full_tgt = load_fomc_target_full()
    print(full_tgt.head())
    print(full_tgt.tail())
    print("Nb obs daily:", len(full_tgt))

    print("\n=== FOMC DECISIONS (rate changes) ===")
    decisions = detect_fomc_decisions(full_tgt)
    print(decisions.head(15))
    print("Nb decisions detected:", len(decisions))

    print("\n=== MACRO SERIES (monthly) ===")
    macro = load_macro_series()
    for name, df in macro.items():
        print(f"\n-- {name} --")
        print(df.head())
        print(df.tail())
        print("Nb obs:", len(df))