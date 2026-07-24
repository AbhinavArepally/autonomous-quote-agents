"""
Data Ingestion & Preprocessing Layer
=====================================
Handles the raw quote CSV (~146K rows) end to end:
  - schema validation & efficient dtypes (for large-volume handling)
  - missingness handling (documented per column, not blanket-imputed)
  - de-duplication (Quote_Num repeats == real re-quotes; Re_Quote flag
    in the raw data is unreliable, so we derive our own signal)
  - feature engineering (ratios, composite scores, recency)
  - encoding for downstream models
  - a structured data-quality report the dashboard can render directly

Design note: at 146K rows this doesn't strictly need Dask/Spark, but the
code is written the way it would need to be at 10-100x this volume:
vectorized pandas ops only (no row-wise .apply loops), categorical dtypes
for repeated strings, chunked-read support for future larger files.
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---- lookup tables for turning range-buckets into usable numeric midpoints
SALARY_MIDPOINT = {
    '<= $ 25 K': 12.5, '> $ 25 K <= $ 40 K': 32.5, '> $ 40 K <= $ 60 K': 50.0,
    '> $ 60 K <= $ 90 K': 75.0, '> $ 90 K': 100.0,
}
VEHICLE_COST_MIDPOINT = {
    '<= $ 10 K': 5.0, '> $ 10 K <= $ 20 K': 15.0, '> $ 20 K <= $ 30 K': 25.0,
    '> $ 30 K <= $ 40 K': 35.0, '> $ 40 K': 50.0,
}
MILES_MIDPOINT = {
    '<= 7.5 K': 3.75, '> 7.5 K & <= 15 K': 11.25, '> 15 K & <= 25 K': 20.0,
    '> 25 K & <= 35 K': 30.0, '> 35 K & <= 45 K': 40.0,
    '> 45 K & <= 55 K': 50.0, '> 55 K': 60.0,
}

CATEGORICAL_COLS = [
    'Agent_Type', 'Region', 'Policy_Type', 'Gender', 'Marital_Status',
    'Education', 'Sal_Range', 'Coverage', 'Veh_Usage',
    'Annual_Miles_Range', 'Vehicl_Cost_Range', 'Re_Quote',
]


@dataclass
class PreprocessingArtifacts:
    """Stats learned from the training set, needed to transform any new
    single row consistently (can't recompute an IQR from one row)."""
    snapshot_date: pd.Timestamp = None
    outlier_bounds: dict = field(default_factory=dict)  # {col: (lo, hi)}


@dataclass
class DataQualityReport:
    rows_in: int = 0
    rows_out: int = 0
    exact_duplicates_dropped: int = 0
    requote_rows_identified: int = 0
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)
    dropped_redundant_cols: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "exact_duplicates_dropped": self.exact_duplicates_dropped,
            "requote_rows_identified": self.requote_rows_identified,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "dropped_redundant_cols": self.dropped_redundant_cols,
            "notes": self.notes,
        }


def load_raw(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load CSV with efficient dtypes for large-volume handling."""
    dtype_map = {c: "category" for c in CATEGORICAL_COLS}
    df = pd.read_csv(path, dtype=dtype_map, nrows=nrows)
    for date_col in ["Q_Creation_DT", "Q_Valid_DT", "Policy_Bind_DT"]:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def clean_and_engineer(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None):
    """Run the full cleaning + feature engineering pass on a TRAINING set.
    Returns (df, report, artifacts) - artifacts must be saved and reused
    for scoring any new single quote later (see transform_new_quote)."""
    report = DataQualityReport()
    report.rows_in = len(df)
    report.missing_before = df.isnull().sum().to_dict()

    if snapshot_date is None:
        snapshot_date = df["Q_Creation_DT"].max()
    artifacts = PreprocessingArtifacts(snapshot_date=snapshot_date)

    df = df.copy()

    # --- exact duplicate rows (true data-entry duplicates, not re-quotes)
    exact_dupe_mask = df.duplicated(keep="first")
    report.exact_duplicates_dropped = int(exact_dupe_mask.sum())
    df = df[~exact_dupe_mask]

    # --- re-quote detection: Quote_Num repeats are the real signal.
    # The raw Re_Quote flag is kept as a feature but we don't trust it alone.
    quote_counts = df.groupby("Quote_Num")["Quote_Num"].transform("count")
    df["true_requote_count"] = quote_counts
    df["is_requote_derived"] = (quote_counts > 1).astype(int)
    report.requote_rows_identified = int((quote_counts > 1).sum())
    report.notes.append(
        "Re_Quote flag in raw data does not reliably align with repeated "
        "Quote_Num values; is_requote_derived is computed directly from "
        "duplicate Quote_Num occurrences and used alongside (not instead "
        "of) the raw flag."
    )

    # --- redundant / non-informative columns
    # Driving_Exp = Driver_Age - 17 for every row in this dataset (perfectly
    # collinear) -> keep Driver_Age, drop Driving_Exp to avoid double-counting.
    if "Driving_Exp" in df.columns:
        diff = (df["Driver_Age"] - df["Driving_Exp"]).nunique()
        if diff == 1:
            df = df.drop(columns=["Driving_Exp"])
            report.dropped_redundant_cols.append("Driving_Exp")
            report.notes.append(
                "Driving_Exp == Driver_Age - 17 for every row (perfectly "
                "collinear) -> dropped as redundant."
            )

    # --- missingness handling (documented per column)
    # Policy_Bind_DT is null for every unbound quote by definition - not
    # missing data, it's structurally absent. Leave as-is (do not impute),
    # and never feed it to a model (it would leak the target).
    report.notes.append(
        "Policy_Bind_DT is null for unbound quotes by construction, not a "
        "data quality issue. It is excluded from all model features to "
        "avoid target leakage."
    )

    df = _engineer_features(df, snapshot_date)

    # --- outlier bounds: LEARNED here on the full training set, then
    # reused as-is when scoring any new single quote (an IQR from n=1 is
    # meaningless, so new rows are checked against these saved bounds).
    for col in ["Quoted_Premium", "miles_mid_k"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        artifacts.outlier_bounds[col] = (lo, hi)
        df[f"{col}_outlier"] = ((df[col] < lo) | (df[col] > hi)).astype(int)

    report.rows_out = len(df)
    report.missing_after = df.isnull().sum().to_dict()
    return df, report, artifacts


def _engineer_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Feature engineering steps that are identical whether applied to the
    full training set or a single new row - shared by both paths."""
    # .str.strip() guards against trailing whitespace present in some raw
    # category values (e.g. '> $ 90 K ' vs. lookup key '> $ 90 K') which
    # otherwise silently produces NaN for ~11-16% of rows on these two cols.
    df["sal_mid_k"] = df["Sal_Range"].astype(str).str.strip().map(SALARY_MIDPOINT).astype(float)
    df["vehcost_mid_k"] = df["Vehicl_Cost_Range"].astype(str).str.strip().map(VEHICLE_COST_MIDPOINT).astype(float)
    df["miles_mid_k"] = df["Annual_Miles_Range"].astype(str).str.strip().map(MILES_MIDPOINT).astype(float)

    df["premium_to_salary"] = df["Quoted_Premium"] / (df["sal_mid_k"] * 1000)
    df["premium_to_vehcost"] = df["Quoted_Premium"] / (df["vehcost_mid_k"] * 1000)
    df["risk_composite_raw"] = (
        df["Prev_Accidents"].astype(int)
        + df["Prev_Citations"].astype(int)
        + (df["Driver_Age"] < 25).astype(int)
        + (df["miles_mid_k"] > 45).astype(int)
    )
    df["quote_age_days"] = (snapshot_date - df["Q_Creation_DT"]).dt.days
    return df


def transform_new_quote(raw_fields: dict, artifacts: PreprocessingArtifacts) -> pd.DataFrame:
    """
    Turn ONE new quote (raw field values from a UI form) into a single-row
    dataframe with exactly the engineered features the trained agents
    expect - using the saved training-set artifacts (snapshot date,
    outlier bounds) rather than recomputing statistics from n=1.
    """
    row = dict(raw_fields)
    row.setdefault("Quote_Num", "NEW-QUOTE")
    row.setdefault("Q_Creation_DT", artifacts.snapshot_date)
    row.setdefault("true_requote_count", 1)
    row.setdefault("is_requote_derived", 0)
    row.setdefault("Policy_Bind", "No")  # unknown at scoring time, placeholder only

    df = pd.DataFrame([row])
    df["Q_Creation_DT"] = pd.to_datetime(df["Q_Creation_DT"])
    df = _engineer_features(df, artifacts.snapshot_date)

    for col, (lo, hi) in artifacts.outlier_bounds.items():
        df[f"{col}_outlier"] = ((df[col] < lo) | (df[col] > hi)).astype(int)

    return df


if __name__ == "__main__":
    raw = load_raw(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Autonomous_QUOTE_AGENTS.csv"))
    clean, rep, artifacts = clean_and_engineer(raw)
    print("Rows in / out:", rep.rows_in, "/", rep.rows_out)
    print("Exact dupes dropped:", rep.exact_duplicates_dropped)
    print("Re-quote rows identified:", rep.requote_rows_identified)
    print("Dropped redundant cols:", rep.dropped_redundant_cols)
    print("\nNotes:")
    for n in rep.notes:
        print(" -", n)
    clean.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clean_quotes.pkl"))
    print("\nSaved clean_quotes.pkl, shape:", clean.shape)
