"""Load and validate the customer file."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import resolve

POSITIVE = {"yes", "y", "true", "t", "1", "churn", "churned", "left", "cancelled", "canceled"}
NEGATIVE = {"no", "n", "false", "f", "0", "retained", "stayed", "active", "not churned"}


def make_target(series: pd.Series) -> pd.Series:
    """Turn a churn column (Yes/No, True/False, 1/0) into 0/1. Unknown labels raise."""
    s = series.dropna()
    if pd.api.types.is_numeric_dtype(s):
        if not set(s.unique()) <= {0, 1}:
            raise ValueError(f"Numeric target must hold only 0 and 1, found {sorted(set(s.unique()))[:10]}")
        return s.astype(int)
    norm = s.astype(str).str.strip().str.lower()
    unknown = set(norm.unique()) - POSITIVE - NEGATIVE
    if unknown:
        raise ValueError(
            f"Unrecognised values in the target column: {sorted(unknown)[:10]}. Use Yes/No, True/False or 1/0."
        )
    return norm.isin(POSITIVE).astype(int)


def clean_frame(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Trim column names and blank impossible negative values. Shared by training and scoring."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    blanked: dict[str, int] = {}
    for col in cfg["data"]["non_negative"]:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col] < 0
        if bad.any():
            blanked[col] = int(bad.sum())
            df.loc[bad, col] = np.nan
    return df, blanked


def load_raw(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Read the CSV, check it, and return it with a data-quality summary."""
    path = resolve(cfg, cfg["data"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Set data.path in config.yaml.")
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    df, blanked = clean_frame(pd.read_csv(path), cfg)
    for needed in (target, id_col):
        if needed not in df.columns:
            raise ValueError(f"Column '{needed}' not found. Columns are: {list(df.columns)}")
    rows_in_file = len(df)
    df = df[df[target].notna()]
    duplicates = int(df.duplicated(subset=[id_col]).sum())
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    quality = {
        "rows_in_file": rows_in_file,
        "rows_used": len(df),
        "duplicate_ids_removed": duplicates,
        "negative_values_blanked": blanked,
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n},
    }
    return df, quality
