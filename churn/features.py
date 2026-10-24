"""Feature engineering and the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and band features when their source columns exist. Safe to call twice."""
    df = df.copy()
    cols = set(df.columns)
    if {"data_used", "calls_made"} <= cols:
        df["data_per_call"] = df["data_used"] / df["calls_made"].where(df["calls_made"] > 0)
    if {"sms_sent", "calls_made"} <= cols:
        df["sms_per_call"] = df["sms_sent"] / df["calls_made"].where(df["calls_made"] > 0)
    if {"estimated_salary", "num_dependents"} <= cols:
        df["salary_per_dependent"] = df["estimated_salary"] / (1 + df["num_dependents"])
    if "age" in cols:
        bands = pd.cut(df["age"], [0, 29, 44, 59, 200], labels=["18-29", "30-44", "45-59", "60+"])
        df["age_band"] = bands.astype(object).where(bands.notna(), np.nan)
    return df


def split_columns(df: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str], dict[str, str]]:
    """Choose model inputs. Returns numeric columns, categorical columns and what was dropped."""
    d = cfg["data"]
    dropped: dict[str, str] = {d["id_column"]: "identifier"}
    for c in d["drop_columns"]:
        dropped[c] = "configured drop"
    candidates = [c for c in df.columns if c != d["target"] and c not in dropped]
    for c in candidates:
        if df[c].nunique(dropna=True) <= 1:
            dropped[c] = "constant"
    keep = [c for c in candidates if c not in dropped]
    numeric = [c for c in keep if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in keep if c not in numeric]
    return numeric, categorical, dropped


def make_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns the model needs: {missing}")
    X = df[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    for c in categorical:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def build_pipeline(estimator, numeric: list[str], categorical: list[str], scale: bool) -> Pipeline:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False),
        ),
    ]
    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), categorical)],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", estimator)])
