"""Load config.yaml and resolve paths relative to it."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "data": {
        "path": "data/raw/dataone.csv",
        "target": "churn",
        "id_column": "customer_id",
        "drop_columns": [],
        "non_negative": [],
    },
    "model": {
        "seed": 42,
        "test_size": 0.2,
        "cv_folds": 5,
        "target_recall": 0.70,
        "candidates": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
    },
    "risk_bands": {"high_top_pct": 10, "medium_top_pct": 30},
    "goals": {"roc_auc": 0.80, "recall": 0.70, "precision": 0.50, "top_decile_lift": 2.5},
    "gate": {"min_roc_auc": 0.75, "min_recall": 0.60},
    "unreliable_below_auc": 0.60,
    "significance_alpha": 0.05,
    "labels": {},
    "actions": {},
    "output_dir": "artifacts",
}


def _merge(base: dict, extra: dict | None) -> dict:
    out = deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    user = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    cfg = _merge(DEFAULTS, user)
    cfg["_root"] = str(p.resolve().parent)
    return cfg


def resolve(cfg: dict, relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def label(cfg: dict, name: str) -> str:
    return cfg["labels"].get(name) or name.replace("_", " ").capitalize()
