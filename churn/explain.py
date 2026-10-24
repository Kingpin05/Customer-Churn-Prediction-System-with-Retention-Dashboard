"""Plain-language reasons for each customer's score, by occlusion.

For each factor we replace the customer's value with a typical one and see how much the model's
score drops. A big drop means that factor pushed this customer's score up.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from churn.config import label


def background_values(X_train: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    bg = {c: float(X_train[c].median()) for c in numeric}
    for c in categorical:
        mode = X_train[c].mode(dropna=True)
        bg[c] = mode.iloc[0] if len(mode) else "missing"
    return bg


def effects(pipe, X: pd.DataFrame, background: dict) -> pd.DataFrame:
    """Change in score when each factor is set to a typical value (positive = raised the score)."""
    base = pipe.predict_proba(X)[:, 1]
    out = {}
    for c in X.columns:
        Xm = X.copy()
        Xm[c] = background[c]
        out[c] = base - pipe.predict_proba(Xm)[:, 1]
    return pd.DataFrame(out, index=X.index)


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def reasons_json(
    X: pd.DataFrame,
    eff: pd.DataFrame,
    background: dict,
    numeric: list[str],
    cfg: dict,
    n_raise: int = 3,
    n_lower: int = 1,
) -> list[str]:
    """One JSON string per customer: the factors that raised and lowered their score."""
    numeric_set = set(numeric)
    out = []
    for i in range(len(X)):
        e = eff.iloc[i]
        row = X.iloc[i]
        picks = [(c, "raises") for c in e[e > 0].sort_values(ascending=False).head(n_raise).index]
        picks += [(c, "lowers") for c in e[e < 0].sort_values().head(n_lower).index]
        items = []
        for c, direction in picks:
            is_num = c in numeric_set
            items.append(
                {
                    "feature": c,
                    "label": label(cfg, c),
                    "kind": "numeric" if is_num else "categorical",
                    "value": _clean(row[c]),
                    "typical": _clean(background[c]) if is_num else None,
                    "direction": direction,
                    "effect": round(float(e[c]), 5),
                }
            )
        out.append(json.dumps(items))
    return out
