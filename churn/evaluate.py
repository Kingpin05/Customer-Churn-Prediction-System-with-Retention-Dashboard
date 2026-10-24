"""Metrics, the contact-size curve, and checks on which factors really separate leavers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import precision_recall_curve, roc_auc_score


def choose_threshold(y_true, proba, target_recall: float) -> float:
    """Highest-precision threshold that still reaches target_recall; otherwise best F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= target_recall
    if ok.any():
        idx = int(np.flatnonzero(ok)[np.argmax(precision[ok])])
    else:
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        idx = int(np.argmax(f1))
    return float(thresholds[idx])


def lift_at(y_true, proba, fraction: float = 0.10) -> float:
    y = np.asarray(y_true)
    k = max(1, int(np.ceil(fraction * len(y))))
    top = np.argsort(-np.asarray(proba), kind="stable")[:k]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else 0.0


def bootstrap_auc_interval(y_true, proba, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% range for ROC-AUC. If it contains 0.5 the model is indistinguishable from guessing."""
    y, p = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        scores.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def contact_curve(y_true, proba, max_pct: int = 50) -> list[dict]:
    """If the team contacts the top X% by risk, what share of leavers does it reach?"""
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    total_leavers, base = y.sum(), y.mean()
    rows = []
    for pct in range(1, max_pct + 1):
        k = max(1, int(np.ceil(len(y) * pct / 100)))
        hit = y[order[:k]]
        precision = float(hit.mean())
        rows.append(
            {
                "pct": pct,
                "contacts": int(k),
                "precision": precision,
                "recall": float(hit.sum() / total_leavers) if total_leavers else 0.0,
                "lift": precision / base if base else 0.0,
            }
        )
    return rows


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.3g}"


def segment_table(
    df: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    alpha: float,
) -> dict:
    """Churn rate by segment for every factor, with a significance test.

    Because many factors are tested at once, the cut-off is alpha divided by the number of tests.
    """
    base = float(y.mean())
    rows: list[dict] = []
    for c in numeric:
        s = df[c]
        ok = s.notna()
        if ok.sum() < 50 or s[ok].nunique() < 3:
            continue
        p = float(mannwhitneyu(s[ok & (y == 1)], s[ok & (y == 0)]).pvalue)
        bins = pd.qcut(s[ok], 5, duplicates="drop")
        g = pd.DataFrame({"g": bins, "y": y[ok]}).groupby("g", observed=True)["y"].agg(["mean", "size"])

        def grp(iv, g=g):
            return {
                "label": f"{_fmt(iv.left)} to {_fmt(iv.right)}",
                "rate": float(g.loc[iv, "mean"]),
                "n": int(g.loc[iv, "size"]),
            }

        rows.append(
            {
                "feature": c,
                "kind": "numeric",
                "p_value": p,
                "high": grp(g["mean"].idxmax()),
                "low": grp(g["mean"].idxmin()),
            }
        )
    for c in categorical:
        s = df[c].astype(object).fillna("missing")
        tab = pd.crosstab(s, y)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        p = float(chi2_contingency(tab)[1])
        g = pd.DataFrame({"mean": tab[1] / tab.sum(axis=1), "size": tab.sum(axis=1)})
        g = g[g["size"] >= 30]
        if len(g) < 2:
            continue

        def grp_c(k, g=g):
            return {"label": str(k), "rate": float(g.loc[k, "mean"]), "n": int(g.loc[k, "size"])}

        rows.append(
            {
                "feature": c,
                "kind": "categorical",
                "p_value": p,
                "high": grp_c(g["mean"].idxmax()),
                "low": grp_c(g["mean"].idxmin()),
            }
        )
    n_tests = max(1, len(rows))
    cutoff = alpha / n_tests
    for r in rows:
        r["reliable"] = bool(r["p_value"] < cutoff)
    rows.sort(key=lambda r: r["p_value"])
    return {"base_rate": base, "n_tests": n_tests, "cutoff": cutoff, "items": rows}
