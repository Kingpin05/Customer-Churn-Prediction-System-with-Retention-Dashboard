"""Train the churn model and write everything the app needs to ./artifacts.

python -m churn.train [--config config.yaml] [--gate] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from churn.config import label, load_config, resolve
from churn.data import load_raw, make_target
from churn.evaluate import (
    bootstrap_auc_interval,
    choose_threshold,
    contact_curve,
    lift_at,
    segment_table,
)
from churn.explain import background_values, effects, reasons_json
from churn.features import build_pipeline, engineer, make_matrix, split_columns


def candidate_models(seed: int) -> dict[str, tuple[object, bool]]:
    """name -> (estimator, needs_scaling)."""
    return {
        "dummy": (DummyClassifier(strategy="prior"), False),
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=seed),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_depth=4,
                class_weight="balanced",
                random_state=seed,
            ),
            False,
        ),
    }


def _r(x, n: int = 4):
    return None if x is None else round(float(x), n)


def run(cfg: dict, gate: bool = False) -> int:
    mcfg = cfg["model"]
    seed = mcfg["seed"]
    out_dir = resolve(cfg, cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, quality = load_raw(cfg)
    y = make_target(raw[cfg["data"]["target"]]).reset_index(drop=True)
    df = engineer(raw)
    numeric, categorical, dropped = split_columns(df, cfg)
    X = make_matrix(df, numeric, categorical)
    n = len(X)
    if min(y.sum(), n - y.sum()) < 2 * mcfg["cv_folds"]:
        raise ValueError("Too few rows in one class for stratified cross-validation.")
    print(f"Customers: {n}  churn rate: {y.mean():.3f}")
    print(f"Numeric: {numeric}\nCategorical: {categorical}\nDropped: {dropped}")

    idx_tr, idx_te = train_test_split(np.arange(n), test_size=mcfg["test_size"], stratify=y, random_state=seed)
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
    skf = StratifiedKFold(n_splits=mcfg["cv_folds"], shuffle=True, random_state=seed)
    catalog = candidate_models(seed)

    cv_scores: dict[str, float] = {}
    for name in ["dummy", *mcfg["candidates"]]:
        est, scale = catalog[name]
        pipe = build_pipeline(est, numeric, categorical, scale)
        cv_scores[name] = float(cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="average_precision").mean())
        print(f"  CV PR-AUC {name:<24}{cv_scores[name]:.4f}")

    best = None
    for name in mcfg["candidates"]:  # simpler models come first and win near-ties
        if best is None or cv_scores[name] > cv_scores[best] + 0.005:
            best = name
    est, scale = catalog[best]
    pipe = build_pipeline(est, numeric, categorical, scale)

    # Every customer gets a score from a model that did not see them:
    # training customers via cross-validation, test customers via the final model.
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_tr, oof, mcfg["target_recall"])
    pipe.fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    scores = np.empty(n)
    scores[idx_tr], scores[idx_te] = oof, p_te
    split = np.where(np.isin(np.arange(n), idx_te), "test", "train")

    pred = (p_te >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_te, p_te))
    ci_lo, ci_hi = bootstrap_auc_interval(y_te, p_te, seed=seed)
    test = {
        "roc_auc": _r(auc),
        "roc_auc_range": [_r(ci_lo), _r(ci_hi)],
        "pr_auc": _r(average_precision_score(y_te, p_te)),
        "precision": _r(precision_score(y_te, pred, zero_division=0)),
        "recall": _r(recall_score(y_te, pred, zero_division=0)),
        "f1": _r(f1_score(y_te, pred, zero_division=0)),
        "top_decile_lift": _r(lift_at(y_te, p_te)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "flagged_share": _r(pred.mean()),
        "churn_rate": _r(y_te.mean()),
    }
    reliable = bool(ci_lo > 0.5 and auc >= cfg["unreliable_below_auc"])

    g = cfg["goals"]
    goals = [
        {
            "name": "How well customers are ranked from safest to riskiest (ROC-AUC)",
            "goal": f"{g['roc_auc']:.2f} or more",
            "result": f"{test['roc_auc']:.2f}",
            "met": bool(test["roc_auc"] >= g["roc_auc"]),
        },
        {
            "name": "Customers who leave that the list catches (recall)",
            "goal": f"{g['recall']:.0%} or more",
            "result": f"{test['recall']:.0%}",
            "met": bool(test["recall"] >= g["recall"]),
        },
        {
            "name": "Contacted customers who really leave (precision)",
            "goal": f"{g['precision']:.0%} or more",
            "result": f"{test['precision']:.0%}",
            "met": bool(test["precision"] >= g["precision"]),
        },
        {
            "name": "Riskiest 10% compared with an average group (lift)",
            "goal": f"{g['top_decile_lift']:.1f} times or more",
            "result": f"{test['top_decile_lift']:.1f} times",
            "met": bool(test["top_decile_lift"] >= g["top_decile_lift"]),
        },
    ]

    # What separates leavers from stayers, and which customers the model pushes up.
    segments = segment_table(df, y, numeric, categorical, cfg["significance_alpha"])
    imp = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=1)
    importance = (
        pd.DataFrame(
            {
                "feature": X_te.columns,
                "label": [label(cfg, c) for c in X_te.columns],
                "importance_mean": imp.importances_mean,
                "importance_std": imp.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .round(5)
    )
    bg = background_values(X_tr, numeric, categorical)
    eff = effects(pipe, X, bg)
    reasons = reasons_json(X, eff, bg, numeric, cfg)

    rank = pd.Series(scores).rank(method="first", ascending=False).astype(int)
    pct = rank / n * 100
    b = cfg["risk_bands"]
    band = np.where(pct <= b["high_top_pct"], "High", np.where(pct <= b["medium_top_pct"], "Medium", "Low"))
    scored = raw.drop(columns=[cfg["data"]["target"]]).copy()
    scored.insert(1, "split", split)
    scored["score"] = np.round(scores, 5)
    scored["risk_rank"] = rank.to_numpy()
    scored["risk_percentile"] = np.round(pct.to_numpy(), 2)
    scored["band"] = band
    scored["actual_churned"] = y.to_numpy()
    scored["reasons_json"] = reasons
    scored = scored.sort_values("risk_rank")

    data_path = resolve(cfg, cfg["data"]["path"])
    metrics = {
        "selected_model": best,
        "cv_pr_auc": {k: _r(v) for k, v in cv_scores.items()},
        "threshold": _r(threshold, 5),
        "test": test,
        "all_customers": {
            "roc_auc": _r(roc_auc_score(y, scores)),
            "top_decile_lift": _r(lift_at(y, scores)),
        },
        "reliable": reliable,
        "goals": goals,
        "curve": contact_curve(y, scores),
        "bands": {k: int((band == k).sum()) for k in ("High", "Medium", "Low")},
        "data": {
            "file": data_path.name,
            "rows": n,
            "train": int(len(idx_tr)),
            "test": int(len(idx_te)),
            "churn_rate": _r(y.mean()),
            "loaded_at": datetime.fromtimestamp(data_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d"),
        },
        "quality": quality,
        "features": {"numeric": numeric, "categorical": categorical, "dropped": dropped},
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    bundle = {
        "pipeline": pipe,
        "threshold": threshold,
        "numeric": numeric,
        "categorical": categorical,
        "id_column": cfg["data"]["id_column"],
        "config": {k: v for k, v in cfg.items() if k != "_root"},
        "model_name": best,
        "trained_at": metrics["trained_at"],
    }
    joblib.dump(bundle, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "segments.json").write_text(json.dumps(segments, indent=2), encoding="utf-8")
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    scored.to_csv(out_dir / "customers_scored.csv", index=False)

    print(f"\nSelected model: {best}   threshold: {threshold:.4f}")
    print(
        f"Held-out ROC-AUC {test['roc_auc']} (95% range {ci_lo:.2f} to {ci_hi:.2f}), "
        f"recall {test['recall']}, precision {test['precision']}, top-decile lift {test['top_decile_lift']}"
    )
    if not reliable:
        print("WARNING: this model is no better than guessing on this data. Do not act on its ranking.")
    real = [s for s in segments["items"] if s["reliable"]]
    print(f"Factors with a reliable link to leaving: {[s['feature'] for s in real] or 'none'}")

    if gate:
        failures = []
        if test["roc_auc"] < cfg["gate"]["min_roc_auc"]:
            failures.append(f"ROC-AUC {test['roc_auc']} < {cfg['gate']['min_roc_auc']}")
        if test["recall"] < cfg["gate"]["min_recall"]:
            failures.append(f"recall {test['recall']} < {cfg['gate']['min_recall']}")
        if failures:
            print("QUALITY GATE FAILED: " + "; ".join(failures), file=sys.stderr)
            return 1
        print("Quality gate passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", help="Override output_dir.")
    p.add_argument("--gate", action="store_true", help="Exit 1 if the quality targets are missed.")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.out:
        cfg["output_dir"] = str(Path(args.out).resolve())
    return run(cfg, gate=args.gate)


if __name__ == "__main__":
    sys.exit(main())
