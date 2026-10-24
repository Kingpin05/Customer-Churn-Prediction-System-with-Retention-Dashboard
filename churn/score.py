"""Score a CSV of customers with the saved model.

python -m churn.score --input customers.csv --output scored.csv [--model artifacts/model.joblib]
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from churn.data import clean_frame
from churn.features import engineer, make_matrix


def score_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    df, _ = clean_frame(df, bundle["config"])
    id_col = bundle["id_column"]
    X = make_matrix(engineer(df), bundle["numeric"], bundle["categorical"])
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    out = df[[id_col]].copy() if id_col in df.columns else pd.DataFrame({"row": df.index})
    out["score"] = np.round(proba, 5)
    out["flagged"] = (proba >= bundle["threshold"]).astype(int)
    out = out.sort_values("score", ascending=False)
    out.insert(0, "risk_rank", np.arange(1, len(out) + 1))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    scored = score_frame(joblib.load(args.model), pd.read_csv(args.input))
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored)} customers, flagged {int(scored['flagged'].sum())}. Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
