"""Customer churn prediction: data, features, training, evaluation and scoring.

Typical use from Python:

    from churn import load_config, train_model, score_customers

    cfg = load_config("config.yaml")
    train_model(cfg)                      # writes ./artifacts
    scored = score_customers("artifacts/model.joblib", "new_customers.csv")
"""

from __future__ import annotations

from pathlib import Path

from churn.config import load_config

__version__ = "1.0.0"
__all__ = ["__version__", "load_config", "score_customers", "train_model"]


def train_model(cfg: dict, gate: bool = False) -> int:
    """Train on the data file named in cfg. Returns 0, or 1 if gate=True and the targets are missed."""
    from churn.train import run  # imported here so `import churn` stays fast

    return run(cfg, gate=gate)


def score_customers(model_path: str | Path, csv_path: str | Path):
    """Score a CSV of customers with a saved model. Returns a DataFrame ranked by risk."""
    import joblib
    import pandas as pd

    from churn.score import score_frame

    return score_frame(joblib.load(model_path), pd.read_csv(csv_path))