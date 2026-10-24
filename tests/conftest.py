"""Shared fixtures: small synthetic customer files in the same shape as data/raw/dataone.csv."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from churn.config import load_config
from churn.train import run

PARTNERS = ["Airtel", "BSNL", "Reliance Jio", "Vodafone"]
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]


def make_telecom(n: int = 1500, seed: int = 0, signal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    calls = rng.integers(-5, 110, n)  # a few impossible negatives, as in the real file
    sms = rng.integers(0, 55, n)
    data = rng.integers(-200, 10000, n)
    partner = rng.choice(PARTNERS, n)
    age = rng.integers(18, 75, n)
    logit = -1.8 + (0.03 * (50 - calls) + 0.0003 * (5000 - data) + 0.8 * (partner == "Vodafone")) * signal
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    if not signal:
        churn = (rng.random(n) < 0.2).astype(int)
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "telecom_partner": partner,
            "gender": rng.choice(["M", "F"], n),
            "age": age,
            "state": rng.choice(["Goa", "Kerala", "Punjab", "Assam"], n),
            "city": rng.choice(CITIES, n),
            "pincode": rng.integers(100000, 999999, n),
            "date_of_registration": "2020/1/1",
            "num_dependents": rng.integers(0, 5, n),
            "estimated_salary": rng.integers(20000, 150000, n),
            "calls_made": calls,
            "sms_sent": sms,
            "data_used": data,
            "churn": churn,
        }
    )


def write_project(folder, df: pd.DataFrame, **overrides):
    """Write data + config into folder and return the config path."""
    df.to_csv(folder / "customers.csv", index=False)
    cfg = {
        "data": {
            "path": "customers.csv",
            "target": "churn",
            "id_column": "customer_id",
            "drop_columns": ["pincode", "date_of_registration"],
            "non_negative": ["age", "estimated_salary", "calls_made", "sms_sent", "data_used"],
        },
        "model": {"seed": 7, "candidates": ["logistic_regression", "hist_gradient_boosting"]},
        "output_dir": "artifacts",
    }
    cfg.update(overrides)
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture(scope="session")
def signal_project(tmp_path_factory):
    folder = tmp_path_factory.mktemp("signal")
    path = write_project(folder, make_telecom(1500, seed=1, signal=True))
    assert run(load_config(path)) == 0
    return path


@pytest.fixture(scope="session")
def noise_project(tmp_path_factory):
    folder = tmp_path_factory.mktemp("noise")
    path = write_project(folder, make_telecom(1500, seed=2, signal=False))
    assert run(load_config(path)) == 0
    return path
