import json

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config
from churn.score import score_frame
from churn.train import run
from tests.conftest import make_telecom, write_project


def read(project, name):
    return project.parent / "artifacts" / name


def metrics(project):
    return json.loads(read(project, "metrics.json").read_text())


def test_artifacts_are_written(signal_project):
    for name in (
        "model.joblib",
        "metrics.json",
        "customers_scored.csv",
        "segments.json",
        "feature_importance.csv",
    ):
        assert read(signal_project, name).exists()


def test_model_beats_the_baseline_when_there_is_signal(signal_project):
    m = metrics(signal_project)
    assert m["test"]["roc_auc"] > 0.65
    assert m["cv_pr_auc"][m["selected_model"]] > m["cv_pr_auc"]["dummy"] + 0.03
    assert m["reliable"] is True


def test_shuffled_noise_is_flagged_as_unreliable(noise_project):
    m = metrics(noise_project)
    assert m["reliable"] is False
    assert m["test"]["roc_auc_range"][0] <= 0.5 or m["test"]["roc_auc"] < 0.6


def test_identifiers_and_dropped_columns_never_reach_the_model(signal_project):
    m = metrics(signal_project)
    used = m["features"]["numeric"] + m["features"]["categorical"]
    assert not {"customer_id", "pincode", "date_of_registration", "churn"} & set(used)
    assert m["features"]["dropped"]["customer_id"] == "identifier"


def test_every_customer_is_scored_once_out_of_sample(signal_project):
    df = pd.read_csv(read(signal_project, "customers_scored.csv"))
    assert df["customer_id"].is_unique and len(df) == 1500
    assert sorted(df["risk_rank"]) == list(range(1, 1501))
    assert set(df["split"]) == {"train", "test"}
    assert (df["split"] == "test").mean() == pytest.approx(0.2, abs=0.01)
    high = (df["band"] == "High").mean()
    assert high == pytest.approx(0.10, abs=0.005)
    assert (df.sort_values("risk_rank")["score"].diff().dropna() <= 1e-9).all()


def test_top_ranked_customers_really_churn_more_when_there_is_signal(signal_project):
    df = pd.read_csv(read(signal_project, "customers_scored.csv")).sort_values("risk_rank")
    top = df.head(150)["actual_churned"].mean()
    assert top > 1.5 * df["actual_churned"].mean()


def test_reasons_are_valid_json_with_known_directions(signal_project):
    df = pd.read_csv(read(signal_project, "customers_scored.csv"))
    for raw in df["reasons_json"].head(50):
        items = json.loads(raw)
        assert len(items) <= 4
        assert {i["direction"] for i in items} <= {"raises", "lowers"}


def test_contact_curve_is_consistent(signal_project):
    curve = metrics(signal_project)["curve"]
    assert len(curve) == 50 and curve[0]["pct"] == 1
    recalls = [c["recall"] for c in curve]
    assert recalls == sorted(recalls)
    assert curve[49]["contacts"] == 750


def test_real_drivers_are_found_with_signal_and_not_in_noise(signal_project, noise_project):
    real = json.loads(read(signal_project, "segments.json").read_text())
    names = {s["feature"] for s in real["items"] if s["reliable"]}
    assert {"calls_made", "data_used", "telecom_partner"} & names
    noise = json.loads(read(noise_project, "segments.json").read_text())
    assert not [s for s in noise["items"] if s["reliable"]]


def test_goals_report_met_and_not_met(noise_project):
    goals = metrics(noise_project)["goals"]
    assert len(goals) == 4 and not all(g["met"] for g in goals)


def test_quality_gate_fails_the_run_on_noise(noise_project):
    assert run(load_config(noise_project), gate=True) == 1


def test_same_seed_gives_the_same_result(tmp_path):
    df = make_telecom(700, seed=3)
    results = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        path = write_project(folder, df)
        run(load_config(path))
        results.append(metrics(path)["test"]["roc_auc"])
    assert results[0] == results[1]


def test_scoring_new_customers(signal_project):
    import joblib

    bundle = joblib.load(read(signal_project, "model.joblib"))
    new = make_telecom(40, seed=9).drop(columns=["churn"])
    out = score_frame(bundle, new)
    assert len(out) == 40 and out["score"].between(0, 1).all()
    assert set(out["customer_id"]) == set(new["customer_id"])
    assert out["risk_rank"].tolist() == list(range(1, 41))


def test_scoring_reports_missing_columns(signal_project):
    import joblib

    bundle = joblib.load(read(signal_project, "model.joblib"))
    bad = make_telecom(10, seed=4).drop(columns=["churn", "calls_made"])
    with pytest.raises(ValueError, match="calls_made"):
        score_frame(bundle, bad)


def test_too_few_churners_gives_a_clear_error(tmp_path):
    df = make_telecom(300, seed=5)
    df["churn"] = 0
    df.loc[:3, "churn"] = 1
    with pytest.raises(ValueError, match="Too few rows"):
        run(load_config(write_project(tmp_path, df)))


def test_ratio_features_do_not_blow_up_on_zero_calls(signal_project):
    from churn.features import engineer

    df = pd.DataFrame(
        {
            "calls_made": [0, 10, np.nan],
            "data_used": [5, 50, 3],
            "sms_sent": [1, 2, 3],
            "estimated_salary": [100, 200, 300],
            "num_dependents": [0, 1, 2],
            "age": [20, 40, 70],
        }
    )
    out = engineer(df)
    assert np.isnan(out.loc[0, "data_per_call"]) and out.loc[1, "data_per_call"] == 5
    assert np.isfinite(out["salary_per_dependent"]).all()
    assert out["age_band"].tolist() == ["18-29", "30-44", "60+"]
