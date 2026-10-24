import pandas as pd
import pytest

from churn.config import load_config
from churn.data import clean_frame, load_raw, make_target
from tests.conftest import make_telecom, write_project


def test_target_accepts_binary_and_yes_no():
    assert make_target(pd.Series([1, 0, 1])).tolist() == [1, 0, 1]
    assert make_target(pd.Series(["Yes", " no ", "YES"])).tolist() == [1, 0, 1]


def test_target_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unrecognised"):
        make_target(pd.Series(["Yes", "No", "Maybe"]))
    with pytest.raises(ValueError, match="only 0 and 1"):
        make_target(pd.Series([0, 1, 2]))


def test_impossible_negative_values_are_blanked_and_counted(tmp_path):
    cfg = load_config(write_project(tmp_path, make_telecom(50)))
    df = pd.DataFrame({"calls_made": [10, -3, 5], "data_used": [-1, -2, 7], "age": [30, 40, 50]})
    cleaned, blanked = clean_frame(df, cfg)
    assert blanked == {"calls_made": 1, "data_used": 2}
    assert cleaned["calls_made"].isna().sum() == 1 and cleaned["age"].notna().all()


def test_load_raw_removes_duplicate_ids_and_reports_quality(tmp_path):
    df = make_telecom(100)
    df = pd.concat([df, df.iloc[:5]])
    cfg = load_config(write_project(tmp_path, df))
    out, quality = load_raw(cfg)
    assert len(out) == 100 and quality["duplicate_ids_removed"] == 5
    assert quality["negative_values_blanked"]["calls_made"] > 0


def test_missing_target_column_names_the_columns(tmp_path):
    cfg = load_config(write_project(tmp_path, make_telecom(60).rename(columns={"churn": "left"})))
    with pytest.raises(ValueError, match="'churn' not found"):
        load_raw(cfg)
