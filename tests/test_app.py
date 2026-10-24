import csv
import io
import re

import pytest

from app import create_app, main_reasons
from tests.conftest import make_telecom, write_project


@pytest.fixture()
def client(signal_project, tmp_path):
    art = signal_project.parent / "artifacts"
    app = create_app(signal_project, artifacts_dir=art, db_path=tmp_path / "app.db")
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture()
def noise_client(noise_project, tmp_path):
    art = noise_project.parent / "artifacts"
    return create_app(noise_project, artifacts_dir=art, db_path=tmp_path / "n.db").test_client()


def text(resp) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", resp.get_data(as_text=True)))


def test_all_screens_render(client):
    for url in ("/", "/customers", "/customers?risk=all", "/customers/5", "/drivers", "/model"):
        assert client.get(url).status_code == 200, url


def test_overview_numbers_match_the_data(client):
    t = text(client.get("/"))
    assert "Customers scored 1,500" in t and "High risk 150" in t and "Medium risk 300" in t


def test_warning_banner_only_when_the_model_is_unreliable(client, noise_client):
    assert "no better than guessing" not in text(client.get("/"))
    for url in ("/", "/customers", "/drivers", "/model"):
        assert "no better than guessing" in text(noise_client.get(url)), url


def test_drivers_screen_does_not_invent_actions_for_noise(noise_client):
    t = text(noise_client.get("/drivers"))
    assert "No factor in this data reliably separates" in t
    assert "No action suggested" in t and "Real link to leaving" not in t


def test_drivers_screen_names_real_drivers_with_signal(client):
    t = text(client.get("/drivers"))
    assert "show a real link to leaving" in t or "shows a real link to leaving" in t


def test_list_filters_and_pagination(client):
    t = text(client.get("/customers"))
    assert "Showing 1 to 10 of 150 high-risk customers" in t
    assert "Showing 11 to 20 of 150" in text(client.get("/customers?page=2"))
    one = text(client.get("/customers?risk=all&partner=Vodafone"))
    assert "Airtel" not in re.findall(r"Showing.*", one)[0] and "Vodafone" in one
    found = text(client.get("/customers?risk=all&q=C-0005"))
    assert "C-0005" in found and "Showing 1 to 1 of 1" in found
    assert "No customers match" in text(client.get("/customers?q=C-99999&risk=all"))
    assert "No customers match" in text(client.get("/customers?q=abc&risk=all"))


def test_csv_download_matches_the_filtered_list(client):
    r = client.get("/customers.csv?risk=high")
    assert r.mimetype == "text/csv" and "attachment" in r.headers["Content-Disposition"]
    rows = list(csv.DictReader(io.StringIO(r.get_data(as_text=True))))
    assert len(rows) == 150 and {x["risk_band"] for x in rows} == {"High"}
    assert rows[0]["risk_rank"] == "1"


def test_customer_detail_and_not_found(client):
    t = text(client.get("/customers/5"))
    assert "C-0005" in t and "Suggested next step" in t and "Telecom partner" in t
    assert client.get("/customers/999999").status_code == 404


def test_follow_up_status_and_note_persist(client):
    cid = 5
    assert client.post(f"/customers/{cid}/status", data={"status": "Called"}).status_code == 302
    assert client.post(f"/customers/{cid}/note", data={"note": "Asked for a longer plan"}).status_code == 302
    page = text(client.get(f"/customers/{cid}"))
    assert "Called" in page and "Asked for a longer plan" in page
    called = text(client.get("/customers?risk=all&follow=Called"))
    assert "C-0005" in called and "Showing 1 to 1 of 1" in called


def test_invalid_form_values_are_rejected(client):
    assert client.post("/customers/5/status", data={"status": "Banana"}).status_code == 400
    assert client.post("/customers/999999/note", data={"note": "x"}).status_code == 400
    assert client.post("/drivers/calls_made", data={"owner": "Nobody", "status": "Done"}).status_code == 400
    assert client.post("/drivers/not_a_feature", data={"owner": "Sales", "status": "Done"}).status_code == 400


def test_driver_owner_and_status_persist(client):
    assert client.post("/drivers/calls_made", data={"owner": "Support", "status": "In progress"}).status_code == 302
    html = client.get("/drivers").get_data(as_text=True)
    row = re.search(r'action="/drivers/calls_made".*?</form>', html, re.S).group(0)
    assert re.search(r"<option selected>Support</option>", row)
    assert re.search(r"<option selected>In progress</option>", row)


def test_model_screen_shows_goals_and_slider_data(client, noise_client):
    t = text(client.get("/model"))
    assert "How many customers can your team contact?" in t and "Retrain on the current file" in t
    assert "Met" in t
    html = noise_client.get("/model").get_data(as_text=True)
    assert "Not met" in html and '"pct": 50' in html


def test_reason_text_is_plain_language():
    import json

    raw = json.dumps(
        [
            {
                "feature": "calls_made",
                "label": "Calls made",
                "kind": "numeric",
                "value": 12.0,
                "typical": 49.0,
                "direction": "raises",
                "effect": 0.1,
            },
            {
                "feature": "telecom_partner",
                "label": "Telecom partner",
                "kind": "categorical",
                "value": "Vodafone",
                "typical": None,
                "direction": "raises",
                "effect": 0.05,
            },
        ]
    )
    assert main_reasons(raw) == "Calls made is 12 (typical 49), Telecom partner is Vodafone"


def test_empty_state_when_no_model_exists(tmp_path):
    cfg = write_project(tmp_path, make_telecom(50))
    app = create_app(cfg, artifacts_dir=tmp_path / "nothing", db_path=tmp_path / "e.db")
    c = app.test_client()
    for url in ("/", "/customers", "/drivers", "/model"):
        assert "No model yet" in text(c.get(url)), url


def test_retrain_button_rebuilds_the_artifacts(tmp_path):
    cfg = write_project(tmp_path, make_telecom(600, seed=11))
    app = create_app(cfg, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "r.db")
    c = app.test_client()
    assert "No model yet" in text(c.get("/"))
    r = c.post("/retrain")
    assert r.status_code == 302 and "Retrained" in r.headers["Location"].replace("+", " ")
    assert "Customers scored 600" in text(c.get("/"))
