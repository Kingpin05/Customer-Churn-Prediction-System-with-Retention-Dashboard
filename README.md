# Churn Watch: customer churn prediction

Trains a churn model on a customer CSV, then serves it in a small web app (overview, at-risk list,
customer detail, drivers and actions, model and data) with CI/CD around it.

## Read this first: what the included data can and cannot do

The included file `data/raw/dataone.csv` (2,026 telecom customers, 19.4% churned) has **no usable
signal for predicting churn**. This is a finding about the data, not a bug in the code.

| Check | Result |
| --- | --- |
| Held-out ROC-AUC of the best model | 0.55 (95% range 0.48 to 0.62; 0.50 is a coin flip) |
| ROC-AUC when every customer is scored out of sample | 0.50 |
| Churn rate in the riskiest 10% vs everyone | 1.0 times (no lift) |
| Smallest p-value among 14 factors tested | 0.28 (needs to be below 0.0036 to count) |

None of partner, gender, age, city, state, dependents, salary, calls, SMS or data use separates the
customers who left from those who stayed. The app says so on every screen instead of showing made-up
drivers, and the CD quality gate fails for this file on purpose. Use data that has real signal
(the PRD's fields: tenure, contract, usage trend, support tickets, billing) and everything here works
unchanged: edit `config.yaml` and retrain.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install          # or: pip install -r requirements.txt -r requirements-dev.txt
make train            # trains on data/raw/dataone.csv, writes ./artifacts
make app              # http://127.0.0.1:8000
make check            # lint + 36 tests
```

To use your own CSV, put it in `data/raw/` and edit `config.yaml`: `data.path`, `data.target`,
`data.id_column`, and `data.drop_columns` (anything recorded after a customer left leaks the answer).
You can also press **Retrain on the current file** on the Model and data screen.

## Layout

```
churn-watch/
├── README.md
├── config.yaml              every setting: data, model, risk bands, goals, gate
├── data/raw/                the customer CSV
├── churn/                   the ML package (TRD stages)
│   ├── config.py            read config.yaml
│   ├── data.py              load, validate, clean
│   ├── features.py          feature engineering and preprocessing pipeline
│   ├── train.py             compare models, pick threshold, write artifacts   (python -m churn.train)
│   ├── evaluate.py          metrics, contact curve, which factors really matter
│   ├── explain.py           plain-language reasons for each customer's score
│   └── score.py             score a new CSV                                    (python -m churn.score)
├── app/                     the web app (Flask)
│   ├── __init__.py          routes for the five screens
│   ├── store.py             reads artifacts; keeps follow-ups and notes in SQLite
│   ├── templates/  static/  the screens and stylesheet
├── tests/                   36 tests: data, training, scoring, every screen
├── docs/                    PRD, TRD, UI/UX design, database schema (read in order)
├── artifacts/               model and reports written by training (not committed)
├── Dockerfile  wsgi.py  Makefile  pyproject.toml  requirements*.txt
└── .github/workflows/       ci.yml and cd.yml
```

## How the screens work

| Screen | What it shows |
| --- | --- |
| Overview | Counts by risk band, the first customers to call, how strongly each factor links to leaving, model check |
| At-risk customers | Filter by risk, partner, follow-up status or ID; page through; download CSV |
| Customer detail | Why the customer ranks where they do, account facts, usage vs typical, notes, mark called or offer sent |
| Drivers and actions | Each factor with churn rate by group and a significance check; owner and status per action |
| Model and data | Contact-size slider from real outcomes, goals met or not met, data in use, retrain |

Risk bands are by rank (top 10% High, next 20% Medium), so they mean the same whatever the model
scores look like. Every customer is scored by a model that did not see them, so the list is honest.
Follow-up status, notes and driver owners are saved in `var/app.db` (SQLite).

Changes from the design mockup, forced by the data: there is no contract, tenure or support-ticket
column in `dataone.csv`, so the screens use the columns that exist (partner, age, city, calls, SMS,
data use), and the mockup's sample figures are replaced by real results.

## CI/CD (GitHub Actions)

- **CI** (`ci.yml`) runs on every pull request and push to `main`: lint, then tests on Python 3.10 and 3.12.
  Tests build their own synthetic customers, so CI never needs your data.
- **CD** (`cd.yml`) runs when you push a tag like `v1.0.0`, or press *Run workflow*: lint and tests, get the
  data, train with the quality gate, upload the model and reports, push a web-app image to GitHub Container
  Registry, and attach the files to a release.
  - The data file comes from the repo, or from a `DATA_URL` repository secret.
  - With `dataone.csv` the gate fails by design. To publish anyway, run the workflow manually and untick
    *Fail if the model misses the quality gate*.

```bash
git init && git add . && git commit -m "Churn Watch"
git branch -M main && git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
git tag v1.0.0 && git push --tags
```

Run the image: `docker run --rm -p 8000:8000 ghcr.io/<you>/<repo>:v1.0.0`

## Not done yet

- The PostgreSQL schema in `docs/04-database-schema/` is not connected. The app uses files plus SQLite;
  the schema is the design for storing results in a real database.
- No SHAP: reasons come from a simpler "replace with a typical value" check (`churn/explain.py`).
- The TRD planned a separate validation split and tuning search; the code uses a train/test split with
  cross-validation. Details are in `docs/README.md`.
