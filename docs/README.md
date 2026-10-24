# Project documents

Read them in this order.

| Order | File | What it answers |
| --- | --- | --- |
| 1 | 01-PRD.pdf | What are we building, for whom, and how do we know it worked? |
| 2 | 02-TRD.pdf | How is it built technically? |
| 3 | 03-UI-UX-design.pdf | What will the manager see and use? |
| 4 | 04-database-schema/001_churn_schema.sql | How would results be stored in a database? |

## How the build differs from the plan

| Plan | Built | Why |
| --- | --- | --- |
| PRD fields: tenure, contract, usage, tickets | Columns in `dataone.csv`: partner, age, city, calls, SMS, data use, salary | The file has no tenure, contract or ticket data |
| 70/15/15 train, validation, test split | 80/20 train and test, with 5-fold cross-validation on train for selection | Smaller data (2,026 rows); validation comes from cross-validation |
| RandomizedSearchCV tuning | Fixed sensible settings for three models | Tuning cannot help when there is no signal; add it when real data arrives |
| SHAP explanations | "Replace with a typical value" check per customer | No extra dependency; works for any model |
| XGBoost or LightGBM | scikit-learn gradient boosting | Fewer dependencies |
| data.py / features.py / train.py / evaluate.py | Same modules under `churn/`, plus `explain.py` and `score.py` | Matches the TRD structure |
| PostgreSQL storage | Files in `artifacts/` plus SQLite for follow-ups | Simpler to run; schema kept for later |
| Mockup: contract, tenure, tickets, "7 in 10" | Real columns and real results; unreliable-model warning | The mockup used sample numbers |

## Data notes for dataone.csv

- 59 impossible negative `data_used` values, 52 in `calls_made` and 62 in `sms_sent` are blanked and filled from the training data.
- `pincode` is almost unique per customer and is dropped as an identifier.
- `date_of_registration` only spans 1 to 11 January 2020, so it carries no tenure information and is dropped.
