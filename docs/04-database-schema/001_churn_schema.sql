-- Customer Churn Prediction: backend schema (PostgreSQL 14+)
-- Format: dbmate-style migration (-- migrate:up / -- migrate:down).
--
-- Design notes
--   * Greenfield migration: all tables are empty, so plain CREATE INDEX and
--     constraints are safe here. Any later change to populated tables should
--     follow expand/contract (add nullable, backfill in a script, then constrain;
--     CREATE INDEX CONCURRENTLY; CHECK ... NOT VALID then VALIDATE).
--   * Internal analytics tables, so primary keys are bigint identity. Nothing here
--     is exposed to third parties; the source system's id is kept in
--     customers.external_id.
--   * Lineage is enforced with composite foreign keys: a model run, its split,
--     its feature set and every customer it touches must all belong to the same
--     dataset, and the database refuses anything else.
--   * ON DELETE policy: containment (a run's metrics, a customer's tickets) is
--     CASCADE; dataset-level parents are RESTRICT so a dataset is never wiped by
--     accident; cross references between lineage tables use NO ACTION so an input
--     (a split or feature set) cannot be deleted while a model run depends on it.
--   * NULL is used only where absence has a distinct meaning, and each such column
--     has a CHECK tying it to the condition that explains the NULL.

-- migrate:up

-- ---------------------------------------------------------------------------
-- Enumerated types (fixed sets)
-- ---------------------------------------------------------------------------
CREATE TYPE split_name        AS ENUM ('train', 'validation', 'test');
CREATE TYPE eval_split        AS ENUM ('validation', 'test');
CREATE TYPE algorithm_name    AS ENUM ('dummy', 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm');
CREATE TYPE run_status        AS ENUM ('trained', 'selected', 'rejected');
CREATE TYPE importance_method AS ENUM ('model', 'permutation', 'shap_mean_abs');

-- ---------------------------------------------------------------------------
-- Datasets and raw customer data
-- ---------------------------------------------------------------------------

-- One row per loaded source file. The churn definition is stored with the data
-- because the label is meaningless without it.
CREATE TABLE datasets (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name             text        NOT NULL UNIQUE,
    source           text        NOT NULL,
    churn_definition text        NOT NULL,
    row_count        integer     NOT NULL CHECK (row_count >= 0),
    loaded_at        timestamptz NOT NULL DEFAULT now()
);

-- One row per customer per dataset, with the churn label.
-- Categorical gaps are stored as the literal 'missing' (see TRD cleaning rules),
-- so these columns stay NOT NULL.
CREATE TABLE customers (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id      bigint        NOT NULL REFERENCES datasets (id) ON DELETE RESTRICT,
    external_id     text          NOT NULL,
    tenure_months   integer       NOT NULL CHECK (tenure_months >= 0),
    contract_type   text          NOT NULL,
    plan            text          NOT NULL,
    payment_method  text          NOT NULL,
    monthly_charges numeric(10,2) NOT NULL CHECK (monthly_charges >= 0),
    churned         boolean       NOT NULL,
    created_at      timestamptz   NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, external_id),
    -- Target for composite foreign keys that must stay inside one dataset.
    UNIQUE (dataset_id, id)
);
CREATE INDEX idx_customers_dataset_churned ON customers (dataset_id, churned);

-- Usage per customer per calendar month. Source for usage_frequency / usage_trend.
CREATE TABLE customer_usage_monthly (
    customer_id  bigint        NOT NULL REFERENCES customers (id) ON DELETE CASCADE,
    period_start date          NOT NULL CHECK (EXTRACT(DAY FROM period_start) = 1),
    sessions     integer       NOT NULL CHECK (sessions >= 0),
    usage_volume numeric(14,2) NOT NULL CHECK (usage_volume >= 0),
    PRIMARY KEY (customer_id, period_start)
);

-- Support tickets. resolved_at is NULL only while the ticket is still open.
CREATE TABLE support_tickets (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id bigint      NOT NULL REFERENCES customers (id) ON DELETE CASCADE,
    opened_at   timestamptz NOT NULL,
    resolved_at timestamptz,
    CHECK (resolved_at IS NULL OR resolved_at >= opened_at)
);
CREATE INDEX idx_support_tickets_customer_opened ON support_tickets (customer_id, opened_at);

-- ---------------------------------------------------------------------------
-- Feature engineering
-- ---------------------------------------------------------------------------

-- A named, versioned definition of the engineered features for one dataset.
CREATE TABLE feature_sets (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id  bigint      NOT NULL REFERENCES datasets (id) ON DELETE RESTRICT,
    version     text        NOT NULL,
    description text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, version),
    UNIQUE (id, dataset_id)
);

-- Engineered feature values. Column list mirrors the TRD feature specification;
-- adding a feature means a new migration plus a new feature_sets.version.
-- NULL has a defined meaning for the three nullable columns (see CHECKs).
CREATE TABLE customer_features (
    feature_set_id            bigint        NOT NULL,
    dataset_id                bigint        NOT NULL,
    customer_id               bigint        NOT NULL,
    tenure_bucket             text          NOT NULL,
    usage_frequency           numeric(12,4) NOT NULL CHECK (usage_frequency >= 0),
    usage_trend               numeric(12,4),   -- NULL: not enough history to compute a trend
    tickets_per_month         numeric(12,4) NOT NULL CHECK (tickets_per_month >= 0),
    avg_resolution_hours      numeric(12,4) CHECK (avg_resolution_hours >= 0), -- NULL: no resolved tickets
    charge_per_usage          numeric(14,4) CHECK (charge_per_usage >= 0),     -- NULL: zero usage
    PRIMARY KEY (feature_set_id, customer_id),
    FOREIGN KEY (feature_set_id, dataset_id) REFERENCES feature_sets (id, dataset_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, customer_id)    REFERENCES customers (dataset_id, id)    ON DELETE CASCADE,
    -- A customer with no tickets cannot have a resolution time.
    CHECK (tickets_per_month > 0 OR avg_resolution_hours IS NULL)
);
CREATE INDEX idx_customer_features_customer ON customer_features (dataset_id, customer_id);

-- ---------------------------------------------------------------------------
-- Train / validation / test splits
-- ---------------------------------------------------------------------------
CREATE TABLE split_runs (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id      bigint       NOT NULL REFERENCES datasets (id) ON DELETE RESTRICT,
    seed            integer      NOT NULL,
    train_frac      numeric(4,3) NOT NULL CHECK (train_frac > 0),
    validation_frac numeric(4,3) NOT NULL CHECK (validation_frac > 0),
    test_frac       numeric(4,3) NOT NULL CHECK (test_frac > 0),
    created_at      timestamptz  NOT NULL DEFAULT now(),
    CHECK (train_frac + validation_frac + test_frac = 1),
    UNIQUE (id, dataset_id)
);
CREATE INDEX idx_split_runs_dataset ON split_runs (dataset_id);

-- Each customer lands in exactly one split per split run (the primary key).
CREATE TABLE split_assignments (
    split_run_id bigint     NOT NULL,
    dataset_id   bigint     NOT NULL,
    customer_id  bigint     NOT NULL,
    split        split_name NOT NULL,
    PRIMARY KEY (split_run_id, customer_id),
    FOREIGN KEY (split_run_id, dataset_id) REFERENCES split_runs (id, dataset_id)  ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, customer_id)  REFERENCES customers (dataset_id, id)   ON DELETE CASCADE
);
CREATE INDEX idx_split_assignments_split    ON split_assignments (split_run_id, split);
CREATE INDEX idx_split_assignments_customer ON split_assignments (dataset_id, customer_id);

-- ---------------------------------------------------------------------------
-- Models, evaluation, predictions, explanations
-- ---------------------------------------------------------------------------
CREATE TABLE model_runs (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id       bigint         NOT NULL REFERENCES datasets (id) ON DELETE RESTRICT,
    split_run_id     bigint         NOT NULL,
    feature_set_id   bigint         NOT NULL,
    algorithm        algorithm_name NOT NULL,
    hyperparameters  jsonb          NOT NULL DEFAULT '{}'::jsonb, -- genuinely free-form per algorithm
    cv_folds         smallint       NOT NULL DEFAULT 5 CHECK (cv_folds >= 2),
    seed             integer        NOT NULL,
    status           run_status     NOT NULL DEFAULT 'trained',
    artifact_uri     text           NOT NULL,                     -- saved model + preprocessing pipeline
    library_versions jsonb          NOT NULL,                     -- reproducibility record
    trained_at       timestamptz    NOT NULL DEFAULT now(),
    FOREIGN KEY (split_run_id, dataset_id)   REFERENCES split_runs (id, dataset_id)   ON DELETE NO ACTION,
    FOREIGN KEY (feature_set_id, dataset_id) REFERENCES feature_sets (id, dataset_id) ON DELETE NO ACTION,
    -- Target for composite foreign keys from predictions.
    UNIQUE (id, split_run_id)
);
CREATE INDEX idx_model_runs_dataset ON model_runs (dataset_id);
CREATE INDEX idx_model_runs_split   ON model_runs (split_run_id, dataset_id);
CREATE INDEX idx_model_runs_feature ON model_runs (feature_set_id, dataset_id);
-- At most one selected model per split run.
CREATE UNIQUE INDEX uq_model_runs_one_selected ON model_runs (split_run_id) WHERE status = 'selected';

-- One evaluation row per model per split. The primary key means the test set can
-- be scored once per model run: the pipeline inserts, it never upserts.
CREATE TABLE model_evaluations (
    model_run_id          bigint           NOT NULL REFERENCES model_runs (id) ON DELETE CASCADE,
    eval_split            eval_split       NOT NULL,
    roc_auc               double precision NOT NULL CHECK (roc_auc BETWEEN 0 AND 1),
    pr_auc                double precision NOT NULL CHECK (pr_auc BETWEEN 0 AND 1),
    threshold             numeric(6,5)     NOT NULL CHECK (threshold BETWEEN 0 AND 1),
    precision_at_threshold double precision NOT NULL CHECK (precision_at_threshold BETWEEN 0 AND 1),
    recall_at_threshold    double precision NOT NULL CHECK (recall_at_threshold BETWEEN 0 AND 1),
    f1_at_threshold        double precision NOT NULL CHECK (f1_at_threshold BETWEEN 0 AND 1),
    top_decile_lift       double precision NOT NULL CHECK (top_decile_lift >= 0),
    evaluated_at          timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (model_run_id, eval_split)
);

-- Churn probability per customer per model run. The composite foreign keys force
-- the customer to be in the model run's own split run.
CREATE TABLE predictions (
    model_run_id      bigint       NOT NULL,
    split_run_id      bigint       NOT NULL,
    customer_id       bigint       NOT NULL,
    churn_probability numeric(6,5) NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    risk_rank         integer      NOT NULL CHECK (risk_rank >= 1), -- 1 = most likely to churn
    scored_at         timestamptz  NOT NULL DEFAULT now(),
    PRIMARY KEY (model_run_id, customer_id),
    UNIQUE (model_run_id, risk_rank), -- also serves "top N at-risk" queries
    FOREIGN KEY (model_run_id, split_run_id)  REFERENCES model_runs (id, split_run_id)                ON DELETE CASCADE,
    FOREIGN KEY (split_run_id, customer_id)   REFERENCES split_assignments (split_run_id, customer_id) ON DELETE NO ACTION
);
CREATE INDEX idx_predictions_split_customer ON predictions (split_run_id, customer_id);

-- Top contributing features (SHAP) per scored customer, for the at-risk list.
CREATE TABLE prediction_drivers (
    model_run_id bigint           NOT NULL,
    customer_id  bigint           NOT NULL,
    driver_rank  smallint         NOT NULL CHECK (driver_rank >= 1),
    feature_name text             NOT NULL,
    shap_value   double precision NOT NULL,
    PRIMARY KEY (model_run_id, customer_id, driver_rank),
    UNIQUE (model_run_id, customer_id, feature_name),
    FOREIGN KEY (model_run_id, customer_id) REFERENCES predictions (model_run_id, customer_id) ON DELETE CASCADE
);

-- Global feature importance per model run and method.
CREATE TABLE feature_importances (
    model_run_id    bigint            NOT NULL REFERENCES model_runs (id) ON DELETE CASCADE,
    method          importance_method NOT NULL,
    feature_name    text              NOT NULL,
    importance      double precision  NOT NULL,
    importance_rank integer           NOT NULL CHECK (importance_rank >= 1),
    PRIMARY KEY (model_run_id, method, feature_name),
    UNIQUE (model_run_id, method, importance_rank)
);

-- Business recommendations derived from the top drivers.
CREATE TABLE recommendations (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_run_id     bigint      NOT NULL REFERENCES model_runs (id) ON DELETE CASCADE,
    priority         smallint    NOT NULL CHECK (priority >= 1),
    driver_feature   text        NOT NULL,
    evidence         text        NOT NULL,
    suggested_action text        NOT NULL,
    caveat           text        NOT NULL DEFAULT 'Association in historical data, not proof of causation.',
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_run_id, priority)
);

-- ---------------------------------------------------------------------------
-- Convenience view: ranked at-risk customers for the selected model
-- ---------------------------------------------------------------------------
CREATE VIEW v_selected_model_at_risk AS
SELECT mr.id            AS model_run_id,
       c.external_id    AS customer_external_id,
       p.churn_probability,
       p.risk_rank
FROM model_runs mr
JOIN predictions p ON p.model_run_id = mr.id
JOIN customers   c ON c.id = p.customer_id
WHERE mr.status = 'selected';

-- migrate:down
-- Reverses this migration completely. Destroys all data in these tables, so it is
-- only for development and for rolling back the initial deploy.

DROP VIEW  IF EXISTS v_selected_model_at_risk;
DROP TABLE IF EXISTS recommendations;
DROP TABLE IF EXISTS feature_importances;
DROP TABLE IF EXISTS prediction_drivers;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS model_evaluations;
DROP TABLE IF EXISTS model_runs;
DROP TABLE IF EXISTS split_assignments;
DROP TABLE IF EXISTS split_runs;
DROP TABLE IF EXISTS customer_features;
DROP TABLE IF EXISTS feature_sets;
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS customer_usage_monthly;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS datasets;
DROP TYPE  IF EXISTS importance_method;
DROP TYPE  IF EXISTS run_status;
DROP TYPE  IF EXISTS algorithm_name;
DROP TYPE  IF EXISTS eval_split;
DROP TYPE  IF EXISTS split_name;
