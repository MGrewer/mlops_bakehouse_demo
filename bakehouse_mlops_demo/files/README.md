# Bakehouse MLOps Demo

End-to-end MLOps lifecycle on Databricks using the `samples.bakehouse` dataset. Three distinct model training approaches, each with its own experiment and model, followed by serving, drift simulation, promotion, and automated monitoring/retraining.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DEV — Interactive Demo Notebooks                               │
│                                                                 │
│  01 Autolog              02 Decorators                          │
│  experiment:             experiment:                             │
│    bakehouse_autolog       bakehouse_decorators                  │
│  model:                  model:                                  │
│    bakehouse_autolog_      bakehouse_decorators_                 │
│    model                   model                                │
│                                                                 │
│  03 Register + Serve (parameterized — pick any model)           │
│  04 Drift Simulation (parameterized — break any model)          │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  05 Promote to Prod (parameterized)                             │
│  Copy dev champion → prod schema for selected model             │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Automated Job (DABs-defined)                                   │
│  Parameterized by environment + model_variant                   │
│                                                                 │
│  06 Monitoring → 07 Evaluation → 08 Retraining                  │
└─────────────────────────────────────────────────────────────────┘
```

## Notebooks

### 01_autolog
Experiment: `bakehouse_autolog` | Model: `bakehouse_autolog_model`
Demonstrates `mlflow.autolog()`. Trains, registers, runs batch inference. Shows that autolog does NOT capture batch evaluation metrics.

### 02_decorators
Experiment: `bakehouse_decorators` | Model: `bakehouse_decorators_model`
Uses `@mlflow.trace` decorators and explicit logging. Trains, registers, runs batch inference with explicit `batch_rmse`/`batch_r2`/`batch_mae`. Writes predictions table.

### 03_register_serve (widget: model_variant)
Serves any registered model. Sets champion alias, deploys serving endpoint, enables AI Gateway inference table, sends test requests, queries the inference table.

### 04_drift_simulation (widget: model_variant)
Introduces feature drift on the selected model. Logs degraded metrics. Reset cell cleans up.

### 05_promote_to_prod (widget: model_variant)
Copies the dev champion to the prod schema for the selected model variant.

### 06_monitoring (widgets: environment, model_variant)
Queries system tables and MLflow client for experiment runs and metrics. Passes values downstream via task values.

### 07_evaluation (widgets: environment, model_variant)
Compares latest metrics against thresholds (RMSE > 5.0, R2 < 0.5). Sets retrain flag.

### 08_retraining (widgets: environment, model_variant)
Conditionally retrains, registers new version, promotes to champion.

## Bundle Structure

```
bakehouse_mlops_demo/
├── databricks.yml
├── README.md
├── 01_autolog.py
├── 02_decorators.py
├── 03_register_serve.py
├── 04_drift_simulation.py
├── 05_promote_to_prod.py
├── 06_monitoring.py
├── 07_evaluation.py
└── 08_retraining.py
```

## Setup

### Prerequisites
- Access to `samples.bakehouse` in Unity Catalog
- Schemas `mjrent_uc_int.dev_mlops` and `mjrent_uc_int.prod_mlops`
- Databricks Runtime ML cluster
- Model serving enabled (for NB03)

### DABs Deployment
```bash
databricks bundle deploy -t dev
databricks bundle deploy -t prod
```

## Demo Flow

### Part 1 — MLflow Logging Progression
1. Run 01_autolog — train, register, see autolog gap
2. Run 02_decorators — train, register, explicit batch metrics
3. Run 03_register_serve — serve either model, inference table

### Part 2 — Healthy Watcher
4. Deploy DABs, run watcher job — healthy metrics, no retrain

### Part 3 — Break It
5. Run 04_drift_simulation — degraded metrics

### Part 4 — Platform Catches and Fixes
6. Run watcher job — catches breach, retrains, new champion

### Part 5 — Promote to Prod
7. Run 05_promote_to_prod — copy champion to prod

### Reset
8. Run 04_drift_simulation cell 3

## Configuration Reference

| Parameter | Dev | Prod |
|---|---|---|
| Autolog model | `dev_mlops.bakehouse_autolog_model` | `prod_mlops.bakehouse_autolog_model` |
| Decorators model | `dev_mlops.bakehouse_decorators_model` | `prod_mlops.bakehouse_decorators_model` |
| Autolog experiment | `bakehouse_autolog` | same |
| Decorators experiment | `bakehouse_decorators` | same |
| RMSE threshold | 5.0 | 5.0 |
| R2 threshold | 0.5 | 0.5 |
| Watcher schedule | Every 6 hours (America/Detroit) | Every 6 hours (America/Detroit) |
