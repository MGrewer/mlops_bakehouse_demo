# Bakehouse MLOps Demo

End-to-end MLOps lifecycle on Databricks using the `samples.bakehouse` dataset. Trains a demand forecasting model on bakery franchise transaction data, deploys it for batch and real-time inference, monitors performance via MLflow system tables, and automatically retrains when metrics degrade.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Interactive Demo Notebooks                                     │
│                                                                 │
│  01 Autolog ──── 02 Decorators ──── 03 Register + Serve         │
│       │                  │                      │               │
│       │                  ▼                      ▼               │
│       │           UC Model Registry      Serving Endpoint       │
│       │           (champion alias)       (inference table)      │
│       ▼                  │                      │               │
│  MLflow Experiment ◄─────┘──────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  04 Drift Simulation                                            │
│  Corrupt features → score → log degraded metrics → break it     │
│  Reset cell cleans up drift runs and predictions table           │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Automated Job (DABs-defined, runs every 6 hours)               │
│                                                                 │
│  05 Monitoring ──► 06 Evaluation ──► 07 Retraining              │
│  (system tables)   (threshold check)  (conditional)             │
└─────────────────────────────────────────────────────────────────┘
```

## Notebooks

### 01_autolog
Demonstrates `mlflow.autolog()`. Joins transaction, customer, and franchise tables, engineers time-based features, trains a RandomForest pipeline. MLflow captures params, metrics, and the model artifact automatically with zero manual logging code.

### 02_decorators
Same data and model, but with explicit control. Uses `@mlflow.trace` decorators on feature prep, pipeline build, and evaluation functions. Manually logs params, metrics, dataset lineage via `mlflow.log_input()`, and registers the model to Unity Catalog.

### 03_register_serve
Sets the champion alias on the latest model version, deploys a serving endpoint with inference table auto-capture enabled, sends test requests, and queries the auto-generated inference table.

### 04_drift_simulation
Introduces feature drift by randomizing numeric columns (unitPrice, hour, day_of_week, month), scores with the champion model, and logs degraded metrics tagged `drift_simulation=true`. The reset cell deletes all drift-tagged runs and drops the predictions table to restore clean demo state.

### 05_monitoring
Queries `system.mlflow.model_versions` for model version history. Uses the MLflow client to pull the latest batch inference metrics. Passes metric values downstream via `dbutils.jobs.taskValues`.

### 06_evaluation
Reads metrics from the monitoring task and compares them against defined thresholds (RMSE > 5.0, R2 < 0.5). Sets a `retrain_needed` flag for the next task.

### 07_retraining
Checks the retrain flag and exits early if metrics are healthy. Otherwise retrains on clean data with tuned hyperparameters, registers a new model version, and promotes it to champion.

## Bundle Structure

```
bakehouse_mlops_demo/
├── databricks.yml
├── README.md
├── 01_autolog.py
├── 02_decorators.py
├── 03_register_serve.py
├── 04_drift_simulation.py
├── 05_monitoring.py
├── 06_evaluation.py
└── 07_retraining.py
```

## Setup

### Prerequisites
- Access to `samples.bakehouse` in Unity Catalog
- Schemas `mjrent_uc_int.dev_mlops` and `mjrent_uc_int.prod_mlops` created
- A compute cluster running **Databricks Runtime ML** (e.g., 16.x ML)
- Model serving enabled on the workspace (for notebook 03)

### DABs Deployment
```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle deploy -t prod
```

Dev deploys to:
`/Workspace/Users/mitchell.grewer@databricks.com/mlops_bakehouse_demo/bakehouse_mlops_demo`

Prod deploys to:
`/Workspace/Shared/bakehouse_mlops_published`

## Demo Flow

### Part 1 — MLflow Logging Progression
1. Run 01_autolog — show the experiment UI, point out what got captured for free
2. Run 02_decorators — compare the run in the experiment UI, show the traces, dataset lineage, explicit metrics
3. Run 03_register_serve — show the model in UC, the serving endpoint, the inference table populating

### Part 2 — Break It
4. Run 04_drift_simulation cells 1-2 — show the degraded metrics in the experiment UI

### Part 3 — Platform Catches and Fixes It
5. Run the watcher job (or step through 05-07 interactively)
6. Show 05 output — system tables surfacing the metric decline
7. Show 06 output — thresholds breached
8. Show 07 output — retrained model with recovered metrics, new version promoted to champion

### Reset
9. Run 04_drift_simulation cell 3 to clean up drift artifacts and restore baseline state

## Configuration Reference

| Parameter | Value |
|---|---|
| Dev catalog.schema | `mjrent_uc_int.dev_mlops` |
| Prod catalog.schema | `mjrent_uc_int.prod_mlops` |
| Model name | `mjrent_uc_int.dev_mlops.bakehouse_demand_model` |
| Serving endpoint | `bakehouse-demand-endpoint` |
| Predictions table | `mjrent_uc_int.prod_mlops.bakehouse_demand_predictions` |
| Experiment path | `/Users/mitchell.grewer@databricks.com/mlops_bakehouse_demo/bakehouse_mlops_demo` |
| RMSE threshold | 5.0 |
| R2 threshold | 0.5 |
| Watcher schedule | Every 6 hours (America/Detroit) |
