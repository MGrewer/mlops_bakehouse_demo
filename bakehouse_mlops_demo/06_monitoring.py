# Databricks notebook source
# Cell 01 — Config

dbutils.widgets.dropdown("environment", "dev", ["dev", "prod"])
dbutils.widgets.dropdown("model_variant", "decorators", ["autolog", "decorators"])
environment = dbutils.widgets.get("environment")
model_variant = dbutils.widgets.get("model_variant")

CATALOG = "mjrent_uc_int"
SCHEMA = "dev_mlops" if environment == "dev" else "prod_mlops"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.bakehouse_{model_variant}_model"
EXPERIMENT_NAME = f"/Users/mitchell.grewer@databricks.com/bakehouse_{model_variant}"

print(f"Environment: {environment} | Model variant: {model_variant}")
print(f"Model: {MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# Cell 02 — Query experiment runs from system tables

experiment_runs_df = spark.sql(f"""
    SELECT
        r.run_id,
        r.run_name,
        r.status,
        r.start_time,
        r.end_time,
        r.experiment_id
    FROM system.mlflow.runs_latest r
    JOIN system.mlflow.experiments_latest e
        ON r.experiment_id = e.experiment_id
    WHERE e.name = '{EXPERIMENT_NAME}'
    ORDER BY r.start_time DESC
    LIMIT 10
""")

display(experiment_runs_df)

# COMMAND ----------

# Cell 03 — Query metric trends from system tables

metrics_history_df = spark.sql(f"""
    SELECT
        m.run_id,
        r.run_name,
        m.metric_name,
        m.metric_value,
        m.metric_time,
        r.start_time AS run_start_time
    FROM system.mlflow.run_metrics_history m
    JOIN system.mlflow.runs_latest r
        ON m.run_id = r.run_id
    JOIN system.mlflow.experiments_latest e
        ON r.experiment_id = e.experiment_id
    WHERE e.name = '{EXPERIMENT_NAME}'
        AND m.metric_name IN ('batch_rmse', 'batch_mae', 'batch_r2', 'rmse', 'mae', 'r2')
    ORDER BY r.start_time DESC, m.metric_name
""")

display(metrics_history_df)

# COMMAND ----------

# Cell 04 — Pull latest batch inference metrics from MLflow

from mlflow import MlflowClient
import mlflow
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.batch_rmse > 0",
    order_by=["start_time DESC"],
    max_results=10,
)

rows = []
for run in runs:
    rows.append({
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "start_time": pd.Timestamp(run.info.start_time, unit="ms"),
        "batch_rmse": run.data.metrics.get("batch_rmse"),
        "batch_mae": run.data.metrics.get("batch_mae"),
        "batch_r2": run.data.metrics.get("batch_r2"),
        "drift_simulation": run.data.tags.get("drift_simulation", "false"),
    })

metrics_pdf = pd.DataFrame(rows)
metrics_sdf = spark.createDataFrame(metrics_pdf)

dbutils.jobs.taskValues.set(key="latest_batch_rmse", value=rows[0]["batch_rmse"] if rows else None)
dbutils.jobs.taskValues.set(key="latest_batch_r2", value=rows[0]["batch_r2"] if rows else None)
dbutils.jobs.taskValues.set(key="latest_run_id", value=rows[0]["run_id"] if rows else None)

display(metrics_sdf)
