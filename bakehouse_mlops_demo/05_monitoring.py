# Databricks notebook source
# Cell 01 — Config

CATALOG = "mjrent_uc_int"
SCHEMA_DEV = "dev_mlops"
MODEL_NAME = f"{CATALOG}.{SCHEMA_DEV}.bakehouse_demand_model"
EXPERIMENT_NAME = "/Users/mitchell.grewer@databricks.com/bakehouse_demand_forecasting"

# COMMAND ----------

# Cell 02 — Query model version history from system tables

model_versions_df = spark.sql(f"""
    SELECT
        name,
        version,
        creation_timestamp,
        last_updated_timestamp,
        status,
        run_id,
        source
    FROM system.mlflow.model_versions
    WHERE name = '{MODEL_NAME}'
    ORDER BY version DESC
""")

display(model_versions_df)

# COMMAND ----------

# Cell 03 — Pull latest batch inference metrics from MLflow

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
