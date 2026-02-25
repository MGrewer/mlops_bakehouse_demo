# Databricks notebook source
# Cell 01 — Config

import mlflow
from pyspark.sql import functions as F
import pandas as pd
import numpy as np

dbutils.widgets.dropdown("model_variant", "decorators", ["autolog", "decorators"])
model_variant = dbutils.widgets.get("model_variant")

CATALOG = "mjrent_uc_int"
SCHEMA = "dev_mlops"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.bakehouse_{model_variant}_model"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.bakehouse_{model_variant}_predictions"
EXPERIMENT_NAME = f"/Users/mitchell.grewer@databricks.com/bakehouse_{model_variant}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Model variant: {model_variant}")
print(f"Model: {MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# Cell 02 — Simulate drift: corrupt features, score, log degraded metrics
#
# ── NOTES ──────────────────────────────────────────────────────────────
#
# TAGS AS OPERATIONAL MARKERS:
#   We tag this run with drift_simulation=true. Tags aren't just labels — they're
#   searchable filters. The reset cell (Cell 03) uses this tag to find and delete
#   only the drift runs without touching baseline runs. In production, you'd use
#   tags like pipeline_stage=staging, triggered_by=scheduler, or alert_id=xyz
#   to organize and filter runs programmatically.
#
# METRICS UNDER DRIFT:
#   The same metric names (batch_rmse, batch_mae, batch_r2) are logged here as
#   in notebook 02's baseline run. This consistency is what makes the monitoring
#   job (notebook 06) work — it queries the latest batch_rmse and batch_r2 values
#   regardless of which run produced them. When drift degrades these metrics past
#   the thresholds, the evaluation job (notebook 07) flags for retraining.
#
# PREDICTIONS TABLE — APPEND FOR DRIFT WINDOWS:
#   The drifted predictions are appended (not overwritten) with scored_at offset
#   by +1 day. This gives Lakehouse Monitoring two distinct time windows to
#   compare: the clean baseline window from notebook 02, and the drifted window
#   from this cell. Without separate windows, the monitor has nothing to measure
#   drift against.
# ────────────────────────────────────────────────────────────────────────────────

from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

transactions = spark.table("samples.bakehouse.sales_transactions")
customers = spark.table("samples.bakehouse.sales_customers")
franchises = spark.table("samples.bakehouse.sales_franchises")

df = (
    transactions
    .join(customers, on="customerID", how="left")
    .join(franchises, on="franchiseID", how="left")
    .select(
        transactions["transactionID"],
        transactions["dateTime"],
        transactions["product"],
        transactions["quantity"],
        transactions["unitPrice"],
        transactions["paymentMethod"],
        customers["gender"],
        customers["state"],
        customers["continent"].alias("customer_continent"),
        franchises["name"].alias("franchise_name"),
        franchises["city"].alias("franchise_city"),
        franchises["size"].alias("franchise_size"),
        franchises["country"].alias("franchise_country"),
    )
)

df_features = (
    df
    .withColumn("hour", F.hour("dateTime"))
    .withColumn("day_of_week", F.dayofweek("dateTime"))
    .withColumn("month", F.month("dateTime"))
    .withColumn("day_of_month", F.dayofmonth("dateTime"))
    .drop("transactionID", "dateTime")
)

pdf = df_features.toPandas()

categorical_cols = ["product", "paymentMethod", "gender", "state", "customer_continent",
                    "franchise_name", "franchise_city", "franchise_size", "franchise_country"]
numeric_cols = ["unitPrice", "hour", "day_of_week", "month", "day_of_month"]
target = "quantity"

np.random.seed(99)
pdf_drifted = pdf.copy()
pdf_drifted["unitPrice"] = pdf_drifted["unitPrice"] * np.random.uniform(0.1, 5.0, size=len(pdf_drifted))
pdf_drifted["hour"] = np.random.randint(0, 24, size=len(pdf_drifted))
pdf_drifted["day_of_week"] = np.random.randint(1, 8, size=len(pdf_drifted))
pdf_drifted["month"] = np.random.randint(1, 13, size=len(pdf_drifted))

X_drifted = pdf_drifted[categorical_cols + numeric_cols]
y_actual = pdf_drifted[target]

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
y_pred = model.predict(X_drifted)

client = MlflowClient()
champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")

rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
mae = float(mean_absolute_error(y_actual, y_pred))
r2 = float(r2_score(y_actual, y_pred))

with mlflow.start_run(run_name=f"batch_inference_drifted_{model_variant}") as run:
    # ── TAG: marks this run so it can be filtered and cleaned up ──
    mlflow.set_tag("drift_simulation", "true")
    # ── METRICS: same names as baseline, degraded values ──
    mlflow.log_metric("batch_rmse", rmse)
    mlflow.log_metric("batch_mae", mae)
    mlflow.log_metric("batch_r2", r2)
    mlflow.log_param("num_rows_scored", len(y_pred))
    mlflow.log_param("model_name", MODEL_NAME)

results_pdf = pd.DataFrame({
    "transactionID": range(len(y_pred)),
    "prediction_timestamp": pd.Timestamp.now(),
    "actual_quantity": y_actual.values,
    "predicted_quantity": y_pred,
    "residual": y_actual.values - y_pred,
    "model_version": str(champion_version.version),
})

# Offset scored_at by +1 day so Lakehouse Monitoring sees a separate time window
results_sdf = (
    spark.createDataFrame(results_pdf)
    .withColumn("scored_at", F.current_timestamp() + F.expr("INTERVAL 1 DAY"))
)
results_sdf.write.mode("append").saveAsTable(PREDICTIONS_TABLE)

print(f"Drifted metrics:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R2:   {r2:.4f}")
print(f"Predictions appended to {PREDICTIONS_TABLE} (offset +1 day for drift window)")

# COMMAND ----------

# MAGIC %skip
# MAGIC # Cell 03 — Reset demo: clean up drift artifacts and restore baseline
# MAGIC #
# MAGIC # ── NOTE ───────────────────────────────────────────────────────────────
# MAGIC #   Tags make this cleanup possible. We search for runs where
# MAGIC #   tags.drift_simulation='true' and delete only those, leaving the baseline
# MAGIC #   training and inference runs untouched. The predictions table cleanup uses the
# MAGIC #   offset timestamp — drifted rows have scored_at in the future, so we can
# MAGIC #   surgically remove them.
# MAGIC # ────────────────────────────────────────────────────────────────────────────────
# MAGIC
# MAGIC from mlflow import MlflowClient
# MAGIC
# MAGIC client = MlflowClient()
# MAGIC
# MAGIC experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
# MAGIC if experiment:
# MAGIC     drift_runs = client.search_runs(
# MAGIC         experiment_ids=[experiment.experiment_id],
# MAGIC         filter_string="tags.drift_simulation = 'true'",
# MAGIC     )
# MAGIC     for run in drift_runs:
# MAGIC         client.delete_run(run.info.run_id)
# MAGIC         print(f"Deleted drift run: {run.info.run_id}")
# MAGIC
# MAGIC     print(f"Removed {len(drift_runs)} drift run(s)")
# MAGIC else:
# MAGIC     print("Experiment not found — nothing to clean")
# MAGIC
# MAGIC try:
# MAGIC     deleted = spark.sql(f"""
# MAGIC         DELETE FROM {PREDICTIONS_TABLE}
# MAGIC         WHERE scored_at > current_timestamp()
# MAGIC     """)
# MAGIC     print(f"Removed drifted rows (offset timestamps) from {PREDICTIONS_TABLE}")
# MAGIC except Exception as e:
# MAGIC     print(f"Could not clean predictions table: {e}")
# MAGIC
# MAGIC print(f"Demo state reset for {model_variant}")
