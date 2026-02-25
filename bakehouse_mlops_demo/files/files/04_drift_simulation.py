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
    mlflow.set_tag("drift_simulation", "true")
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

# Cell 03 — Reset demo: clean up drift artifacts and restore baseline

from mlflow import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment:
    drift_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.drift_simulation = 'true'",
    )
    for run in drift_runs:
        client.delete_run(run.info.run_id)
        print(f"Deleted drift run: {run.info.run_id}")

    print(f"Removed {len(drift_runs)} drift run(s)")
else:
    print("Experiment not found — nothing to clean")

try:
    deleted = spark.sql(f"""
        DELETE FROM {PREDICTIONS_TABLE}
        WHERE scored_at > current_timestamp()
    """)
    print(f"Removed drifted rows (offset timestamps) from {PREDICTIONS_TABLE}")
except Exception as e:
    print(f"Could not clean predictions table: {e}")

print(f"Demo state reset for {model_variant}")
