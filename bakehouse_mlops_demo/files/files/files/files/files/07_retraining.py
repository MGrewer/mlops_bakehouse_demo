# Databricks notebook source
# Cell 01 — Check if retraining was requested

retrain_needed = dbutils.jobs.taskValues.get(
    taskKey="evaluation",
    key="retrain_needed",
)

if not retrain_needed:
    print("No retraining needed — metrics are within thresholds")
    dbutils.notebook.exit("skipped")

print("Retraining triggered — proceeding with model refresh")

# COMMAND ----------

# Cell 02 — Retrain on clean data, register new version, promote to champion

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CATALOG = "mjrent_uc_int"
SCHEMA_DEV = "dev_mlops"
MODEL_NAME = f"{CATALOG}.{SCHEMA_DEV}.bakehouse_demand_model"
EXPERIMENT_NAME = "/Users/mitchell.grewer@databricks.com/bakehouse_demand_forecasting"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

transactions = spark.table("samples.bakehouse.sales_transactions")
customers = spark.table("samples.bakehouse.sales_customers")
franchises = spark.table("samples.bakehouse.sales_franchises")

df = (
    transactions
    .join(customers, on="customerID", how="left")
    .join(franchises, on="franchiseID", how="left")
    .select(
        "product", "quantity", "unitPrice", "paymentMethod",
        customers["gender"], customers["state"],
        customers["continent"].alias("customer_continent"),
        franchises["name"].alias("franchise_name"),
        franchises["city"].alias("franchise_city"),
        franchises["size"].alias("franchise_size"),
        franchises["country"].alias("franchise_country"),
        F.hour("dateTime").alias("hour"),
        F.dayofweek("dateTime").alias("day_of_week"),
        F.month("dateTime").alias("month"),
        F.dayofmonth("dateTime").alias("day_of_month"),
    )
)

pdf = df.toPandas()

categorical_cols = ["product", "paymentMethod", "gender", "state", "customer_continent",
                    "franchise_name", "franchise_city", "franchise_size", "franchise_country"]
numeric_cols = ["unitPrice", "hour", "day_of_week", "month", "day_of_month"]
target = "quantity"

X = pdf[categorical_cols + numeric_cols]
y = pdf[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

params = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 3,
    "random_state": 42,
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(**params)),
])

with mlflow.start_run(run_name="retrain_demand_rf") as run:
    mlflow.set_tag("retrain_trigger", "metric_threshold_breach")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_train, y_pred)
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=MODEL_NAME,
    )

    print(f"Retrained model — Run ID: {run.info.run_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

client = MlflowClient()
latest_version = max(
    client.search_model_versions(f"name='{MODEL_NAME}'"),
    key=lambda v: int(v.version),
)
client.set_registered_model_alias(MODEL_NAME, "champion", latest_version.version)
print(f"Promoted version {latest_version.version} to 'champion'")
