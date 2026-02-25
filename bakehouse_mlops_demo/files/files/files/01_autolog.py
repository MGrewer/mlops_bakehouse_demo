# Databricks notebook source
# Cell 01 — Config and load data

from pyspark.sql import functions as F
import pandas as pd

CATALOG = "mjrent_uc_int"
SCHEMA = "dev_mlops"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.bakehouse_autolog_model"
EXPERIMENT_NAME = "/Users/mitchell.grewer@databricks.com/bakehouse_autolog"

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

display(df.limit(5))

# COMMAND ----------

# Cell 02 — Feature engineering and train/test split

from sklearn.model_selection import train_test_split

df_features = (
    df
    .withColumn("hour", F.hour("dateTime"))
    .withColumn("day_of_week", F.dayofweek("dateTime"))
    .withColumn("month", F.month("dateTime"))
    .withColumn("day_of_month", F.dayofmonth("dateTime"))
    .drop("transactionID", "dateTime")
)

pdf = df_features.toPandas()

target = "quantity"
categorical_cols = ["product", "paymentMethod", "gender", "state", "customer_continent",
                    "franchise_name", "franchise_city", "franchise_size", "franchise_country"]
numeric_cols = ["unitPrice", "hour", "day_of_week", "month", "day_of_month"]

X = pdf[categorical_cols + numeric_cols]
y = pdf[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# COMMAND ----------

# Cell 03 — Train with mlflow.autolog() and register to Unity Catalog

import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
])

with mlflow.start_run(run_name="autolog_demand_rf") as run:
    pipeline.fit(X_train, y_train)
    print(f"Run ID: {run.info.run_id}")
    print(f"Autolog captured params and metrics automatically")

mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
print(f"Model registered to {MODEL_NAME}")

# COMMAND ----------

# Cell 04 — Batch inference: autolog does NOT capture evaluation metrics on new data

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = pipeline.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print(f"Batch inference results (NOT captured by autolog):")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R2:   {r2:.4f}")
print()
print("Autolog captures training metrics automatically, but batch inference")
print("evaluation on new data requires explicit logging — see notebook 02.")
