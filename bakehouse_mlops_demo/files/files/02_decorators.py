# Databricks notebook source
# Cell 01 — Config and load data

from pyspark.sql import functions as F
import pandas as pd

CATALOG = "mjrent_uc_int"
SCHEMA = "dev_mlops"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.bakehouse_decorators_model"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.bakehouse_decorators_predictions"
EXPERIMENT_NAME = "/Users/mitchell.grewer@databricks.com/bakehouse_decorators"

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

# Cell 02 — Define traced training functions

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

categorical_cols = ["product", "paymentMethod", "gender", "state", "customer_continent",
                    "franchise_name", "franchise_city", "franchise_size", "franchise_country"]
numeric_cols = ["unitPrice", "hour", "day_of_week", "month", "day_of_month"]
target = "quantity"


@mlflow.trace(name="prepare_features")
def prepare_features(df):
    df_features = (
        df
        .withColumn("hour", F.hour("dateTime"))
        .withColumn("day_of_week", F.dayofweek("dateTime"))
        .withColumn("month", F.month("dateTime"))
        .withColumn("day_of_month", F.dayofmonth("dateTime"))
        .drop("transactionID", "dateTime")
    )
    pdf = df_features.toPandas()
    X = pdf[categorical_cols + numeric_cols]
    y = pdf[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


@mlflow.trace(name="build_pipeline")
def build_pipeline(params):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(**params)),
    ])


@mlflow.trace(name="evaluate_model")
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


print("Traced functions defined: prepare_features, build_pipeline, evaluate_model")

# COMMAND ----------

# Cell 03 — Train with explicit logging, register to Unity Catalog

from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(disable=True)

X_train, X_test, y_train, y_test = prepare_features(df)

params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42,
}

with mlflow.start_run(run_name="decorators_demand_rf") as run:
    mlflow.set_tag("training_method", "explicit_decorators")
    mlflow.set_tag("dataset", "samples.bakehouse")
    mlflow.log_params(params)

    dataset = mlflow.data.from_pandas(
        X_train.assign(quantity=y_train),
        source="samples.bakehouse.sales_transactions",
        name="bakehouse_training_data",
    )
    mlflow.log_input(dataset, context="training")

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_train, pipeline.predict(X_test))
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=MODEL_NAME,
    )

    print(f"Run ID: {run.info.run_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# COMMAND ----------

# Cell 04 — Batch inference with explicit metric logging

from mlflow import MlflowClient

client = MlflowClient()
latest_version = max(
    client.search_model_versions(f"name='{MODEL_NAME}'"),
    key=lambda v: int(v.version),
)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest_version.version}")

y_pred = model.predict(X_test)

batch_metrics = {
    "batch_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "batch_mae": float(mean_absolute_error(y_test, y_pred)),
    "batch_r2": float(r2_score(y_test, y_pred)),
}

with mlflow.start_run(run_name="batch_inference_baseline") as run:
    mlflow.log_metrics(batch_metrics)
    mlflow.log_param("num_rows_scored", len(y_pred))
    mlflow.log_param("model_name", MODEL_NAME)

results_pdf = pd.DataFrame({
    "transactionID": X_test.index,
    "prediction_timestamp": pd.Timestamp.now(),
    "actual_quantity": y_test.values,
    "predicted_quantity": y_pred,
    "residual": y_test.values - y_pred,
    "model_version": str(latest_version.version),
})

results_sdf = spark.createDataFrame(results_pdf).withColumn("scored_at", F.current_timestamp())
results_sdf.write.mode("overwrite").saveAsTable(PREDICTIONS_TABLE)

spark.sql(f"""
    ALTER TABLE {PREDICTIONS_TABLE}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print(f"Batch inference metrics (explicitly logged):")
for k, v in batch_metrics.items():
    print(f"  {k}: {v:.4f}")
print(f"Predictions written to {PREDICTIONS_TABLE} (CDF enabled)")
