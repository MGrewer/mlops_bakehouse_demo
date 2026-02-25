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
#
# ── NOTE: TRACING ──────────────────────────────────────────────────────
#   @mlflow.trace decorators capture the inputs, outputs, and execution time of
#   each function call. This is separate from params/metrics/tags — tracing gives
#   you a step-by-step audit trail of the training pipeline. In the MLflow UI,
#   each traced function shows up as a span in the run's trace view, so you can
#   see exactly what happened and how long each step took.
# ────────────────────────────────────────────────────────────────────────────────

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

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

# Cell 03 — Hyperparameter sweep: compare configurations
#
# ── NOTES ──────────────────────────────────────────────────────────────
#
# EXPERIMENT COMPARISON:
#   This cell trains 4 models with different hyperparameter configs. Each run
#   logs the same set of params, metrics, tags, and dataset metadata — making
#   them directly comparable in the experiment UI.
#
#   After running, go to the experiment page, select all 4 runs, and click
#   "Compare". The parallel coordinates plot shows how n_estimators, max_depth,
#   and min_samples_split trade off against rmse, mae, and r2. This is the
#   model selection workflow — data scientists iterate here before promoting
#   the best version to the registry.
#
# WHY EXPERIMENTS AND MODELS ARE SEPARATE:
#   All 4 runs live in the experiment. Only the winner gets registered as a
#   model version. The experiment is the workbench; the registry is the shelf.
# ────────────────────────────────────────────────────────────────────────────────

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(disable=True)

X_train, X_test, y_train, y_test = prepare_features(df)

sweep_configs = [
    {"n_estimators": 50,  "max_depth": 5,  "min_samples_split": 10, "random_state": 42},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5,  "random_state": 42},
    {"n_estimators": 150, "max_depth": 15, "min_samples_split": 3,  "random_state": 42},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2,  "random_state": 42},
]

sweep_results = []

for i, params in enumerate(sweep_configs):
    run_name = f"sweep_{i+1}_trees{params['n_estimators']}_depth{params['max_depth']}"

    with mlflow.start_run(run_name=run_name) as run:
        # ── TAGS: organizational metadata ──
        mlflow.set_tag("training_method", "explicit_decorators")
        mlflow.set_tag("dataset", "samples.bakehouse")
        mlflow.set_tag("sweep_run", "true")
        mlflow.set_tag("sweep_index", str(i + 1))

        # ── PARAMETERS: training configuration ──
        mlflow.log_params(params)

        # ── DATASET METADATA: data lineage tracking ──
        dataset = mlflow.data.from_pandas(
            X_train.assign(quantity=y_train),
            source="samples.bakehouse.sales_transactions",
            name="bakehouse_training_data",
        )
        mlflow.log_input(dataset, context="training")

        pipeline = build_pipeline(params)
        pipeline.fit(X_train, y_train)

        # ── METRICS: model performance measurements ──
        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        sweep_results.append({
            "run_id": run.info.run_id,
            "run_name": run_name,
            "params": params,
            "metrics": metrics,
            "pipeline": pipeline,
        })

        print(f"  {run_name}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")

print(f"\nCompleted {len(sweep_configs)} sweep runs")
print("Go to the experiment page → select all runs → Compare")

# COMMAND ----------

# Cell 04 — Select best model, register to Unity Catalog
#
# ── NOTES ──────────────────────────────────────────────────────────────
#
# MODEL SELECTION:
#   We pick the run with the lowest RMSE and register only that one. This is the
#   boundary between experiment (messy iteration) and registry (curated versions).
#   The other 3 runs stay in the experiment for reference but never become model
#   versions.
#
# MODEL ARTIFACT (mlflow.sklearn.log_model):
#   The trained pipeline is serialized with its flavor (sklearn). MLflow creates:
#     - MLmodel file: describes how to load and serve the model
#     - conda.yaml + requirements.txt: environment reproducibility
#     - Model signature: input/output schema (inferred from data here)
#     - Input example: sample data for testing the serving endpoint
#   registered_model_name registers it to Unity Catalog in a single call.
# ────────────────────────────────────────────────────────────────────────────────

best_run = min(sweep_results, key=lambda r: r["metrics"]["rmse"])

print(f"Best run: {best_run['run_name']}")
print(f"  RMSE: {best_run['metrics']['rmse']:.4f}")
print(f"  MAE:  {best_run['metrics']['mae']:.4f}")
print(f"  R2:   {best_run['metrics']['r2']:.4f}")
print(f"  Config: {best_run['params']}")

with mlflow.start_run(run_id=best_run["run_id"]):
    # ── MODEL ARTIFACT: serialized model with signature and reproducibility ──
    signature = infer_signature(X_train, best_run["pipeline"].predict(X_test))
    mlflow.sklearn.log_model(
        sk_model=best_run["pipeline"],
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=MODEL_NAME,
    )

print(f"Registered {MODEL_NAME} from run {best_run['run_id']}")

# COMMAND ----------

# Cell 05 — Batch inference with explicit metric logging
#
# ── NOTE ───────────────────────────────────────────────────────────────
#   This is the cell autolog could never produce. We load the registered model,
#   score new data, and explicitly log batch evaluation metrics. These batch_*
#   metrics are what the monitoring job (notebook 06) watches for threshold
#   breaches. Without explicit logging here, the automated watcher has nothing
#   to evaluate.
#
#   The predictions table also feeds Lakehouse Monitoring for drift detection.
#   The model_version column lets the monitor slice metrics per model version,
#   and CDF (Change Data Feed) enables incremental processing on refreshes.
# ────────────────────────────────────────────────────────────────────────────────

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
