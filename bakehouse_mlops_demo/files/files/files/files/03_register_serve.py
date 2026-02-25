# Databricks notebook source
# Cell 01 — Config and review registered model versions

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

CATALOG = "mjrent_uc_int"
SCHEMA_DEV = "dev_mlops"
SCHEMA_PROD = "prod_mlops"
MODEL_NAME = f"{CATALOG}.{SCHEMA_DEV}.bakehouse_demand_model"
ENDPOINT_NAME = "bakehouse-demand-endpoint"
INFERENCE_TABLE_CATALOG = CATALOG
INFERENCE_TABLE_SCHEMA = SCHEMA_PROD

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for v in sorted(versions, key=lambda x: int(x.version)):
    print(f"  Version {v.version} | Run ID: {v.run_id} | Status: {v.status}")

# COMMAND ----------

# Cell 02 — Set champion alias on the latest version

latest_version = max(
    client.search_model_versions(f"name='{MODEL_NAME}'"),
    key=lambda v: int(v.version),
)

client.set_registered_model_alias(MODEL_NAME, "champion", latest_version.version)
print(f"Set alias 'champion' on {MODEL_NAME} version {latest_version.version}")

# COMMAND ----------

# Cell 03 — Deploy serving endpoint with inference table enabled

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AutoCaptureConfigInput,
)

w = WorkspaceClient()

served_entities = [
    ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=latest_version.version,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

auto_capture_config = AutoCaptureConfigInput(
    catalog_name=INFERENCE_TABLE_CATALOG,
    schema_name=INFERENCE_TABLE_SCHEMA,
    enabled=True,
)

try:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
            auto_capture_config=auto_capture_config,
        ),
    )
    print(f"Creating endpoint: {ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e).lower():
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_entities=served_entities,
            auto_capture_config=auto_capture_config,
        )
        print(f"Updating existing endpoint: {ENDPOINT_NAME}")
    else:
        raise e

print(f"Inference table will be written to {INFERENCE_TABLE_CATALOG}.{INFERENCE_TABLE_SCHEMA}")

# COMMAND ----------

# Cell 04 — Send test requests to the serving endpoint

import time
from databricks.sdk.service.serving import EndpointStateReady

endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
while endpoint.state.ready != EndpointStateReady.READY:
    print(f"Endpoint state: {endpoint.state.ready} — waiting 30s...")
    time.sleep(30)
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)

print(f"Endpoint {ENDPOINT_NAME} is ready")

transactions = spark.table("samples.bakehouse.sales_transactions")
customers = spark.table("samples.bakehouse.sales_customers")
franchises = spark.table("samples.bakehouse.sales_franchises")

sample_records = (
    transactions
    .join(customers, on="customerID", how="left")
    .join(franchises, on="franchiseID", how="left")
    .select(
        transactions["product"],
        transactions["unitPrice"],
        transactions["paymentMethod"],
        customers["gender"],
        customers["state"],
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
    .limit(10)
    .toPandas()
)

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=sample_records.to_dict(orient="records"),
)

print(f"Sent {len(sample_records)} requests")
print(f"Predictions: {response.predictions}")

# COMMAND ----------

# Cell 05 — Query the auto-generated inference table

import time

inference_table = f"{INFERENCE_TABLE_CATALOG}.{INFERENCE_TABLE_SCHEMA}.`{ENDPOINT_NAME}_payload`"

print(f"Waiting for inference table to populate: {inference_table}")
print("(This can take a few minutes after the first requests)")

max_attempts = 10
for attempt in range(max_attempts):
    try:
        inf_df = spark.table(inference_table)
        count = inf_df.count()
        if count > 0:
            print(f"Inference table has {count} rows")
            display(inf_df.limit(10))
            break
    except Exception:
        pass
    print(f"  Attempt {attempt + 1}/{max_attempts} — waiting 30s...")
    time.sleep(30)
else:
    print("Inference table not yet populated — check back shortly")
    print("The inference table auto-captures every request and response")
    print("with zero additional logging code — the platform handles it.")
