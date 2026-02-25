# Databricks notebook source
# Cell 01 — Config

import mlflow
from mlflow import MlflowClient

dbutils.widgets.dropdown("model_variant", "decorators", ["autolog", "decorators"])
model_variant = dbutils.widgets.get("model_variant")

CATALOG = "mjrent_uc_int"
SCHEMA_DEV = "dev_mlops"
SCHEMA_PROD = "prod_mlops"
MODEL_NAME_DEV = f"{CATALOG}.{SCHEMA_DEV}.bakehouse_{model_variant}_model"
MODEL_NAME_PROD = f"{CATALOG}.{SCHEMA_PROD}.bakehouse_{model_variant}_model"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

print(f"Promoting {model_variant}: {MODEL_NAME_DEV} -> {MODEL_NAME_PROD}")

# COMMAND ----------

# Cell 02 — Get dev champion and copy to prod

dev_champion = client.get_model_version_by_alias(MODEL_NAME_DEV, "champion")
dev_run_id = dev_champion.run_id

print(f"Dev champion: {MODEL_NAME_DEV} version {dev_champion.version}")
print(f"  Run ID: {dev_run_id}")

result = mlflow.register_model(
    model_uri=f"runs:/{dev_run_id}/model",
    name=MODEL_NAME_PROD,
)

print(f"Registered to {MODEL_NAME_PROD} as version {result.version}")

# COMMAND ----------

# Cell 03 — Set champion alias on prod model

client.set_registered_model_alias(MODEL_NAME_PROD, "champion", result.version)
print(f"Set alias 'champion' on {MODEL_NAME_PROD} version {result.version}")

prod_versions = client.search_model_versions(f"name='{MODEL_NAME_PROD}'")
for v in sorted(prod_versions, key=lambda x: int(x.version)):
    aliases = v.aliases if hasattr(v, 'aliases') else []
    print(f"  Version {v.version} | Aliases: {aliases} | Run ID: {v.run_id}")
