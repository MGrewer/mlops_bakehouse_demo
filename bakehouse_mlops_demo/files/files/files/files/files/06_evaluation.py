# Databricks notebook source
# Cell 01 — Config and thresholds

RMSE_THRESHOLD = 5.0
R2_THRESHOLD = 0.5

# COMMAND ----------

# Cell 02 — Evaluate latest metrics against thresholds

latest_rmse = dbutils.jobs.taskValues.get(
    taskKey="monitoring",
    key="latest_batch_rmse",
)
latest_r2 = dbutils.jobs.taskValues.get(
    taskKey="monitoring",
    key="latest_batch_r2",
)
latest_run_id = dbutils.jobs.taskValues.get(
    taskKey="monitoring",
    key="latest_run_id",
)

rmse_breach = latest_rmse is not None and latest_rmse > RMSE_THRESHOLD
r2_breach = latest_r2 is not None and latest_r2 < R2_THRESHOLD
retrain_needed = rmse_breach or r2_breach

print(f"Latest RMSE: {latest_rmse} (threshold: {RMSE_THRESHOLD}) — {'BREACH' if rmse_breach else 'OK'}")
print(f"Latest R2:   {latest_r2} (threshold: {R2_THRESHOLD}) — {'BREACH' if r2_breach else 'OK'}")
print(f"Retrain needed: {retrain_needed}")

dbutils.jobs.taskValues.set(key="retrain_needed", value=retrain_needed)
dbutils.jobs.taskValues.set(key="trigger_run_id", value=latest_run_id)
