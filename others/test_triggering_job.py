# Databricks notebook source
import requests


db_url = "https://dbc-3dffe409-a605.cloud.databricks.com/api/2.1/jobs/run-now"
token = "dapi46fddf17b4d59c8319cf037f3f05b753"
run_test = True
payload = {'job_id': 304950363592978, "notebook_params": {"Message1": "Hello Mars", "Message2": "This is Earth"}}

if run_test:
    res = requests.post(db_url, headers={"Authorization": "Bearer %s" % token}, json=payload)
    

# COMMAND ----------

res
