# Databricks notebook source
import torch
import tqdm
import mlflow
import pandas as pd
import mlflow.pytorch
from torch.utils.data import DataLoader, random_split
from src.data.dataloader import InvSalesData, collate_fn


MODEL_PATH = 'runs:/f2b98f0de6d741dc822e58a056121697/RL_test_f2b98f0de6d741dc822e58a056121697_epo_9'
DATA_CONFIG = {
    'profile': {'table_name': 'rl_data.profile_simulated', 'version': 3},
    'action': {'table_name': 'rl_data.action_simulated', 'version': 0},
    'digital': {'table_name': 'rl_data.digital_interaction_simulated', 'version': 2},
    'meeting': {'table_name': 'rl_data.meeting_interaction_simulated', 'version': 2}
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCALING_FACTOR = 1

dbutils.widgets.text("run_dt", "")
run_dt = dbutils.widgets.get("run_dt")
print(run_dt)

def read_table(data_config, log_info=True):
    tables = list()
    for data_name, config in data_config.items():
        if log_info:
            mlflow.set_tag("data_version", config['version'])
            mlflow.set_tag("data_source", config['table_name'])
        
        df = spark.read.option("versionAsOf", config['version']).table(config['table_name']).toPandas()
        tables.append(df)
    
    return tables


def predict():
    # Load model as a PyFuncModel.
    model = mlflow.pytorch.load_model(MODEL_PATH)
    
    profile_data, action_data, digi_data, meeting_data = read_table(DATA_CONFIG, log_info=False)    
    dataset = InvSalesData(action_data, digi_data, meeting_data, profile_data, key_column='ids')    
    inference_set, _ = random_split(dataset=dataset, lengths=[5000, 95000])
    
    inference_loader = DataLoader(inference_set, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model.to(DEVICE)
    all_predictions = list()
    all_ids = list()
    with torch.no_grad():
        for batch in tqdm.tqdm(inference_loader):
            cust_profile, meeting_history, campaign_history, actions, rewards, ids = batch
            x = [
                cust_profile.div(SCALING_FACTOR).double().to(DEVICE),
                campaign_history.div(SCALING_FACTOR).double().to(DEVICE),
                meeting_history.div(SCALING_FACTOR).double().to(DEVICE),
                actions.double().to(DEVICE)
            ]

            outputs = model(x)
            predictions = outputs.flatten().cpu().numpy().tolist()
            all_predictions.extend(predictions)
            all_ids.extend(ids)
    
    return pd.DataFrame({'ids': all_ids, 'prediction': all_predictions})

# We can also use model from model artifact repo directly here
# from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
# import os

# model_name = "RL test"
# model_uri = f"models:/{model_name}/Staging"
# local_path = ModelsArtifactRepository(model_uri).download_artifacts("")

# model = mlflow.pytorch.load_model(model_uri)

# COMMAND ----------

outputs = predict()
outputs = outputs.assign(run_dt=run_dt)
pred_df = spark.createDataFrame(outputs)
pred_df.write.mode("overwrite").saveAsTable("rl_data.predictions")
