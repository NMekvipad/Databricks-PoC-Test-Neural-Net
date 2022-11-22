# Databricks notebook source
import torch
import tqdm
import mlflow
import pandas as pd
import mlflow.pytorch
from torch.utils.data import DataLoader
from src.data.dataloader import InvSalesData, collate_fn
from torch.utils.utils import read_table, log_scalar


MODEL_PATH = 'runs:/f2b98f0de6d741dc822e58a056121697/RL_test_f2b98f0de6d741dc822e58a056121697_epo_9'
DATA_CONFIG = {
    'profile': {'table_name': 'rl_data.profile_simulated', 'version': 3},
    'action': {'table_name': 'rl_data.action_simulated', 'version': 0},
    'digital': {'table_name': 'rl_data.digital_interaction_simulated', 'version': 2},
    'meeting': {'table_name': 'rl_data.meeting_interaction_simulated', 'version': 2}
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCALING_FACTOR = 1


def predict():
    # Load model as a PyFuncModel.
    model = mlflow.pytorch.load_model(MODEL_PATH)
    profile_data, action_data, digi_data, meeting_data = read_table(DATA_CONFIG)
    dataset = InvSalesData(action_data, digi_data, meeting_data, profile_data, key_column='ids')
    inference_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model.to(DEVICE)
    all_predictions = list()
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

    outputs = pd.DataFrame({'ids': ids, 'predictions': all_predictions})

    return outputs

# COMMAND ----------

if __name__ == "__main__":
    outputs = predict()
    pred_df = spark.createDataFrame(outputs)
    digi_df.write.mode("overwrite").saveAsTable("rl_data.predictions")
