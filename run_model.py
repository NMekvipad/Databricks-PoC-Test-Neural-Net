# Databricks notebook source
import torch
import mlflow.pytorch
from mlflow import MlflowClient
from databricks import feature_store
from torch.utils.data import DataLoader
from src.data.dataloader import InvSalesData, collate_fn
from src.model.nn import InvSalesCritic

batch_size = 64
model_config = {
    'in_channels_profile': 30,
    'in_channels_campaign': 15,
    'in_channels_meeting': 12,
    'in_channels_action': 4,
    'profile_hist_len': 12,
    'out_channel_profile': 15,
    'hidden_size_campaign': 8,
    'hidden_size_meeting': 5,
    'kernel_size': 3,
    'out_channel': 1
}

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
profile_data = fs.read_table(
  name='rl_data.profile_simulated',
).toPandas()

# COMMAND ----------

action_data = spark.sql("""select * from  rl_data.action_simulated""").toPandas()
digi_data = spark.sql("""select * from  rl_data.digital_interaction_simulated""").toPandas()
meeting_data = spark.sql("""select * from  rl_data.meeting_interaction_simulated""").toPandas()

# COMMAND ----------

dataset = InvSalesData(action_data, digi_data, meeting_data, profile_data, key_column='ids')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

# COMMAND ----------

model = InvSalesCritic(**model_config)

# COMMAND ----------

mlflow.pytorch.autolog()

# COMMAND ----------


