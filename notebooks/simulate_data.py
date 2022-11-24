# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog hive_metastore;
# MAGIC CREATE SCHEMA IF NOT EXISTS rl_data;

# COMMAND ----------

import numpy as np
import pandas as pd
from src.data.simulate_data import simulate_samples_history, transform_history


# COMMAND ----------

mu = 0
sigma = 1
n_samples = 100000
hist_len = 12
feature_vec_size = 30
feat_name_prefix = 'profile_feat'

profile_df = simulate_samples_history(mu, sigma, n_samples, hist_len, feature_vec_size, feat_name_prefix)
prof_df = spark.createDataFrame(profile_df)
prof_df.write.mode("overwrite").saveAsTable("rl_data.profile_simulated")

# COMMAND ----------

mu = 0
sigma = 1
n_samples = 100000
hist_len = 4
feature_vec_size = 15
feat_name_prefix = 'digi_feat'

digi_hist_df = simulate_samples_history(mu, sigma, n_samples, hist_len, feature_vec_size, feat_name_prefix)
digi_df = spark.createDataFrame(digi_hist_df)
digi_df.write.mode("overwrite").saveAsTable("rl_data.digital_interaction_simulated")

# COMMAND ----------

mu = 0
sigma = 1
n_samples = 100000
hist_len = 4
feature_vec_size = 12
feat_name_prefix = 'meeting_feat'

meeting_df = simulate_samples_history(mu, sigma, n_samples, hist_len, feature_vec_size, feat_name_prefix)
meet_df = spark.createDataFrame(meeting_df)
meet_df.write.mode("overwrite").saveAsTable("rl_data.meeting_interaction_simulated")

# COMMAND ----------

tf_profile_df = transform_history(profile_df, sample_ids='ids', w_mu=1, w_sigma=2, hidden_size=20, feat_name_prefix='tf_profile')
tf_digi_df = transform_history(digi_hist_df, sample_ids='ids', w_mu=1, w_sigma=2, hidden_size=10, feat_name_prefix='tf_digi')
tf_meeting_df = transform_history(meeting_df, sample_ids='ids', w_mu=1, w_sigma=2, hidden_size=8, feat_name_prefix='tf_meeting')


# COMMAND ----------

feature_df = tf_profile_df.merge(tf_digi_df, how='left', on='ids').merge(tf_meeting_df, how='left', on='ids')

# COMMAND ----------

actions = np.random.choice([0, 1], size=(n_samples, 4), replace=True)
action_df = pd.DataFrame(actions, columns=['a1', 'a2', 'a3', 'a4'])
action_df.insert(0, "ids", list(range(n_samples)))
tf_action_df = transform_history(action_df, sample_ids='ids', w_mu=1, w_sigma=2, hidden_size=feature_df.shape[1] - 1, feat_name_prefix='action_tf')

# COMMAND ----------

last_hidden = feature_df.iloc[:, 1:].values + tf_action_df.iloc[:, 1:].values
last_hidden_df = pd.DataFrame(last_hidden) 
last_hidden_df.insert(0, "ids", feature_df['ids'])

# COMMAND ----------

last_hidden_df.head()

# COMMAND ----------

reward_df = transform_history(last_hidden_df, sample_ids='ids', w_mu=1, w_sigma=2, hidden_size=1, feat_name_prefix='reward', add_noise=True, e_mu=0, e_sigma=1)

# COMMAND ----------

action_df = action_df.merge(reward_df, how='left', on='ids')

# COMMAND ----------

a_df = spark.createDataFrame(action_df)
a_df.write.mode("overwrite").saveAsTable("rl_data.action_simulated")
