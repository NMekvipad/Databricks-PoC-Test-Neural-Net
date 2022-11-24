# Databricks notebook source
from functools import reduce
from operator import add
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from databricks import feature_store

# COMMAND ----------

mode = 'update' ## mode can be create or update
profile_df = spark.sql("""select * from rl_data.profile_simulated""")

# COMMAND ----------

def rolling_average_row(df, columns, partition_by, order_by, start, end):
    w = Window.partitionBy(partition_by).orderBy(order_by).rangeBetween(start, end)   
    new_columns = ['rolling_avg_' + col for col in columns]
    
    for col, new_col in zip(columns, new_columns):    
        df = df.withColumn(new_col, F.avg(col).over(w))
        
    new_columns = [partition_by, order_by] + new_columns
        
    rolling_avg_data = df.select(*new_columns)    
        
    return rolling_avg_data


def sum_feature(df, sum_cols, primary_keys):
    new_df = df.withColumn('total', reduce(add, [F.col(x) for x in sum_cols]))
    return_cols = primary_keys + ['total']
    
    return new_df.select(*return_cols)
    

# COMMAND ----------

profile_rolling = rolling_average_row(profile_df, columns=[ 'profile_feat_1', 'profile_feat_2', 'profile_feat_3', 'profile_feat_4'], partition_by='ids', order_by='profile_feat_0', start=0, end=3)
profile_sum = sum_feature(profile_df, sum_cols=[ 'profile_feat_1', 'profile_feat_2', 'profile_feat_3', 'profile_feat_4'], primary_keys=['ids', 'profile_feat_0'])

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
#fs.drop_table(
#  name='rl_data.rolling_avg_profile'
#)
#fs.drop_table(
#  name='rl_data.sum_profile'
#)

if mode == 'create':
    fs = feature_store.FeatureStoreClient()
    fs.create_table(
        name="rl_data.rolling_avg_profile",
        primary_keys=["ids", "profile_feat_0"],
        df=profile_rolling,
        description="RL model: 3 months rolling average on all columns"
    )
    
    fs.create_table(
        name="rl_data.sum_profile",
        primary_keys=["ids", "profile_feat_0"],
        df=profile_sum,
        description="RL model: Sum of first 4 columns"
    )
elif mode == 'update':
    fs.write_table(
      name="rl_data.rolling_avg_profile",
      df=profile_rolling,
      mode="overwrite",
    )
    
    fs.write_table(
      name="rl_data.sum_profile",
      df=profile_sum,
      mode="overwrite",
    )
    
    
