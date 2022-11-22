# Databricks notebook source
import torch
import tqdm
import random
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from databricks import feature_store
from torch.utils.data import DataLoader, random_split
from src.data.dataloader import InvSalesData, collate_fn
from src.model.nn import InvSalesCritic
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

data_size = 100000
TRAIN_SIZE = int(0.6 * data_size)
VALIDATION_SIZE = int(0.2 * data_size)
TEST_SIZE = int(data_size - TRAIN_SIZE - VALIDATION_SIZE)

# input parameters
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
SCALING_FACTOR = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 1
EPOCHS = 10
BATCH_SIZE = 64
MODE = 'train'
MODEL_HYPER_PARAMS = {
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

data_config = {
    'profile': {'table_name': 'rl_data.profile_simulated', 'version': 3},
    'action': {'table_name': 'rl_data.action_simulated', 'version': 0},
    'digital': {'table_name': 'rl_data.digital_interaction_simulated', 'version': 2},
    'meeting': {'table_name': 'rl_data.meeting_interaction_simulated', 'version': 2}
}


# COMMAND ----------

def read_table(data_config):
    tables = list()
    for data_name, config in data_config.items():
        mlflow.set_tag("data_version", config['version'])
        mlflow.set_tag("data_source", config['table_name'])
        
        df = spark.read.option("versionAsOf", config['version']).table(config['table_name']).toPandas()
        tables.append(df)
    
    return tables


def log_scalar(tb_writer, name, value, step, flush=False):
    """Log a scalar value or dictionary of scalar to both MLflow and TensorBoard"""

    if type(value) == dict:
        tb_writer.add_scalar(name, value, step)

        for key, val in value.items():
            mlflow.log_metric(key, val, step=step)
    else:
        tb_writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value, step=step)

    if flush:
        tb_writer.flush()


# adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(loader, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    lost_list = list()
    step_no = list()

    for i, batch in enumerate(tqdm.tqdm(loader)):
        cust_profile, meeting_history, campaign_history, actions, rewards, _ = batch  # _, _, _,
        x = [
            cust_profile.div(SCALING_FACTOR).double().to(DEVICE),
            campaign_history.div(SCALING_FACTOR).double().to(DEVICE),
            meeting_history.div(SCALING_FACTOR).double().to(DEVICE),
            actions.double().to(DEVICE)
        ]

        # Set gradient to 0
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x)

        # Loss calculation & backward pass
        rewards = rewards.div(SCALING_FACTOR).flatten().double()
        loss = loss_fn(outputs.flatten(), rewards.to(DEVICE))

        loss.backward()

        # gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2.0)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 0:
            last_loss = running_loss / 50  # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * TRAIN_SIZE + (i + 1) * BATCH_SIZE
            log_scalar(tb_writer, 'Loss/train', last_loss, tb_x)
            running_loss = 0.

    return lost_list, step_no


def evaluate(loader):
    running_validation_loss = 0.0

    for i, batch in enumerate(tqdm.tqdm(loader)):
        cust_profile, meeting_history, campaign_history, actions, rewards, _ = batch
        x = [
            cust_profile.div(SCALING_FACTOR).double().to(DEVICE),
            campaign_history.div(SCALING_FACTOR).double().to(DEVICE),
            meeting_history.div(SCALING_FACTOR).double().to(DEVICE),
            actions.double().to(DEVICE)
        ]

        outputs = model(x)
        rewards = rewards.div(SCALING_FACTOR).flatten().double()
        loss = loss_fn(outputs.flatten(), rewards.to(DEVICE))
        running_validation_loss += loss

    return running_validation_loss


def predict(loader):
    pred = list()
    truth = list()
    prob = list()
    cust_id = list()
    action_vec = list()

    for i, batch in enumerate(tqdm.tqdm(loader)):
        cust_profile, meeting_history, campaign_history, actions, rewards, ids = batch
        x = [
            cust_profile.div(SCALING_FACTOR).double().to(DEVICE),
            campaign_history.div(SCALING_FACTOR).double().to(DEVICE),
            meeting_history.div(SCALING_FACTOR).double().to(DEVICE),
            actions.double().to(DEVICE)
        ]

        outputs = model(x)
        rewards = rewards.div(SCALING_FACTOR).flatten().double()
        pred.append(outputs)

        truth.append(rewards)
        action_vec.append(actions[:, :, 0].flatten())
        cust_id.extend(ids)

    return truth, (pred, prob), action_vec, cust_id

# COMMAND ----------

if __name__ == '__main__':
    experiment_name = "/Users/odl_user_794888@databrickslabs.com/RL_test"
    mlflow.set_experiment(experiment_name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        run_id = run.info.run_id
        # Log our parameters into mlflow
        for key, value in MODEL_HYPER_PARAMS.items():
            mlflow.log_param(key, value)
        
        mlflow.log_param('Feature scaling factor', SCALING_FACTOR)
        mlflow.log_param('Model type', 'Regressor')
        mlflow.log_param('Number of epoch', EPOCHS)
        mlflow.log_param('Batch size', BATCH_SIZE)
        mlflow.log_param('Number of epoch', EPOCHS)
        mlflow.log_param('Loss function', 'MSE')
        mlflow.log_param('Optimization algorithm', 'SGD with momentum')
        mlflow.log_param('Learning rate', LEARNING_RATE)
        mlflow.log_param('Momentum', LEARNING_RATE)
        
        # Load data set (has builting logging for data version in the function)         
        profile_data, action_data, digi_data, meeting_data = read_table(data_config)
        
        # convert to pytorch dataset
        dataset = InvSalesData(action_data, digi_data, meeting_data, profile_data, key_column='ids')
        train_set, validation_set, test_set = random_split(dataset=dataset, lengths=[50000, 20000, 30000])
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
        validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)

        out_channel = 1
        loss_fn = torch.nn.MSELoss()

        # Init model
        print("Initialize model")
        model = InvSalesCritic(**MODEL_HYPER_PARAMS).double().to(DEVICE)
        mlflow.log_param('Model architecture', str(type(model)))

        # Init loss and optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        # Initiate writer. Adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_validation_loss = 1_000.
        writer = SummaryWriter('runs/{}_{}'.format(experiment_name, run_id))

        print("Start model training")
        for epoch in range(EPOCHS):
            # train loop
            model.train(True)
            avg_loss = 0

            print('EPOCH {}:'.format(epoch_number + 1))
            print('START EPOCH {}:'.format(epoch_number + 1))

            avg_loss = train_one_epoch(train_loader, epoch_number, writer)

            # We don't need gradients for validation performance calculation
            model.train(False)
            running_validation_loss = 0

            running_validation_loss_epo = evaluate(validation_loader)
            running_validation_loss += running_validation_loss_epo

            avg_validation_loss = running_validation_loss / len(validation_loader)
            print('LOSS train {} valid {}'.format(avg_loss, avg_validation_loss))

            log_scalar(
                tb_writer=writer,
                name='Training Loss',
                value=avg_loss,
                step=epoch_number + 1
            )
            
            log_scalar(
                tb_writer=writer,
                name='Validation Loss',
                value=avg_validation_loss,
                step=epoch_number + 1,
                flush=True
            )

            # Track best performance, and save the model's state
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                # relpace with mlflow log artifacts                
                mlflow.pytorch.log_model(model, '{}_{}_epo_{}'.format(experiment_name, run_id, epoch_number))

            epoch_number += 1
