import os
import numpy as np
import torch
import pickle
import tqdm
import random
import argparse
import yaml
from sklearn.metrics import f1_score
from src.data.dataloader import make_db_connection, make_generic_load_sample_fn, get_num_batch, rebatch_data
from src.data.data_processing import (
    sales_model_meeting_only_collate_fn, profile_query, meeting_query,
    campaign_query, action_vector_query, process_profile_data, process_action_data,
    process_meeting_data, process_campaign_data, sort_sample_in_batch_meeting_only
)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.model.nn import InvSalesCriticGRU
from src.utils.utils import load_yaml
from src.data.predict import generate_prediction_table


parser = argparse.ArgumentParser()
parser.add_argument('--mode',  required=True, type=str)
parser.add_argument('--model_dir',  default=os.path.join('artifacts', 'model'), type=str)
parser.add_argument('--trained_model_path',  default='', type=str)
parser.add_argument('--output_dir',  default=os.path.join('artifacts', 'outputs'), type=str)
parser.add_argument('--dataset_dir',  default=os.path.join('artifacts', 'dataset'), type=str)
parser.add_argument('--train_config_dir',  default=os.path.join('artifacts', 'training_config'), type=str)
parser.add_argument('--model_config', default=os.path.join('config', 'model_config.yaml'), type=str)
parser.add_argument('--input_config',  default=os.path.join('config', 'input_config.yaml'), type=str)
parser.add_argument('--prediction_config',  default=os.path.join('config', 'prediction_config.yaml'), type=str)

args = parser.parse_args()
MODE = args.mode.lower()
TRAINED_MODEL_PATH = args.trained_model_path
MODEL_DIR = args.model_dir
OUTPUT_DIR = args.output_dir
DATASET_DIR = args.dataset_dir
CONFIG_DIR = args.train_config_dir

DIRS = [MODEL_DIR, OUTPUT_DIR, DATASET_DIR, CONFIG_DIR]
for dir_path in DIRS:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

server_config = load_yaml(os.path.join('config', 'data_source_config.yaml'))
model_config = load_yaml(args.model_config)
input_config = load_yaml(args.input_config)
prediction_config = load_yaml(args.prediction_config)


# model parameters
if MODE == 'predict':
    MODEL_NAME = prediction_config['data_name']
else:
    MODEL_NAME = model_config['model_name']

CLASSIFIER = True if model_config['model_type'] == 'classifier' else False
EPOCHS = model_config['epoch']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = model_config['num_class']
CLASS_WEIGHT = torch.tensor(model_config['class_weight']).double().to(DEVICE)
LOSS_RECORD_STEP = model_config['loss_recording_step']
MODEL_HYPER_PARAMS = model_config['model_hyperparameters']

# input parameters
SCALING_FACTOR = input_config['feature_scaling_factor']

if MODE == 'predict':
    if TRAINED_MODEL_PATH == '':
        raise ValueError("Please provide model path when choose 'predict' mode")

    READ_FROM_SQL = prediction_config['read_data_from_sql']
    REWARD_DATA = prediction_config['prediction_table']
    BATCH_SIZE = prediction_config['batch_size']
    REBATCH = prediction_config['rebatch']

    if prediction_config['create_prediction_table']:
        generate_prediction_table(data_source_config=server_config, prediction_config=prediction_config)
else:
    READ_FROM_SQL = input_config['read_data_from_sql']
    REWARD_DATA = input_config['target_table']
    BATCH_SIZE = model_config['batch_size']
    REBATCH = input_config['rebatch']


if REBATCH:
    rebatch_data(batch_size=BATCH_SIZE, reward_table=REWARD_DATA)

data_loading_steps = [
    (
        profile_query, process_profile_data,
        {
            'db': server_config['input_db'], 'schema': server_config['schema'], 'target_table': REWARD_DATA,
            'feature_table': input_config['profile_table'], 'factor_table': input_config['factor_table']
        }
    ),
    (
        campaign_query, process_campaign_data,
        {
            'db': server_config['input_db'], 'schema': server_config['schema'],
            'target_table': REWARD_DATA, 'feature_table': input_config['campaign_table']
        }
    ),
    (
        meeting_query, process_meeting_data,
        {
            'db': server_config['input_db'], 'schema': server_config['schema'],
            'target_table': REWARD_DATA, 'feature_table': input_config['meeting_table']
        }
    ),
    (
        action_vector_query, process_action_data,
        {'db': server_config['input_db'], 'schema': server_config['schema'], 'target_table': REWARD_DATA}
    ),
]

N_BATCH = get_num_batch(reward_table=REWARD_DATA)
N_BATCH_PER_READ = input_config['num_mini_batch_per_batch']
TRAIN_FRAC = input_config['train_fraction']
VALIDATION_FRAC = input_config['validation_fraction']
TRAIN_SIZE = int(N_BATCH * TRAIN_FRAC)
VALIDATION_SIZE = int(N_BATCH * VALIDATION_FRAC)
TRAIN_BATCHES = (0, TRAIN_SIZE - 1)
VALIDATION_BATCHES = (TRAIN_BATCHES[1] + 1, TRAIN_BATCHES[1] + VALIDATION_SIZE)
RESERVED_BATCHES = (VALIDATION_BATCHES[1], N_BATCH)
PREDICTION_BATCHES = (0, N_BATCH - 1)
PREDICTION_SIZE = N_BATCH
N_LOAD = int(TRAIN_SIZE // N_BATCH_PER_READ + (0 if TRAIN_SIZE % N_BATCH_PER_READ == 0 else 1))
N_LOAD_VALIDATION = int(VALIDATION_SIZE // N_BATCH_PER_READ + (0 if VALIDATION_SIZE % N_BATCH_PER_READ == 0 else 1))
N_LOAD_PREDICTION = int(PREDICTION_SIZE // N_BATCH_PER_READ + (0 if PREDICTION_SIZE % N_BATCH_PER_READ == 0 else 1))

load_samples = make_generic_load_sample_fn(
    steps=data_loading_steps, sort_fn=sort_sample_in_batch_meeting_only, is_single_batch=False,
    collate_fn=sales_model_meeting_only_collate_fn
)


# adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(loader, start_idx, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    train_data = list(loader.values())
    random.shuffle(train_data)

    if CLASSIFIER:
        pred_labels = list()
        true_labels = list()

    for i, batch in enumerate(train_data):
        (cust_profile, actions, rewards, campaign_history, meeting_history), _ = batch  # _, _, _,
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
        if CLASSIFIER:
            rewards = rewards.flatten().long()
            loss = loss_fn(outputs, rewards.to(DEVICE))

            pred_labels.append(torch.clone(torch.argmax(outputs, dim=1)).detach())
            true_labels.append(torch.clone(rewards).detach())
        else:
            rewards = rewards.div(SCALING_FACTOR).flatten().double()
            loss = loss_fn(outputs.flatten(), rewards.to(DEVICE))

        loss.backward()

        # gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2.0)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if (i + start_idx) % LOSS_RECORD_STEP == LOSS_RECORD_STEP - 1:
            last_loss = running_loss / LOSS_RECORD_STEP  # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = (epoch_index * TRAIN_SIZE) + start_idx + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    if CLASSIFIER:
        pred_labels = torch.cat(pred_labels).flatten().cpu().detach().numpy()
        true_labels = torch.cat(true_labels).flatten().cpu().detach().numpy()

        if NUM_CLASS > 2:
            f1_epoch = f1_score(true_labels, pred_labels, average='macro')
        else:
            f1_epoch = f1_score(true_labels, pred_labels, average='binary')

        return last_loss, f1_epoch

    else:
        return last_loss


def evaluate(loader):
    running_validation_loss = 0.0

    if CLASSIFIER:
        pred_labels = list()
        true_labels = list()

    for i, batch in loader.items():
        (cust_profile, actions, rewards, campaign_history, meeting_history), _ = batch
        x = [
            cust_profile.div(SCALING_FACTOR).double().to(DEVICE),
            campaign_history.div(SCALING_FACTOR).double().to(DEVICE),
            meeting_history.div(SCALING_FACTOR).double().to(DEVICE),
            actions.double().to(DEVICE)
        ]

        outputs = model(x)

        if CLASSIFIER:
            rewards = rewards.flatten().long()
            loss = loss_fn(outputs, rewards.to(DEVICE))

            pred_labels.append(torch.clone(torch.argmax(outputs, dim=1)).detach())
            true_labels.append(torch.clone(rewards).detach())
        else:
            rewards = rewards.div(SCALING_FACTOR).flatten().double()
            loss = loss_fn(outputs.flatten(), rewards.to(DEVICE))

        running_validation_loss += loss

    if CLASSIFIER:
        pred_labels = torch.cat(pred_labels).flatten().cpu().detach().numpy()
        true_labels = torch.cat(true_labels).flatten().cpu().detach().numpy()

        if NUM_CLASS > 2:
            f1_epoch = f1_score(true_labels, pred_labels, average='macro')
        else:
            f1_epoch = f1_score(true_labels, pred_labels, average='binary')

        return running_validation_loss, f1_epoch

    else:
        return running_validation_loss


def predict(loader):
    pred = list()
    truth = list()
    prob = list()
    cust_id = list()
    action_vec = list()

    for i, batch in enumerate(tqdm.tqdm(loader.values())):
        (cust_profile, actions, rewards, campaign_history, meeting_history), ids = batch
        x = [
            cust_profile.div(SCALING_FACTOR).double().to(DEVICE),
            campaign_history.div(SCALING_FACTOR).double().to(DEVICE),
            meeting_history.div(SCALING_FACTOR).double().to(DEVICE),
            actions.double().to(DEVICE)
        ]

        outputs = model(x)

        if CLASSIFIER:
            rewards = rewards.flatten().long()
            pred.append(torch.argmax(outputs, dim=1))
            prob.append(outputs)
        else:
            rewards = rewards.div(SCALING_FACTOR).flatten().double()
            pred.append(outputs)

        truth.append(rewards)
        action_vec.append(actions[:, :, 0].flatten())
        cust_id.extend(ids)

    return truth, (pred, prob), action_vec, cust_id


def get_batch_range(batch_size, batch_idx, start_at, end_at):
    start_batch = batch_idx * batch_size + start_at
    end_batch = (batch_idx + 1) * batch_size - 1 + start_at

    # final train batch
    if end_batch > end_at:
        end_batch = end_at

    return start_batch, end_batch


def load_and_write_batch(epoch, read_from_sql, start_batch, end_batch, data_name):
    print("Load dataset")
    if read_from_sql and epoch == 0:  # Only read data for 1st epoch
        conn = make_db_connection()
        batch = load_samples(batch_nr=(start_batch, end_batch), conn=conn)

        with open(os.path.join(DATASET_DIR, '{}_{}_{}.pickle'.format(data_name, start_batch, end_batch)), 'wb') as f:
            pickle.dump(batch, f)
    else:
        with open(os.path.join(DATASET_DIR, '{}_{}_{}.pickle'.format(data_name, start_batch, end_batch)), 'rb') as f:
            batch = pickle.load(f)

    return batch


if __name__ == '__main__':
    if CLASSIFIER:
        loss_fn = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT)
    else:
        out_channel = 1
        loss_fn = torch.nn.MSELoss()

    # Init model
    print("Initialize model")
    model = InvSalesCriticGRU(**MODEL_HYPER_PARAMS).double().to(DEVICE)

    if MODE == 'train':
        # Init loss and optimizer
        optimizer = torch.optim.SGD(model.parameters(), **model_config['training_parameters'])

        # Initiate writer. Adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/{}_{}'.format(MODEL_NAME, timestamp))
        epoch_number = 0
        best_validation_loss = 1_000.

        with open(os.path.join(CONFIG_DIR, '{}_{}_model_config.yaml'.format(MODEL_NAME, timestamp)), 'w') as f:
            yaml.dump(model_config, f)

        with open(os.path.join(CONFIG_DIR, '{}_{}_input_config.yaml'.format(MODEL_NAME, timestamp)), 'w') as f:
            yaml.dump(input_config, f)

        print("Start model training")
        for epoch in range(EPOCHS):
            # train loop
            model.train(True)
            avg_loss = 0
            for idx in range(N_LOAD):
                start_batch, end_batch = get_batch_range(
                    batch_size=N_BATCH_PER_READ, batch_idx=idx, start_at=TRAIN_BATCHES[0], end_at=TRAIN_BATCHES[1] - 1
                )
                train_bulk = load_and_write_batch(
                    epoch=epoch, read_from_sql=READ_FROM_SQL, start_batch=start_batch, end_batch=end_batch,
                    data_name='train_' + MODEL_NAME
                )

                print('EPOCH {} bulk {}:'.format(epoch_number + 1, idx))
                print('START EPOCH {} bulk {}:'.format(epoch_number + 1, idx))

                if CLASSIFIER:
                    avg_loss, f1_epoch_train = train_one_epoch(train_bulk, start_batch, epoch_number, writer)
                else:
                    avg_loss = train_one_epoch(train_bulk, start_batch, epoch_number, writer)

            # We don't need gradients for validation performance calculation
            model.train(False)
            running_validation_loss = 0
            for idx in range(N_LOAD_VALIDATION):
                start_batch, end_batch = get_batch_range(
                    batch_size=N_BATCH_PER_READ, batch_idx=idx, start_at=VALIDATION_BATCHES[0],
                    end_at=VALIDATION_BATCHES[1] - 1
                )
                validation_bulk = load_and_write_batch(
                    epoch=epoch, read_from_sql=READ_FROM_SQL, start_batch=start_batch, end_batch=end_batch,
                    data_name='validation_' + MODEL_NAME
                )

                if CLASSIFIER:
                    running_validation_loss_epo, f1_epoch_test = evaluate(validation_bulk)
                else:
                    running_validation_loss_epo = evaluate(validation_bulk)

                running_validation_loss += running_validation_loss_epo

            avg_validation_loss = running_validation_loss / VALIDATION_SIZE
            print('LOSS train {} valid {}'.format(avg_loss, avg_validation_loss))

            if CLASSIFIER:
                print('F1 train {} valid {}'.format(f1_epoch_train, f1_epoch_test))
                writer.add_scalars(
                    'Training vs. Validation F1',
                    {'Training': f1_epoch_train, 'Validation': f1_epoch_test},
                    epoch_number + 1
                )

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_validation_loss},
                               epoch_number + 1)

            writer.flush()

            # Track best performance, and save the model's state
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                model_path = os.path.join(MODEL_DIR, '{}_{}_{}'.format(MODEL_NAME, timestamp, epoch_number))
                torch.save(model.state_dict(), model_path)

            epoch_number += 1

    elif MODE == 'predict':
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device(DEVICE)))
        truth = list()
        pred = list()
        prob = list()
        action_vec = list()
        cust_id = list()

        for idx in range(N_LOAD_PREDICTION):
            start_batch, end_batch = get_batch_range(
                batch_size=N_BATCH_PER_READ, batch_idx=idx,
                start_at=PREDICTION_BATCHES[0], end_at=PREDICTION_BATCHES[1] - 1
            )
            evaluation_bulk = load_and_write_batch(
                epoch=0, read_from_sql=READ_FROM_SQL, start_batch=start_batch, end_batch=end_batch,
                data_name='predict_' + MODEL_NAME
            )

            truth_batch, (pred_batch, prob_batch), action_vec_batch, cust_id_batch = predict(evaluation_bulk)
            truth.extend(truth_batch)
            pred.extend(pred_batch)
            prob.extend(prob_batch)
            action_vec.extend(action_vec_batch)
            cust_id.extend(cust_id_batch)

        action_vec = torch.cat(action_vec).cpu().detach().numpy()
        pred = torch.cat(pred).cpu().detach()  # .numpy()
        prob = torch.cat(prob).cpu().detach()  # .numpy()
        prob_score = prob.gather(1, pred.unsqueeze(1)).flatten().numpy()
        pred = pred.numpy()

        outputs = [cust_id, prob_score, pred]
        outputs_name = ['cust_id', 'prob_score', 'pred']

        for outputs_obj, filename in zip(outputs, outputs_name):
            np.save(os.path.join(OUTPUT_DIR, filename + '.npy'), outputs_obj)




