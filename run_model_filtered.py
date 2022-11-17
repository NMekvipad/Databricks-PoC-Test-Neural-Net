import os
import torch
import pickle
import tqdm
from src.data.dataloader import (
    make_db_connection, sales_model_collate_fn, get_num_batch, make_load_sample_fn
)
from src.model.nn import InvSalesNet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# Training parameters
READ_FROM_SQL = True
EPOCHS = 40
BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data parameters
REWARD_DATA = 'AA_MODEL_REWARD_FILTERED'
PROFILE_DATA = 'AA_CUST_PROFILE_NORM'
CAMPAIGN_DATA = 'AA_CAMPAIGN_HISTORY_NORM'
MEETING_DATA = 'AA_MEETING_HISTORY_NORM'

n_batch = get_num_batch()
n_batch_per_read = 10000
train_size = int(n_batch * 0.2)
validation_size = int(n_batch * 0.05)
reserved_size = int(n_batch - train_size - validation_size)
n_load = train_size // n_batch_per_read + (0 if train_size % n_batch_per_read == 0 else 1)
n_load_validation = validation_size // n_batch_per_read + (0 if validation_size % n_batch_per_read == 0 else 1)

load_samples = make_load_sample_fn(
    reward_table=REWARD_DATA, profile_table=PROFILE_DATA,
    campaign_table=CAMPAIGN_DATA, meeting_table=MEETING_DATA,
    is_single_batch=False, collate_fn=sales_model_collate_fn
)


# adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(loader, start_idx, epoch_index, train_size, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, batch in enumerate(tqdm.tqdm(loader.values())):
        (cust_profile, actions, rewards, campaign_history, meeting_history), _ = batch  # _, _, _,
        x = [
            cust_profile.div(10000).double(), campaign_history.div(10000).double(), meeting_history.div(10000).double()
        ]

        # Set gradient to 0
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x)

        # Gather predicted state action value for specific action took in historical data
        state_action_values = outputs.gather(1, actions.unsqueeze(1))

        # Loss calculation & backward pass
        loss = loss_fn(state_action_values.flatten(), rewards.div(10000).double())
        loss.backward()

        # gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2.0)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if (i + start_idx) % 500 == 499:
            last_loss = running_loss / 500  # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = (epoch_index * train_size) + start_idx + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def evaluate(loader):
    running_validation_loss = 0.0
    for i, batch in tqdm.tqdm(loader.items()):
        (cust_profile, actions, rewards, campaign_history, meeting_history), _ = batch
        x = [
            cust_profile.div(10000).double(), campaign_history.div(10000).double(), meeting_history.div(10000).double()
        ]
        outputs = model(x)
        state_action_values = outputs.gather(1, actions.unsqueeze(1))
        loss = loss_fn(state_action_values.squeeze(0), rewards.div(10000).double())
        running_validation_loss += loss

    return running_validation_loss


if __name__ == '__main__':
    # Init model
    print("Initialize model")
    model = InvSalesNet(
        in_channels_profile=68,
        in_channels_campaign=24,
        in_channels_meeting=9,
        profile_hist_len=12,
        out_channel_profile=10, hidden_size_campaign=10, hidden_size_meeting=5,
        out_channel=3, kernel_size=3, dropout=0.2
    ).double()

    # Init loss and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Initiate writer. Adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/inv_sales_model_norm_{}'.format(timestamp))
    epoch_number = 0
    best_validation_loss = 1_000.

    print("Start model training")
    for epoch in range(EPOCHS):
        # train loop
        model.train(True)
        avg_loss = 0
        for idx in range(n_load):
            start_batch = idx * n_batch_per_read
            end_batch = (idx + 1) * n_batch_per_read - 1

            # final train batch
            if (idx + 1) == n_load:
                end_batch = train_size - 1

            print("Load dataset")
            if READ_FROM_SQL and epoch == 0:  # Only read data for 1st epoch
                conn = make_db_connection()
                train_bulk = load_samples(batch_nr=(start_batch, end_batch), conn=conn)

                with open('./artifacts/dataset/train_bulk_{}.pickle'.format(idx), 'wb') as f:
                    pickle.dump(train_bulk, f)
            else:
                with open('./artifacts/dataset/train_bulk_{}.pickle'.format(idx), 'rb') as f:
                    train_bulk = pickle.load(f)

            print('EPOCH {} bulk {}:'.format(epoch_number + 1, idx))
            print('START EPOCH {} bulk {}:'.format(epoch_number + 1, idx))
            avg_loss = train_one_epoch(train_bulk, start_batch, epoch_number, train_size, writer)

        # We don't need gradients for validation performance calculation
        model.train(False)
        running_validation_loss = 0
        for idx in range(n_load_validation):
            start_batch = train_size + idx * n_batch_per_read
            end_batch = train_size + (idx + 1) * n_batch_per_read - 1

            # final validation batch
            if (idx + 1) == n_load_validation:
                end_batch = train_size + validation_size - 1

            print("Load dataset")
            if READ_FROM_SQL and epoch == 0:  # Only read data for 1st epoch
                conn = make_db_connection()
                validation_bulk = load_samples(batch_nr=(start_batch, end_batch), conn=conn)

                with open('./artifacts/dataset/validation_bulk_{}.pickle'.format(idx), 'wb') as f:
                    pickle.dump(validation_bulk, f)
            else:
                with open('./artifacts/dataset/validation_bulk_{}.pickle'.format(idx), 'rb') as f:
                    validation_bulk = pickle.load(f)

            running_validation_loss += evaluate(validation_bulk)

        avg_validation_loss = running_validation_loss / validation_size
        print('LOSS train {} valid {}'.format(avg_loss, avg_validation_loss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_validation_loss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            model_path = os.path.join('artifacts', 'model', 'model_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_path)

        epoch_number += 1



