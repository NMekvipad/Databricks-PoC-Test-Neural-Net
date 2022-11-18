import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def pad_sequences_to_fixed_length(sequences, max_len, pad_last_dim=False, value=0):

    if not isinstance(sequences, list):
        raise ValueError("sequences must be a list of tensor or list of tensor like structure")

    padded_seq = list()

    for seq in sequences:
        seq = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq)

        if pad_last_dim:
            seq_len = seq.shape[-1]
            pad_len = 0 if max_len - seq_len < 0 else int(max_len - seq_len)
            padded_seq.append(F.pad(input=seq, pad=(0, pad_len), mode="constant", value=value))

        else:
            seq_len = seq.shape[0]
            pad_len = 0 if max_len - seq_len < 0 else int(max_len - seq_len)
            padded_seq.append(F.pad(input=seq.t(), pad=(0, pad_len), mode="constant", value=value).t())

    return padded_seq


class InvSalesData(Dataset):
    def __init__(
            self, action_df, digital_df, meeting_df, profile_df,
            key_column='ids'
    ):
        # take in make connection function instead of connection object
        # to prevent overhead when using dataset with multiple worker in loader
        self.action_df = action_df
        self.digital_df = digital_df
        self.meeting_df = meeting_df
        self.profile_df = profile_df
        self.num_samples = self.action_df.shape[0]
        self.key_column = key_column

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # start = round(time.time() * 1000)
        # worker = torch.utils.data.get_worker_info()
        # worker_id = worker.id if worker is not None else -1

        profile = self.profile_df[self.profile_df[self.key_column] == idx].iloc[:, 1:].values
        meeting = self.meeting_df[self.meeting_df[self.key_column] == idx].iloc[:, 1:].values
        digital = self.digital_df[self.digital_df[self.key_column] == idx].iloc[:, 1:].values
        action = self.action_df[self.action_df[self.key_column] == idx].iloc[:, 1:-1].values
        reward = self.action_df[self.action_df[self.key_column] == idx].values[0][-1]

        sample = (profile, meeting, digital, action, reward), idx

        return sample


def collate_fn(batch):
    ids = list()
    profile_data = list()
    meeting_data = list()
    digital_data = list()
    action_data = list()
    reward_data = list()

    for sample in batch:
        (profile, meeting, digital, action, reward), idx = sample
        profile_data.append(torch.tensor(profile, dtype=torch.double))
        meeting_data.append(torch.tensor(meeting, dtype=torch.double))
        digital_data.append(torch.tensor(digital, dtype=torch.double))
        action_data.append(torch.tensor(action, dtype=torch.double))
        reward_data.append(reward)
        ids.append(idx)

    profile_data = pad_sequences_to_fixed_length(profile_data, max_len=12)
    meeting_data = pad_sequence(meeting_data, batch_first=True)
    digital_data = pad_sequence(digital_data, batch_first=True)

    return (
               profile_data, meeting_data, digital_data, torch.tensor(action_data), torch.tensor(reward_data).flatten()
           ), ids
