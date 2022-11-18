from torch.utils.data import DataLoader
from src.data.simulate_data import create_dummy_inv_model_data
from src.data.dataloader import InvSalesData, collate_fn


mu = 0
sigma = 1
n_samples = 20
hist_len = [12, 3, 3, 1]
feature_vec_size = [10, 5, 5, 5]
batch_size = 4
feat_name_prefix = ['profile', 'digital', 'meeting', 'action']
dataset = create_dummy_inv_model_data(mu, sigma, n_samples, hist_len, feature_vec_size, feat_name_prefix)

profile_df, digital_df, meeting_df, action_df = dataset
dataset_torch = InvSalesData(action_df, digital_df, meeting_df, profile_df, key_column='ids')
dataloader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

sample = None
for idx, sample_batched in enumerate(dataloader):
    if idx > 0:
        break
    sample = sample_batched

feature_vec_size[-1] = feature_vec_size[-1] - 1

for idx, (history_len, vec_size) in enumerate(zip(hist_len, feature_vec_size)):
    assert sample[idx].shape == (batch_size, history_len, vec_size)

