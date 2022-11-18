import numpy as np
import pandas as pd


def simulate_samples_history(mu, sigma, n_samples, hist_len, feature_vec_size, feat_name_prefix):
    sample_ids = list()

    for i in range(n_samples):
        sample_ids.extend([i] * hist_len)

    feats = np.random.normal(mu, sigma, (hist_len * n_samples, feature_vec_size))
    feat_names = [feat_name_prefix + '_' + str(i) for i in range(feature_vec_size)]

    df = pd.DataFrame(feats, columns=feat_names)
    df.insert(0, "ids", sample_ids)

    return df


def transform_history(df, sample_ids, w_mu, w_sigma, hidden_size, feat_name_prefix, add_noise=False, e_mu=0, e_sigma=1):
    agg_df = df.groupby(sample_ids).mean().reset_index()
    keys = agg_df[sample_ids]

    values = agg_df.iloc[:, 1:].values
    w = np.random.normal(w_mu, w_sigma, (values.shape[1], hidden_size))
    feat_names = [feat_name_prefix + '_' + str(i) for i in range(hidden_size)]

    tf_values = np.matmul(values, w)

    if add_noise:
        tf_values = tf_values + np.random.normal(e_mu, e_sigma, tf_values.shape)

    tf_df = pd.DataFrame(tf_values, columns=feat_names)
    tf_df.insert(0, "ids", keys)

    return tf_df


def create_dummy_inv_model_data(mu, sigma, n_samples, hist_len, feature_vec_size, feat_name_prefix):

    dataset = list()

    for data_len, feat_size, feat_name in zip(hist_len, feature_vec_size, feat_name_prefix):
        df = simulate_samples_history(mu, sigma, n_samples, data_len, feat_size, feat_name)
        dataset.append(df)

    return dataset


