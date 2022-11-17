import yaml
import torch
import pandas as pd
import tqdm
import torch.nn.functional as F


def load_yaml(path):
    """
    :param path: path to yaml file
    :return: dictionary of yaml file
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def grant_access(conn, table_name, dbname, user):
    conn.execute("COMMIT")
    conn.execute(
        """
        use {dbname}
        grant select on {table_name} to {user}
        """.format(dbname=dbname, table_name=table_name, user=user)
    )
    conn.execute("COMMIT")


def make_key(iterable):
    return '-'.join([str(i).rstrip().replace(' ', '_') for i in iterable])


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


def get_total_number_of_param(model):
    total_params = 0
    for parameter in model.parameters():
        total_params = total_params + torch.prod(torch.tensor(parameter.shape)).tolist()

    return total_params


def read_sql_chunk(sql, con, chunksize=100000):
    dfs = list()

    print('Start loading data')
    for df_chunk in tqdm.tqdm(pd.read_sql(sql=sql, con=con, chunksize=chunksize)):
        dfs.append(df_chunk)

    return pd.concat(dfs, axis=0)






