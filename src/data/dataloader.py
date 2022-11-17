import os
import pandas as pd
import tqdm
from torch.utils.data import Dataset
from src.utils.utils import load_yaml, read_sql_chunk
from src.utils.custom_sqlconnector import make_sql_connection


server_config = load_yaml(os.path.join('config', 'data_source_config.yaml'))


def make_db_connection():
    conn = make_sql_connection(
        server_name=server_config['input_server'], dbname=server_config['input_db'], conn_loc='edge_node'
    )
    return conn


def get_num_batch(reward_table):
    conn = make_db_connection()
    n_batch = pd.read_sql(
        """select max(new_batch_nr) from {db}.{schema}.{reward_table}""".format(
            db=server_config['input_db'], schema=server_config['schema'], reward_table=reward_table
        ),
        conn
    ).values[0][0]
    conn.close()

    return n_batch


def rebatch_data(batch_size=64, reward_table='AA_MODEL_REWARD'):
    conn = make_db_connection()
    # calculate number  of batches
    data_len = pd.read_sql(
        """select count_big(*) from {db}.{schema}.{reward_table}""".format(
            db=server_config['input_db'], schema=server_config['schema'], reward_table=reward_table
        ),
        conn
    ).values[0][0]

    if data_len % batch_size == 0:
        num_batch = data_len // batch_size
    else:
        num_batch = data_len // batch_size + 1
    print("Batch into ", num_batch)

    columns = list(
        pd.read_sql(
            """select top 1 * from {db}.{schema}.{reward_table}""".format(
                db=server_config['input_db'], schema=server_config['schema'], reward_table=reward_table
            ), conn
        ).columns
    )

    columns = [col for col in columns if col != 'new_batch_nr']
    column_names = ', '.join(columns)

    # re-batching query
    query = """
    drop table if exists #tmp
    select  {column_names},
            (row_number() over (order by rand_row_id) % {num_batch}) as new_batch_nr --shuffle then batch
    into #tmp
    from
    (   select  {column_names},
                CRYPT_GEN_RANDOM(3) rand_row_id --for shuffling
        from {db}.{schema}.{reward_table}
    ) a
    
    drop table if exists {db}.{schema}.{reward_table}
    select *
    into {db}.{schema}.{reward_table}
    from #tmp
    
    drop index if exists r_idx on {db}.{schema}.{reward_table}
    create nonclustered index r_idx on {db}.{schema}.{reward_table} (CustPrimeIpId, Sample_dt, new_batch_nr)
    """.format(
        db=server_config['input_db'], schema=server_config['schema'],
        reward_table=reward_table, num_batch=num_batch, column_names=column_names
    )

    conn.execute("COMMIT")
    conn.execute(query)
    conn.execute("COMMIT")
    conn.close()

    return num_batch


# TODO: Add prediction mode to data loader
def make_batch_query(base_query, batch_nr=None, batch_range=None, **kwargs):
    if batch_range is not None:
        batch_condition = "new_batch_nr between {start} and {end}".format(start=batch_range[0], end=batch_range[1])
    elif batch_nr is not None:
        batch_condition = "new_batch_nr = {batch_nr}".format(batch_nr=batch_nr)
    else:
        raise ValueError('Please provide either batch_range or batch_nr')

    query = base_query.format(batch_condition=batch_condition, **kwargs)

    return query


def load_data(conn, query, process_fn, is_single_batch, batch_by=['new_batch_nr']):
    if is_single_batch:
        df = pd.read_sql(query, conn)
    else:
        df = read_sql_chunk(query, conn)

    if is_single_batch:
        return process_fn(df)

    else:
        outputs = dict()
        for group, data_df in tqdm.tqdm(df.groupby(batch_by)):
            batch = process_fn(data_df)
            outputs[group] = batch

    return outputs


# created as make function instead of class to ensure compatibility with multiprocess in torch loader
# (pickle-able object)
def make_generic_load_sample_fn(steps, sort_fn, is_single_batch=True, collate_fn=None, batch_by=['new_batch_nr']):
    """
    Args:
        steps: List of tuple of data processing steps
        sort_fn:
        reward_table: list of ('source_name', 'query_template', process_fn, {'template_args': 'table_name'})
        is_single_batch:
        collate_fn:

    Returns:

    """
    def load_samples(batch_nr, conn):
        single_batch = is_single_batch

        batches = list()
        for query_template, process_fn, queries_tables in steps:
            if single_batch:
                query = make_batch_query(
                    base_query=query_template, batch_nr=batch_nr, batch_range=None,
                    **queries_tables
                )
            else:
                query = make_batch_query(
                    base_query=query_template, batch_nr=None, batch_range=batch_nr,
                    **queries_tables
                )

            data_item = load_data(
                conn, query=query, process_fn=process_fn, is_single_batch=single_batch, batch_by=batch_by
            )
            batches.append(data_item)

        if single_batch:
            model_batch, ids = sort_fn(*batches)
            return model_batch, ids

        else:
            outputs = dict()
            for batch_id in batches[0].keys():
                batch = [data_item[batch_id] for data_item in batches]

                if collate_fn is not None:
                    outputs[batch_id] = collate_fn(sort_fn(*batch))
                else:
                    outputs[batch_id] = sort_fn(*batch)

            return outputs

    return load_samples


# Partly adapted from https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
# Generic class for streaming pre-batched SQL data into PyTorch model
# Pre-batched data are suitable for any SQL dataset with multiple tables that cannot be joined into single table
# and stream with pandas chunk/Dask to parallelize data loading
class SQLBatchData(Dataset):
    def __init__(self, num_batch, make_conn_fn, load_sample_fn):
        # take in make connection function instead of connection object
        # to prevent overhead when using dataset with multiple worker in loader
        self.num_batch = num_batch
        self.make_conn_fn = make_conn_fn
        self.load_sample_fn = load_sample_fn

    def __len__(self):
        return self.num_batch

    def __getitem__(self, idx):
        # start = round(time.time() * 1000)
        # worker = torch.utils.data.get_worker_info()
        # worker_id = worker.id if worker is not None else -1

        conn = self.make_conn_fn()
        # beware. Samples return here must be tuples/list of torch tensor compatible objects
        # unless you define default collate_fn or don't want to use it with DataLoader
        samples = self.load_sample_fn(batch_nr=idx, conn=conn)
        # end = round(time.time() * 1000)

        return samples  # , worker_id, start, end
