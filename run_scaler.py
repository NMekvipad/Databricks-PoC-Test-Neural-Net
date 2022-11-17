import os
from src.utils.utils import load_yaml
from src.data.scaler import sql_min_max_scaling
from src.data.dataloader import make_db_connection


server_config = load_yaml(os.path.join('config', 'data_source_config.yaml'))

MAX_SCALING = 10000  # set scaling range to 0-10000 to avoid underflow in SQL
conn = make_db_connection()

if __name__ == '__main__':
    profile_key = ['FactDt', 'CustPrimeIpId']
    sql_min_max_scaling(
        conn,
        table_name='{db}.{schema}.{table_name}'.format(
            db=server_config['input_db'], schema=server_config['schema'], table_name=server_config['profile_table_name']
        ),
        key=profile_key,
        scale_max=MAX_SCALING, scaling_parameter_path=os.path.join('config', 'scaler_params', 'profile_scaler.csv')
    )

    campaign_key = ['CustPrimeIpId', 'CONTACT_DT', 'CAMPAIGN_CD', 'Campaign_hist_id']
    sql_min_max_scaling(
        conn,
        table_name='{db}.{schema}.{table_name}'.format(
            db=server_config['input_db'], schema=server_config['schema'],
            table_name=server_config['campaign_table_name']
        ),
        key=campaign_key,
        scale_max=MAX_SCALING, scaling_parameter_path=os.path.join('config', 'scaler_params', 'campign_scaler.csv')
    )

    meeting_key = ['CustPrimeIpId', 'MeetingDt', 'Meeting_hist_id']
    sql_min_max_scaling(
        conn,
        table_name='{db}.{schema}.{table_name}'.format(
            db=server_config['input_db'], schema=server_config['schema'], table_name=server_config['meeting_table_name']
        ),
        key=meeting_key,
        scale_max=MAX_SCALING, scaling_parameter_path=os.path.join('config', 'scaler_params', 'meeting_scaler.csv')
    )

    query = """
    create nonclustered index r_idx on H5CO_CommercialAnalytics.dbo.{profile_table} (CustPrimeIpId, FactDt)
    create nonclustered index r_idx on H5CO_CommercialAnalytics.dbo.{meeting_table} (CustPrimeIpId, MeetingDt)
    create nonclustered index r_idx on H5CO_CommercialAnalytics.dbo.{campaign_table} (CustPrimeIpId, CONTACT_DT)
    """.format(
        profile_table=server_config['profile_table_name'],
        meeting_table=server_config['meeting_table_name'],
        campaign_table=server_config['campaign_table_name']
    )
    conn.execute(query)



