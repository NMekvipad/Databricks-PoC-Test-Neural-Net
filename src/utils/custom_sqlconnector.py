import urllib
import sqlalchemy as sa
import pandas as pd

connection_dict = {
    'JUNE': {
        'local': 'y79494',
        'edge_node': 'y79494.danskenet.net'
    },
    'ETPEW': {
        'local': 'etpew\inst004',
        'edge_node': 'K126S01.danskenet.net,50004'
    },
    'DPL_R': {
        'local': 'DWH-SQL-PROD',
        'edge_node': 'WK2312.danskenet.net'
    },
    'DPL_RW': {
        'local': 'DWH-SQL-PROD-RW',
        'edge_node': 'WK2311.danskenet.net'
    },
    'MART': {
        'local': 'MART-SQL-PROD',
        'edge_node': 'K284S01.danskenet.net'
    },
}


def make_sql_connection(server_name, dbname, conn_loc='local', verbose=False):

    if conn_loc == 'local':
        driver = 'SQL Server Native Client 11.0'
    elif conn_loc == 'edge_node':
        driver = 'ODBC Driver 13 for SQL Server'

    server = connection_dict[server_name][conn_loc]

    connection_string = """DRIVER={};SERVER={};DATABASE={};Trusted_Connection=yes""".format(driver, server, dbname)
    params = urllib.parse.quote_plus(connection_string)
    engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))

    try:
        conn = engine.connect()
        if verbose:
            print("Successfully connect to {}.{}".format(server, dbname))

        return conn
    except KeyError:
        print('Cannot connect to {}.{}'.format(server, dbname))

        return None

def grant_permission(conn, db_name, schema, name_pattern, user):

    query = """select TABLE_NAME from INFORMATION_SCHEMA.TABLES 
    where TABLE_NAME like '%{}%'""".format(name_pattern)
    df = pd.read_sql(query, conn)
    table_names = list(df['TABLE_NAME'])

    for name in table_names:
        tmp_table = '.'.join([db_name, schema, name])
        grant_query = """grant select on {} to {}""".format(tmp_table, user)
        print(grant_query)
        conn.execute(grant_query)