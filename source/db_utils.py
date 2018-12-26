import os
import sqlite3

import numpy as np
import pandas as pd
from config_utils import get_pars_from_ini

sqlite3.register_adapter(np.int64, lambda val: int(val))


def create_db(table_name='raw_data', trigger=False, dt_trigger=None):
    """

    :param table_name:
    :param trigger: True for creating trigger
    :param dt_trigger: Dictionary with window bounds for a given area
    :return:
    """
    db_path = '../db/db_rayden.db'
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    query = """
    CREATE TABLE IF NOT EXISTS {}(
    datetime DATETIME NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    height REAL NOT NULL,
    type INTEGER NOT NULL,
    amperage REAL NOT NULL,
    error REAL NOT NULL
    );
    """.format(table_name)
    cur.execute(query)

    query = """
    CREATE UNIQUE INDEX idx_{0} ON {0} (datetime, latitude, longitude);
    """.format(table_name)
    cur.execute(query)

    if trigger:
        x_min = dt_trigger['x_min']
        x_max = dt_trigger['x_max']
        y_min = dt_trigger['y_min']
        y_max = dt_trigger['y_max']

        query = """
        CREATE TRIGGER tgr_{0} BEFORE INSERT ON raw_data
        WHEN (
            NEW.longitude BETWEEN {1} AND {2} 
            AND NEW.latitude BETWEEN {3} AND {4} 
            AND ABS(NEW.amperage) >= 10
            AND NEW.error <= 1
            )
        BEGIN
            INSERT OR REPLACE INTO {0} (datetime, latitude, longitude, height, type, amperage, error)
            VALUES(NEW.datetime, NEW.latitude, NEW.longitude, NEW.height, NEW.type, NEW.amperage, NEW.error);
        END;
        """.format(table_name, x_min, x_max, y_min, y_max)
        cur.execute(query)

    con.close()


def main():
    create_db()
    dt_bogota = get_pars_from_ini()['Bogota']
    create_db('data_bogota', True, dt_bogota)

    data_path = '../data'
    list_files = ['{}/{}'.format(data_path, i) for i in sorted(os.listdir(data_path))]

    db_path = '../db/db_rayden.db'

    for data_file in list_files:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        print(data_file)
        df_data = pd.read_csv(data_file)
        df_data.sort_values('Date', inplace=True)
        to_db = [tuple(df_data.loc[i]) for i in df_data.index]
        query = """
        INSERT OR REPLACE INTO raw_data(datetime, latitude, longitude, height, type, amperage, error)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """
        cur.executemany(query, to_db)
        con.commit()
        con.close()


if __name__ == '__main__':
    main()
    pass
