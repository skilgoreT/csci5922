#%load drop_from_panda.py
import pandas as pd
import os
def load_data(filename):
    csv_path = os.path.join(".", filename)
    return pd.read_csv(csv_path)

room_data = load_data('train_data.txt')
room_data.head()
# Drop colums from dataframe - axis=0 (row), axis=1 (column)
room_data = room_data.drop(['date','Occupancy'], axis=1)
room_data.describe()