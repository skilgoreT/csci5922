#%load read_data.py
import pandas as pd
import os
def load_data(filename):
    csv_path = os.path.join(".", filename)
    return pd.read_csv(csv_path)

room_data = load_data('train_data.txt')
room_data.head()
