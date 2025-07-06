import pandas as pd
import os

def load_data(data_dir, columns=None):

    data = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training-set.csv'))
    if columns is None:
        columns = data.columns

    acceptable = ['1.csv', '2.csv', '3.csv', '4.csv']
    for filename in os.listdir(data_dir):
        if any(filename.endswith(suffix) for suffix in acceptable):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path, header=0, names=columns, skiprows=1, low_memory=False)
            data = pd.concat([data, df], ignore_index=True)
    return data


data = load_data('data')

pd.set_option('display.max_columns', None)
print(data.dtypes)