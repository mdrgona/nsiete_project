import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_dataset(filename):
    return pd.read_csv(filename, delimiter=',')

def split_dataset(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=1)

def encode_values(ids):
    return ids.astype('category').cat.codes.values




def get_data(df, batch_size=False):
    if not batch_size:
        batch_size = len(df)
        
    user_ids = df['USER_ID'][0:batch_size].tolist()
    joke_ids = df['JOKE_ID'][0:batch_size].tolist()
    ratings = df['Rating'][0:batch_size].tolist()
    
    return user_ids, joke_ids, ratings


# def load_dataset(filename, nrows=0):
#     if nrows == 0:
#         return pd.read_csv(filename, delimiter=',')
#     return pd.read_csv(filename, delimiter=',', nrows=nrows)


# def load_train_data():
#     return pd.read_csv('../../data/Jester-train.csv', delimiter=',')

# def load_test_data():
#     return pd.read_csv('../../data/Jester-test.csv', delimiter=',')

# def encode(arr):
#     return to_categorical(arr)
