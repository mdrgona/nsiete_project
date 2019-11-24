import numpy as np
import pandas as pd

def get_data(df, batch_size=False):
    
    if batch:
        batch_size = len(df)
       
    user_ids = df['USER_ID'][0:batch_size].tolist()
    joke_ids = df['JOKE_ID'][0:batch_size].tolist()
    ratings = df['Rating'][0:batch_size].tolist()
    
    return user_ids, joke_ids, ratings


def load_dataset(filename):
    return pd.read_csv(filename, delimiter=',')

