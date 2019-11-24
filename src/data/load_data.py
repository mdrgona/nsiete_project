import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(df, batch_size=False):
    
    if not batch_size:
        batch_size = len(df)
        
        
    user_ids = df['USER_ID'][0:batch_size].tolist()
    joke_ids = df['JOKE_ID'][0:batch_size].tolist()
    ratings = df['Rating'][0:batch_size].tolist()
    
    return user_ids, joke_ids, ratings


def load_dataset(filename):
    return pd.read_csv(filename, delimiter=',')


def split_dataset(df, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)