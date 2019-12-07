import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def get_data(df, batch_size=False):
    
    if not batch_size:
        batch_size = len(df)
        
        
    user_ids = df['USER_ID'][0:batch_size].tolist()
    joke_ids = df['JOKE_ID'][0:batch_size].tolist()
    ratings = df['Rating'][0:batch_size].tolist()
    
    return user_ids, joke_ids, ratings


#def load_dataset(filename, nrows=0):
#    if nrows == 0:
#        return pd.read_csv(filename, delimiter=',')
#    return pd.read_csv(filename, delimiter=',', nrows=nrows)



def load_dataset():
	train = pd.read_csv('../../data/Jester-train.csv', delimiter=',')
	test = pd.read_csv('../../data/Jester-test.csv', delimiter=',')
	return train, test


def encode(arr):
    return to_categorical(arr)
 
def encode_values(ids):
    return ids.astype('category').cat.codes.values
