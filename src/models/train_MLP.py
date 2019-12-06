import sys
sys.path.insert(0, "..")

from datetime import datetime
import numpy as np
import os
from tensorflow import keras
from src.data.load_data import *
from src.models.model_MLP import JokeRecommender
from tensorflow.keras.layers import Input

emb_output_dim = 5

# load and preprocess data

df = load_dataset(filename='../../data/Jester-Dataset-ratings.csv')
df = df[:20000]   # delete later

df['USER_ID'] = encode_values(df['USER_ID'])
df['JOKE_ID'] = encode_values(df['JOKE_ID'])

number_users = len(df['USER_ID'].unique())
number_jokes = len(df['JOKE_ID'].unique())

train, test = split_data(df)
y_true = test['Rating']


model = JokeRecommender(emb_output_dim, number_users, number_jokes)

model.compile(
    optimizer='adam', 
    loss='mean_absolute_error', 
)

model.fit(
    [np.array(train['USER_ID']), np.array(train['JOKE_ID'])],
    np.array(train['Rating']), 
    epochs=2, 
    verbose=1, 
    validation_split=0.1
)


# Evaluation
print(model.summary())

print(test['USER_ID'])
print(test['JOKE_ID'])
from sklearn.metrics import mean_absolute_error
y_pred = model.predict([test['USER_ID'], test['JOKE_ID']])
print("MSE")
print(mean_absolute_error(y_true, y_pred))

