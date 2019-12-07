import sys
sys.path.append('../..')

import numpy as np
import os
from tensorflow import keras
from src.data.load_data import *
from src.models.model_MLP import JokeRecommender
from src.models.predict import *

# Set constants
emb_output_dim = 5
tb_cb = keras.callbacks.TensorBoard(log_dir=os.path.join("../../logs", str(datetime.datetime.now())),histogram_freq=1)


# Load and preprocess data

df = load_dataset(filename='../../data/Jester-Dataset-ratings.csv')
df['USER_ID'] = encode_values(df['USER_ID'])
df['JOKE_ID'] = encode_values(df['JOKE_ID'])
number_users = len(df['USER_ID'].unique())
number_jokes = len(df['JOKE_ID'].unique())

train, test = split_data(df)
y_true = test['Rating']


# Create, complie and fit model

model = JokeRecommender(emb_output_dim, number_users, number_jokes)
model.compile(optimizer='adam', loss='mean_absolute_error')

model.fit(
    [np.array(train['USER_ID']), np.array(train['JOKE_ID'])],
    np.array(train['Rating']), 
    epochs=30, 
    verbose=1,
    validation_split=0.1,
    callbacks=[tb_cb]
)

model.save('../../models/MLP_1') 


# Evaluate metrics

y_pred = predict(model, test)
evaluate(y_true, y_pred)
# get_precision(model, train, test)
