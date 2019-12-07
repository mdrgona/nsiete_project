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

train, _ = load_dataset()
train['USER_ID'] = encode_values(train['USER_ID'])
train['JOKE_ID'] = encode_values(train['JOKE_ID'])
number_users = len(train['USER_ID'].unique())
number_jokes = len(train['JOKE_ID'].unique())


# Create, compile and fit model

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