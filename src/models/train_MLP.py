import sys
sys.path.insert(0, "..")

from datetime import datetime
import numpy as np
import os
from tensorflow import keras
from src.data.load_data import *
from src.models.model_MLP import JokeRecommender

# load data

df = load_dataset(filename='../../data/Jester-Dataset-ratings.csv')
user_ids, joke_ids, ratings = get_data(df, batch_size=20000)

# prepare data

user_encoded = encode(user_ids)
joke_encoded = encode(joke_ids)

final_vector = []
for u, i in zip(user_encoded, joke_encoded):
    final_vector.append(np.concatenate((u, i), axis=None))
    

model = JokeRecommender(final_vector)

model.compile(
    optimizer='adam', 
    loss='mean_squared_error', 
    metrics=['accuracy']
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=os.path.join("../../logs", str(datetime.now())),
    histogram_freq=1)

model.fit(
    x=np.array(final_vector),
    y=np.array(ratings),
    batch_size=100, 
    epochs=50,
    callbacks=[tensorboard_callback],
    validation_split=0.2
)