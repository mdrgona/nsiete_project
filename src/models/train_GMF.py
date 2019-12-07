import sys
sys.path.insert(0, "..")

from datetime import datetime
import numpy as np
import os
from tensorflow import keras
from src.data.load_data import *
from src.models.model_GMF import JokeRecommender

# load data

df = load_dataset(filename='../../data/Jester-Dataset-ratings.csv')
user_ids, joke_ids, ratings = get_data(df, batch_size=20000)

# prepare data

user_encoded = encode(user_ids)
joke_encoded = encode(joke_ids)

n_users = len(user_encoded[0])
n_jokes = len(joke_encoded[0])
num_classes = n_users if n_users > n_jokes else n_jokes

user_onehot = to_categorical(user_ids, num_classes=num_classes)
joke_onehot = to_categorical(joke_ids, num_classes=num_classes)

final_vector = np.concatenate([user_onehot, joke_onehot], axis=1)

n_users = len(user_onehot[0])
n_jokes = len(joke_onehot[0])

print(n_users, n_jokes)

model = JokeRecommender(n_users, n_jokes)

model.compile(
    optimizer='adam', 
    loss='mean_squared_error'
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=os.path.join("../../logs", str(datetime.now())),
    histogram_freq=1)

model.fit(
    x=np.array(final_vector),
    y=np.array(ratings),
    batch_size=100, 
    epochs=10,
    #callbacks=[tensorboard_callback],
    validation_split=0.2
)