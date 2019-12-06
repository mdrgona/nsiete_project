import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dropout, Lambda, Reshape, Dot, Multiply
from tensorflow.keras.backend import transpose
import numpy as np

class JokeRecommender(keras.Model):
    def __init__(self, n_users, n_jokes, n_rows, units1 = 64, units2 = 32, units3 = 8, dropout_rate=0.2):
        super(JokeRecommender, self).__init__()

        self.split_user = Lambda(lambda x: x[:, :n_users])
        self.split_joke = Lambda(lambda x: x[:, n_users:])
        # Embedding layers for jokes
        self.joke_embedding = Embedding(
            input_dim=n_jokes, 
            output_dim=3,
            input_length=1
        )
        self.joke_flatten = Flatten()
        self.joke_dropout = Dropout(dropout_rate)

        # Embedding layers for users
        self.user_embedding = Embedding(
            input_dim=n_users,
            output_dim=3,
            input_length=1
        )
        self.user_flatten = Flatten()
        self.user_dropout = Dropout(dropout_rate)

        self.merge_dot = Dot(axes=1, normalize=True)
        # self.merge_dot = Multiply()
        
        self.layer_1 = Dense(units=units1, activation='relu')
        self.layer_2 = Dense(units=units2, activation='relu')
        # # self.layer_3 = Dense(units=units3, activation='relu')
        
        self.my_output = Dense(units=1, activation='sigmoid')
        

    def call(self, x):
        user = self.split_user(x)
        user = self.user_dropout(self.user_flatten(self.user_embedding(user)))
        
        joke = self.split_joke(x)
        joke = self.joke_dropout(self.joke_flatten(self.joke_embedding(joke)))

        # print(user, joke)
        x = self.merge_dot([user, joke])

        # x = self.layer_1(x)
        # x = self.layer_2(x)

        x = self.my_output(x)
        return x