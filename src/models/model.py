import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dropout, dot, Lambda, Reshape, Dot
from tensorflow.keras.backend import transpose
import numpy as np

class JokeRecommender(keras.Model):
    
    def __init__(self, n_users, n_jokes, units1 = 100, units2 = 100, units3 = 8, dropout_rate=0.2):
        super(JokeRecommender, self).__init__()
                
        self.n_users = n_users        
        self.n_jokes = n_jokes

        self.split_user = Lambda(lambda x: x[:, :n_users])
        self.split_joke = Lambda(lambda x: x[:, n_users:])
        # Embedding layers
        self.joke_embedding = Embedding(
            n_jokes, 
            n_users,
            # input_length=len(movies[0])
        )
        self.joke_flatten = Flatten()
        self.joke_dropout = Dropout(dropout_rate)

        # Embedding layers
        self.user_embedding = Embedding(
            n_users, 
            n_jokes,
            # input_length=len(input_vector[0])
        )
        self.user_flatten = Flatten()
        self.user_dropout = Dropout(dropout_rate)


        # self.user_transpose = Reshape((-1, -1))
        # self.joke_transpose = Reshape((-1, -1))
        self.merge_dot = Dot(axes=-1)
        
        self.layer_1 = Dense(units=units1, activation='relu')
        self.layer_2 = Dense(units=units2, activation='relu')
        # # self.layer_3 = Dense(units=units3, activation='relu')
        # # self.dropout = Dropout(0.5)
        
        self.my_output = Dense(units=1, activation='tanh')
        

    def call(self, x):
        # print(x.value_index())
        user = self.split_user(x)
        user = self.user_dropout(self.user_flatten(self.user_embedding(user)))
        
        joke = self.split_joke(x)
        joke = self.joke_dropout(self.joke_flatten(self.joke_embedding(joke)))

        # print(user, transpose(user))
        # x = dot([user, joke], axes=-1)
        x = self.merge_dot([user, joke])

        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.my_output(x)
        return x