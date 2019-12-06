import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dropout
import numpy as np

class JokeRecommender(keras.Model):
    
    def __init__(self, emb_output_dim, number_users, number_jokes):
        super(JokeRecommender, self).__init__()
        
        self.user_input = keras.layers.Input(shape=[1])
        self.user_emb = keras.layers.Embedding(number_users + 1, emb_output_dim)
        self.user_vector = keras.layers.Flatten()
        
        self.joke_input = keras.layers.Input(shape=[1])
        self.joke_emb = keras.layers.Embedding(number_jokes + 1, emb_output_dim)
        self.joke_vector = keras.layers.Flatten()
        
        self.dense = keras.layers.Dense(units=32, activation='relu')
        self.dense_2 = keras.layers.Dense(units=16, activation='relu')
        self.dense_3 = keras.layers.Dense(units=12, activation='relu')
        self.dense_4 = keras.layers.Dense(units=1, activation='relu')
        

    def call(self, x):
        
        print(x)
        
        user = self.user_input(x[0])
        user = self.user_emb(user)
        user = self.user_vector(user)
        
        joke = self.joke_input(x[1])
        joke = self.joke_emb(joke)
        joke = self.joke_vector(joke)
        
        x = concatenate([user, joke])
        
        x = self.dense(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x