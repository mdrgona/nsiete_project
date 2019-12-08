import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dropout, BatchNormalization
import numpy as np

class JokeRecommender(keras.Model):
    
    def __init__(self, emb_output_dim, number_users, number_jokes):
        super(JokeRecommender, self).__init__()
        
        self.user_emb = keras.layers.Embedding(number_users + 1, emb_output_dim)
        self.user_vector = keras.layers.Flatten()
        self.user_drop = keras.layers.Dropout(0.2)
        
        self.joke_emb = keras.layers.Embedding(number_jokes + 1, emb_output_dim)
        self.joke_vector = keras.layers.Flatten()
        self.joke_drop = keras.layers.Dropout(0.2)
        
        self.concat_drop = keras.layers.Dropout(0.2)
        
        self.dense = keras.layers.Dense(units=150, activation='relu')
        self.batch_1 = keras.layers.BatchNormalization()
        self.drop_1 = keras.layers.Dropout(0.2)
        
        self.dense_2 = keras.layers.Dense(units=100, activation='relu')
        self.batch_2 = keras.layers.BatchNormalization()
        self.drop_2 = keras.layers.Dropout(0.2)
        
        self.dense_3 = keras.layers.Dense(units=50, activation='relu')
        self.dense_4 = keras.layers.Dense(units=20, activation='relu')      
        self.dense_5 = keras.layers.Dense(units=1, activation='relu')  
        

    def call(self, x):
        user = self.user_emb(x[0])
        user = self.user_vector(user)
        user = self.user_drop(user)
        
        joke = self.joke_emb(x[1])
        joke = self.joke_vector(joke)
        joke = self.joke_drop(joke)
        
        x = concatenate([user, joke])
        x = self.concat_drop(x)
        
        x = self.dense(x)
        x = self.batch_1(x)
        x = self.drop_1(x)
        
        x = self.dense_2(x)
        x = self.batch_2(x)
        x = self.drop_2(x)
        
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        return x