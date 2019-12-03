import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dropout
import numpy as np

class JokeRecommender(keras.Model):
    
    def __init__(self, input_vector, units1 = 32, units2 = 16, units3 = 8):
        super(JokeRecommender, self).__init__()
        
        # # Input layers
        # user_input = Input(shape=(1,), dtype='int32')
        # joke_input = Input(shape=(1,), dtype='int32')
                
        # Embedding layers
        self.embedding = Embedding(
            input_dim=len(input_vector), 
            output_dim=int(units1 / 2),
            input_length=len(input_vector[0])
        )
        self.flatten = Flatten()

        # self.user_embedding = Embedding(
        #     input_dim=len(user_ids), 
        #     output_dim=int(units1 / 2),
        #     input_length=1
        # )
        
        # self.joke_embedding = Embedding(
        #     input_dim=len(joke_ids),
        #     output_dim=int(units1 / 2),
        #     input_length=1
        # )
        
        # # Concatenate user and joke embeddings
        # self.user_flatten = Flatten()
        # self.joke_flatten = Flatten()
        
        self.layer_1 = Dense(units=units1, activation='relu')
        self.layer_2 = Dense(units=units2, activation='relu')
        self.layer_3 = Dense(units=units3, activation='relu')

        self.dropout = Dropout(0.5)
        
        self.my_output = Dense(units=10, activation='tanh')
        

    def call(self, x):
        # user = self.user_embedding(x[0])
        # joke = self.joke_embedding(x[1])
        # x = concatenate([user, joke])
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.dropout(x)
        x = self.my_output(x)
        return x