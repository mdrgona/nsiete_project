import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dropout, Lambda, Reshape, Dot, Multiply
from tensorflow.keras.backend import transpose
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

        self.merge_dot = Dot(axes=1, normalize=True)
        

    def call(self, x):
        user = self.user_emb(x[0])
        user = self.user_vector(user)
        user = self.user_drop(user)
        
        joke = self.joke_emb(x[1])
        joke = self.joke_vector(joke)
        joke = self.joke_drop(joke)
        
        x = self.merge_dot([user, joke])
   
        return x