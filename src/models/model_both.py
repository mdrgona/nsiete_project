import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dropout, dot, Lambda, Reshape, Dot
from tensorflow.keras.backend import transpose
import numpy as np

class JokeRecommender(keras.Model):
    def __init__(self, emb_output_dim, number_users, number_jokes):
        super(JokeRecommender, self).__init__()
        
        # MLP part
                
        self.user_emb_mlp = keras.layers.Embedding(number_users + 1, emb_output_dim)
        self.user_vector_mlp = keras.layers.Flatten()
        self.user_drop_mlp = keras.layers.Dropout(0.2)
        
        self.joke_emb_mlp = keras.layers.Embedding(number_jokes + 1, emb_output_dim)
        self.joke_vector_mlp = keras.layers.Flatten()
        self.joke_drop_mlp = keras.layers.Dropout(0.2)
        
        self.concat_drop = keras.layers.Dropout(0.2)
        
        self.dense = keras.layers.Dense(units=32, activation='relu')
        self.dense_2 = keras.layers.Dense(units=16, activation='relu')
        
        self.dense_3 = keras.layers.Dense(units=12, activation='relu')
        self.final_mlp = keras.layers.Dense(units=1, activation='relu')  
        
        # GMF part
        
        self.user_emb_gmf = keras.layers.Embedding(number_users + 1, emb_output_dim)
        self.user_vector_gmf = keras.layers.Flatten()
        self.user_drop_gmf = keras.layers.Dropout(0.2)
        
        self.joke_emb_gmf = keras.layers.Embedding(number_jokes + 1, emb_output_dim)
        self.joke_vector_gmf = keras.layers.Flatten()
        self.joke_drop_gmf = keras.layers.Dropout(0.2)
        
        self.final_gmf = Dot(axes=1, normalize=True)

        # Merge
        
        self.dense_output = keras.layers.Dense(1)
        

    def call(self, x):
        user_mlp = self.user_emb_mlp(x[0])
        user_mlp = self.user_vector_mlp(user_mlp)
        user_mlp = self.user_drop_mlp(user_mlp)
        
        joke_mlp = self.joke_emb_mlp(x[1])
        joke_mlp = self.joke_vector_mlp(joke_mlp)
        joke_mlp = self.joke_drop_mlp(joke_mlp)
        
        mlp = concatenate([user_mlp, joke_mlp])
        mlp = self.dense(mlp)
        mlp = self.dense_2(mlp)
        mlp = self.dense_3(mlp)
        mlp = self.final_mlp(mlp)
        
        
        user_gmf = self.user_emb_gmf(x[0])
        user_gmf = self.user_vector_gmf(user_gmf)
        user_gmf = self.user_drop_gmf(user_gmf)
        
        joke_gmf = self.joke_emb_gmf(x[1])
        joke_gmf = self.joke_vector_gmf(joke_gmf)
        joke_gmf = self.joke_drop_gmf(joke_gmf)
        
        gmf = self.final_gmf([user_gmf, joke_gmf])
        
        x = concatenate([mlp, gmf])
        x = self.dense_output(x)
        return x