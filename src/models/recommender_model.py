import pandas as pd
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datetime
import os
import sys
sys.path.append('..')

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time;

from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate
from src.data.load_data import *
import itertools

class RecommenderModel:

    def __init__(self, model):
        self.model = model
        logger.info('RecommenderModel initialized')

    def fit(self, ratings):
        """
        Train a model.
 |      
 |      Args:
 |          ratings: the ratings data frame.
 |      
 |      Returns:
 |          The algorithm (for chaining).
        """
        self.ratings = ratings

        logger.info('Data handling')
        ratings_pivot = ratings.pivot(index='USER_ID', columns='JOKE_ID', values='Rating').fillna(0)
        users = ratings_pivot.index
        jokes = ratings_pivot.columns
        data_input = np.array(list(itertools.product(users, jokes)))
        user_ids = data_input[:, 0]
        joke_ids = data_input[:, 1]
        
        logger.info('Predicting')
        predicted_ratings = self.model.predict([user_ids, joke_ids])

        logger.info('Saving')
        predicted_table = pd.DataFrame(data={'USER_ID': user_ids, 'JOKE_ID': joke_ids, 'score': predicted_ratings.T[0]})
        self.predictions = predicted_table.pivot(index='JOKE_ID', columns='USER_ID', values='score').fillna(0)
        
        logger.info('Done')
              
        return self

    def recommend(self, user, n=None):
        """
        Compute recommendations for a user.
 |      
 |      Args:
 |          user: the user ID
 |          n(int): the number of recommendations to produce (``None`` for unlimited)
 |      
 |      Returns:
 |          pandas.DataFrame:
 |              a frame with an ``item`` column; if the recommender also produces scores,
 |              they will be in a ``score`` column.
        """      
        if user not in self.predictions:
            return pd.DataFrame(columns=['JOKE_ID', 'score'])
        
        seenJokes = self.ratings[self.ratings['USER_ID'] == user]['JOKE_ID'].unique()
        userPredictions = self.predictions[user]#.drop(seenMovies)
        userPredictions = userPredictions.sort_values(ascending=False)[:n]
        userPredictions = userPredictions.reset_index()
        # logger.info(userPredictions)
        return userPredictions.rename(columns={userPredictions.columns[1]: 'score'})
