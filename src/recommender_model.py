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

class RecommenderModel:

    def __init__(self, model):
        self.model = model
        logger.info('RecommenderModel initialized')

    def fit(self, ratings, predicted_ratings):
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

#         logger.info('Data handling')        
#         user_ids, joke_ids, ratings = get_data(df=ratings, batch_size=20000)
#         logger.info('Predicting')
#         predicted_ratings = self.model.predict([np.array(user_ids), np.array(joke_ids)])

        logger.info('Saving')
        predicted_table = pd.DataFrame(data={'USER_ID': ratings['USER_ID'], 'JOKE_ID': ratings['JOKE_ID'], 'score': predicted_ratings.T[0]})
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
