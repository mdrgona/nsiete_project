from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Svd:

    def __init__(self, singular_values = 1):
        self.singular_values = singular_values
        logger.info('SVD model initialized')

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
        logger.info('Creating pivot table')
        matrix = ratings.pivot(index='USER_ID', columns='JOKE_ID', values='Rating').fillna(0)
        logger.info('Sparse rating with shape: {0}'.format(matrix.shape))

        logger.info('Decompomposing matrix')
        u, sigma, vt = svds(matrix, k = self.singular_values)
        
        sigma = np.diag(sigma)
        self.users_ids = ratings['USER_ID'].unique()

        logger.info('Calculating prediction')
        predicted_ratings = np.dot(np.dot(u, sigma), vt)
        self.predictions = pd.DataFrame(predicted_ratings, columns = matrix.columns, index = matrix.index).transpose()
        
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
        
        seenMovies = self.ratings[self.ratings['USER_ID'] == user]['JOKE_ID'].unique()
        userPredictions = self.predictions[user].drop(seenMovies)
        userPredictions = userPredictions.sort_values(ascending=False)[:n]
        userPredictions = userPredictions.reset_index()
        #logger.info(user_predictions.columns)
        return userPredictions.rename(columns={userPredictions.columns[1]: 'score'})
