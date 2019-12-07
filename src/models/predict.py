import sys
sys.path.insert(0, "..")

from sklearn.metrics import mean_absolute_error
import numpy as np
from src.recommender_model import *
from src.svd import *

def predict(model, test):
    return model.predict([np.array(test['USER_ID']), np.array(test['JOKE_ID'])])

def evaluate(y_true, y_pred):
    print("---------------------------")
    print("Mean absolute error: " + str(np.round(mean_absolute_error(y_true, y_pred), 4)))
    print("Precision:       TODO")
    print("---------------------------")
    
    
def get_precision(model, train, test):
    recommender_model = RecommenderModel(model)
    recommender_model.fit(train)
    svd = Svd()
    svd.fit(train)

    users = test['USER_ID'].unique()
    precision_model = 0
    precision_svd = 0
    
    for user in users:
        user_rec_model = list(recommender_model.recommend(user, 10)['JOKE_ID'])
        user_rec_svd = list(svd.recommend(user, 10)['JOKE_ID'])
        user_test = list(test[test['USER_ID'] == user]['JOKE_ID'])
    
        user_precision_model = len(set(user_rec_model).intersection(user_test)) / 10
        user_precision_svd = len(set(user_rec_svd).intersection(user_test)) / 10
    
        precision_model = precision_model + user_precision_model
        precision_svd = precision_svd + user_precision_svd

    print('precision@10 model', precision_model / len(users))
    print('precision@10 svd', precision_svd / len(users))
    