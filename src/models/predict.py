import sys
sys.path.insert(0, "..")

from sklearn.metrics import mean_absolute_error
import numpy as np
from src.recommender_model import *
from src.svd import *
from src.data.load_data import *

def predict(model, test):
    return model.predict([np.array(test['USER_ID']), np.array(test['JOKE_ID'])])


def evaluate(y_pred, model, test):
    # TODO: refactor. 
    
    y_true = test['Rating']
    
    MAE = np.round(mean_absolute_error(y_true, y_pred), 4)
    precision = np.round(get_precision(model, test, y_pred), 4)
    
    print("----------------")
    print("Evaluating model MLP")
    print("Mean absolute error: " + str(MAE))
    print("Precision@10:      : " + str(precision))
    print("---------------------------")
    print("Results saved into file: evaluation_results.txt")
    
    save_results(MAE, precision)
    
    
def save_results(MAE, precision):
    f = open("../../logs/evaluation_results.txt", "a")
    f.write("Evaluating model MLP\n")
    f.write("Mean absolute error: " + str(MAE))
    f.write("\nPrecision:         : " + str(precision))
    f.write("\n\n\n\n")
    f.close()
    
    
    
def get_precision(model, test, y_pred):
    
    recommender_model = RecommenderModel(model)
    recommender_model.fit(test, y_pred)

    users = test['USER_ID'].unique()
    precision_model = 0
    
    for user in users:
        user_rec_model = list(recommender_model.recommend(user, 10)['JOKE_ID'])
        user_test = list(test[test['USER_ID'] == user]['JOKE_ID'])
    
        user_precision_model = len(set(user_rec_model).intersection(user_test)) / 10
    
        precision_model = precision_model + user_precision_model

    return precision_model / len(users)
    