from sklearn.metrics import mean_absolute_error

def evaluate(model, test, y_true):
    y_pred = model.predict([test['USER_ID'], test['JOKE_ID']])
    print("Mean absolute error: " + mean_absolute_error(y_true, y_pred))