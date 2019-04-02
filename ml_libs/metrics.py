import numpy as np

def mse(y, y_pred):
    return np.sum(( y - y_pred ) ** 2)

def rmse(y, y_pred):
    return mse(y, y_pred) ** 0.5

