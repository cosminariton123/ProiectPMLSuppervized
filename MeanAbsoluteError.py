import numpy as np

def mean_absolute_error(predicted, true):
    return np.sum(np.absolute(predicted - true))/len(predicted)