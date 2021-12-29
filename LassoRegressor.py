from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt

from DataLoader import *
from BOW import *
from MeanAbsoluteError import *
from CrossValidation import *

def get_error_lasso(model, train_data, train_labels, validation_data, validation_labels):

    bag_of_words = BOW()

    bag_of_words.extract_vocabulary(train_data)
    train_features = bag_of_words.extract_features(train_data)
    validation_features = bag_of_words.extract_features(validation_data)

    model.fit(train_features, train_labels)

    predicted_labels = model.predict(validation_features)

    predicted_labels = model.predict(validation_features)
    predicted_labels = np.around(predicted_labels, 2)
    predicted_labels = np.array([0 if x < 0 else x for x in predicted_labels])

    return mean_absolute_error(predicted_labels, validation_labels)



def alpha_tuning_lasso():
    
    min = np.inf
    errors = list()

    for i in range(1, 100):
        
        model = Lasso(alpha = i)
        current_error = cross_validation(get_error_lasso, model, 5)
        current_error = np.mean(current_error)
        if current_error < min:
            min = current_error

        errors.append(current_error)

    plt.plot(errors)
    plt.title("Erorile pentru modelul Lasso pentru diferiti alpha\n Eroarea minima este data de alpha=" + str(min))
    plt.show()