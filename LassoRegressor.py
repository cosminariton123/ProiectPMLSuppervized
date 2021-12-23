from sklearn.linear_model import Lasso
import numpy as np

from DataLoader import *
from BOW import *
from MeanAbsoluteError import *

def get_error(alpha):
    train_data, train_labels, validation_data, validation_labels = load_training_data("train_full.txt")

    bag_of_words = BOW()

    bag_of_words.extract_vocabulary(train_data)
    train_features = bag_of_words.extract_features(train_data)
    validation_features = bag_of_words.extract_features(validation_data)


    model = Lasso(alpha = alpha)

    model.fit(train_features, train_labels)

    predicted_labels = model.predict(validation_features)

    predicted_labels = model.predict(validation_features)
    predicted_labels = np.around(predicted_labels, 2)
    predicted_labels = np.array([0 if x < 0 else x for x in predicted_labels])

    return mean_absolute_error(predicted_labels, validation_labels)