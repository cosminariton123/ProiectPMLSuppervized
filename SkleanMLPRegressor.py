import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from BOW import *
from MeanAbsoluteError import *
from DataLoader import *
from CrossValidation import *


def get_error_for_configuration(model, train_data, train_labels, validation_data, validation_labels):
    bag_of_words = BOW()

    bag_of_words.extract_vocabulary(train_data)
    train_features = bag_of_words.extract_features(train_data)
    validation_features = bag_of_words.extract_features(validation_data)

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    validation_features = scaler.transform(validation_features)

    model.fit(train_features, train_labels)
    predicted_labels = model.predict(validation_features)
    predicted_labels = np.around(predicted_labels, 2)
    predicted_labels = np.array([0 if x < 0 else x for x in predicted_labels])

    return mean_absolute_error(predicted_labels, validation_labels)


def kagel_train(hidden_layer_sizes):
    train_data, train_labels = load_data("train_full.txt")
    idx, submission_data = load_submission_data("test.txt")

    bag_of_words = BOW()

    bag_of_words.extract_vocabulary(train_data)
    train_features = bag_of_words.extract_features(train_data)
    submission_features = bag_of_words.extract_features(submission_data)
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    submission_features = scaler.transform(submission_features)

    model = MLPRegressor(random_state=1, hidden_layer_sizes= hidden_layer_sizes, solver = "adam", early_stopping=True, max_iter=500)
    model.fit(train_features, train_labels)

    submission_predicted_labels = model.predict(submission_features)
    submission_predicted_labels = np.around(submission_predicted_labels, 2)
    submission_predicted_labels = np.array([0 if x < 0 else x for x in submission_predicted_labels])
    make_submission(idx, submission_predicted_labels)



def tuning():
    min = 99999999
    best_config = None

    try:
        for _ in tqdm(range(10)):
            hidden_layer_sizes = [np.random.randint(1, 1000) for _ in range(np.random.randint(1, 5))]
            hidden_layer_sizes = tuple(hidden_layer_sizes)

            model = MLPRegressor(random_state=1, hidden_layer_sizes=tuple(hidden_layer_sizes), solver = "adam", early_stopping=True, max_iter=500)

            errors = cross_validation(get_error_for_configuration, model, 5)
            mean_error = mean(errors)

            if mean_error < min:
                min = mean_error
                best_config = hidden_layer_sizes
    except:
        print("Existed early")
    
    finally:
        print("Best error is: " + str(min))
        print("Best configuration is " + str(best_config))