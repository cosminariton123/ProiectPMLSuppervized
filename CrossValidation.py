import numpy as np
from sklearn.utils import shuffle

from DataLoader import *


def cross_validation(error_function_to_call , model , k):
    data, labels = load_data("train_full.txt")
    data, labels = shuffle(data, labels)
    data = np.array_split(data, k)
    labels = np.array_split(labels, k)


    data = [list(x) for x in data]
    labels = [list(x) for x in labels]

    errors = list()

    for d_chunk, l_chunk in zip(data, labels):
        training_data = list()
        training_labels = list()
        validation_data = d_chunk
        validation_labels = l_chunk
        for d_chunk2, l_chunk2 in zip(data, labels):
            if d_chunk != d_chunk2:
                training_data.append(d_chunk2)
                training_labels.append(l_chunk2)
        
        training_data = [elem for sublist in training_data for elem in sublist]
        training_labels = [elem for sublist in training_labels for elem in sublist]

        errors.append(error_function_to_call(model, training_data, training_labels, validation_data, validation_labels))
    
    return errors