import matplotlib.pyplot as plt

from SkleanMLPRegressor import *
from CrossValidation import *
from LassoRegressor import *
from DataLoader import *

#hidden_layer_sizes = (2, 3, 2)
#model = MLPRegressor(random_state=1, hidden_layer_sizes= hidden_layer_sizes, solver = "adam", early_stopping=True, max_iter=500)
#print(cross_validation(get_error_for_configuration, model, 5))

def main():
    hidden_layer_sizes = (700, 800, 1000, 800, 700)
    #model = MLPRegressor(random_state=1, hidden_layer_sizes= hidden_layer_sizes, solver = "adam", early_stopping=True, max_iter=500)
    #print(cross_validation(get_error_for_configuration, model, 5))
    kagel_train(hidden_layer_sizes)

if __name__ == '__main__':
    main()