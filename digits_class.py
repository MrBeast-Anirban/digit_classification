# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import preprocess_data, tune_hparams, predict_and_eval, split_data, train_model, read_digits, train_test_dev_split
from itertools import product
#import pdb


# 1. Get the dataset
X, y = read_digits()


# Hyper Parameter Tuning
# take all combinations of gamma and c and check the model performance
gamma_ranges = [0.01, 0.1, 1, 10, 100]
c_ranges = [0.1, 1, 2, 5, 10]
test_size = [0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]


gamma_c_combination = list(product(gamma_ranges, c_ranges))  #list of tuples
dev_test_combination = list(product( dev_size, test_size))

for pair in dev_test_combination:
    # 3. Data Splitting to create train , dev and test set
    X_train, X_dev, X_test, y_train, y_dev, y_test = train_test_dev_split(X, y, pair[1], pair[0])
    #4. Data Preprocessing
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)
    X_test = preprocess_data(X_test)
    best_model, optimal_gamma, optimal_c = tune_hparams(X_train, y_train, X_dev, y_dev, gamma_c_combination)
    train_accuracy = predict_and_eval(best_model, X_train, y_train)
    dev_accuracy = predict_and_eval(best_model, X_dev, y_dev)
    test_accuracy = predict_and_eval(best_model, X_test, y_test)
    print("Test size = ", pair[1], "Dev Size = ", pair[0], "Train Size = ", 1-pair[0]-pair[1],"Train Accuracy = ", train_accuracy*100, "Dev Accuracy = ", dev_accuracy*100, "Test Accuracy = ", test_accuracy*100)

