# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import preprocess_data, predict_and_eval, split_data, train_model, read_digits, train_test_dev_split
from itertools import product
#import pdb
# 1. Get the dataset
X, y = read_digits()

# 3. Data Splitting to create train and test set
X_train, X_dev, X_test, y_train, y_dev, y_test = train_test_dev_split(X, y, 0.3, 0.2)
#4. Data Preprocessing
X_train = preprocess_data(X_train)
X_dev = preprocess_data(X_dev)
X_test = preprocess_data(X_test)


# Hyper Parameter Tuning
# take all combinations of gamma and c and check the model performance
gamma_ranges = [0.01, 0.1, 1, 10, 100]
c_ranges = [0.1, 1, 2, 5, 10]
best_accuracy = -1
best_model = None

gamma_c_combination = list(product(gamma_ranges, c_ranges))  #list of tuples

for pair in gamma_c_combination: #pair is a tuple
    #trining over current parameters
    current_model = train_model(X_train, y_train, {'gamma' : pair[0], 'C' : pair[1]}, model_type = 'svm')
    #evaluating model over current parameters
    current_accuracy = predict_and_eval(current_model, X_dev, y_dev)
    #get hyperparameter with best model accuracy
    if current_accuracy > best_accuracy:
       print("The new best accuracy is : ", current_accuracy)
       best_accuracy = current_accuracy
       best_model = current_model
       optimal_gamma = pair[0]
       optimal_c = pair[1]
    

print("Optimal Gamma -> ", optimal_gamma, '\n',"Optimal C Value -> ", optimal_c)

#5. Model Training
#model = train_model(X_train, y_train, {'gamma' : optimal_gamma, 'C' : optimal_c}, model_type = 'svm')

# 6. Getting model accuracy on test set
test_accuracy = predict_and_eval(best_model, X_test, y_test)
print("The test accuracy of model is ", test_accuracy)

