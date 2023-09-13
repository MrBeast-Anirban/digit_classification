# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import preprocess_data, predict_and_eval, split_data, train_model, read_digits, train_test_dev_split

#import pdb


gamma_ranges = [0.01, 0.1, 1, 10, 100]
c_ranges = [0.1, 1, 2, 5, 10]

# 1. Get the dataset
X, y = read_digits()

# 3. Data Splitting to create train and test set
#X_train, y_train, X_test, y_test = split_data(X, y, test_size = 0.3)
X_train, X_dev, X_test, y_train, y_dev, y_test = train_test_dev_split(X, y, 0.3, 0.2)
#4. Data Preprocessing
X_train = preprocess_data(X_train)
X_dev = preprocess_data(X_dev)
X_test = preprocess_data(X_test)


# Hyper Parameter Tuning
# take all combinations of gamma and c and check the model performance
best_accuracy = -1
best_model = None
for current_gamma in gamma_ranges:
    for current_c in c_ranges:
        #training over current parameters
        current_model = train_model(X_train, y_train, {'gamma' : current_gamma, 'C' : current_c}, model_type = 'svm')
        #evaluation of model over current parameters
        current_accuracy = predict_and_eval(current_model, X_dev, y_dev)
        #get the hyperparameter with best model accuracy
        if current_accuracy > best_accuracy:
            print("The new best accuracy is ", current_accuracy)
            best_accuracy = current_accuracy
            best_model = current_model
            optimal_gamma = current_gamma
            optimal_c = current_c

print("Optimal Gamma -> ", optimal_gamma, '\n',"Optimal C Value -> ", optimal_c)

#5. Model Training
#model = train_model(X_train, y_train, {'gamma' : optimal_gamma, 'C' : optimal_c}, model_type = 'svm')

# 6. Getting model accuracy on test set
test_accuracy = predict_and_eval(best_model, X_test, y_test)
print("The test accuracy of model is ", test_accuracy)

