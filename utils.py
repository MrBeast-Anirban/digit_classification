from sklearn.model_selection import train_test_split
from sklearn import datasets, svm

# here we will put utils
def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state = 1):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

#train the model of choice with model parameter
def train_model(x, y, model_params, model_type):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    #train the model
    #pdb.set_trace()
    model.fit(x, y)
    return model

#creating a train test and validation split function
def train_test_dev_split(x, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 1) 
    #spliting the training set into training set and validation set
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = dev_size, random_state = 1)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

#prediction over the training data
def predict_and_eval(model, X_test, y_test):
    prediction = model.predict(X_test)
    return prediction