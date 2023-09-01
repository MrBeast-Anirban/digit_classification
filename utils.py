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
    return X_train, y_train, X_test, y_test

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

def train_test_dev_split(x, y, test_size, random_state = 1):
    dev_ratio = 0.10

    X_train, y_train, X_test, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state) 
    #spliting the training set into training set and validation set
    X_train, y_train, X_dev, y_dev = train_test_split(X_train, y_train, test_size = dev_ratio, random_state = random_state)
    return X_train, y_train, X_test, y_test, X_dev, y_dev
