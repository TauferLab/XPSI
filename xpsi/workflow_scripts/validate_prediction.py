import configparser as cp
import feature_extraction
from knn_angles import KNeighborsAngleRegressor
from math import acos
import numpy as np
import numpy.linalg as la
import os
import optparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import sys
import time 
from utils import angular_difference


parser = optparse.OptionParser()
parser.set_defaults(k=0, trees=0, knn_average='vector_mean')
parser.add_option('--k', action='store', dest='k', type='int')
parser.add_option('--trees', action='store', dest='trees', type='int')
parser.add_option('--knn-average', action='store', dest='knn_average', type='string')
(options, args) = parser.parse_args()


# Read configuration file
conf = cp.ConfigParser()
conf.read(args[0])

val_euler_angles = np.loadtxt(conf['validationData']['eulerAngleFilePath'], usecols=(0, 1))
train_euler_angles = np.loadtxt(conf['data']['eulerAngleFilePath'], usecols=(0, 1))

X_val, inds_val = feature_extraction.load_dataset(conf, mode='validation')
X_train, inds_train = feature_extraction.load_dataset(conf)
autoencoder, encoder, decoder = feature_extraction.train_autoencoder(conf, X_train)


def do_trials(model, num_trials=10):

    start_time = time.perf_counter()
    # Train the model on training data
    model.fit(X_train, Y_train)
    train_time = time.perf_counter() - start_time

    # Predict
    start_time = time.perf_counter()
    val_predictions = model.predict(X_val)
    test_time = time.perf_counter() - start_time

    val_error = [angular_difference(predicted, actual)
                 for predicted, actual in zip(val_predictions, val_euler_angles)]

    print("Total Train Time: ", train_time)
    print("Total Test Time: ", test_time)

    return val_errors


def save_trial(directory, val_error):
    error_path = conf['output']['errorPath']
    os.makedirs(error_path+directory, exist_ok=True)
    
    training_errors = np.array(train_errors)
    testing_errors = np.array(test_errors)
    
    with open(error_path + directory + "val_errors.txt", "a") as f:
        for error in val_error:
            f.write("%s " % error)
        f.write("\n")



if options.k != 0:
    print("NN trials", flush=True)

    # Perform Nearest Neighbors regression
    model = KNeighborsAngleRegressor(n_neighbors=options.k, average=options.knn_average)
    val_error = do_trials(model, options.knn_trials)

    save_trial(str(options.k) + " Nearest Neighbors/", val_error)


if options.trees != 0:    
    print("RF trials", flush=True)
 
    # Perform Random Forest regression
    model = RandomForestRegressor(n_estimators = options.trees, random_state = 42)
    val_errors,test_errors = do_trials(model, options.rf_trials)

    save_trial("Random "+str(options.trees)+" Tree Forest/", val_error)

