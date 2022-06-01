import configparser as cp
from knn_angles import KNeighborsAngleRegressor
from math import acos
import numpy as np
import numpy.linalg as la
import os
import optparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import sys
import time 
from utils import angular_difference
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder 
from confusionmatrix_plot import cm_analysis
import pickle



def do_trials(rf, rf_1, rf_2, filename, labels_names_conf, names_conf, labels_names_type, names_type, features, labels, indices, num_trials=10) :
    train_errors = []
    test_errors = []
    psi_train_errors = []
    psi_test_errors = []
    train_accuracies_conf = []
    test_accuracies_conf  = []
    train_accuracies_type = []
    test_accuracies_type = []

    train_time = []
    test_time = []
    train_indices = []
    test_indices = []
    predictions_real_test = []
    confusionmatrix_test = []

    for j in range(num_trials):
        #print("j="+str(j), flush=True)
        
        # Split data
        X_train, X_test, Y_train, Y_test, train_ind, test_ind = train_test_split(
            features, labels, indices, test_size=0.1, random_state=j
        )
        
        # Move all angles from [0,360] to [0,180], so we remove the symmetry present in the diffraction patterns
        Y_train[:,2]=np.where(Y_train[:,2]>180, Y_train[:,2]-180, Y_train[:,2])
        Y_test[:,2]=np.where(Y_test[:,2]>180, Y_test[:,2]-180, Y_test[:,2])

        # Train all knns	
        start_time = time.perf_counter()
        rf.fit(X_train, Y_train[:,:-1]) # Orientation
        rf_1.fit(X_train, Y_train[:,-2]) # Conformation 
        rf_2.fit(X_train, Y_train[:,-1]) # Protein Type
        train_time.append(time.perf_counter() - start_time)
        
        
        # Predict
        train_predictions = rf.predict(X_train)
        train_predictions_conf = rf_1.predict(X_train)
        train_predictions_type = rf_2.predict(X_train)

    

        # Orientation error for training data
        train_error = []
        psi_train_error = []
        for i in range(len(train_predictions)):
            predicted_angles = train_predictions[i]
            actual_angles = Y_train[i]

            error, psi_error = angular_difference(actual_angles,predicted_angles)

            train_error.append(error)
            psi_train_error.append(psi_error)

        # Conformation accuracy for training data
        train_accuracies_conf.append(accuracy_score(Y_train[:,-2], train_predictions_conf))
        #train_accuracies.append(f1_score(Y_train[:,2], train_predictions_conf, average='weighted'))
        cm=confusion_matrix(Y_train[:,-2], train_predictions_conf) 
        #print('Training cm for conformation:\n', cm)

        # Protein type accuracy for training data
        train_accuracies_type.append(accuracy_score(Y_train[:,-1], train_predictions_type))
        cm=confusion_matrix(Y_train[:,-1], train_predictions_type) 
        #print('Training cm for protein type:\n', cm)

        # Testing knns
        start_time = time.perf_counter()
        test_predictions = rf.predict(X_test) # Orientation
        test_predictions_conf = rf_1.predict(X_test) # Conformation 
        test_predictions_type = rf_2.predict(X_test) # Protein Type
        test_time.append(time.perf_counter() - start_time) 

        # Orientation error for testing data
        test_error = []
        psi_test_error = []
        for i in range(len(test_predictions)):
            predicted_angles = test_predictions[i]
            actual_angles = Y_test[i]
            
            # Save trials of predictions and actual testing values
            predicted=predicted_angles.tolist()
            real=actual_angles[:-2].tolist()
            actual_conformation=Y_test[i,-2]
            predicted_conformation=np.rint(test_predictions_conf[i])
            actual_type=Y_test[i,-1]
            predicted_type=np.rint(test_predictions_type[i])
            orientation = real+predicted
            conformation=list([actual_conformation, predicted_conformation])
            prot_type=list([actual_type, predicted_type])
            orie_confor = orientation + conformation + prot_type
            predictions_real_test.append(orie_confor)           

            error, psi_error = angular_difference(actual_angles, predicted_angles)

            test_error.append(error)
            psi_test_error.append(psi_error)

	# Conformation accuracy for testing data
        test_accuracies_conf.append(accuracy_score(Y_test[:,-2], np.rint(test_predictions_conf)))
        if not os.path.exists(filename):
            os.makedirs(filename)
        cm_analysis(Y_test[:,-2], test_predictions_conf, filename+'conformations_cm_'+str(j)+'.jpg', labels_names_conf, names_conf, ymap=None, figsize=(10,10))
        cm_test=confusion_matrix(Y_test[:,-2], test_predictions_conf)  
        #print('Testing cm for conformation:\n', cm_test)
        cm_list=cm_test.tolist()
        confusionmatrix_test.append(cm_list)

	# Conformation accuracy for testing data
        test_accuracies_type.append(accuracy_score(Y_test[:,-1], np.rint(test_predictions_type)))
        if not os.path.exists(filename):
            os.makedirs(filename)
        cm_analysis(Y_test[:,-1], test_predictions_type, filename+'types_cm_'+str(j)+'.jpg', labels_names_type, names_type, ymap=None, figsize=(10,10))
        cm_test=confusion_matrix(Y_test[:,-1], test_predictions_type)  
        #print('Testing cm for type:\n', cm_test)
        
        # save models
        model_save_directory = os.path.abspath(os.path.join(filename ,"../"))+"/"
        pickle.dump(rf, open(model_save_directory+'kNN_orientation_model.pkl', 'wb'))
        pickle.dump(rf_1, open(model_save_directory+'kNN_conformation_model.pkl', 'wb'))
        pickle.dump(rf_2, open(model_save_directory+'kNN_protein_type_model.pkl', 'wb'))

        # Append errors and indices for orientation
        train_errors.append(train_error)
        test_errors.append(test_error)
        psi_train_errors.append(psi_train_error)
        psi_test_errors.append(psi_test_error)
        train_indices.append(train_ind)
        test_indices.append(test_ind)


    # Other metrics
    #print("Total Train Time: ", np.mean(train_time))
    #print("  Stdev: ", np.std(train_time))
    #print("Total Test Time: ", np.mean(test_time))
    #print("  Stdev: ", np.std(test_time))
    
    #print("Error Degree") 
    test_means = np.mean(test_errors, axis=1)
    #print("Mean Mean Test Error:", round(np.mean(test_means), 4))
    #print("  Stdev:", round(np.std(test_means), 4))
    #print("  Min/Med/Max:", round(np.min(test_means), 4), round(np.median(test_means), 4), round(np.max(test_means), 4))
    test_medians = np.median(test_errors, axis=1)
    #print("Mean Median Test Error:", round(np.mean(test_medians), 4))
    #print("  Stdev:", round(np.std(test_medians), 4))
    #print("  Min/Med/Max:", round(np.min(test_medians), 4), round(np.median(test_medians), 4), round(np.max(test_medians), 4))

    test_rmse = np.sqrt(np.mean(np.square(test_errors), axis=1))
    #print("Mean RMSE Test Error:", round(np.mean(test_rmse), 4))
    #print("  Stdev:", round(np.std(test_rmse), 4))
    #print("  Min/Med/Max:", round(np.min(test_rmse), 4), round(np.median(test_rmse), 4), round(np.max(test_rmse), 4))

    #print("Psi Error") 
    test_means = np.mean(psi_test_errors, axis=1)
    #print("Mean Mean Test Error:", round(np.mean(test_means), 4))
    #print("  Stdev:", round(np.std(test_means), 4))
    #print("  Min/Med/Max:", round(np.min(test_means), 4), round(np.median(test_means), 4), round(np.max(test_means), 4))
    test_medians = np.median(psi_test_errors, axis=1)
    #print("Mean Median Test Error:", round(np.mean(test_medians), 4))
    #print("  Stdev:", round(np.std(test_medians), 4))
    #print("  Min/Med/Max:", round(np.min(test_medians), 4), round(np.median(test_medians), 4), round(np.max(test_medians), 4))

    test_rmse = np.sqrt(np.mean(np.square(psi_test_errors), axis=1))
    #print("Mean RMSE Test Error:", round(np.mean(test_rmse), 4))
    #print("  Stdev:", round(np.std(test_rmse), 4))
    #print("  Min/Med/Max:", round(np.min(test_rmse), 4), round(np.median(test_rmse), 4), round(np.max(test_rmse), 4))
	
    
    return (train_errors, test_errors, train_accuracies_conf, test_accuracies_conf, train_accuracies_type, test_accuracies_type, psi_train_errors, psi_test_errors, train_indices, test_indices, predictions_real_test, confusionmatrix_test)

def save_trial(error_path, directory, train_errors, test_errors, train_accuracies_conf, test_accuracies_conf, train_accuracies_type, test_accuracies_type, psi_train_errors, psi_test_errors, train_ind, test_ind, predictions_real_test, confusionmatrix_test, conf):
    os.makedirs(error_path+directory, exist_ok=True)

    training_errors = np.array(train_errors)
    testing_errors = np.array(test_errors)
    training_accuracies_conf = np.array(train_accuracies_conf)
    testing_accuracies_conf = np.array(test_accuracies_conf)
    training_accuracies_type = np.array(train_accuracies_type)
    testing_accuracies_type = np.array(test_accuracies_type)
    train_ind = np.array(train_ind)
    test_ind = np.array(test_ind)
    predictions_real_test = np.array(predictions_real_test)
    confusionmatrix_test = np.array(confusionmatrix_test)


    f = open(error_path + directory + "rf_train_errors.txt", "w")
    for errors in training_errors:
        for error in errors:
            f.write("%s " % error)
        f.write("\n")
    f.close()
    
    f = open(error_path + directory + "rf_test_errors.txt", "w")
    for errors in testing_errors:
        for error in errors:
            f.write("%s " % error)
        f.write("\n")
    f.close()

    f = open(error_path + directory + "psi_train_errors.txt", "w")
    for errors in psi_train_errors:
        for error in errors:
            f.write("%s " % error)
        f.write("\n")
    f.close()

    f = open(error_path + directory + "psi_test_errors.txt", "w")
    for errors in psi_test_errors:
        for error in errors:
            f.write("%s " % error)
        f.write("\n")
    f.close()

    '''
    f = open(error_path + directory + "predicted_real_train.txt", "w")
    for errors in predictions_real_train:
        for error in errors:
            f.write("%s " % error)
        f.write("\n")
    f.close()
    '''

    f = open(error_path + directory + "real-predicted_test.txt", "w")
    for errors in predictions_real_test:
        for error in errors:
            f.write("%s " % error)
        f.write("\n")
    f.close()

    f = open(error_path + directory + "cm_test.txt", "w")
    for errors in confusionmatrix_test:
        for error in errors:
            f.write("%s " % error)
        f.write("\n")
    f.close()

    f = error_path + directory + "rf_train_accuracies_conf.txt"
    np.savetxt(f, training_accuracies_conf, delimiter=',')


    f = error_path + directory + "rf_test_accuracies_conf.txt"
    np.savetxt(f, testing_accuracies_conf, delimiter=',')

    f = error_path + directory + "rf_train_accuracies_type.txt"
    np.savetxt(f, training_accuracies_type, delimiter=',')


    f = error_path + directory + "rf_test_accuracies_type.txt"
    np.savetxt(f, testing_accuracies_type, delimiter=',')

    f = open(error_path + directory + "train_ind.txt", "w")
    for indices in train_ind:
        for index in indices:
            f.write("%s " % index)
        f.write("\n")
    f.close()
    
    f = open(error_path + directory + "test_ind.txt", "w")
    for indices in test_ind:
        for index in indices:
            f.write("%s " % index)
        f.write("\n")
    f.close()
    
#

def predict_properties_main(configs_path, parameters):
    '''
    For a set of dataset, the function performs prediction of orientation, conformation, 
    and protein type of protein structure diffraction patterns using a kNN angle regressor
    and kNN-classificator.
    
    Parameters
    ----------
    configs_path: str; required; path to the configs file used for prediction
    parameters: required; values for no. of kNN trials, k value for k nearest neighbor/s, 
        and kNN average method
    
        
    Returns
    ----------
    returns prediction error and accuracy stats, along with confusion matrices for the
    corresponding property prediction (orientation, conformation, protein type).
    '''
    
    # Read configuration file
    conf = cp.ConfigParser()
    conf.read(configs_path)
    '''
    conformations = conf['data']['conformations'].split(';')
    intensity = conf['data']['intensity']
    all_conformations = []
    for i,val in enumerate(conformations):
        something=val.split(',')
        for k in something:
            all_conformations.append(k)
    all_conformations = "-".join(all_conformations)
    dir_name = "_".join([str(code_size), intensity, all_conformations])+'/'
    features = np.loadtxt(conf['dataset']['outputpath']+dir_name+conf['dataset']['featurefile'])
    indices = np.loadtxt(conf['dataset']['outputpath']+dir_name+conf['dataset']['indexfile']).astype(int)
    labels_path = conf['dataset']['outputpath']+dir_name+conf['dataset']['labelsfile']
    error_path = conf['dataset']['outputpath']+dir_name+conf['output']['errorpath']
    '''
    
    features = np.loadtxt(conf['dataset']['featureFile'])
    indices = np.loadtxt(conf['dataset']['indexFile']).astype(int)
    error_path = conf['output']['errorPath']
    #Selection of number of angles to predict (to include the rotation angle or not)
    if parameters['knn_average'] == 'vector_mean':
        euler_angles = np.loadtxt(conf['dataset']['labelsFile'], usecols=(0, 1))
    else:
        euler_angles = np.loadtxt(conf['dataset']['labelsFile'], usecols=(0, 1, 2))
    
    #Process the conformation 
    conformation = np.loadtxt(conf['dataset']['labelsFile'], usecols=(3), dtype=str)
    le = LabelEncoder()
    conformation = le.fit_transform(conformation)
    #print('Conformations: ', le.classes_)
    conformation = conformation.reshape(len(conformation),1)
    
    #Process the protein type 
    ptypes = np.loadtxt(conf['dataset']['labelsFile'], usecols=(4), dtype=str)
    le_type = LabelEncoder()
    ptypes = le_type.fit_transform(ptypes)
    #print('Protein types: ', le_type.classes_)
    ptypes = ptypes.reshape(len(ptypes),1)
    
    #Label of orientation, conformation, and protein type
    labels = np.hstack((euler_angles, conformation, ptypes))
    
    if parameters['knn_trials'] != 0:
        #print("NN trials", flush=True)
        # Perform Nearest Neighbors regression
        # knn for orientation (regressor)
        knn_orientation = KNeighborsAngleRegressor(n_neighbors=parameters['k'], average=parameters['knn_average'])
        # knn for conformation (classifier)
        knn_conformation = KNeighborsClassifier(n_neighbors=parameters['k'], algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
        # knn for protein type (classifier)
        knn_ptype = KNeighborsClassifier(n_neighbors=parameters['k'], algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
        print("Initiating for ", parameters['k'], " nearest neighbors")
        # Do trials 
        train_errors,test_errors, train_accuracies_conf, test_accuracies_conf, train_accuracies_type, test_accuracies_type, psi_train_errors, psi_test_errors, train_ind, test_ind,  predictions_real_test, confusionmatrix_test = do_trials(knn_orientation, knn_conformation, knn_ptype, error_path+str(parameters['k'])+" Nearest_Neighbors/visual_knn/", range(len(le.classes_)), le.classes_, range(len(le_type.classes_)), le_type.classes_, features, labels, indices, parameters['knn_trials'])
        
        #Save trials 
        print("Saving trials", parameters['k'], " nearest neighbors")
        save_trial(error_path, str(parameters['k']) + " Nearest_Neighbors/", train_errors, test_errors, train_accuracies_conf, test_accuracies_conf, train_accuracies_type, test_accuracies_type, psi_train_errors, psi_test_errors, train_ind, test_ind, predictions_real_test, confusionmatrix_test, conf)
    
'''
if __name__ == '__main__':

    #Parse information from command line
    parser = optparse.OptionParser()
    parser.set_defaults(knn_trials=10,rf_trials=1, k=1, trees=1000, knn_average='vector_mean')
    parser.add_option('--knn-trials', action='store', dest='knn_trials', type='int')
    parser.add_option('--rf-trials', action='store', dest='rf_trials', type='int')
    parser.add_option('--k', action='store', dest='k', type='int')
    parser.add_option('--trees', action='store', dest='trees', type='int')
    parser.add_option('--knn-average', action='store', dest='knn_average', type='string')
    (options, args) = parser.parse_args()
    print(options)
    print(args)
  
    # Read configuration file
    conf = cp.ConfigParser()
    if len(args) > 0:
        conf.read(args[0])
    else:
        conf.read("configs/global.ini")
    
    features = np.loadtxt(conf['dataset']['featureFile'])
    indices = np.loadtxt(conf['dataset']['indexFile']).astype(int)

    #Selection of number of angles to predict (to include the rotation angle or not)
    if options.knn_average == 'vector_mean':
        euler_angles = np.loadtxt(conf['dataset']['labelsFile'], usecols=(0, 1))
    else:
        euler_angles = np.loadtxt(conf['dataset']['labelsFile'], usecols=(0, 1, 2))

    #Process the conformation 
    conformation = np.loadtxt(conf['dataset']['labelsFile'], usecols=(3), dtype=str)
    le = LabelEncoder()
    conformation = le.fit_transform(conformation)
    print('Conformations: ', le.classes_)
    conformation = conformation.reshape(len(conformation),1)
    #print(conformation.shape, euler_angles.shape)

    #Process the protein type 
    ptypes = np.loadtxt(conf['dataset']['labelsFile'], usecols=(4), dtype=str)
    le_type = LabelEncoder()
    ptypes = le_type.fit_transform(ptypes)
    print('Protein types: ', le_type.classes_)
    ptypes = ptypes.reshape(len(ptypes),1)

    #Label of orientation, conformation, and protein type
    labels = np.hstack((euler_angles, conformation, ptypes))
    #print(labels)

    if options.knn_trials != 0:
        print("NN trials", flush=True)
        # Perform Nearest Neighbors regression
        # knn for orientation (regressor)
        knn_orientation = KNeighborsAngleRegressor(n_neighbors=options.k, average=options.knn_average)
        # knn for conformation (classifier)
        knn_conformation = KNeighborsClassifier(n_neighbors=options.k, algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
        # knn for protein type (classifier)
        knn_ptype = KNeighborsClassifier(n_neighbors=options.k, algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

        # Do trials 
        train_errors,test_errors, train_accuracies_conf, test_accuracies_conf, train_accuracies_type, test_accuracies_type, psi_train_errors, psi_test_errors, train_ind, test_ind,  predictions_real_test, confusionmatrix_test = do_trials(knn_orientation, knn_conformation, knn_ptype, conf['output']['errorPath']+str(options.k)+" Nearest_Neighbors/visual_knn/", range(len(le.classes_)), le.classes_, range(len(le_type.classes_)), le_type.classes_, options.knn_trials)
        
        #Save trials 
        save_trial(str(options.k) + " Nearest_Neighbors/", train_errors, test_errors, train_accuracies_conf, test_accuracies_conf, train_accuracies_type, test_accuracies_type, psi_train_errors, psi_test_errors, train_ind, test_ind, predictions_real_test, confusionmatrix_test)
'''

'''
####Random Forest Model
if options.rf_trials != 0:    
    print("RF trials", flush=True)
 
    # Perform Random Forest regression
    rf = RandomForestRegressor(n_estimators = options.trees, random_state = 42)
    train_errors,test_errors = do_trials(rf, options.rf_trials)

    save_trial("Random "+str(options.trees)+" Tree Forest/", train_errors, test_errors)
'''

