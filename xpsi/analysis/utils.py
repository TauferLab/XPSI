import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Build dataframe for index and angles 
def build_df (indices, euler_angles):
    df = pd.DataFrame({'indices':indices,'euler_angle1':euler_angles[:,0], 'euler_angle2':euler_angles[:,1]})
    return df

#Function to read the data from the files 
def read_from_files(conf):
    features = np.loadtxt(conf['dataset']['featureFile'])
    indices = np.loadtxt(conf['dataset']['indexFile']).astype(int)
    euler_angles = np.loadtxt(conf['data']['eulerAngleFilePath'], usecols=(0, 1))
    euler_angles = euler_angles[indices]
    return (features, indices, euler_angles)