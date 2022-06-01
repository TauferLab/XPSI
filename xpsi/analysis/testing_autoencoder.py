import sys
sys.path.insert(0, "xpsi/workflow_scripts/")
import feature_extraction
import numpy as np
import configparser as cp
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from scipy.spatial import distance
import scipy.stats
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib.image as mpimg


# Image visualization
def show_image(x):
    plt.imshow(x, cmap='gray')

def visualize(dataset,encoder,decoder,save_path):
    reconstruction_images = []
    for i in range(3):
        img = dataset[i]
        code = encoder.predict(img[None])[0]  # img[None] is the same as img[np.newaxis, :]
        reco = decoder.predict(code[None])[0]
        reconstruction_images.append(reco)
        
    fig, axes = plt.subplots(3, 2, figsize=(5, 5))
    axes[0, 0].imshow(np.reshape(dataset[0], (128,128)),cmap='gray')
    axes[0, 0].set_title('Original:')
    axes[0, 1].imshow(np.reshape(reconstruction_images[0], (128,128)),cmap='gray')
    axes[0, 1].set_title('Reconstruction:')
    axes[1, 0].imshow(np.reshape(dataset[1], (128,128)),cmap='gray')
    axes[1, 1].imshow(np.reshape(reconstruction_images[1], (128,128)),cmap='gray')
    axes[2, 0].imshow(np.reshape(dataset[2], (128,128)),cmap='gray')
    axes[2, 1].imshow(np.reshape(reconstruction_images[2], (128,128)),cmap='gray')
    
    plt.savefig(str(save_path)+'model_comparison.png')
    plt.show()
    

def plot_recontructed_images(config_file_path):
    '''
    Given a config file, renders the original images and reconstructed images based on the autoencoder.
    The plot will consist of three original images and three reconstructed images for a given resolution 
    of dataset for comparison.
    
    Parameter
    ----------
    config_file_path: str; required; path to config file
    
    Returns
    ----------
    renders a 3x2 plot of images; original and reconstructed in a colum each
    
    '''
    
    # Read configuration file
    conf = cp.ConfigParser()
    feature_extraction.conf = conf # use the same configuration in feature_extraction
    conf.read(config_file_path)
    
    # Initialize
    print("Initializing data...")
    euler_angle1 = np.loadtxt(conf['data']['eulerAngleFilePath'], usecols=0)
    euler_angle2 = np.loadtxt(conf['data']['eulerAngleFilePath'], usecols=1)
    euler_angle3 = np.loadtxt(conf['data']['eulerAngleFilePath'], usecols=2)
    
    print("Configuring path to image...")
    data_size = int(conf['data']['dataSize'])
    image_output_path = conf['output']['imagePath']
    os.makedirs(image_output_path, exist_ok=True)
    
    print("Loading data...")
    dataset, data_indices, conformation_labels, labels_type = feature_extraction.load_dataset(conf)
    print("Loading autoencoder...")
    autoencoder, encoder, decoder = feature_extraction.load_autoencoder(conf)
    
    reconstruction_mse = autoencoder.evaluate(dataset, dataset, verbose=0)
    print("Convolutional autoencoder MSE:", reconstruction_mse)
    visualize(dataset,encoder,decoder, image_output_path)

    
    
def compare_reconstructions(config_one, config_two):
    '''
    Given config files for two testcases, the function displays the reconstruction images side by side.
    
    Parameter
    ----------
    config_one: str; required; path to first config file, usually to the default test case
    config_two: str; required; path to second config file
    
    Returns
    ----------
    displays orginal and reconstructed images for two test cases 
    
    '''
    conf = cp.ConfigParser()
    conf.read(config_one)
    first_comparison = conf['output']['imagePath']
    conf.read(config_two)
    second_comparison = conf['output']['imagePath']
    
    image1 = mpimg.imread(first_comparison+'model_comparison.png')
    image2 = mpimg.imread(second_comparison+'model_comparison.png')
    
    # display images
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].imshow(image1)
    axes[0].set_title('With default autoencoder')
    axes[0].set_axis_off()
    
    axes[1].imshow(image2)
    axes[1].set_title('With new autoencoder')
    axes[1].set_axis_off()
    
    plt.show()
    
def test_case():
    print('testing testing!')