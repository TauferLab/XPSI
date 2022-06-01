import configparser as cp
import keras_utils as ku
import numpy as np
import pandas as pd
import os
from PIL import Image
import random
import sys
import tensorflow as tf
import tensorflow.keras.layers as L
import time
import utils
#from autoencoder_architecture import build_autoencoder_layers


#There are some images with a different size. Therefore, we cut them or expand them all to have the same size. 
def processdataset(dataset, conf):
    # Process dataset
    dataset_min = np.min(dataset)
    #print(dataset_min)
    datatype = conf['autoencoder']['precision']
    if datatype == 'mixed_float16':
        datatype = 'float32'
    elif datatype == 'mixed_bfloat16':
        datatype = 'float32'
    elif datatype == 'bfloat16':
        datatype = 'float32'
    #print('SIZE OF IMAGE PRE: ', dataset[0].shape)
    dataset = (dataset.astype(datatype) - dataset_min) / (np.max(dataset) - dataset_min)
    dataset = dataset[..., np.newaxis]
    #print('SIZE OF IMAGE: ', dataset[0].shape)
    #print('DATASET SIZE NEW', dataset.shape)
    raw_image_shape = dataset.shape[1:]
    if raw_image_shape[0] < 128 and raw_image_shape[1] < 128:
        dataset = np.pad(dataset, pad_width=((0, 0),
                                             (0, 128-raw_image_shape[0]),
                                             (0, 128-raw_image_shape[1]),
                                             (0, 0)),  mode='constant')
    return (dataset)

def load_dataset(conf, subset=False, mode='base'):
    global IMG_SHAPE
    IMG_SHAPE = (128, 128, 1)
    
    if mode == 'base':
        conf_data = conf['data']
    elif mode == 'validation':
        conf_data = conf['validationdata']
    elif mode == 'decoded':
        conf_data = conf['decodedData']
    else:
        raise ValueError('Unknown dataset mode: "{}"'.format(mode))
    if subset == False:
        data_size = int(conf_data['datasize'])
        data_path = conf_data['dataPath']
    
        #Load angles
        data_labels=pd.read_csv(conf_data['eulerAngleFilePath'], sep='\t', header=None)
    else:
        data_size = int(conf_data['subsetdatasize'])
        data_path = conf_data['subsetDataPath']
    
        #Load angles
        data_labels=pd.read_csv(conf_data['subsetFilePath'], sep='\t', header=None)

   
    # Load dataset
    dataset_list = []
    labels_list_conf = []
    labels_list_type = []
    indices = []
    angles1 = [] 
    angles2 = []
    angles3 = []
    
    for image in os.listdir(data_path):
        im = Image.open(os.path.join(data_path, image), "r")
        labels_list_type.append(image.split('_')[0])
        labels_list_conf.append(image.split('_')[1])
        # Gets the right index of the dp read by checking that the labels from file and the name of the image match
        ind = (data_labels.loc[(data_labels[3]==str(image.split('_')[1])) & (data_labels[5]==int(image.split('_')[2].split('.')[0][3:]))]).index.values[0]
	# Extracts the right angles from that specific diffraction pattern 
        angles1.append(data_labels.at[ind,0])
        angles2.append(data_labels.at[ind,1])
        angles3.append(data_labels.at[ind,2])

        indices.append(image.split('ptm')[1].split('.')[0])
        data = np.array(im)*100 ## See if changing the intensity affects 
        dataset_list.append(data)
    dataset = np.array(dataset_list)
    dataset = processdataset (dataset, conf)

    if mode == 'base':
        #print("Labels file")
        #print(time.time())
        os.makedirs(os.path.dirname(conf['dataset']['labelsFile']), exist_ok=True)
        with open(conf['dataset']['labelsFile'], "w") as f:
            for e1,e2,e3,label_conf, label_type, ind in zip(angles1,angles2,angles3,labels_list_conf, labels_list_type, indices):
                f.write("%s\t%s\t%s\t%s\t%s\t%s\n" %(e1, e2, e3, label_conf, label_type, ind))
        #print("Finish Printing",time.time())
    else :
        #print("Labels file")
        #print(time.time())
        os.makedirs(os.path.dirname(conf['validationdata']['labelsFile']), exist_ok=True)
        with open(conf['validationdata']['labelsFile'], "w") as f:
            for e1,e2,e3,label_conf, label_type, ind in zip(angles1,angles2,angles3,labels_list_conf, labels_list_type, indices):
                f.write("%s\t%s\t%s\t%s\t%s\t%s\n" %(e1, e2, e3, label_conf, label_type, ind))
        #print("Finish Printing",time.time())
    return dataset, indices, labels_list_conf, labels_list_type


def compile_autoencoder(conf, encoder, decoder, img_shape, optimizer):
    inp = L.Input(IMG_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)

    autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer, loss=conf['autoencoder']['loss'])
    
    return autoencoder


def train_autoencoder(conf, dataset, code_size, epochs, batch_size, optimizer, build_autoencoder_layers, target_dataset=None):
    if target_dataset is None:
        target_dataset = dataset
    
    # Compile autoencoder
    s = ku.reset_tf_session()

    img_shape = dataset.shape[1:]

    # set randomization seeds to improve consistency between runs
    tf.compat.v1.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    encoder, decoder = build_autoencoder_layers(conf, img_shape, code_size)

    #print()
    encoder.summary()
    #print()
    decoder.summary()

    autoencoder = compile_autoencoder(conf, encoder, decoder, img_shape, optimizer)
    # Train autoencoder
    history_filename = conf['output']['historyFile']
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)
    try: # need to remove any previous history to avoid appending to it
        os.remove(history_filename)
    except FileNotFoundError:
        pass
    start_time = time.perf_counter()
    autoencoder.fit(x=dataset, y=target_dataset, epochs=epochs,
                validation_split=0.0,
                callbacks=[ku.LossHistory(history_filename)],
                batch_size=batch_size,
                initial_epoch=0)
    end_time = time.perf_counter()
    #print('Took ' + str(end_time - start_time) + ' seconds to train the autoencoder')
    return autoencoder, encoder, decoder


def load_autoencoder(conf, build_autoencoder_layers, code_size, optimizer):
    encoder, decoder = build_autoencoder_layers(conf, IMG_SHAPE, code_size)
    encoder.load_weights(conf['output']['encoderFile'])
    decoder.load_weights(conf['output']['decoderFile'])
    
    autoencoder = compile_autoencoder(conf, encoder, decoder, IMG_SHAPE, optimizer)
    return autoencoder, encoder, decoder

def feature_extraction_main(config_file_path, parameters, build_autoencoder_layers, subset=False):
    '''
    Given a pre-processed data, hyperparameters for the autoencoder, and the architecture 
    of the autoencoder, the function extracts feature and creates tensor representation of 
    these features. The decoder reconstructs the images with the information from the 
    tensor representation.  
    
    Parameters
    ----------
    config_file_path: str; required; path to the configs file used for feature extraction
    parameters: required; code size (latent space dimension), epochs, batch size, and 
                optimizer values
    build_autoencoder_layers: function; required; function holding the encoder and decoder 
                structure
    
        
    Returns
    ----------
    returns an output folder with the saved encoder and decoder weights, along with the 
    loss for each epoch of the feature extraction run.
    
    '''
    
    #print("loading data and configuration")
    # Read configuration file
    conf = cp.ConfigParser()
    conf.read(config_file_path)
    
    #Load dataset from configs, if Y is not provided, set Y to X, meaning that the desired output of the autoencoders are the original images
    X, inds, labels_conf, labels_type = load_dataset(conf, subset)
    #print("dataset loaded")
    if conf.has_option('data', 'decodedData'):
        Y, _ = load_dataset(conf, subset, mode='decoded')
    else:
        Y = X
    print("Training autoencoder")
    #Train autoencoder
    #print("training autoencoder")
    autoencoder, encoder, decoder = train_autoencoder(conf, X, parameters['code_size'], parameters['epochs'], parameters['batch_size'], parameters['optimizer'], build_autoencoder_layers, Y)
    print("Outputing results...")
    
    
    # Extract and save features
    os.makedirs(os.path.dirname(conf['dataset']['featureFile']), exist_ok=True)
    with open(conf['dataset']['featureFile'], "w") as f:
        for img in X:
            features = encoder.predict(img[None])[0]
            for feature in features:
                f.write("%s " % feature)
            f.write("\n")
    
    os.makedirs(os.path.dirname(conf['dataset']['indexFile']), exist_ok=True)
    with open(conf['dataset']['indexFile'], "w") as f:
        for ind in inds:
            f.write("%s\n" % ind)

    
    # Save trained weights
    encoderFile = conf['output']['encoderFile']
    os.makedirs(os.path.dirname(encoderFile), exist_ok=True)
    encoder.save_weights(encoderFile)
    
    decoderFile = conf['output']['decoderFile']
    os.makedirs(os.path.dirname(decoderFile), exist_ok=True)
    decoder.save_weights(decoderFile)
    
    