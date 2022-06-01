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
from autoencoder_architecture import build_autoencoder_layers


#There are some images with a different size. Therefore, we cut them or expand them all to have the same size. 
def processdataset(dataset, conf):
    # Process dataset
    dataset_min = np.min(dataset)
    print(dataset_min)
    datatype = conf['autoencoder']['precision']
    if datatype == 'mixed_float16':
        datatype = 'float32'
    elif datatype == 'mixed_bfloat16':
        datatype = 'float32'
    elif datatype == 'bfloat16':
        datatype = 'float32'
    print('SIZE OF IMAGE PRE: ', dataset[0].shape)
    dataset = (dataset.astype(datatype) - dataset_min) / (np.max(dataset) - dataset_min)
    dataset = dataset[..., np.newaxis]
    print('SIZE OF IMAGE: ', dataset[0].shape)
    print('DATASET SIZE NEW', dataset.shape)
    raw_image_shape = dataset.shape[1:]
    if raw_image_shape[0] < 128 and raw_image_shape[1] < 128:
        dataset = np.pad(dataset, pad_width=((0, 0),
                                             (0, 128-raw_image_shape[0]),
                                             (0, 128-raw_image_shape[1]),
                                             (0, 0)),  mode='constant')
    return (dataset)

def load_dataset(conf, mode='base'):
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

    data_size = int(conf_data['dataSize'])
    data_path = conf_data['dataPath']
    
    #Load angles
    data_labels=pd.read_csv(conf_data['eulerAngleFilePath'], sep='\t', header=None)
   
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
        print("Labels file")
        print(time.time())
        os.makedirs(os.path.dirname(conf['dataset']['labelsFile']), exist_ok=True)
        with open(conf['dataset']['labelsFile'], "w") as f:
            for e1,e2,e3,label_conf, label_type, ind in zip(angles1,angles2,angles3,labels_list_conf, labels_list_type, indices):
                f.write("%s\t%s\t%s\t%s\t%s\t%s\n" %(e1, e2, e3, label_conf, label_type, ind))
        print("Finish Printing",time.time())
    else :
        print("Labels file")
        print(time.time())
        os.makedirs(os.path.dirname(conf['validationdata']['labelsFile']), exist_ok=True)
        with open(conf['validationdata']['labelsFile'], "w") as f:
            for e1,e2,e3,label_conf, label_type, ind in zip(angles1,angles2,angles3,labels_list_conf, labels_list_type, indices):
                f.write("%s\t%s\t%s\t%s\t%s\t%s\n" %(e1, e2, e3, label_conf, label_type, ind))
        print("Finish Printing",time.time())
    return dataset, indices, labels_list_conf, labels_list_type

'''
# Autoencoder
def build_autoencoder_layers(conf, img_shape, code_size):
    """Convolutional Autoencoder"""

    #tf.keras.mixed_precision.experimental.set_policy(conf['autoencoder']['precision'])

    # encoder
    encoder = tf.keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    
    encoder.add(L.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    print(encoder.output_shape)
    encoder.add(L.MaxPool2D(pool_size=(3, 3), padding='same'))
    print(encoder.output_shape)
    encoder.add(L.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    print(encoder.output_shape)
    encoder.add(L.MaxPool2D(pool_size=(3, 3), padding='same'))
    print(encoder.output_shape)
    encoder.add(L.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    print(encoder.output_shape)
    encoder.add(L.MaxPool2D(pool_size=(3, 3), padding='same'))
    print(encoder.output_shape)
    encoder.add(L.Flatten())
    print(encoder.output_shape)
    encoder.add(L.Dense(code_size))
    print(encoder.output_shape)
    
    # decoder
    decoder = tf.keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    print(decoder.output_shape)

    decoder.add(L.Dense(800))
    print(decoder.output_shape)
    decoder.add(L.Reshape((5, 5, 32)))
    print(decoder.output_shape)
    decoder.add(L.UpSampling2D((3, 3)))
    print(decoder.output_shape)
    decoder.add(L.Conv2D(64, (3, 3), activation='relu', padding='same'))
    print(decoder.output_shape)
    decoder.add(L.UpSampling2D((3, 3)))
    print(decoder.output_shape)
    decoder.add(L.Conv2D(128, (3, 3), activation='relu'))
    print(decoder.output_shape)
    decoder.add(L.UpSampling2D((3, 3)))
    print(decoder.output_shape)
    decoder.add(L.Conv2D(1, (2, 2), activation=None))
    print(decoder.output_shape)
    
    return encoder, decoder
'''

def compile_autoencoder(conf, encoder, decoder, img_shape):
    inp = L.Input(IMG_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)

    autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer=conf['autoencoder']['optimizer'], loss=conf['autoencoder']['loss'])
    
    return autoencoder



def train_autoencoder(conf, dataset, target_dataset=None):
    if target_dataset is None:
        target_dataset = dataset
    
    # Compile autoencoder
    s = ku.reset_tf_session()

    img_shape = dataset.shape[1:]

    # set randomization seeds to improve consistency between runs
    tf.compat.v1.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    encoder, decoder = build_autoencoder_layers(conf, img_shape, code_size=int(conf['autoencoder']['code_size']))

    print()
    encoder.summary()
    print()
    decoder.summary()

    autoencoder = compile_autoencoder(conf, encoder, decoder, img_shape)
    # Train autoencoder
    history_filename = conf['output']['historyFile']
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)
    try: # need to remove any previous history to avoid appending to it
        os.remove(history_filename)
    except FileNotFoundError:
        pass
    start_time = time.perf_counter()
    autoencoder.fit(x=dataset, y=target_dataset, epochs=int(conf['autoencoder']['epochs']),
                validation_split=0.0,
                callbacks=[ku.LossHistory(history_filename)],
                batch_size=int(conf['autoencoder']['batchSize']),
                initial_epoch=0)
    end_time = time.perf_counter()
    print('Took ' + str(end_time - start_time) + ' seconds to train the autoencoder')
    return autoencoder, encoder, decoder


def load_autoencoder(conf):
    encoder, decoder = build_autoencoder_layers(conf, IMG_SHAPE, code_size=int(conf['autoencoder']['code_size']))
    encoder.load_weights(conf['output']['encoderFile'])
    decoder.load_weights(conf['output']['decoderFile'])
    
    autoencoder = compile_autoencoder(conf, encoder, decoder, IMG_SHAPE)
    return autoencoder, encoder, decoder


if __name__ == '__main__':
    
    print("loading data and configuration")
    # Read configuration file
    conf = cp.ConfigParser()
    if len(sys.argv) != 1:
        conf.read(sys.argv[1])
    else:
        conf.read("configs/global.ini")

    #Load dataset from configs, if Y is not provided, set Y to X, meaning that the desired output of the autoencoders are the original images
    X, inds, labels_conf, labels_type = load_dataset(conf)
    if conf.has_option('data', 'decodedData'):
        Y, _ = load_dataset(conf, mode='decoded')
    else:
        Y = X

    #Train autoencoder
    print("training autoencoder")
    autoencoder, encoder, decoder = train_autoencoder(conf, X, Y)
    
    print("outputing results")
    
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
