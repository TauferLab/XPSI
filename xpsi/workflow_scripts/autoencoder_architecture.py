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

