import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape, BatchNormalization, Conv2DTranspose, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow import keras

from model.SpectralNormalizationKeras import DenseSN, ConvSN2D

import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



# Fully Connected GAN
'''

def build_generator(latent_dim):
        # 100->32->64->128->784

        model = Sequential(name='generator')
        model.add(Dense(32, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(784, activation='tanh'))
        model.add(Reshape((28, 28, 1)))

        return model

def build_discriminator():
        # 784->128->64->32->1
        model = Sequential(name='discriminator')
        model.add(Dense(1,input_shape=(28,28,1)))
        model.add(Flatten())
        
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(1, activation='sigmoid'))
        return model
'''


# DCGAN - 1
'''
'''
# DCGAN implementation, following the example from the tutorial in TensorFlow
def build_generator(latent_dim):

        model = Sequential(name='generator')

        model.add(Dense(7*7*256, use_bias = False, input_shape=(latent_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))         # Default alpha value = 0.3


        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)


        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))


        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))


        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model


def build_discriminator():

        model = Sequential(name='discriminator')
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=(28,28,1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model



# SN-GAN
'''
'''
def build_generator(latent_dim):

        model = Sequential(name='generator')

        model.add(Dense(7*7*256, use_bias = False, input_shape=(latent_dim,)))
        model.add(BatchNormalization(epsilon=0.00002, momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))         # Default alpha value = 0.3


        model.add(Reshape((7, 7, 256)))
        #assert model.output_shape == (None, 7, 7, 256)


        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        #assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization(epsilon=0.00002, momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))


        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        #assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization(epsilon=0.00002, momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))


        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        #assert model.output_shape == (None, 28, 28, 1)

        return model


def build_discriminator():

        # Applying strided convolution: helps network to extract features at different scales

        model = Sequential(name='discriminator')


        model.add(ConvSN2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=(28,28,1)))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.3))
        model.add(ConvSN2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(28,28,1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))


        model.add(ConvSN2D(128, (3, 3), strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.3))
        model.add(ConvSN2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))        
        model.add(Dropout(0.3))


        model.add(ConvSN2D(256, (3, 3), strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.3))
        model.add(ConvSN2D(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))        
        model.add(Dropout(0.3))        


        model.add(Flatten())
        model.add(DenseSN(1, activation='sigmoid'))

        return model





def build_optimizer(lr=0.002, beta_1=0.5):
    return Adam(lr=lr, beta_1=beta_1)

##test area
if __name__ == '__main__':
    print("test area")
    latent_dim = 100
    generator = build_generator(latent_dim)
    generator.summary()
    discriminator = build_discriminator()
    discriminator.summary()
    optimizer = build_optimizer()
    print("test area end")