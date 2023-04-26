import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow import keras

import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def build_generator(latent_dim):
        #100->32->64->128->784

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
        #784->128->64->32->1
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