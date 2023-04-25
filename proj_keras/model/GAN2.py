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

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        
        return
    def load_Data(self, train_path, test_path):
        #load csv from path
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        #convert to numpy array
        train = train.to_numpy()
        test = test.to_numpy()
        #first column is label, second column is pixel
        train_y = train[:,0]
        train_X = train[:,1:]
        test_y = test[:,0]
        test_X = test[:,1:]

        #preprocess
        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        train_X = train_X / 255.0
        train_X = train_X * 2 - 1

        test_X = test_X / 255.0
        test_X = test_X * 2 - 1

        train_X = train_X.reshape(-1,28,28,1)
        test_X = test_X.reshape(-1,28,28,1)
        
        return train_X, train_y, test_X, test_y

    def build_generator(self):
        #100->32->64->128->784

        model = Sequential(name='generator')
        model.add(Dense(32, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Dense(784))
        model.add(Reshape((28, 28, 1)))

        return model
    
    def build_discriminator(self):
        #784->128->64->32->1
        model = Sequential(name='discriminator')
        model.add(Dense(1,input_shape=[28,28,1]))
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
    
    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def discriminator_optimizer(self):
        return tf.keras.optimizers.Adam(0.002, 0.5)

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def generator_optimizer(self):
        return tf.keras.optimizers.Adam(0.002, 0.5)

    def create_Checkpoint(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer(),
                                         discriminator_optimizer=self.discriminator_optimizer(),
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        return checkpoint, checkpoint_prefix

    