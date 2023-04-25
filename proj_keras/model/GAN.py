
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

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
        self.gan = self.build_gan(self.generator, self.discriminator)

        return
    
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

    def build_gan(self, gen, disc):
        optimizer = Adam(0.002, 0.5)
        
       

        gan = Sequential([gen, disc], name='gan')
        disc.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        disc.trainable = False
        gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return gan
    
    def train_from_csv(self, csv_path, epochs=500, batch_size=256):
        #load csv
        train_data = pd.read_csv(csv_path)
        
        #get X_train
        X_train = train_data.drop('label', axis=1)
        X_train = X_train.values
        X_train = X_train.reshape(-1, 28, 28, 1)

        self.train(X_train, epochs, batch_size)
        return
    
    def train(self, X_train, epochs=500, batch_size=256):
        #assume X_train is from 0 to 255 --> convert to float32 --> normalize to [-1, 1]
        X_train = X_train.astype('float32')
        X_train = X_train / 255.0
        X_train = X_train * 2 - 1
        
        #train
        for epoch in range(epochs):
            for __ in range(X_train.shape[0] // batch_size):
                #train generator
                noise = np.random.normal(loc=0, scale=1, size=(batch_size, self.latent_dim))
                gen_imgs = self.generator.predict_on_batch(noise)
                y_gen = np.ones((batch_size, 1))
                self.discriminator.trainable = False
                gan_loss = self.gan.train_on_batch(noise, y_gen)

                #train discriminator
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                noise = np.random.normal(loc=0, scale=1, size=(batch_size, self.latent_dim))
                gen_imgs = self.generator.predict_on_batch(noise)
                y_real = np.ones((batch_size, 1))
                y_fake = np.zeros((batch_size, 1))
                self.discriminator.trainable = True
                disc_loss_real = self.discriminator.train_on_batch(imgs, y_real)
                disc_loss_fake = self.discriminator.train_on_batch(gen_imgs, y_fake)
                disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

            print(f'Epoch {epoch+1}/{epochs} | GAN Loss: {gan_loss[0]} | GAN Accuracy: {gan_loss[1]} | Discriminator Loss: {disc_loss[0]} | Discriminator Accuracy: {disc_loss[1]}')           
        return
    
    def save_model(self, folder_path):
        #create folder if not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #save model
        path_generator = os.path.join(folder_path, f'generator.h5')
        path_discriminator = os.path.join(folder_path, f'discriminator.h5')
        path_gan = os.path.join(folder_path, f'gan.h5')

        self.generator.save(path_generator)
        self.discriminator.save(path_discriminator)
        self.gan.save(path_gan)
        return
    def load_model(self, folder_path):
        #load model
        path_generator = os.path.join(folder_path, f'generator.h5')
        path_discriminator = os.path.join(folder_path, f'discriminator.h5')
        path_gan = os.path.join(folder_path, f'gan.h5')

        self.generator = load_model(path_generator)
        self.discriminator = load_model(path_discriminator)
        self.gan = load_model(path_gan)
        return


    def genSampleFromGenerator(self, isShow=False, samples=10):
        #generate sample from generator
        
        x_fake = self.generator.predict(np.random.normal(loc=0, scale=1, size=(samples, 100)))

        rows = samples // 5
        cols = 5


        for k in range(samples):
            plt.subplot(rows, cols, k+1)
            plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
            plt.axis('off')


        plt.tight_layout()
        plt.show()
        return 
    
##test code
if __name__ == '__main__':
    gan = GAN()
    print(gan.generator.summary())
    print(gan.discriminator.summary())
    print(gan.gan.summary()) 
    
