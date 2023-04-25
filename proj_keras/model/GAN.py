
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import torch.nn as nn
import numpy as np
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
        X_train = np.loadtxt(csv_path, delimiter=',', dtype='str')
        #ignore first row
        X_train = X_train[1:]
        #first col is label, others are pixel values
        X_train = X_train[:, 1:]
        #astype to float32
        X_train = X_train.astype('float32')
        #reshape to 28x28
        X_train = X_train.reshape(-1, 28, 28, 1)

        self.train(X_train, epochs, batch_size)

        return
    
    def train(self, X_train, epochs=500, batch_size=256):
        #train model
        for epoch in range(epochs):
        
            #train discriminator
            #get random batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]
            #generate fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_imgs = self.generator.predict(noise)
            #train discriminator
            d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
            d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)

            #train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss, g_acc = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            
            #print loss
            print(f'epoch: {epoch}, d_loss: {d_loss}, d_acc: {d_acc} g_loss: {g_loss}, g_acc: {g_acc}')
            
            #save model
            #if epoch % 100 == 0:
            #    self.save_model(epoch)
        
            
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
    
