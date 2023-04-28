import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class CGAN():
    def __init__(self, latent_dim, save_path):
        
       

        self.opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.laten_dim = latent_dim

        self.discriminator = self.build_discriminator(latent_dim = latent_dim)
        self.generator = self.build_generator(latent_dim = latent_dim)
        self.gan_model = self.build_gan(self.discriminator, self.generator)
        self.losses = []
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        return
    

    def load_dataset(self, csv_path):
        data = pd.read_csv('proj_keras/train.csv')
        data = data.to_numpy()
        print(data.shape)
        #first column is the label and the rest is the image
        labels = data[:,0]
        images = data[:,1:]
        images = images.reshape(-1, 28, 28, 1)
        images = images.astype('float32')
        images = (images - 127.5) / 127.5 # Normalize the images to [-1, 1]
        return [images, labels]
    
    def get_dataset_samples(self, dataset, num_samples):
        Xs, lbls = dataset
        idx = np.random.randint(0, Xs.shape[0], num_samples)
        X , lbl = Xs[idx], lbls[idx]
        y = np.ones((num_samples, 1))


        return [X, lbl] , y
    
    def generate_noise(self, noise_dim, n_samples, num_classes = 10):
        x_input = np.random.randn(noise_dim * n_samples)
        z_input = x_input.reshape(n_samples, noise_dim)
        labels = np.random.randint(0, num_classes, n_samples)
        return [z_input, labels]
    
    def generate_fake_samples(self, latent_dim, n_samples):
        z_input , labels_input = self.generate_noise(latent_dim, n_samples)
        images = self.generator.predict([z_input, labels_input])
        y = np.zeros((n_samples, 1))
        return [images, labels_input], y
    
    def build_generator(self, latent_dim = 50, num_classes = 10):
        # label input
        in_label = keras.layers.Input(shape=(1,))
        # embedding for categorical input
        li = keras.layers.Embedding(num_classes, latent_dim)(in_label)
        # linear multiplication
        n_nodes = 7 * 7
        li = keras.layers.Dense(n_nodes)(li)
        # reshape to additional channel
        li = keras.layers.Reshape((7, 7, 1))(li)
        # image generator input
        in_lat = keras.layers.Input(shape=(latent_dim,))
        # foundation for 7x7 image
        n_nodes = 256 * 7 * 7
        gen = keras.layers.Dense(n_nodes)(in_lat)

        gen = keras.layers.LeakyReLU(alpha=0.2)(gen)
        gen = keras.layers.Reshape((7, 7, 256))(gen)
        # merge image gen and label input
        merge = keras.layers.Concatenate()([gen, li])
        # upsample to 14x14
        gen = keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', 
                                           activation=keras.layers.LeakyReLU(alpha=0.2))(merge)
        gen = keras.layers.BatchNormalization()(gen)
        # upsample to 28x28
        gen = keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', 
                                           activation=keras.layers.LeakyReLU(alpha=0.2))(gen)
        gen = keras.layers.BatchNormalization()(gen)
        # output
        out_layer = keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        # define model
        model = keras.Model([in_lat, in_label], out_layer)
        
        return model

    def build_discriminator(self, latent_dim, input_shape = (28,28,1), num_classes = 10):
        in_label = keras.layers.Input(shape=(1,))
        li = keras.layers.Embedding(num_classes, latent_dim)(in_label)

        n_nodes = input_shape[0] * input_shape[1]
        li=keras.layers.Dense(n_nodes)(li)

        li=keras.layers.Reshape((input_shape[0], input_shape[1], 1))(li)

        in_image = keras.layers.Input(shape=input_shape)

        merge = keras.layers.Concatenate()([in_image, li])

        #downsample
        fe=keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', 
                               activation=keras.layers.LeakyReLU(alpha=0.2))(merge)
        fe=keras.layers.Dropout(0.4)(fe)

        #downsample
        fe=keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', 
                               activation=keras.layers.LeakyReLU(alpha=0.2))(fe)
        fe=keras.layers.Dropout(0.4)(fe)

        fe = keras.layers.Flatten()(fe)

        out_layer = keras.layers.Dense(1, activation='sigmoid')(fe)

        model = keras.Model([in_image, in_label], out_layer)

        opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
    
    def build_gan(self, discriminator, generator):
        discriminator.trainable = False
        gen_noise, gen_label = generator.input
        gen_output = generator.output
        gan_output = discriminator([gen_output, gen_label])

        model= keras.Model([gen_noise, gen_label], gan_output)
        opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train_gan(self, generator, discriminator, GAN, dataset, noise_size=100, n_epochs=30, n_batch=512):
        steps = int(dataset[0].shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for e in range(n_epochs):
            # enumerate batches over the training set
            for s in range(steps):
                #TRAIN THE DISCRIMINATOR
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = self.get_dataset_samples(dataset, half_batch)
                # update discriminator model weights
                d_loss1, _ = discriminator.train_on_batch([X_real, labels_real], y_real)
                # generate 'fake' examples
                [X_fake, labels], y_fake = self.generate_fake_samples(noise_size, half_batch)
                # update discriminator model weights
                d_loss2, _ = discriminator.train_on_batch([X_fake, labels], y_fake)

                #TRAIN THE GENERATOR
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_noise(noise_size, n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = GAN.train_on_batch([z_input, labels_input], y_gan)
                # summarize loss on this batch
                print('>%d, d1=%.3f, d2=%.3f g=%.3f' %
                (e+1, d_loss1, d_loss2, g_loss))
            #self.plot_results(X_fake, 8)  
            #log the losses
            self.losses.append((d_loss1, d_loss2, g_loss))
            



        # save the generator model
        generator.save(os.path.join(self.save_path,'cgan_generator.h5'))
        # save the discriminator model
        discriminator.save(os.path.join(self.save_path,'cgan_discriminator.h5'))
        # log all losses to a file
        self.save_loss_log(self.losses)
    
    def save_loss_log(self, losses):
        epochs = len(losses)
        txt_path = os.path.join(self.save_path,'losses.txt')

        with open(txt_path, 'w') as f:
            f.write('epoch, d_loss1, d_loss2, g_loss\n')
            for i in range(epochs):
                row = '%d,%.10f,%.10f,%.10f\n' % (i+1, losses[i][0], losses[i][1], losses[i][2])
                f.write(row)

    def plot_results(self, images, n_cols=None):
        '''visualizes fake images'''
        #display.clear_output(wait=False)  

        n_cols = n_cols or len(images)
        n_rows = (len(images) - 1) // n_cols + 1

        if images.shape[-1] == 1:
            images = np.squeeze(images, axis=-1)

        plt.figure(figsize=(n_cols, n_rows))

        for index, image in enumerate(images):
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(image, cmap = "binary")
            plt.axis("off")
        plt.show()

    def train(self, dataset, n_epochs=30, n_batch=512):
        self.train_gan(self.generator, self.discriminator, self.gan_model, dataset, self.laten_dim, n_epochs, n_batch)
        return
##test area
if __name__ == "__main__":
    
    
    gan = CGAN(latent_dim=100, save_path='models/cgan2_ep100_lat100')
    dataset = gan.load_dataset('proj_keras/train.csv')
    gan.train(dataset, n_epochs=100, n_batch=512)
    latent_points, labels = gan.generate_noise(100, 20)

    labels = np.ones(20) * 5
    X = gan.generator.predict([latent_points, labels])
    gan.plot_results(X, 10)


    print(gan.generator.summary())
    #print(gan.discriminator.summary())
    
    
    

    

