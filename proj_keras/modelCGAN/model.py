import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CGAN():
    def __init__(self):
        
       

        self.opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan_model = self.build_gan(self.discriminator, self.generator)

        return
    

    def load_dataset(self, csv_path):
        train_data = pd.read_csv(csv_path)
        X_train = train_data.iloc[:,1:].values.astype(np.float32)
        X_train = X_train.reshape(-1,28,28,1) * 2. - 1.
        y_train = train_data.iloc[:,0].values.astype(np.int32)

        return [X_train, y_train]
    
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
        li = keras.layers.Embedding(num_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 7 * 7
        li = keras.layers.Dense(n_nodes)(li)
        # reshape to additional channel
        li = keras.layers.Reshape((7, 7, 1))(li)
        # image generator input
        in_lat = keras.layers.Input(shape=(latent_dim,))
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        gen = keras.layers.Dense(n_nodes)(in_lat)

        gen = keras.layers.LeakyReLU(alpha=0.2)(gen)
        gen = keras.layers.Reshape((7, 7, 128))(gen)
        # merge image gen and label input
        merge = keras.layers.Concatenate()([gen, li])
        # upsample to 14x14
        gen = keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', 
                                           activation=keras.layers.LeakyReLU(alpha=0.2))(merge)
        gen = keras.layers.BatchNormalization()(gen)
        # upsample to 28x28
        gen = keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', 
                                           activation=keras.layers.LeakyReLU(alpha=0.2))(gen)
        gen = keras.layers.BatchNormalization()(gen)
        # output
        out_layer = keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        # define model
        model = keras.Model([in_lat, in_label], out_layer)
        
        return model

    def build_discriminator(self, input_shape = (28,28,1), num_classes = 10):
        in_label = keras.layers.Input(shape=(1,))
        li = keras.layers.Embedding(num_classes, 50)(in_label)

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
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
            (e+1, s+1, steps, d_loss1, d_loss2, g_loss))
          #self.plot_results(X_fake, 8)  

        # save the generator model
        generator.save('cgan_generator.h5')

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

    def train(self, dataset, noise_size=50, n_epochs=30, n_batch=512):
        self.train_gan(self.generator, self.discriminator, self.gan_model, dataset, noise_size, n_epochs, n_batch)
        return
##test area
if __name__ == "__main__":
    gan = CGAN()

    dataset = gan.load_dataset('proj_keras/train.csv')
    gan.train(dataset, noise_size=50, n_epochs=30, n_batch=512)
    gan.plot_results(gan.generate_fake_samples(gan.generator, 100, 8)[0], 8)    