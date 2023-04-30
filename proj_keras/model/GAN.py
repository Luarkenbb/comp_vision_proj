import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


#tf.config.run_functions_eagerly(False)

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        #self.discriminator_train = True

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    

    # Origin train_step() function: always training the discriminator first then generator
    '''
    '''
    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)


        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
    
    
    ########################################################################

    '''
    
    # def call(self, inputs, *args, **kwargs):
    #   return self.model(inputs)
    #def call(self, inputs):
        #latent_vectors, labels = inputs
        #return self.generator([latent_vectors, labels])
    '''

    
    # New train_step() function: interchange the order of the training of discriminator and generator
    '''
    
    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))


        if self.discriminator_train:
        
            self.discriminator_train = False

            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            # Update metrics: discriminator
            self.d_loss_metric.update_state(d_loss)

        else:
        
            self.discriminator_train = True

            # Sample random points in the latent space
            #random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Assemble labels that say "all real images"
            misleading_labels = tf.zeros((batch_size, 1))

            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            with tf.GradientTape() as tape:
                predictions = self.discriminator(self.generator(random_latent_vectors))
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Update metrics: generator
            self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
    '''


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, folder_path, num_img=3, latent_dim=100):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.folder_path = folder_path
        return
    
    def on_epoch_end(self, epoch, logs=None):
        #create folder if not exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images += 1
        generated_images *= 127.5
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img_path = os.path.join(self.folder_path, "generated_img_%03d_%d.png" % (epoch, i))

            img.save(img_path)


        if epoch % 2 == 0:

            ckpt_path = os.path.join('modelCheckPoint', 
                                     "ckpt_epoch_{epoch:04d}--g_loss_{g_loss:.4f}--d_loss_{d_loss:.4f}.h5".format(epoch=epoch, 
                                                                                                                  g_loss= logs['g_loss'], 
                                                                                                                  d_loss= logs['d_loss']))
            self.model.generator.save(ckpt_path)


        return


'''
def getModelCheckPoint(folder, save_freq='epoch', monitor='g_loss', save_weights_only=True, period=2):
    #create folder if not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    ckpt_path = os.path.join(folder, "ckpt_{epoch:04d}--{g_loss:.4f}--{d_loss:.4f}.h5")

    return ModelCheckpoint(ckpt_path, save_freq=save_freq, monitor=monitor, save_weights_only=save_weights_only, period=period)
'''