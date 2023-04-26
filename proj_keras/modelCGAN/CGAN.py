import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

import os

class CGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, image_size=28, num_classes=10):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

        self.image_size = image_size
        self.num_classes = num_classes

    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        real_images, one_hot_labels = data
        


        return


