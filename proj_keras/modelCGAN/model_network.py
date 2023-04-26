import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape, Input, Embedding, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import Model

def build_generator(latent_dim, num_classes = 10):
    generator_in_channel = latent_dim + num_classes

    model = Sequential(name = 'generator')
    model.add(Input(shape = (generator_in_channel,)))
    model.add(Dense(7 * 7 * generator_in_channel))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, generator_in_channel)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7, 7), padding="same", activation="sigmoid"))

    return model

def build_discriminator(img_shape = (28,28,1), num_classes = 10):
    discriminator_in_channels = num_classes + 1
    model = Sequential(name = 'discriminator')
    model.add(Input(shape = (28, 28, discriminator_in_channels)))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1))

    return model

if __name__ == '__main__':
    print("test area")
    latent_dim = 50
    generator = build_generator(latent_dim)
    generator.summary()
    discriminator = build_discriminator()
    discriminator.summary()
    print("test area end")