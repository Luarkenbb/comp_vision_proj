import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape, Input, Embedding, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import Model

def build_generator(latent_dim, num_classes = 10):
    in_label = Input(shape=(1,))
    li = Embedding(num_classes, 50)(in_label)
    ##remarks from discriminator
    # 7 * 7 * 64
    li = Dense(7 * 7)(li)
    li = Reshape((7, 7, 1))(li)

    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 64 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 64))(gen)

    # merge image gen and label input
    merge = Concatenate()([gen, li])
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)

    model = Model([in_lat, in_label], out_layer, name='generator')

    return model

def build_discriminator(img_shape = (28,28,1), num_classes = 10):
    #label input
    in_label = Input(shape=(1,))
    #embedding for categorical input
    li = Embedding(num_classes, 50)(in_label)
    li = Dense(img_shape[0]*img_shape[1])(li)
    # reshape to additional C
    li = Reshape((img_shape[0], img_shape[1], 1))(li)

    #image input
    in_image = Input(shape=img_shape)
    #concat li and in_image
    merge = Concatenate()([in_image, li])

    #784->128->64->32->1
    #fe = Conv2DTranspose(128, strides=(2,2), padding='same')(merge)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.3)(fe)

    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.3)(fe)

    fe = Flatten()(fe)
    fe = Dense(1, activation='sigmoid')(fe)

    model = Model([in_image, in_label], fe, name='discriminator')
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    print("test area")
    latent_dim = 100
    generator = build_generator(latent_dim)
    generator.summary()
    discriminator = build_discriminator()
    discriminator.summary()
    print("test area end")