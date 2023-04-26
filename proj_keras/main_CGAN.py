import tensorflow as tf

from modelCGAN.model_network import build_discriminator, build_generator
from modelCGAN.CGAN import CGAN

from tensorflow import keras
import pandas as pd
import numpy as np
    
    


def check_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def runGAN():
    check_gpu()
    gan = CGAN(discriminator=build_discriminator(), generator=build_generator(100), latent_dim=100)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5), 
        g_optimizer=keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    # Scale the pixel values to [0, 1] range, add a channel dimension to
    # the images, and one-hot encode the labels.
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    all_labels = keras.utils.to_categorical(all_labels, 10)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    print(f"Shape of training images: {all_digits.shape}")
    print(f"Shape of training labels: {all_labels.shape}")
    
    """
    #get x_train, y_train
    train_data = pd.read_csv('proj_keras/train.csv')
    X_train = train_data.iloc[:,1:].values.astype('float32')
    X_train = np.reshape(X_train, (-1,28,28,1))
    X_train = (X_train - 127.5) / 127.5 # Normalize the images to [-1, 1]

    y_train = train_data.iloc[:,0].values.astype('int32')
    y_train = keras.utils.to_categorical(y_train, 10)

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    print(f"Shape of training images: {X_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")


    gan.fit(
        dataset,
        epochs=1,
        batch_size=256,
        steps_per_epoch=256,
        shuffle=True,
    )


    return
runGAN()