import tensorflow as tf

from modelCGAN.model_network import build_discriminator, build_generator
from modelCGAN.CGAN import CGAN

from tensorflow import keras
import pandas as pd

    
    


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
    
    #get x_train, y_train
    train_data = pd.read_csv('proj_keras/train.csv')
    X_train = train_data.iloc[:,1:].values.astype('float32')
    X_train = X_train.reshape(-1,28,28,1)
    X_train = (X_train - 127.5) / 127.5 # Normalize the images to [-1, 1]

    y_train = train_data.iloc[:,0].values.astype('float32')
    y_train = keras.utils.to_categorical(y_train, 10)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(256)

    gan.fit(
        dataset,
        epochs=50,
    )


    return
runGAN()