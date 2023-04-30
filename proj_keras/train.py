import tensorflow as tf
from model.model_network import build_discriminator, build_generator, build_optimizer
#from model.GAN import GAN, GANMonitor, getModelCheckPoint
from model.GAN import GAN, GANMonitor
import pandas as pd
import os
    

def check_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def main():
    

    if not os.path.exists('./modelCheckPoint'):
        os.makedirs('./modelCheckPoint')


    gan = GAN(discriminator=build_discriminator(), generator=build_generator(100), latent_dim=100)
    gan.compile(
        d_optimizer=build_optimizer(), 
        g_optimizer=build_optimizer(),
        loss_fn=tf.keras.losses.BinaryCrossentropy(),
    )
    
    #get x_train, y_train
    train_data = pd.read_csv('proj_keras/train.csv')
    X_train = train_data.iloc[:,1:].values.astype('float32')
    X_train = X_train.reshape(-1,28,28,1)
    X_train = (X_train - 127.5) / 127.5 # Normalize the images to [-1, 1]

    
    '''
    
    gan.fit(
        X_train,
        epochs=250,
        callbacks=[getModelCheckPoint('./modelCheckPoint'),
                   GANMonitor('images',num_img=1, latent_dim=100)]
    )
    '''


    gan.fit(
        X_train,
        epochs=250,
        callbacks=[GANMonitor('images',num_img=1, latent_dim=100)]
    )




    return

main()