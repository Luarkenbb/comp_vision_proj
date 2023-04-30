import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from scipy import linalg

#import PIL
#from PIL import Image

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import load_model


np.random.seed(42)
tf.random.set_seed(42)


def plot_images(generated_images, plot_name):

    fig, ax = plt.subplots(3, 3, figsize=(5,5))
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout(pad=0)

    for i in range(3):
        for j in range(3):
            #ax[i, j].imshow(cv2.cvtColor(keras.preprocessing.image.array_to_img(generated_images[i*3 + j]), cv2.COLOR_BGR2GRAY))
            ax[i, j].imshow(keras.preprocessing.image.array_to_img(generated_images[i*3 + j]), cmap='gray')
            ax[i, j].xaxis.set_visible(False)
            ax[i, j].yaxis.set_visible(False)
    #plt.show()

    plt.savefig(f'{plot_name}.png')


def main():

    '''
    # Load test set
    test_data = pd.read_csv('proj_keras/test.csv')
    #test_data = test_data.to_numpy()

    test_image = test_data.iloc[:,1:].values.astype('uint8')
    test_image = test_image.reshape(-1, 28, 28, 1)

    #X_test = test_data[:,1:]
    X_test = test_data.iloc[:,1:].values.astype('float32')
    #X_test = X_test.to_numpy()
    X_test = X_test.reshape(-1, 28, 28, 1)
    #X_test = X_test.astype('float32')
    X_test = (X_test - 127.5) / 127.5 # Normalize the images to [-1, 1]

    
    X_test = test_image.iloc[:,1:].values.astype('float32')
    X_test = X_test.reshape(-1,28,28,1)
    X_test = (X_test - 127.5) / 127.5 # Normalize the images to [-1, 1]
    '''



    ##########################################################
    model_type = 'FCGAN'
    #model_type = 'DCGAN'
    #model_type = 'SNGAN'

    weights = 'epoch_50'
    #weights = 'epoch_100'
    #weights = 'epoch_150'
    #weights = 'epoch_200'
    #weights = 'epoch_250'

    model_path = os.path.join('weights', model_type, weights+'.h5')

    # Load model
    generator = load_model(model_path)

    # Generate images
    random_latent_vectors = tf.random.normal(shape=(9, 100))
    generated_images_2 = generator(random_latent_vectors)

    generated_images_2 = (generated_images_2 + 1) / 2.0

    plot_name = model_type+'_'+weights

    plot_images(generated_images_2, plot_name)


    ##########################################################

    return

main()