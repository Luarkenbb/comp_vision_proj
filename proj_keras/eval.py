import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import linalg

import pandas as pd
import numpy as np

from model.model_network import build_discriminator, build_generator, build_optimizer
from model.GAN import GAN, GANMonitor, getModelCheckPoint


def calculate_fid(real_images, generated_images):
    """
    Calculates the FID between two sets of images.

    Args:
        real_images: A NumPy array representing the real images. The shape should be (n, height, width, channels).
        generated_images: A NumPy array representing the generated images. The shape should be (n, height, width, channels).

    Returns:
        A float representing the FID between the two sets of images.
    """

    # Reshape images to (n, channels, height, width)
    real_images = np.transpose(real_images, (0, 3, 1, 2))
    generated_images = np.transpose(generated_images, (0, 3, 1, 2))

    # Compute mean and covariance matrices for real and generated images
    mu1, sigma1 = np.mean(real_images, axis=0), np.cov(real_images, rowvar=False)
    mu2, sigma2 = np.mean(generated_images, axis=0), np.cov(generated_images, rowvar=False)

    # Compute square root of product between covariance matrices
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Compute FID score
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid


def main():


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

    '''
    X_test = test_image.iloc[:,1:].values.astype('float32')
    X_test = X_test.reshape(-1,28,28,1)
    X_test = (X_test - 127.5) / 127.5 # Normalize the images to [-1, 1]
    '''

    # Load model
    gan = GAN(discriminator=build_discriminator(), generator=build_generator(100), latent_dim=100)
    #gan = GAN(discriminator=build_SNGAN_discriminator(), generator=build_SNGAN_generator(100), latent_dim=100)
    '''
    
    gan.compile(
        d_optimizer=build_optimizer(), 
        g_optimizer=build_optimizer(),
        loss_fn=tf.keras.losses.BinaryCrossentropy(),
    )
    '''
    gan.build(input_shape=(None, 28, 28, 1))
    #gan(tf.ones([1, 28, 28, 1]))
    gan.load_weights('modelCheckPoint_SNGAN_1_epoch_250/ckpt_0250--1.3375--0.4472.h5')


    # Generate fake images with the generator
    batch_size = 32
    num_batches = (len(X_test) + batch_size - 1) // batch_size  # avoid missing the last batch
    generated_images = []

    for i in range(num_batches):
        noise = tf.random.normal([batch_size, 100])
        batch_images = gan.generator.predict(noise)
        generated_images.append(batch_images)
    generated_images = np.concatenate(generated_images, axis=0)

    # Rescale generated images to [0, 1] range
    generated_images = (generated_images + 1) / 2.0

    # Evaluate the model using FID or another metric
    fid = calculate_fid(test_image, generated_images)
    print("FID: {:.3f}".format(fid))

    return

main()