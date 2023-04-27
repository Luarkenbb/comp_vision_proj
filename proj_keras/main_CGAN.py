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
    


    return
runGAN()