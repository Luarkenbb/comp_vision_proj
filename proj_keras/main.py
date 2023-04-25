import tensorflow as tf
from model.GAN import GAN


    
    


def check_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def main():
    check_gpu()
    gan = GAN()
    gan.train_from_csv('proj_keras/train.csv',epochs=5000, batch_size=256)
    #gan.save_model('model_test')
    #gan.load_model('model_test')
    #gan.genSampleFromGenerator(isShow=True, samples=50)

    return

main()