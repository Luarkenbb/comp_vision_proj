import pandas as pd
from tensorflow import keras


##import data set from train.csv
train_data = pd.read_csv('proj_keras/train.csv')
X_train = train_data.iloc[:,1:].values.astype('float32')
X_train = X_train.reshape(-1,28,28,1)
y_train = train_data.iloc[:,0].values.astype('int32')

print(X_train.shape, y_train.shape)

##img parameters
img_rows, img_cols, channels = 28, 28, 1
img_shape = (img_rows, img_cols, channels)
num_classes = 10
z_latent_dim = 100



