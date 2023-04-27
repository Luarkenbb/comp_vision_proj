# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2


# Load the dataset from CSV files
train_df = pd.read_csv("/Users/josephyim/Documents/COMP/COMP4423 Computer Vision/Project/train.csv")
test_df = pd.read_csv("/Users/josephyim/Documents/COMP/COMP4423 Computer Vision/Project/test.csv")
print("dataset loaded.")

# Preprocess the dataset
def preprocess(df):
    labels = df["label"].values
    images = df.drop("label", axis=1).values
    images = images.reshape(-1, 28, 28)
    return images, labels

train_images, train_labels = preprocess(train_df)
test_images, test_labels = preprocess(test_df)
print("dataset preprocessed.")

# Visualize example images
label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def plot_images(images, labels, nrows=2, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(label_names[labels[i]])
        ax.axis("off")
        # Save a png of each 28x28 image at "vis_images/img_i.png"
        cv2.imwrite("vis_images/img_{}.png".format(i), images[i])
    plt.show()

plot_images(train_images, train_labels)
