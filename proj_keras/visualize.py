import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2

train_df = pd.read_csv("/Users/josephyim/Documents/COMP/COMP4423 Computer Vision/Project/train.csv")
test_df = pd.read_csv("/Users/josephyim/Documents/COMP/COMP4423 Computer Vision/Project/test.csv")


# Preprocess the dataset
def preprocess(df):
    labels = df["label"].values
    images = df.drop("label", axis=1).values
    images = images.reshape(-1, 28, 28)
    return images, labels

train_images, train_labels = preprocess(train_df)
test_images, test_labels = preprocess(test_df)
#print("dataset preprocessed.")

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

#plot_images(train_images, train_labels)

# Plot n images of a specific label
def plot_label_images(images, labels, label_index, nrows=2, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4))
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            while labels[i] != label_index:
                i += 1
            axes[row, col].imshow(images[i], cmap="gray")
            axes[row, col].axis("off")
            i += 1
    plt.show()

# # Plot 10 images of trousers
# plot_label_images(train_images, train_labels, 1)

# # Plot 10 images of coats
# plot_label_images(train_images, train_labels, 4)

# # Plot 10 images of sandals
# plot_label_images(train_images, train_labels, 5)

# Plot 10 images of each label
for i in range(10):
    plot_label_images(train_images, train_labels, i)