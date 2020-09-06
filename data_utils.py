# This script contains functions for loading the dataset and labels from the MNIST Dataset

# Importing the required libraries
import gzip
import random
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_dir, verbose = False):
    """This function loads the data for the MNIST Dataset"""

    # MNIST Image Size
    IMAGE_SIZE = 28

    # Unzipping the dataset and reading it
    with gzip.open(data_dir, mode = "r") as f:

        f.read(16)
        buf = f.read()

        # Using np.frombuffer() function to decode the data
        data = np.frombuffer(buf, dtype = np.uint8).astype(np.float32)

        # Reshaping it to dimension(num_imgs, IMAGE_SIZE, IMAGE_SIZE, 1)
        data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    # If verbose, then the loaded data is randomly visualized
    if verbose:
        idx = random.randint(0, len(data))

        # Removing Channel Dimension
        image = np.asarray(data[idx]).squeeze()

        plt.imshow(image, cmap = "gray")
        plt.show()

    return data

def load_labels(labels_dir, verbose = False):
    """This function loads the labels for the MNIST Dataset"""

    # Unzipping the dataset and reading it
    with gzip.open(labels_dir, mode = "r") as f:
        f.read(8)
        buf = f.read()

        # Using np.frombuffer() function to decode the data
        data = np.frombuffer(buf, dtype = np.uint8).astype(np.int64)

    # If verbose, then the first 10 values of the loaded labels is printed
    if verbose:
        print(data[:10])

    return data

if __name__ == "__main__":
    # Loading the data
    train_data_dir = "data/train-images-idx3-ubyte.gz"
    train_data = load_data(train_data_dir, verbose = True)

    # Loading the label
    train_labels_dir = "data/train-labels-idx1-ubyte.gz"
    train_labels = load_labels(train_labels_dir, verbose = True)

    # Randomly visualizing the loaded dataset and it's corresponding label
    idx = random.randint(0, len(train_data))
    plt.imshow(train_data[idx].squeeze(), cmap = "gray")
    plt.title(train_labels[idx])
    plt.show()