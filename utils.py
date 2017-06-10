import os
import cv2
import csv
import sklearn
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

def make_lenet():
    """
    Build a LeNet model using keras
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))
    # Crop 50 rows from the top, 20 rows from the bottom
    # Crop 0 columns from the left, 0 columns from the right
    model.add(Cropping2D(cropping=((60, 23), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def csv2samples(file):
    """
    Read a csv file and save into a list
    """
    samples = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def generator(path, samples, batch_size=32):
    """
    Generate a batch of training data
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            for image_index in range(0,3):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                measurements = []
                for sample in batch_samples:
                    file_name = path + sample[image_index]
                    image = cv2.imread(file_name)
                    images.append(image)
                    mea = float(sample[3])
                    measurements.append(mea)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(measurements)
                yield sklearn.utils.shuffle(X_train, y_train)


def generate_train_data(path):
    """
    Generate a fix batch of random training data from a path
    """
    csv_data = np.genfromtxt(path + "driving_log.csv", delimiter=',',
                             names=True, dtype=None)
    FILE_NUM = 1000
    print("Read ", len(csv_data), " files, only use first", FILE_NUM, " ones")
    measurements_raw = [t[3] for t in csv_data]
    file_names = [[t[0].decode('utf-8'), t[1].decode('utf-8'),
                   t[2].decode('utf-8')] for t in csv_data]
    measurements_raw = measurements_raw[0:FILE_NUM]
    file_names = file_names[0:FILE_NUM]

    images = []
    measurements = []

    selected = [i for i in range(0, len(file_names))]
    shuffle(selected)
    print("Shuffle inputs.. ", selected[0:10])
    for i in range(0, FILE_NUM):
        files = file_names[selected[i]]
        mea = measurements_raw[selected[i]]
        for file in files:
            if file[0] == ' ':    # Stupid space...
                file = file[1:]
            file_name = path + file
            image = cv2.imread(file_name)
            #image_flipped = np.fliplr(image)
            images.append(image)
            measurements.append(mea)

    X_train = np.array(images)
    y_train = np.array(measurements)
    print("X_train = ", X_train.shape)
    print("y_train = ", y_train.shape)
    return X_train, y_train
