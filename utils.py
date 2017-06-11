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

# Image type definition in samples
CENTER_IMAGE = 0
LEFT_IMAGE = 1
RIGHT_IMAGE = 2

# Index definition for samples
INDEX_PATH =0
INDEX_STEER =1
INDEX_TYPE =2

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


def csv2samples(path, file):
    """
    Read a csv file and save into a list
    The format of sample : [path, steering, image type]
    """
    samples_raw = []
    with open(path + file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples_raw.append(line)

    samples = []
    for row in samples_raw:
        steering = float(row[3])
        samples.append([path + row[CENTER_IMAGE], steering, CENTER_IMAGE])
        samples.append([path + row[LEFT_IMAGE], steering, LEFT_IMAGE])
        samples.append([path + row[RIGHT_IMAGE], steering, RIGHT_IMAGE])
    return samples


def generator(path, samples, batch_size=32):
    """
    Generate a batch of training data
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            for image_index in range(0, 3):
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


def augment_steering(samples, steering_center=0, steering_left=0, steering_right=0):
    # Augment the data
    # [-25,25] -> [left,right]
    for i in range(0, len(samples)):
        if samples[i][INDEX_TYPE] == CENTER_IMAGE:
            samples[i][INDEX_STEER] = samples[i][INDEX_STEER] + steering_center
        elif samples[i][INDEX_TYPE] == LEFT_IMAGE:
            samples[i][INDEX_STEER] = samples[i][INDEX_STEER] + steering_left
        elif samples[i][INDEX_TYPE] == RIGHT_IMAGE:
            samples[i][INDEX_STEER] = samples[i][INDEX_STEER] + steering_right
    return samples


def generate_train_data(path):
    """
    Generate a fix batch of random training data from a path
    """
    samples = csv2samples(path + "track1-center/", "driving_log.csv")
    samples = augment_steering(samples, 0, 0.25,-0.25)

    samples1 = csv2samples(path + "track1-center1/", "driving_log.csv")
    samples1 = augment_steering(samples, -0.25,0,-1)
    samples= samples + samples1

    images = []
    measurements = []
    shuffle(samples)
    FILE_NUM = 7000
    print("Read ", len(samples), " files, only use first", FILE_NUM, " ones")
    samples = samples[0:FILE_NUM]
    print("Shuffle inputs.. ", samples[0:5])
    for sample in samples:
        image = cv2.imread(sample[0])
        images.append(image)
        measurements.append(sample[1])

    X_train = np.array(images)
    y_train = np.array(measurements)
    print("X_train = ", X_train.shape)
    print("y_train = ", y_train.shape)
    return X_train, y_train
