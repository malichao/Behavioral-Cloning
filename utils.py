import os
import cv2
import csv
import sklearn
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, ELU
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
# Image type definition in samples
CENTER_IMAGE = int(0)
LEFT_IMAGE = int(1)
RIGHT_IMAGE = int(2)

# Index definition for samples
INDEX_PATH = 0
INDEX_STEER = 1
INDEX_TYPE = 2

def make_preprocess_layers():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))
    # Crop 50 rows from the top, 20 rows from the bottom
    # Crop 0 columns from the left, 0 columns from the right
    model.add(Cropping2D(cropping=((60, 23), (0, 0)), input_shape=(160, 320, 3)))
    return model

def make_lenet():
    """
    Build a LeNet model using keras
    """
    model = make_preprocess_layers()
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def make_lenet2():
    """
    Build a LeNet model using keras
    """
    model = make_preprocess_layers()
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(Dropout(.5))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(Dropout(.5))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def make_commaai():
    model = make_preprocess_layers()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model

def make_commaai2():
    model = make_preprocess_layers()
    model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode='same',
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(16, 5, 5, subsample=(4, 4), border_mode="same",
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same",
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", 
                            init = 'he_normal'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))

    return model

def make_nvidia():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = make_preprocess_layers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
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
        if abs(steering) > 0.01:
            samples.append([path + row[LEFT_IMAGE], steering, LEFT_IMAGE])
            samples.append([path + row[RIGHT_IMAGE], steering, RIGHT_IMAGE])
    print("Read ", len(samples))
    return samples


def generator(samples, batch_size=32):
    """
    Generate a batch of training data
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            images = []
            measurements = []
            batch_samples = samples[offset:offset + batch_size]
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[INDEX_PATH])
                images.append(image)
                mea = batch_sample[INDEX_STEER]
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


def reduce_straight_steering(samples, keep_ratio=0.5):
    keep_num = len(samples) * keep_ratio
    new_samples = []
    straight_count = 0
    for sample in samples:
        if abs(sample[INDEX_STEER]) < 0.01:
            if straight_count < keep_num:
                new_samples.append(sample)
                straight_count += 1
        else:
            new_samples.append(sample)
    return new_samples


def plot_steering_distribution(samples):
    num = int(200)
    bars = [i * 0.1 for i in range(int(-num/2), int(num/2), 1)]
    counts = [[0] * num,[0] * num,[0] * num]
    for sample in samples:
        index = int(int(sample[INDEX_STEER] * num/2) + num/2)
        index = min(max(0, index), num - 1)
        type_ = sample[INDEX_TYPE]
        counts[type_][index] = counts[type_][index] + 1
    plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    plt.bar(bars, counts[0],color='r', label='center')
    plt.subplot(1,3,2)
    plt.bar(bars, counts[1],color='g', label='left')
    plt.subplot(1,3,3)
    plt.bar(bars, counts[2],color='b', label='right')
    plt.show()

def plot_steering_over_time(samples):
    steering = []
    for sample in samples:
        if sample[INDEX_TYPE]==CENTER_IMAGE:
            steering.append(sample[INDEX_STEER])
    plt.figure(figsize=(30,8))
    plt.plot(steering,'r')
    plt.show()

# def plot_steering_distribution(samples):
#     bars = [i * 0.1 for i in range(-100, 100, 1)]
#     counts = [0] * 200
#     for sample in samples:
#         index = int(sample[INDEX_STEER] * 100) + 100
#         index = min(max(0, index), len(counts) - 1)
#         counts[index] = counts[index] + 1
#     center=plt.bar(bars, counts,color='r', label='center')
#     plt.show()
    


def preproccess_samples(samples, file_num=10000):
    shuffle(samples)
    # plot_steering_distribution(samples)
    samples = reduce_straight_steering(samples, 0.3)
    # plot_steering_distribution(samples)
    samples = augment_steering(samples, 0, 0.25, -0.25)

    # samples1 = csv2samples(path + "track1-center/", "driving_log.csv")
    # samples1 = augment_steering(samples1, 0, 0.25,-0.25)

    # samples= samples + samples1
    # shuffle(samples)
    # # plot_steering_distribution(samples)
    # samples = reduce_straight_steering(samples,0.3)
    # # plot_steering_distribution(samples)

    if len(samples) > file_num:
        print("Read ", len(samples), " files, only use first", file_num, " ones")
        samples = samples[0:file_num]
    return samples


def generate_train_data2(path, plot =False):
    samples = csv2samples(path + "track1-center1/", "driving_log.csv")
    if plot :
        plot_steering_distribution(samples)
        plot_steering_over_time(samples)
    # samples1 = csv2samples(path + "track1-center1/", "driving_log.csv")
    # plot_steering_distribution(samples1)
    # samples = samples + samples1
    samples = preproccess_samples(samples)
    if plot:
        plot_steering_distribution(samples)
    train_samples, validation_samples = train_test_split(
        samples, test_size=0.2)
    train_size = len(train_samples)
    validation_size = len(validation_samples)
    print("Read ", len(samples), " samples")
    print("train_samples      = ", train_size)
    print("validation_samples = ", validation_size)

    BATCH_SIZE = 32
    train_size = int(train_size / BATCH_SIZE) * BATCH_SIZE
    validation_size = int(validation_size / BATCH_SIZE) * BATCH_SIZE
    train_samples = train_samples[0:train_size]
    validation_samples = validation_samples[0:validation_size]
    print("Resize sample size to avoid Keras warning")
    print("train_samples      = ", train_size)
    print("validation_samples = ", validation_size)
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(
        validation_samples, batch_size=BATCH_SIZE)

    print("samples_per_epoch", len(train_samples))
    return train_generator, validation_generator, len(train_samples), len(validation_samples)


def generate_train_data(path):
    """
    Generate a fix batch of random training data from a path
    """
    samples = csv2samples(path + "track1-center/", "driving_log.csv")
    samples = preproccess_samples(samples)

    images = []
    measurements = []
    for sample in samples:
        image = cv2.imread(sample[0])
        images.append(image)
        measurements.append(sample[1])

        # Flip the image
        images.append(np.fliplr(image))
        measurements.append(-sample[1])

    X_train = np.array(images)
    y_train = np.array(measurements)
    print("X_train = ", X_train.shape)
    print("y_train = ", y_train.shape)
    return X_train, y_train
