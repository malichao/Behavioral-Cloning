import os
import cv2
import csv
import sklearn
import numpy as np
from random import shuffle, random
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
    model.add(Cropping2D(cropping=((60, 36), (0, 0)), input_shape=(160, 320, 3)))
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


def make_commaai():
    # model = Sequential()
    # model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))
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


def make_nvidia():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = make_preprocess_layers()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    # model.add(Dropout(.5))
    model.add(Dense(50))
    # model.add(Dropout(.5))
    model.add(Dense(10))
    # model.add(Dropout(.5))
    model.add(Dense(1))

    return model

def csv2samples(path):
    """
    Read a csv file and save into a list
    The format of sample : [path, steering, image type]
    The samples are appended in center, left, right order so that we could use
    filter to filter the steerings
    """
    file = "driving_log.csv"
    samples_raw = []
    with open(path + file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples_raw.append(line)

    samples = []
    samples_l = []
    samples_c = []
    samples_r = []
    for row in samples_raw:
        steering = float(row[3])
        samples_c.append([path + row[CENTER_IMAGE], steering, CENTER_IMAGE])
        # if abs(steering) > 0.01:
        samples_l.append([path + row[LEFT_IMAGE], steering, LEFT_IMAGE])
        samples_r.append([path + row[RIGHT_IMAGE], steering, RIGHT_IMAGE])
    samples = samples_c + samples_l + samples_r
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
                mea = batch_sample[INDEX_STEER]
                # if random()<0.5:
                #     mea=-mea
                #     image=np.fliplr(image)
                images.append(image)
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


def reduce_straight_steering(samples, step=0, plot=False):
    """
    Down sample the straight steering data every N steps. If step =0, it does nothing
    """
    if step == 0:
        return samples

    samples_new = []
    count = 0

    for sample in samples:
        if abs(sample[INDEX_STEER]) < 0.01:
            count = count + 1
            if count >= step:
                count = 0
                samples_new.append(sample)
        else:
            count = 0
            samples_new.append(sample)

    print("Samples are reduce from ", len(samples), " to ", len(samples_new))
    plot_steering_over_time(samples_new, plot)
    return samples_new


def plot_steering_over_time(samples, plot=False):
    if plot == False:
        return
    steerings = [[], [], []]
    for sample in samples:
        if sample[INDEX_TYPE] == CENTER_IMAGE:
            steerings[0].append(sample[INDEX_STEER])
        elif sample[INDEX_TYPE] == LEFT_IMAGE:
            steerings[1].append(sample[INDEX_STEER])
        else:
            steerings[2].append(sample[INDEX_STEER])

    plt.figure(figsize=(30, 8))
    plt.plot(steerings[0], 'r', steerings[1], 'g', steerings[2], 'b')
    plt.show()


def plot_steering_distribution(samples):
    num = int(200)
    bars = [i * 0.1 for i in range(int(-num / 2), int(num / 2), 1)]
    counts = [[0] * num, [0] * num, [0] * num]
    for sample in samples:
        index = int(int(sample[INDEX_STEER] * num / 2) + num / 2)
        index = min(max(0, index), num - 1)
        type_ = sample[INDEX_TYPE]
        counts[type_][index] = counts[type_][index] + 1
    plt.subplot(1, 3, 1)
    plt.bar(bars, counts[0], color='r', label='center')
    plt.subplot(1, 3, 2)
    plt.bar(bars, counts[1], color='g', label='left')
    plt.subplot(1, 3, 3)
    plt.bar(bars, counts[2], color='b', label='right')
    plt.show()


def running_mean(x, N):
    # modes = ['full', 'same', 'valid']
    x_ = [s * 1.1 for s in x]
    # x_ = x
    return np.convolve(x_, np.ones((N,)) / N, mode='same')


def filter_steering(samples, N, plot=False):
    steerings = [sample[INDEX_STEER] for sample in samples]
    new_steerings = running_mean(steerings, N)
    new_steerings = running_mean(new_steerings, int(N/2))
    # new_steerings = running_mean(new_steerings, int(N/4))
    # new_steerings = running_mean(new_steerings, int(N/8))

    # new_steerings = running_mean(steerings, N)
    filtered_samples = []
    # print("{} -> {}".format(len(steerings),len(new_steerings)))
    for sample, steering in zip(samples, new_steerings):
        sample_ = sample[:]
        sample_[INDEX_STEER] = steering
        filtered_samples.append(sample_)

    if plot:
        plot_length = int(len(samples) / 3)
        steerings = [sample[INDEX_STEER] for sample in samples[0:plot_length]]
        steerings_f = [sample[INDEX_STEER]
                       for sample in filtered_samples[0:plot_length]]
        plt.figure(figsize=(30, 8))
        plt.plot(steerings, 'r', steerings_f, 'b')
        plt.show()

    return filtered_samples


def preproccess_samples(samples, plot=False):
    # samples = reduce_straight_steering(samples, 2, plot)
    samples = filter_steering(samples, 32, plot)
    samples = augment_steering(samples, 0, 0.1, -0.1)
    plot_steering_over_time(samples, plot)
    shuffle(samples)
    return samples

def load_data(path, plot=False):
    samples = csv2samples(path)
    samples = preproccess_samples(samples, plot)
    return samples

def generate_train_data(samples):
    train_samples, validation_samples = train_test_split(
        samples, test_size=0.2)
    train_size = len(train_samples)
    validation_size = len(validation_samples)
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

    return train_generator, validation_generator, len(train_samples), len(validation_samples)
