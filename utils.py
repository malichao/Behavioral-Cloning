import os
import cv2
import csv
import sklearn
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split

def csv2samples(file):
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
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for sample in batch_samples:
                files = [sample[0], sample[1], sample[2]]
                mea = float(sample[3])
                # Read left, center and right images
                # for file in files:
                #     file_name = path + file
                #     image = cv2.imread(file_name)
                #     images.append(image)
                #     measurements.append(mea)


                file_name = path + files[0]
                image = cv2.imread(file_name)
                images.append(image)
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
    FILE_NUM = 1200
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
