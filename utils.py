import os
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
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
from keras.models import load_model

# Image type definition in samples
CENTER_IMAGE = int(0)
LEFT_IMAGE = int(1)
RIGHT_IMAGE = int(2)

# Index definition for samples
INDEX_PATH = 0
INDEX_STEER = 1
INDEX_TYPE = 2
INDEX_ID = 2

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80


def open_image(img_path):
    img = cv2.imread(img_path)
    return preprocess_image(img)


def preprocess_image(img):
    # [y1:y2, x1:x2]
    img = img[60:120, 0:320]
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return np.asarray(img)


def make_preprocess_layers():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5,
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    # Crop 50 rows from the top, 20 rows from the bottom
    # Crop 0 columns from the left, 0 columns from the right
    # (90,320,3)
    # model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
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
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model


# def make_nvidia():
#     """
#     Creates nVidea Autonomous Car Group model
#     """
#     model = make_preprocess_layers()
#     model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
#     model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
#     model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(100))
#     # model.add(Dropout(.5))
#     model.add(Dense(50))
#     # model.add(Dropout(.5))
#     model.add(Dense(10))
#     # model.add(Dropout(.5))
#     model.add(Dense(1))

    # return model


def make_nvidia():
    model = make_preprocess_layers()
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
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
        samples_l.append([path + row[LEFT_IMAGE], steering, LEFT_IMAGE])
        samples_r.append([path + row[RIGHT_IMAGE], steering, RIGHT_IMAGE])
    samples = samples_c + samples_l + samples_r
    print("Read ", len(samples))
    return samples


def random_shear(image, steering, shear_range=100):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    #    print('dx',dx)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering += dsteering

    return image, steering


def augment_data(image, steering):
    if random() < 0.5:
        steering=-steering
        image=np.fliplr(image)
    # if random() < .5:
    #     image,steering= random_shear(image,steering)
    return image, steering


def augment_data_test(image, steering):
    if random() < 0.5:
        steering = -steering
        image = np.fliplr(image)
    # if random() < .5:
    #     image,steering= random_shear(image,steering)
    return image, steering


def generator(samples, batch_size=32, debug=False):
    """
    Generate a batch of training data
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            images = []
            measurements = []
            batch_samples = samples[offset:offset + batch_size]
            for batch_sample in batch_samples:
                image = open_image(batch_sample[INDEX_PATH])
                mea = batch_sample[INDEX_STEER]
                image, mea = augment_data(image, mea)
                images.append(image)
                measurements.append(mea)

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
        if abs(sample[INDEX_STEER]) <= 0.1:
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


def plot_steering_over_time(samples, plot=True):
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
    plt.grid(True)
    plt.show()


def plot_steering_distribution(samples, plot=True):
    if plot == False:
        return
    num = int(200)
    bars = [i * 0.1 for i in range(int(-num / 2), int(num / 2), 1)]
    counts = [[0] * num, [0] * num, [0] * num]
    for sample in samples:
        index = int((sample[INDEX_STEER] + 1.0) * num / 2)
        # index = min(max(0, index), num - 1)
        type_ = sample[INDEX_TYPE]
        counts[type_][index] = counts[type_][index] + 1
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.bar(bars, counts[0], color='r', label='center')
    plt.subplot(1, 3, 2)
    plt.bar(bars, counts[1], color='g', label='left')
    plt.subplot(1, 3, 3)
    plt.bar(bars, counts[2], color='b', label='right')
    plt.show()


def running_mean(x, N):
    # modes = ['full', 'same', 'valid']
    # x_ = [s * 1.2 for s in x]
    x_ = x
    return np.convolve(x_, np.ones((N,)) / N, mode='same')


def filter_steering(samples, N, plot=False):
    steerings = [sample[INDEX_STEER] for sample in samples]
    new_steerings = running_mean(steerings, N)
    new_steerings = running_mean(new_steerings, int(N / 2))
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
        # plot_length = int(len(samples) / 3)
        plot_length = len(samples)
        steerings = [sample[INDEX_STEER] for sample in samples[0:plot_length]]
        steerings_f = [sample[INDEX_STEER]
                       for sample in filtered_samples[0:plot_length]]
        plt.figure(figsize=(30, 8))
        plt.plot(steerings, 'r', steerings_f, 'b')
        plt.grid(True)
        plt.show()

    return filtered_samples


def preprocess_samples(samples, shuffule_data=True, plot=False):
    samples = filter_steering(samples, 100, plot)
    # samples = reduce_straight_steering(samples, 3, plot)
    samples = augment_steering(samples, 0, 0.08, -0.08)
    plot_steering_over_time(samples, plot)
    plot_steering_distribution(samples, plot)
    if shuffule_data:
        shuffle(samples)
    return samples


def load_data(path, plot=False):
    samples = csv2samples(path)
    samples = preprocess_samples(samples, plot)
    return samples


def load_correction(path, steering=-0.15):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    samples = []
    for file in files:
        samples.append([file, steering, CENTER_IMAGE])
    shuffle(samples)
    return samples


def generate_train_data(train_samples, validation_samples):
    # train_samples, validation_samples = train_test_split(
    #     samples, test_size=0.2)

    train_size = len(train_samples)
    validation_size = len(validation_samples)
    print("train_samples      = ", train_size)
    print("validation_samples = ", validation_size)

    BATCH_SIZE = 32
    # if train_size > BATCH_SIZE:
    #     train_size = int(train_size / BATCH_SIZE) * BATCH_SIZE
    # if validation_size > BATCH_SIZE:
    #     validation_size = int(validation_size / BATCH_SIZE) * BATCH_SIZE
    train_samples = train_samples[0:train_size]
    validation_samples = validation_samples[0:validation_size]
    print("Resize sample size to avoid Keras warning")
    print("train_samples      = ", train_size)
    print("validation_samples = ", validation_size)
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(
        validation_samples, batch_size=BATCH_SIZE)

    return train_generator, validation_generator, len(train_samples), len(validation_samples)


def predict_samples(samples,model,step=25):
    results = []
    gts = []
    pds = []
    i = 0
    for sample in samples:
        image_array = open_image(sample[INDEX_PATH])
        steering = float(model.predict(
            image_array[None, :, :, :], batch_size=1))
        delta = steering - sample[INDEX_STEER]
        gts.append(sample[INDEX_STEER])
        pds.append(steering)
        if i % step == 0:
            results.append(
                [i, image_array, sample[INDEX_STEER], steering, delta])
        i = i + 1
    return results

def test_model(model_path, test_path, plot_image=False):
    """
    Test the model against a test data set and plot the result
    """
    model = load_model(model_path)
    samples = csv2samples(test_path)
    samples = preprocess_samples(samples,shuffule_data=False)
    samples_new = []
    # for sample in samples:
    #     if sample[INDEX_TYPE] == CENTER_IMAGE:
    #         samples_new.append(sample)
    samples_new = samples

    if len(samples_new) > 500:
        samples = [samples_new[i] for i in range(0, len(samples_new), 2)]
    else:
        samples = samples_new
    print("Testing prediction on {} images".format(len(samples)))

    results = []
    gts = []
    pds = []
    i, step = 0, 25
    for sample in samples:
        image_array = open_image(sample[INDEX_PATH])
        steering = float(model.predict(
            image_array[None, :, :, :], batch_size=1))
        delta = steering - sample[INDEX_STEER]
        gts.append(sample[INDEX_STEER])
        pds.append(steering)
        if i % step == 0:
            results.append(
                [i, image_array, sample[INDEX_STEER], steering, delta])
        i = i + 1
        # print("{} {} {}".format(sample[INDEX_PATH],sample[INDEX_STEER],steering,delta))

    print("Plotting results ", len(results))

    plt.figure(figsize=(30, 8))
    plt.plot(gts, 'r--', pds, 'g')
    plt.rc('grid', linestyle="--", color='grey')
    plt.grid(True, 'both')
    plt.show()

    if plot_image:
        font_size=30
        fig = plt.figure(figsize=(50, 50))
        col = 6
        row = int(len(results) / col) + 1
        print("Result: [Ground truth | Prediction | Error]")
        i = 1
        for data in results:
            plt.subplot(row, col, i)
            plt.imshow(data[1])
            plt.axis('off')
            gt = data[2] * 25
            pd = data[3] * 25
            error = data[4] / 2.0 * 100
            if abs(error) > 2.0 and gt*pd<0:
                title = plt.title("{}: {:2.1f},{:2.1f},{:2.1f}%".format(
                            data[0], gt, pd, error), fontsize=font_size,fontweight='bold',color='red')
            elif abs(error) > 2.0 :
                title = plt.title("{}: {:2.1f},{:2.1f},{:2.1f}%".format(
                            data[0], gt, pd, error), fontsize=font_size,color='red')
            else:
                title = plt.title("{}: {:2.1f},{:2.1f},{:2.1f}%".format(
                            data[0], gt, pd, error), fontsize=font_size)
            i = i + 1
        plt.tight_layout()
        plt.show()
    print("Test completed")
