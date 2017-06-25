import os
import shutil
import cv2
import csv
import math
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

STEERING_GAIN = 1.05


def open_image(img_path):
    """
    Read a image from a file and convert it to a numpy array
    Note!! The default format of cv2 image is BGR not RGB
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img)


def preprocess_image(img):
    """
    Crop and resize the image
    """
    # [y1:y2, x1:x2]
    img = img[60:120, 0:320]
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return np.asarray(img)


def make_preprocess_layers():
    """
    Build first layer of the network, normalize the pixels to [-1,1]
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5,
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
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
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(.5))
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
    """
    Randomly augment the images
    """
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

def plot_generator(gen):
    """
    A function to plot the training data in one batch
    """
    images,steerings = next(gen)
    print("Generated a fresh batch")
    fig = plt.figure(figsize=(20, 20))
    col = 8
    row = int(len(images) / col) + 1
    i = 1
    for img,steering in zip(images,steerings):
        plt.subplot(row, col, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title("{}: {:2.2f}".format(i, steering), fontsize=20)
        i = i + 1
    plt.tight_layout()
    plt.show()

def offset_steering(samples, steering_center=0, steering_left=0, steering_right=0):
    # Augment the data
    # [-25,25] -> [left,right]
    for i in range(0, len(samples)):
        if samples[i][INDEX_TYPE] == CENTER_IMAGE:
            samples[i][INDEX_STEER] = samples[i][INDEX_STEER]
        elif samples[i][INDEX_TYPE] == LEFT_IMAGE:
            samples[i][INDEX_STEER] = samples[i][INDEX_STEER] + offset
        elif samples[i][INDEX_TYPE] == RIGHT_IMAGE:
            samples[i][INDEX_STEER] = samples[i][INDEX_STEER] - offset
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
        if abs(sample[INDEX_STEER]) <= 0.01:
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
    line1, = plt.plot(steerings[0], 'r', label='center')
    line2, = plt.plot(steerings[1], 'g', label='left')
    line3, = plt.plot(steerings[2], 'b', label='right')
    plt.title("Steering for different images")
    plt.legend(handles=[line1,line2,line3], loc=1)
    plt.grid(True)
    plt.show()


def plot_steering_distribution(samples, plot=True):
    if plot == False:
        return
    steerings = [[],[],[]]
    bin_size = 50
    for sample in samples:
        type_ = sample[INDEX_TYPE]
        steerings[type_].append(sample[INDEX_STEER])

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.hist(steerings[0], bins=bin_size, color='r')
    plt.title("center")
    plt.subplot(1, 3, 2)
    plt.hist(steerings[1], bins=bin_size, color='g')
    plt.title("left")
    plt.subplot(1, 3, 3)
    plt.hist(steerings[2], bins=bin_size, color='b')
    plt.title("right")
    plt.show()


def running_mean(x, N, gain=1.0):
    """
    Smooth the steering data. Since filtering reduce the magnitude of the
    steering, we need to amplify the steering by some factor
    """
    N = int(N)
    x_ = [s * gain for s in x]
    # modes = ['full', 'same', 'valid']
    return np.convolve(x_, np.ones((N,)) / N, mode='same')


def filter_steering(samples, N, plot=False):
    """
    Filter the steering data in the samples
    """
    steerings = [sample[INDEX_STEER] for sample in samples]
    new_steerings = []
    
    new_steerings = running_mean(steerings, 2,1.5)
    new_steerings = running_mean(new_steerings, 8,1.0)
    new_steerings = running_mean(new_steerings, 16,1.0)
    new_steerings = running_mean(new_steerings, 32,1.0)
    new_steerings = running_mean(new_steerings, 24,1.0)

    # Track1 parameters, smooth
    new_steerings = running_mean(steerings, 2,1.5)
    new_steerings = running_mean(new_steerings, 8,1.0)
    new_steerings = running_mean(new_steerings, 16,1.0)
    new_steerings = running_mean(new_steerings, 32,1.0)

    # Track2 parameters,aggressive
    # new_steerings = running_mean(steerings, 2,1.1)
    # new_steerings = running_mean(new_steerings, 4,1.0)
    # new_steerings = running_mean(new_steerings, 6,1.0)
    # new_steerings = running_mean(new_steerings, 8,1.0)
    # new_steerings = running_mean(new_steerings, 16,1.0)


    filtered_samples = []

    for sample, steering in zip(samples, new_steerings):
        sample_ = sample[:]
        sample_[INDEX_STEER] = steering
        filtered_samples.append(sample_)

    if plot:
        plot_length = int(len(samples) / 3)
        # plot_length = len(samples)
        steerings = [sample[INDEX_STEER] for sample in samples[0:plot_length]]
        steerings_f = [sample[INDEX_STEER]
                       for sample in filtered_samples[0:plot_length]]
        plt.figure(figsize=(30, 8))
        line1, = plt.plot(steerings, 'r', label='Raw steerings')
        line2, =  plt.plot(steerings_f, 'b', label='Filtered steerings')
        plt.legend(handles=[line1,line2], loc=1)
        plt.grid(True)
        plt.show()

    return filtered_samples


def preprocess_samples(samples, shuffule_data=True, plot=False):
    """
    Filter and offset the steering data
    """
    samples = filter_steering(samples, 128, plot)
    samples = reduce_straight_steering(samples, 10, plot)
    samples = offset_steering(samples, 0, 0.25, -0.25)
    plot_steering_over_time(samples, plot)
    plot_steering_distribution(samples, plot)
    if shuffule_data:
        shuffle(samples)
    return samples


def load_data(path, plot=False):    
    """
    Load and preprocess the data for trainning
    """
    samples = csv2samples(path)
    samples = preprocess_samples(samples, True, plot)
    return samples

def generate_train_data(train_samples, validation_samples):
    train_size = len(train_samples)
    validation_size = len(validation_samples)
    print("train_samples      = ", train_size)
    print("validation_samples = ", validation_size)

    BATCH_SIZE = 32
    train_samples = train_samples[0:train_size]
    validation_samples = validation_samples[0:validation_size]
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(
        validation_samples, batch_size=BATCH_SIZE)

    return train_generator, validation_generator, len(train_samples), len(validation_samples)

from sklearn.metrics import mean_squared_error
def test_model(model_path, test_path, plot_image=False):
    """
    Test the model against a test data set and plot the result
    """
    model = load_model(model_path)
    samples = csv2samples(test_path)
    samples = preprocess_samples(samples, shuffule_data=False)
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

    mse = mean_squared_error(gts,pds)
    print("MSE: {:.6f}".format(mse))

    print("Plotting results ", len(results))

    plt.figure(figsize=(30, 8))
    plt.plot(gts, 'r--', pds, 'g')
    plt.rc('grid', linestyle="--", color='grey')
    plt.grid(True, 'both')
    plt.show()

    if plot_image:
        font_size = 30
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
            if abs(error) > 2.0 and gt * pd < 0:
                title = plt.title("{}: {:2.1f},{:2.1f},{:2.1f}%".format(
                    data[0], gt, pd, error), fontsize=font_size, fontweight='bold', color='red')
            elif abs(error) > 2.0:
                title = plt.title("{}: {:2.1f},{:2.1f},{:2.1f}%".format(
                    data[0], gt, pd, error), fontsize=font_size, color='red')
            else:
                title = plt.title("{}: {:2.1f},{:2.1f},{:2.1f}%".format(
                    data[0], gt, pd, error), fontsize=font_size)
            i = i + 1
        plt.tight_layout()
        plt.show()
    print("Test completed")


def plot_log(images_path, log_file, model_path):
    """
    Load the logged images and plot the steering data as well as an offline prediction
    """
    raw = []
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            raw.append(line)

    raw = raw[3:]
    log_result = []
    offline_result = []
    model = load_model(model_path)
    for sample in raw:
        image_file = images_path + sample[0] + ".jpg"
        image_array = open_image(image_file)
        steering = float(model.predict(
            image_array[None, :, :, :], batch_size=1))

        steering = STEERING_GAIN * steering
        log_result.append(float(sample[1]))
        offline_result.append(steering)

    plt.figure(figsize=(30, 8))
    plt.plot(log_result, 'r', label='log result')
    plt.plot(offline_result, 'g', label='offline result')
    plt.rc('grid', linestyle="--", color='grey')
    plt.grid(True, 'both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def draw_line(start_x, start_y, steering, length):
    """
    Draw a line on an image to indicate the steering angle
    """
    angle = steering * math.pi / 4.0
    dx = length * math.sin(angle)
    dy = length * math.cos(angle)
    end_x, end_y = int(start_x + dx), int(start_y - dy)

    return end_x, end_y


def visualize_log(images_path, log_file, output_path, model_path=''):
    """
    Visualize the logged image with steering data
    """
    samples_raw = []
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples_raw.append(line)

    samples_raw = samples_raw[3:]
    samples = []
    for sample in samples_raw:
        steering = float(sample[1])
        samples.append([sample[0] + ".jpg", steering, CENTER_IMAGE])

    print("Open ", len(samples), " files")

    output_path = os.path.join(output_path,"")
    print("Creating image folder at {}".format(output_path))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    if model_path:
        model = load_model(model_path)
        print("Open model ", model_path)
    for sample in samples:
        file = os.path.join(images_path ,sample[INDEX_PATH])
        if not os.path.isfile(file):
            print("Cannot open file ", file)
            continue

        camera_view = open_image(file)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start_x, start_y = int(img.shape[1] / 2), img.shape[0]
        length = img.shape[0] / 3
        end_x, end_y = draw_line(start_x, start_y, sample[INDEX_STEER], length)
        cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 120, 255), 3)
        if model_path:
            steering = float(model.predict(
                camera_view[None, :, :, :], batch_size=1))
            steering = STEERING_GAIN * steering
            end_x, end_y = draw_line(start_x, start_y, steering, length)
            cv2.line(img, (start_x, start_y), (end_x, end_y), (255, 120,0 ), 3)

        img[:camera_view.shape[0], :camera_view.shape[1]] = camera_view
        pil_im = Image.fromarray(img)
        pil_im.save(output_path + sample[INDEX_PATH])

    print("Process completed")

from moviepy.editor import ImageSequenceClip
def make_video(image_folder,fps=60):
    video_file = image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, fps))
    clip = ImageSequenceClip(image_folder, fps=fps)
    clip.write_videofile(video_file)

import imageio
def make_gif(image_folder,down_sample=1):
    images = []
    files = [x for x in os.listdir(image_folder) if x.endswith('.jpg')]
    files.sort()
    files = [files[i] for i in range(0,len(files),down_sample)]
    for file in files:
        images.append(imageio.imread(os.path.join(image_folder,file)))

    imageio.mimsave(image_folder + '.gif', images)
    print("Image is saved to ",image_folder + '.gif')
