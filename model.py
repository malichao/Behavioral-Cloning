import csv
import cv2
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda 
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

csv_data = np.genfromtxt("data/driving_log.csv", delimiter=',',
                         names=True, dtype=None)
FILE_NUM = 800
print("Read ", len(csv_data), " files, only use first", FILE_NUM, " ones")
measurements_raw = [t[3] for t in csv_data]
file_names = [[t[0].decode('utf-8'), t[1].decode('utf-8'),
               t[2].decode('utf-8')] for t in csv_data]
measurements_raw = measurements_raw[0:FILE_NUM]
file_names = file_names[0:FILE_NUM]

images = []
measurements=[]
source_path = "data/"
selected = [i for i in range(0,len(file_names))]
random.shuffle(selected)
print("Shuffle inputs.. ",selected[0:10])
for i in range(0,FILE_NUM):
    files = file_names[selected[i]]
    mea = measurements_raw[selected[i]]
    for file in files:
        if file[0]==' ':    # Stupid space...
            file = file[1:]
        file_name = source_path + file
        image = cv2.imread(file_name)
        image_flipped = np.fliplr(image)
        images.append(image)
        measurements.append(mea)

#measurements = [-mea for mea in measurements]

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))
# Crop 50 rows from the top, 20 rows from the bottom
# Crop 0 columns from the left, 0 columns from the right
model.add(Cropping2D(cropping=((60,23), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save("model.h5")
