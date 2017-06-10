import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout ,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

csv_data = np.genfromtxt("data/driving_log.csv", delimiter=',',\
                         names=True, dtype=None)
FILE_NUM=1000
measurements = [t[3] for t in csv_data]
file_names =[t[0].decode('utf-8') for t in csv_data]
print("Read ",len(file_names)," files, only use first",FILE_NUM," ones")
measurements = measurements[0:FILE_NUM]
file_names = file_names[0:FILE_NUM]

images = []
source_path ="data/"
for file_name in file_names:
    file_name = source_path + file_name
    image = cv2.imread(file_name)
    image_flipped = np.fliplr(image)
    images.append(image)

measurements = [ -mea for mea in measurements]

X_train =  np.array(images)
y_train =  np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x /255. -.5,input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save("model.h5")
