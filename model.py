import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

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
    images.append(image)

X_train =  np.array(images)
y_train =  np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)

model.save("model.h5")
