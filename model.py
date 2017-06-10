import utils
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

# My simple sample reading
#X_train,y_train = utils.generate_train_data("data/")

path = 'data/'
samples = utils.csv2samples(path + 'driving_log.csv')
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = utils.generator(path,train_samples, batch_size=900)
validation_generator = utils.generator(path,validation_samples, batch_size=900)


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

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=6)

model.save("model.h5")
