import utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

# model = utils.make_lenet()
model = utils.make_lenet2()
# model = utils.make_commaai()
# model = utils.make_commaai2()
# model = utils.make_nvidia()
model.compile(loss='mse', optimizer='adam')

# # My simple sample reading
# X_train, y_train = utils.generate_train_data("data/")
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=8)


path = 'data/'
train_generator, validation_generator, train_size, valid_size = \
    utils.generate_train_data2(path)

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

model.fit_generator(train_generator, samples_per_epoch=train_size,
                    validation_data=validation_generator,
                    nb_val_samples=valid_size, nb_epoch=10, callbacks=[checkpoint])

model.save("model.h5")
