import utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import  optimizers
# from keras.utils import plot_model

model = utils.make_nvidia()
#model = utils.make_commaai()
sgd = optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='mse', optimizer=sgd)

# # My simple sample reading
# X_train, y_train = utils.generate_train_data("data/")
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=8)


path = 'data/'
train_generator, validation_generator, train_size, valid_size = \
    utils.generate_train_data2(path)

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=False,
                             mode='auto')

model.fit_generator(train_generator, samples_per_epoch=train_size,
                    validation_data=validation_generator,
                    nb_val_samples=valid_size, nb_epoch=20, callbacks=[checkpoint])

model.save("model.h5")

# plot_model(model,to_file='model.png')

