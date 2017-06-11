import utils
import numpy as np

model = utils.make_lenet()
model.compile(loss='mse', optimizer='adam')

# My simple sample reading
X_train, y_train = utils.generate_train_data("data/")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=8)


# path = 'data/'
# samples = utils.csv2samples(path + 'driving_log.csv')
# train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# train_size = len(train_samples)
# validation_size = len(validation_samples)
# print("Read ", len(samples), " samples")
# print("train_samples      = ", train_size)
# print("validation_samples = ", validation_size)

# BATCH_SIZE = 100
# train_size = int(train_size / BATCH_SIZE) * BATCH_SIZE
# validation_size = int(validation_size / BATCH_SIZE) * BATCH_SIZE
# train_samples = train_samples[0:train_size]
# validation_samples = validation_samples[0:validation_size]
# print("Resize sample size to avoid Keras warning")
# print("train_samples      = ", train_size)
# print("validation_samples = ", validation_size)
# train_generator = utils.generator(path, train_samples, batch_size=BATCH_SIZE)
# validation_generator = utils.generator(
#     path, validation_samples, batch_size=BATCH_SIZE)

# print("samples_per_epoch", len(train_samples))
# model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
#                     validation_data=validation_generator,
#                     nb_val_samples=len(validation_samples), nb_epoch=3)

model.save("model.h5")
