import utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model
import argparse


def make_model(model_type, learning_rate=1e-3):
    if model_type == 'lenet':
        model = utils.make_lenet()
        print("Build a new model [lenet]")
    elif model_type == 'nvidia':
        model = utils.make_nvidia()
        print("Build a new model [nvidia]")
    elif model_type == 'commaai':
        model = utils.make_commaai()
        print("Build a new model [commaai]")
    elif model_type == 'unknown':
        model = utils.make_commaai()
        print("Build a new model [unknown]")
    else:
        print("Error! No such model")
        raise ("Error! No such model")

    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6,
                         momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adam')
    return model


def main():

    parser = argparse.ArgumentParser(description='Behavioral Cloning')

    parser.add_argument('-d', help='data directory',
                        dest='data_dir', type=str, default='')

    parser.add_argument('-g', help='load data from log directory',
                        dest='log_dir', type=str, default='')

    parser.add_argument('-l', help='load model',
                        dest='load_model', type=str, default='')

    parser.add_argument('-m', help='make model',
                        dest='make_model', type=str, default='nvidia')

    parser.add_argument('-e', help='epochs',
                        dest='epochs', type=int, default=20)

    parser.add_argument('-o', help='model output name',
                        dest='output_name', type=str, default='model.h5')

    parser.add_argument('-r', help='learning rate',
                        dest='learning_rate', type=float, default=1e-3)

    parser.add_argument('-w', help='load weights',
                        dest='weights', type=str, default="")

    args = parser.parse_args()

    # Model initialization order:
    # 1.If there is a "weights" file, load it with the given model type
    # 2.If there is a "model" file, load the model, ignore -m option
    # 3.Make a new model
    if args.weights:
        model = make_model(args.make_model, args.learning_rate)
        model.load_weights(args.weights + ".w")
        print("Loaded weights from [{}]".format(args.weights + ".w"))
    elif args.load_model:
        model = load_model(args.load_model + ".h5")
        print("Loaded model from [{}]".format(args.load_model))
    else:
        model = make_model(args.make_model, args.learning_rate)

    # path = 'data/track1-center1/'
    samples = []
    if args.data_dir:
        paths = args.data_dir.split(":")
        for path in paths:
            samples.extend(utils.load_data(path))
    else:
        raise("No training data")

    print("Training data on [{}]".format(paths))
    print("Training epochs {}".format(args.epochs))
    print("Saving model to [{}.h5]".format(args.output_name))
    print("Learning rate [{}]".format(args.learning_rate))

    # val_samples = utils.csv2samples("data/track1-val/")
    # val_samples = utils.preprocess_samples(val_samples)
    val_samples = []

    train_generator, validation_generator, train_size, valid_size = \
        utils.generate_train_data(samples, val_samples)

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')

    model.fit_generator(train_generator, samples_per_epoch=train_size,
                        nb_val_samples=valid_size, nb_epoch=args.epochs, callbacks=[checkpoint])

    model.save(args.output_name + ".h5")
    model.save_weights(args.output_name + ".w")


if __name__ == '__main__':
    main()
