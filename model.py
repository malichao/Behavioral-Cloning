import utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model
import argparse

def make_model(model_type):
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

    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    return model


def main():

    parser = argparse.ArgumentParser(description='Behavioral Cloning')

    parser.add_argument('-d', help='data directory',
                        dest='data_dir', type=str, default='data/track1-center1/')

    parser.add_argument('-l', help='load model',
                        dest='model', type=str, default='')

    parser.add_argument('-m', help='make model',
                        dest='make_model', type=str, default='')

    parser.add_argument('-e', help='epochs',
                        dest='epochs', type=int, default=20)

    parser.add_argument('-o', help='model output name',
                        dest='output_name', type=str, default='model.h5')

    args = parser.parse_args()

    if args.model == '':
        model = make_model(args.make_model)
    else:
        model = load_model(args.model)
        print("Loaded model from [{}]".format(args.model))

    # path = 'data/track1-center1/'
    path = args.data_dir
    print("Training data on [{}]".format(path))
    print("Training epochs {}".format(args.epochs))
    print("Saving model to [{}]".format(args.output_name))
    train_generator, validation_generator, train_size, valid_size = \
        utils.generate_train_data(utils.load_data(path))

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')

    model.fit_generator(train_generator, samples_per_epoch=train_size,
                        validation_data=validation_generator,
                        nb_val_samples=valid_size, nb_epoch=args.epochs, callbacks=[checkpoint])

    model.save(args.output_name)


if __name__ == '__main__':
    main()
