"""
Stacked Denoising AutoEncoder

Usage:
    python MLP_SDAE.py Univariate Mallat --itr "1" --epochs=200 --epochs_final=400 --batch_size=8
Author:
    Baptiste Lafabregue 2019.06.20
"""

from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
import math
import utils

trained_encoder = None


def img_to_code(img_input):
    if trained_encoder is not None:
        return trained_encoder.predict(img_input)
    else:
        return img_input

def adjust_learning_rate(epoch, lr):
    decay_rate = 0.1
    decay_step = 75
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


def step_decay_schedule(initial_lr=0.1, decay_factor=0.1, step_size=90):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))


    return LearningRateScheduler(schedule)

def noise(code, rate):
    noise_code = np.copy(code)
    for i in range(len(noise_code)):
        idx = np.around(np.random.uniform(0., code.shape[1] - 1, size=int(code.shape[1] * rate))).astype(np.int)
        noise_code[i, idx] = 0
    return noise_code


def train_one_by_one(x, dims, input_shape, lr=0.1, batch_size=128, epochs=400, corrupt=0.2, act = 'relu'):
    delta = int(epochs/5)
    input_layer = Input(input_shape)
    output_layer = Flatten()(input_layer)
    trained_encoder = Model(input_layer, output_layer)

    decoders = []
    encoders = []
    num_batch = int(math.ceil(1.0*x.shape[0]/batch_size))

    for i in range(len(dims) - 1):
        encoder_input = Input(shape=(dims[i],))

        if i == len(dims) - 2:
            encoder = Dense(units=dims[i + 1], name='encoder_%d' % i)
            code = encoder(encoder_input)
        else:
            encoder = Dense(units=dims[i + 1], activation=act, name='encoder_%d' % i)
            code = encoder(encoder_input)
            code = Dropout(corrupt)(code)

        decoder = Dense(units=dims[i], activation=act, name='decoder_%d' % i)
        reconstruct_code = decoder(code)
        combined = Model(inputs=encoder_input, outputs=reconstruct_code)
        optimizer = SGD(lr=lr, momentum=0.9)
        callbacks = [LearningRateScheduler(adjust_learning_rate, verbose=1)]
        combined.compile(loss='mse', optimizer=optimizer)

        # Convert the vector to code by the previously trained encoder. If there is no trained encoder, the code is image.
        code = x if (trained_encoder is None) else trained_encoder.predict(x)
        combined.fit(code, code, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=callbacks)

        # After the run above, we train a layer of encoder to merge the encoder with the previous one.
        if trained_encoder is None:
            input_layer = Input(input_shape)
            input_layer = Flatten()(input_layer)
            trained_encoder = Model(input_layer, encoder(input_layer))
        else:
            input_layer = Input(input_shape)
            input_code = trained_encoder(input_layer)
            code = encoder(input_code)
            trained_encoder = Model(input_layer, code)

        # Save the decoder and encoder corresponding to the current layer
        encoders.append(encoder)
        decoders.append(decoder)
        epochs += delta  # Add iterations per layer

    input_layer = Input(shape=input_shape)
    decoders.reverse()
    # last_decode = trained_encoder(img_input)
    last_ae = Flatten()(input_layer)
    for encoder in encoders:
        last_ae = encoder(last_ae)
    for decoder in decoders:
        last_ae = decoder(last_ae)

    last_ae = Reshape(input_shape)(last_ae)
    combined = Model(input_layer, last_ae)
    combined.compile(loss='mse', optimizer=optimizer)
    return trained_encoder, combined

def autoencoder(dims, input_shape, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=input_shape, name='input')
    h = Flatten()(x)

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = Dense(dims[0], name='decoder_0_d')(h)

    h= Reshape(input_shape, name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


def train(x, combined, epochs=10, corrupt=0.2, lr = 0.1, batch_size=64):
    inputs_cor = noise(x, corrupt)
    callbacks = [step_decay_schedule()]
    combined.fit(inputs_cor, x, batch_size=batch_size, epochs=epochs, verbose=True, callbacks=callbacks)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('archive_name', default='Univariate')
    parser.add_argument('dataset', default='Mallat')

    parser.add_argument('--itr', default='0')
    parser.add_argument('--dimensions', default=None, type=str)
    parser.add_argument('--epochs', default='250', type=int)
    parser.add_argument('--epochs_final', default='400', type=int)
    parser.add_argument('--batch_size', default='16', type=int)
    args = parser.parse_args()
    print(args)

    classifier_name = 'mlp'
    dataset_name = args.dataset
    archive_name = args.archive_name
    feature_count = args.features
    itr = args.itr
    root_dir = '.'
    z = 10
    suffix='_z'+str(z)

    batch_size = args.batch_size
    epochs = args.epochs
    final_epochs = args.epochs_final
    output_directory = root_dir + '/ae_weights/'+classifier_name+'/' + dataset_name + itr
    output_directory = utils.create_directory(output_directory)

    train_dict = utils.read_multivariate_dataset(root_dir, archive_name, dataset_name, True)
    x_train = train_dict[dataset_name][0]
    y_train = train_dict[dataset_name][1]

    if args.dimensions is None:
        dimensions = [500, 500, 2000, z]
    else:
        dimensions = eval(args.dimensions)
    encoder, combined = train_one_by_one(x_train, [x_train.shape[1]*x_train.shape[2]] + dimensions,
                                         x_train.shape[1:], epochs=epochs, batch_size=batch_size, corrupt=0.0)
    combined.summary()
    train(x_train, combined, epochs=(int(final_epochs)), corrupt=0, batch_size=batch_size)
    combined.save_weights(output_directory + '/%s-pretrain-model-%d%s.h5' % (dataset_name, epochs, suffix))
