"""
Fully convolution Autoencoder  with1D-Convolution

Usage:
    python FCNN_AE.py Univariate Mallat --itr "1" --epochs=700 --batch_size=8
Author:
    Baptiste Lafabregue 2019.06.20
"""

from time import time
import numpy as np
import math

from keras import layers
from keras.models import Model
from keras import callbacks
from keras.optimizers import Adam, SGD
from keras.constraints import Constraint
import keras.backend as K
import tensorflow as tf

import utils


def step_decay_schedule(initial_lr=0.1, decay_factor=0.1, step_size=90):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))


def noise(code, rate):
    """
    Adds masking noise to an input set of vectors.
    """
    noise_code = np.copy(code)
    for i in range(len(noise_code)):
        idx = np.around(np.random.uniform(0., code.shape[1] - 1, size=int(code.shape[1] * rate))).astype(np.int)
        noise_code[i, idx] = 0
    return noise_code


def Conv1DTranspose(input_tensor, filters, kernel_size, id, strides=2, padding='same', name='Conv1D_Transpose', activation=None):
    """
    Transforms a 1D Tensor into a 2D tensor to apply a Conv2DTranspose layer
    """
    x = layers.Lambda(lambda x: K.expand_dims(x, axis=2), name='lambda_i_%d' % id)(input_tensor)
    x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                               strides=(strides, 1), padding=padding, name=name, activation=activation)(x)
    x = layers.Lambda(lambda x: K.squeeze(x, axis=2), name='lambda_o_%d' % id)(x)
    return x


def autoencoder(input_shape, filters, kernels, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    """
    n_stacks = len(filters) - 1
    if input_shape[0] % 8 == 0:
        pad_last = 'same'
    else:
        pad_last = 'valid'

    # input
    x = layers.Input(shape=input_shape, name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = layers.Conv1D(filters=filters[i], kernel_size=kernels[i],
                          padding='same', strides=2, name='encoder_%d' % i)(h)
        h = layers.normalization.BatchNormalization()(h)
        h = layers.Activation(activation=act)(h)

    h = layers.Conv1D(filters=filters[n_stacks-1], kernel_size=kernels[n_stacks-1],
                      padding=pad_last, strides=2, name='encoder_%d' % (n_stacks-1))(h)
    h = layers.normalization.BatchNormalization()(h)
    h = layers.Activation(activation=act)(h)

    # h = layers.pooling.GlobalAveragePooling1D()(h)
    h = layers.Flatten()(h)
    # hidden layer
    h = layers.Dense(filters[-1], name='encoder_%d' % n_stacks)(h)  # hidden layer, features are extracted from here
    h = layers.Dense(filters[n_stacks-1]*(int(input_shape[0]/8)))(h)

    h = layers.Reshape((int(input_shape[0]/8), filters[n_stacks-1]))(h)
    # internal layers in decoder
    h = Conv1DTranspose(h, filters[n_stacks-2], kernels[n_stacks-2], n_stacks-1, strides=2,
                        activation=act, padding=pad_last, name='decoder_%d' % (n_stacks-1))
    for i in range(n_stacks-3, -1, -1):
        h = Conv1DTranspose(h, filters[i], kernels[i], i+1, strides=2,
                            activation=act, name='decoder_%d' % (i+1))

    # output
    h = Conv1DTranspose(h, input_shape[1], kernels[0], 0, activation=None, name='decoder_0')

    return Model(inputs=x, outputs=h)

def autoencoder_a2cnes(input_shape, filters, dense_filters, kernels, act='relu', drop=0.0):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(filters) - 1
    d_stacks = len(dense_filters) -1

    # input
    x = layers.Input(shape=input_shape, name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks):
        h = layers.Conv1D(filters=filters[i], kernel_size=kernels[i],
                          padding='same', strides=1, name='encoder_%d' % i)(h)
        h = layers.normalization.BatchNormalization()(h)
        h = layers.Activation(activation=act)(h)

    h = layers.Conv1D(filters=filters[n_stacks], kernel_size=kernels[n_stacks],
                      padding='same', strides=1, name='encoder_%d' % (n_stacks))(h)
    h = layers.normalization.BatchNormalization()(h)
    h = layers.Activation(activation=act)(h)

    h = layers.pooling.GlobalAveragePooling1D()(h)
    # h = layers.Flatten()(h)

    for j in range(d_stacks):
        h = layers.Dense(dense_filters[j], name='encoder_d_%d' % j)(h)
    # hidden layer
    h = layers.Dense(dense_filters[-1], name='encoder_d_%d' % (d_stacks))(h)  # hidden layer, features are extracted from here

    for j in range(d_stacks-1, -1, -1):
        h = layers.Dense(dense_filters[j], name='decoder_d_%d' % j)(h)

    h = layers.Dense(filters[n_stacks] * (int(input_shape[0])))(h)
    h = layers.Reshape((int(input_shape[0]), filters[n_stacks]))(h)
    # internal layers in decoder
    h = Conv1DTranspose(h, filters[n_stacks], kernels[n_stacks], n_stacks+1, strides=1,
                        activation=act, padding='same', name='decoder_%d' % (n_stacks+1))
    for i in range(n_stacks-1, -1, -1):
        h = Conv1DTranspose(h, filters[i], kernels[i], i+1, strides=1, padding='same',
                            activation=act, name='decoder_%d' % (i+1))

    # output
    h = Conv1DTranspose(h, input_shape[1], kernels[0], 0, strides=1,
                        padding='same', activation=None, name='decoder_0')

    return Model(inputs=x, outputs=h)


def train(x, combined, epochs=10, corrupt=0.2, lr = 0.1, batch_size=16, save_file='', verbose=1):
    losses = []
    num_batch = int(math.ceil(1.0*x.shape[0]/batch_size))
    for i in range(epochs):

        loss = 0
        for batch_idx in range(num_batch):
            if (batch_idx + 1) * batch_size > x.shape[0]:
                pred = x[batch_idx * batch_size::]
            else:
                pred = x[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            x_noise = noise(pred, corrupt)
            loss += combined.train_on_batch(x=x_noise, y=pred)
        print("epoch:%d,loss:%f" % (i, loss))
        losses.append(loss)
        if i >= 5 and i % 5 == 0:
            combined.save_weights(save_file)
            print('save to '+save_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('archive_name', default='Univariate')
    parser.add_argument('dataset', default='Mallat')

    parser.add_argument('--itr', default='0')
    parser.add_argument('--dimensions', default=None, type=str)
    parser.add_argument('--epochs', default='500', type=int)
    parser.add_argument('--batch_size', default='16', type=int)
    args = parser.parse_args()
    print(args)

    classifier_name = 'fcnn'
    dataset_name = args.dataset
    archive_name = args.archive_name
    itr = args.itr
    root_dir = '.'
    z = 10
    suffix = '_z' + str(z)

    sep = '\t'
    optimizer = 'adam'

    batch_size = args.batch_size
    epochs = args.epochs
    output_directory = root_dir + '/ae_weights/'+classifier_name+'/' + dataset_name + itr
    output_directory = utils.create_directory(output_directory)

    train_dict = utils.read_multivariate_dataset(root_dir, archive_name, dataset_name, True)
    x_train = train_dict[dataset_name][0]
    y_train = train_dict[dataset_name][1]
    input_shape = x_train.shape[1:]

    if args.dimensions is None:
        filters, dense_filters, kernels = [128, 256, 128], [z], [9, 5, 3]
    else:
        filters, dense_filters, kernels = eval(args.dimensions)

    # define the model
    model = autoencoder_a2cnes(input_shape, filters, dense_filters, kernels)
    model.summary()

    # compile the model and callbacks
    model.compile(optimizer=optimizer, loss='mse')

    # begin training
    t0 = time()
    x_noisy = noise(x_train, 0.0)
    tmp_path = output_directory + '/%s-saved-pretrain-model-%d%s.h5' % (dataset_name, epochs, suffix)
    save_weights_cb = callbacks.ModelCheckpoint(tmp_path, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=20)
    train(x_train, model, batch_size=batch_size, epochs=epochs, save_file=tmp_path, verbose=1, corrupt=0.2)
    print('Training time: ', time() - t0)
    model.save_weights(output_directory + '/%s-pretrain-model-%d%s.h5' % (dataset_name, epochs, suffix))
