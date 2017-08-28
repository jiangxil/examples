'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.contrib.keras.python.keras.datasets.cifar import load_batch
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D

import os
import pickle
import numpy as np
import argparse


def download_data(num_classes=10):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def get_data(path, num_classes=10):
    num_train_samples = 50000
  
    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000:i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000:i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def get_model(input_shape, lr, lr_decay, num_classes=10):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=lr, decay=lr_decay)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    return model


def train_model(model, xy_train, xy_test,
                tensorboard_dir,
                data_augmentation=False, epochs=200, batch_size=32):

    x_train, y_train = xy_train
    x_test, y_test = xy_test
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        tb_cb = keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            callbacks=[tb_cb],
                            validation_data=(x_test, y_test))

        # Evaluate model with test data set and share sample prediction results
        evaluation = model.evaluate_generator(datagen.flow(x_test, y_test,
                                            batch_size=batch_size),
                                            steps=x_test.shape[0] // batch_size)

        print('Model Accuracy = %.2f' % (evaluation[1]))


def save_model(model, save_dir, filename='keras_cifar10_trained_model.h5'):
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on CIFAR-10.')
    parser.add_argument('data', type=str,
                        help='path to CIFAR-10 data')
    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr-decay', type=float,
                        default=1e-6,
                        help='learning rate decay')
    parser.add_argument('--epochs', type=int,
                        default=20,
                        help='number of epochs to train')
    parser.add_argument('--augment-data', type=bool,
                        default=True,
                        help='Whether to augment data')
    parser.add_argument('--tensorboard-dir', type=str,
                        default='tensorboard',
                        help='Where to output tensorboard summaries')
                        
    args = parser.parse_args()
    xy_train, xy_test = download_data()
    input_shape = xy_train[0].shape[1:]
    model = get_model(input_shape, args.lr, args.lr_decay)
    print('Learning rate: %s' % (args.lr))
    print('Learning rate decay: %s' % (args.lr_decay))
    train_model(model, xy_train, xy_test, args.tensorboard_dir,
                epochs=args.epochs,
                data_augmentation=args.augment_data)
