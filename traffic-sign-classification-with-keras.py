
# coding: utf-8

import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras.datasets import cifar10

def normalize(image):
    return cv2.normalize(image,None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def pre_process(X_train, y_train, nb_classes):
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1)
    X_train = np.array([normalize(image) for image in X_train], dtype=np.float32)
    X_val = np.array([normalize(image) for image in X_val], dtype=np.float32)
    
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_val = np_utils.to_categorical(y_val, nb_classes)
    
    return X_train, y_train, X_val, y_val

def train_model(model, X_train, y_train, nb_classes):
    X_train, y_train, X_val, y_val = pre_process(X_train, y_train, nb_classes)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, batch_size=25, nb_epoch=2,
                        verbose=1, validation_data=(X_val, y_val))
    return history, model

def evaluate_model(model, X_test, y_test, nb_classes):
    
    X_test = np.array([normalize(image) for image in X_test], dtype=np.float32)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    
    results = model.evaluate(X_test, y_test)
    names = model.metrics_names
    return {k:v for k,v in zip(names, results)}
    
def get_layers(nb_filters, kernel_size, pool_size,imshape, nb_classes):
    feature_layers = [
        Convolution2D(nb_filters, kernel_size, kernel_size,
                      border_mode='valid',
                      input_shape=imshape),
        Activation('relu'),
        Convolution2D(nb_filters, kernel_size, kernel_size),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Activation('relu'),
        Dropout(0.25),
        Flatten(),
        ]
    
    classification_layers = [
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
        ]
    return feature_layers, classification_layers


def train_gtsrb():
    with open('train.p', 'rb') as f:
        train_data = pickle.load(f)
    with open('test.p', 'rb') as f:
        test_data = pickle.load(f)
    X_train, y_train = train_data['features'], train_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']
    
    nb_filters,kernel_size, pool_size = 32, 3, 2
    imshape = X_train.shape[1:]
    nb_classes = 43
    feature_layers, classification_layers = get_layers(nb_filters,
                                                       kernel_size,
                                                       pool_size,
                                                       imshape,
                                                       nb_classes
                                                      )
    model = Sequential(feature_layers + classification_layers)
    model.summary()
    history, model = train_model(model, X_train, y_train, nb_classes)
    metrics = evaluate_model(model, X_test, y_test, nb_classes)
    print(metrics)
    return model

def build_cifar10(model, nb_classes):
    for layer in model.layers:
        layer.trainable = False
    model.layers.pop()
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


def train_cifar10(model):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    nb_filters,kernel_size, pool_size = 32, 3, 2
    print(X_train.shape)
    imshape = X_train.shape[1:]
    nb_classes = 10
    model = build_cifar10(model, nb_classes)
    model.summary()
    history, model = train_model(model, X_train, y_train, nb_classes)
    metrics = evaluate_model(model, X_test, y_test, nb_classes)
    print(metrics)
    
def main():
    model = train_gtsrb()
    train_cifar10(model)
    
if __name__ == '__main__':
    main()
