# -*- coding: utf-8 -*-
"""Net_fruit_on_GPU.ipynb""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FhRSMxlDgPXJ7rKUT7_KiXDIiFbCLEM8
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import numpy as np


def load_train(path):
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255, 
        horizontal_flip=True)

   
    train_datagen_flow = (train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        class_mode='sparse',
        subset='training',
        batch_size=16,
        seed=12345))
    

    return train_datagen_flow


def create_model(input_shape):
    
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=5, activation='relu',
                                  padding='same', input_shape=input_shape))
    model.add(AvgPool2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1), padding='valid', 
                    activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1),
                    activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))

    optimizer = Adam(lr=0.001) 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    


    return model


def train_model(model, train_data, test_data, epochs=15, batch_size=None,
               steps_per_epoch=None, validation_steps=None):
  
    train_datagen_flow = train_data

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data, 
              validation_data=(test_data), epochs=epochs,
              steps_per_epoch=steps_per_epoch, batch_size=batch_size,
              validation_steps=validation_steps, 
              verbose=2, shuffle=True)
    return model