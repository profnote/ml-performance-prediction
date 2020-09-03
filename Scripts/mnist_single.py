import pandas as pd
import numpy as np

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import time

from os import path, getcwd, chdir


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


def mnist_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape=input_shape))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10,activation= 'softmax' ))
    
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    
    return(model)    
    
    
class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append((time.time() - self.epoch_time_start) * 1000)  # time in miliseconds

        
model = mnist_model()
model.summary()

time_callback = TimeHistory()

# Change parameters here
epochs = 4
batchsize = 96

t1 = time.time()

history = model.fit(x=x_train,
          y=y_train,
          batch_size = batchsize,
          callbacks=[time_callback],
          epochs=epochs)

t2 = time.time()

times = time_callback.times 

print("Total times : ", times )





























