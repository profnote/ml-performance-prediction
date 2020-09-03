#Importing the necessary libraries
from tensorflow import keras
import tensorflow as tf
import os,datetime
import tensorflow_datasets as tfds
import pandas as pd

import numpy as np
import time
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

print(tfds.__version__)           


#Loading the data from tensorflow_datasets
df, info = tfds.load('patch_camelyon', with_info = True, as_supervised = True)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#Getting the train, validation and test data
train_data = df['train']
valid_data = df['validation']
test_data = df['test']


num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
    


#A function to help scale the images
def preprocess(image, labels):
  image = tf.cast(image, tf.float32)
  image /= 255.
  return image, labels
  
  
buffer_size = 1024  
  

BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
train_data = train_data.map(preprocess).shuffle(buffer_size).batch(GLOBAL_BATCH_SIZE)
valid_data = valid_data.map(preprocess)
test_data = test_data.map(preprocess)

print("Type of train data :", type(train_data) )

#Checking the image shape
print("Train images shape : ", train_images.shape)
print("Total examples: ", num_train_examples)




#import the necessary libraries
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers



class TimeHistory(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)    




with strategy.scope():
    model = Sequential([tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3)),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3)),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3)),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3)),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3)),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3)),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(1028, activation='relu'),
                        tf.keras.layers.Dense(512, activation='relu'),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])




class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)




num_epochs = 4

time_callback = TimeHistory()

history = model.fit(train_images, train_labels, epochs=num_epochs,  callbacks=[time_callback])

times = time_callback.times 

print("Total times : ", times )
