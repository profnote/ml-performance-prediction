###############

# Importing the necessary libraries

from tensorflow import keras
import tensorflow as tf
import os, datetime
import tensorflow_datasets as tfds
import pandas as pd
import time

import numpy as np

print("Pycharm code")

# Loading the data from tensorflow_datasets
df, info = tfds.load('patch_camelyon', with_info=True, as_supervised=True)

# Getting the train, validation and test data
train_data = df['train']
valid_data = df['validation']
test_data = df['test']

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples


# A function to help scale the images
def preprocess(image, labels):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, labels


# Applying the preprocess function we the use of map() method
train_data = train_data.map(preprocess)
valid_data = valid_data.map(preprocess)
test_data = test_data.map(preprocess)

# Shuffling the train_data
buffer_size = 1000
train_data = train_data.shuffle(buffer_size)

# Batching and prefetching
batch_size = 3072  #512 #256 #16384 #65536 #  262144  # 131072
train_data = train_data.batch(batch_size).prefetch(1)
# valid_data = valid_data.batch(batch_size).prefetch(1)
# test_data = test_data.batch(batch_size).prefetch(1)





# Seperating image and label into different variables
train_images, train_labels = next(iter(train_data))
valid_images, valid_labels = next(iter(valid_data))
test_images, test_labels = next(iter(test_data))


#train_images = np.ones([batch_size, 6, 6, 512])

#train_labels = np.ones([batch_size, 3, 3, 1024])

#train_labels = np.ones([batch_size, 128])


# Checking the label shape
#valid_labels.shape

# Checking the image shape
print("Train images shape : ", train_images.shape)

print("Total examples: ", num_train_examples)

# train_images_2 = pd.DataFrame(train_images)


# train_images_2 = np.array(train_data)

print("checkpoint")
# print(type(train_data))
# print(type(train_data))


# import the necessary libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

print("checkpoint2")


class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_callback = TimeHistory()

num_epochs = 10


print("checkpoint3")




#batch_size_list = [8, 16, 24, 32, 48, 64, 128, 256]



df_all = pd.DataFrame()


def create_patch_cam_model_full():

#    train_images = np.ones([batch_size, 96, 96, 3])

#    train_labels = np.ones([batch_size, 1])
    
    train_images, train_labels = next(iter(train_data))

    model = Sequential([
                        tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3))
                        ,tf.keras.layers.MaxPooling2D(),
                         tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu' )
                        ,tf.keras.layers.MaxPooling2D(),
                         tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu' )
                        ,tf.keras.layers.MaxPooling2D(),
                         tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu' ),
                        tf.keras.layers.MaxPooling2D(),
                         tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu' ),
                         tf.keras.layers.MaxPooling2D()
                        ,tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu'),
                        tf.keras.layers.MaxPooling2D()
                        ,tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(1028, activation='relu'),
                        tf.keras.layers.Dense(512, activation='relu'),
                        tf.keras.layers.Dense(128, activation='relu')
                        ,tf.keras.layers.Dense(1, activation='sigmoid')
                        ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)
    
    
    
    
def create_patch_cam_model_l1():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 48, 48, 256])

    model = Sequential([
                        tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(96, 96, 3))
                        ,tf.keras.layers.MaxPooling2D()])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l2():

    train_images = np.ones([batch_size, 48, 48, 256])

    train_labels = np.ones([batch_size, 24, 24, 256])

    model = Sequential([
                        tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(48, 48, 256))
                        ,tf.keras.layers.MaxPooling2D()])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l3():

    train_images = np.ones([batch_size, 24, 24, 256])

    train_labels = np.ones([batch_size, 12, 12, 512])

    model = Sequential([
                        tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(24, 24, 256))
                        ,tf.keras.layers.MaxPooling2D()])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l4():

    train_images = np.ones([batch_size, 12, 12, 512])

    train_labels = np.ones([batch_size, 6, 6, 512])

    model = Sequential([
                        tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(12, 12, 512))
                        ,tf.keras.layers.MaxPooling2D()])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l5():

    train_images = np.ones([batch_size, 6, 6, 512])

    train_labels = np.ones([batch_size, 3, 3, 1024])

    model = Sequential([tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(6, 6, 512))
                        ,tf.keras.layers.MaxPooling2D()])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l6():

    train_images = np.ones([batch_size, 3, 3, 1024])

    train_labels = np.ones([batch_size, 1, 1, 1024])

    model = Sequential([
                        tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform',
                                               activation='relu', input_shape=(3, 3, 1024))
                        ,tf.keras.layers.MaxPooling2D()])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l7():

    train_images = np.ones([batch_size,  1024])

    train_labels = np.ones([batch_size,  1028])

    model = Sequential([tf.keras.layers.Dense(1028, activation='relu', input_shape=(1024,) )   ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l8():

    train_images = np.ones([batch_size,  1028])

    train_labels = np.ones([batch_size,  512])

    model = Sequential([tf.keras.layers.Dense(512, activation='relu', input_shape=(1028,) )   ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l9():

    train_images = np.ones([batch_size,  512])

    train_labels = np.ones([batch_size,  128])

    model = Sequential([tf.keras.layers.Dense(128, activation='relu', input_shape=(512,) )   ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l10():

    train_images = np.ones([batch_size,  128])

    train_labels = np.ones([batch_size,  1])

    model = Sequential([tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(128,) )])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l1_l2():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 24, 24, 256])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D() ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l1_l3():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 12, 12, 512])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D() ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l1_l4():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 6, 6, 512])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()

                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D() ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l1_l5():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 3, 3, 1024])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()

                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D() 
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D() ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l1_l6():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 1, 1, 1024])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()

                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D() 
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D() ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l1_l7():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 1028])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()

                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D() 
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten()
                   ,tf.keras.layers.Dense(1028, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l1_l8():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size,  512])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()

                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D() 
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten()
                   ,tf.keras.layers.Dense(1028, activation='relu')
                   
                   ,tf.keras.layers.Dense(512, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l1_l9():

    train_images = np.ones([batch_size, 96, 96, 3])

    train_labels = np.ones([batch_size, 128])

    model = Sequential([
                    tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(96, 96, 3))
                    ,tf.keras.layers.MaxPooling2D()
                             
                    ,tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()

                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D() 
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten()
                   ,tf.keras.layers.Dense(1028, activation='relu')
                   
                   ,tf.keras.layers.Dense(512, activation='relu') 
                   
                    ,tf.keras.layers.Dense(128, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)






def create_patch_cam_model_l9_l10():

    train_images = np.ones([batch_size, 512])

    train_labels = np.ones([batch_size,  1])

    model = Sequential([tf.keras.layers.Dense(128, activation='relu', input_shape=(512,)) 
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l8_l10():

    train_images = np.ones([batch_size, 1028])

    train_labels = np.ones([batch_size, 1])

    model = Sequential([tf.keras.layers.Dense(512, activation='relu', input_shape=(1028,)),
    
                        tf.keras.layers.Dense(128, activation='relu') 
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l7_l10():

    train_images = np.ones([batch_size, 1024])

    train_labels = np.ones([batch_size, 1])

    model = Sequential([tf.keras.layers.Dense(1028, activation='relu', input_shape=(1024,)),
    
                        tf.keras.layers.Dense(128, activation='relu'),
    
                        tf.keras.layers.Dense(128, activation='relu')
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)





def create_patch_cam_model_l6_l10():

    train_images = np.ones([batch_size, 3, 3, 1024])

    train_labels = np.ones([batch_size, 1])

    model = Sequential([tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(3, 3, 1024))
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten(),
                    
                    tf.keras.layers.Dense(1028, activation='relu', input_shape=(1024,)),
    
                        tf.keras.layers.Dense(128, activation='relu'),
    
                        tf.keras.layers.Dense(128, activation='relu')
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l5_l10():

    train_images = np.ones([batch_size, 6, 6, 512])

    train_labels = np.ones([batch_size, 1])

    model = Sequential([tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape=(6, 6, 512))
                    ,tf.keras.layers.MaxPooling2D()
    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten(),
                    
                    tf.keras.layers.Dense(1028, activation='relu'),
    
                        tf.keras.layers.Dense(512, activation='relu'),
    
                        tf.keras.layers.Dense(128, activation='relu') 
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)





def create_patch_cam_model_l4_l10():

    train_images = np.ones([batch_size, 12, 12, 512])

    train_labels = np.ones([batch_size, 1])

    model = Sequential([tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape = (12, 12, 512) )
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten(),
                    
                    tf.keras.layers.Dense(1028, activation='relu'),
    
                        tf.keras.layers.Dense(512, activation='relu'),
    
                        tf.keras.layers.Dense(128, activation='relu') 
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)




def create_patch_cam_model_l3_l10():

    train_images = np.ones([batch_size, 24, 24, 256])

    train_labels = np.ones([batch_size, 1])

    model = Sequential([tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape = (24, 24, 256) )
                    ,tf.keras.layers.MaxPooling2D()
    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu' )
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten(),
                    
                    tf.keras.layers.Dense(1028, activation='relu'),
    
                        tf.keras.layers.Dense(512, activation='relu'),
    
                        tf.keras.layers.Dense(128, activation='relu') 
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



def create_patch_cam_model_l2_l10():

    train_images = np.ones([batch_size, 48, 48, 256])

    train_labels = np.ones([batch_size, 1])

    model = Sequential([tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu', input_shape = (48, 48, 256) )
                    ,tf.keras.layers.MaxPooling2D()
    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu' )
                    ,tf.keras.layers.MaxPooling2D()
    
                    ,tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu' )
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
    
                    ,tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer='he_uniform', activation='relu')
                    ,tf.keras.layers.MaxPooling2D()
                    
                    ,tf.keras.layers.Flatten(),
                    
                    tf.keras.layers.Dense(1028, activation='relu'),
    
                        tf.keras.layers.Dense(512, activation='relu'),
    
                        tf.keras.layers.Dense(128, activation='relu') 
                   
                    ,tf.keras.layers.Dense(1, activation='relu') ])

    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

    return (model, train_images, train_labels)



#batch_size_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 112 ]


batch_size_list = [16, 32, 48]



for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l2()
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())
    
    times = time_callback.times 
        
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    
    df_results['model'] = 'l1_l2'
    
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_new_l1_l2_v2.csv',index=False)

#del df_all

##########################################################################################################


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l3()
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())

    
    times = time_callback.times 
    

        
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l3'    
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_bs64_l1_l3_v2.csv',index=False)

#del df_all

#####################################################################################################


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l4()
    
    print("model: l1_l4")
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())

    
    times = time_callback.times 
    

        
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l4'        
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_bs64_l1_l4_v2.csv',index=False)

#del df_all

#####################################################################################################


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l5()
    
    print("model: l1_l5")
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())

    
    times = time_callback.times 
        
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l5'
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_new_l1_l5_v2.csv',index=False)

#del df_all

#####################################################################################################


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l6()
    
    print("model: l1_l6")
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())
    
    times = time_callback.times 
    
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l6'
    
    
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_new_l1_l6_v2.csv',index=False)

#del df_all


#####################################################################################################


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l7()
    
    print("model: l1_l7")
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())

    
    times = time_callback.times 
    

        
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l7'
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_new_l1_l7_v2.csv',index=False)

#del df_all 

#####################################################################################################


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l8()
    
    print("model: l1_l8")
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())
    
    times = time_callback.times 
    
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l8'
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_new_l1_l8_v2.csv',index=False)

#del df_all 


#####################################################################################################


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_l1_l9()
    
    print("model: l1_l9")
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())
    
    times = time_callback.times 
        
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l9'
    
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_new_l1_l9_v2.csv',index=False)

#del df_all 


for batch_size_rep in batch_size_list:
    print("batch_size : ", batch_size_rep)
    time_callback = TimeHistory()

    model, train_images, train_labels = create_patch_cam_model_full()
    
    print("model: l1_l10_full")
    
    print("Shape of train_images : ", train_images.shape)
    
    print("Shape of train_labels : ", train_labels.shape)
    
    time_start = time.time()
    
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size_rep, callbacks=[time_callback])
    time_end = time.time()
    print("Total time : ", time_end - time_start)
    
    print(model.summary())
    
    times = time_callback.times 
        
    print([time*1000 for time in times])
    
    print("Median time : ", np.median(times[1:])*1000)
    
    times.append(time_end - time_start)
    
    epoch_list = [i+1 for i in range(num_epochs)]
    epoch_list.append('Total_time')
    
    df_results = pd.DataFrame({'epoch_no' :  epoch_list,
                               'time_run' : times})
                               
    df_results['batch_size_per_replica'] = batch_size_rep
      
    df_results['optimizer'] = 'Adam'
    df_results['model'] = 'l1_l10'
    print(df_results)

#    df_results.to_csv('total_time_' + str(batch_size_rep) + '_pc_v1.csv')

    df_all = pd.concat([df_all, df_results], axis = 0)
    
                               
print(df_all.shape)

df_all.to_csv('new_results_v3/time_patch_cam_new_l1_l10_full_v3.csv',index=False)

#del df_all 


    



