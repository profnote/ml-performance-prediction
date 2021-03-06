"""Benchmark convolution"""

import tensorflow as tf
import numpy as np
import time

        
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append((time.time() - self.epoch_time_start) * 1000)
                

class convolution(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self,
                 batchsize,
                 matsize,
                 kernelsize,
                 channels_in,
                 channels_out,
                 strides,
                 precision,
                 padding,
                 activation_fct,
                 use_bias,
                 optimizer,
                 strategy,
                 iterations_warmup,
                 iterations_benchmark,
                 backprop):
        """Initialize convolution

        Args:
            args: Input arguments

        """
        
        self.matsize = matsize
        self.kernelsize = kernelsize
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.strides = strides
        self.batchsize = batchsize
        self.padding = padding
        self.use_bias = use_bias
        self.precision = precision
        self.activation_fct = activation_fct
        self.opt = optimizer
        self.strategy = strategy
        self.iterations_warmup = iterations_warmup
        self.iterations_benchmark = iterations_benchmark
        self.backprop = backprop

    
    def run_benchmark(self):
        """Run a tensorflow mirrorStrategy

        Returns:
            timeUsed: time used per iteration in ms
        """
        datatype = eval('tf.float%d' %(self.precision))
        
        if self.padding == 'SAME':
            target_size = np.ceil(np.float(self.matsize)/self.strides)
        else:
            target_size = np.ceil(np.float((self.matsize-(self.kernelsize-1)))/self.strides)
        target_size = int(target_size)
        
        GLOBAL_BATCH_SIZE = self.batchsize * self.strategy.num_replicas_in_sync
        
        train_images = tf.Variable(tf.ones([GLOBAL_BATCH_SIZE, self.matsize, self.matsize,
                    self.channels_in], dtype=datatype))
        target = tf.Variable(tf.ones([GLOBAL_BATCH_SIZE, target_size, target_size,
                    self.channels_out], dtype=datatype))
        
        with self.strategy.scope():
            input_shape = (self.matsize, self.matsize, self.channels_in)
            act = eval(self.activation_fct)
            model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.channels_out,
                kernel_size=(self.kernelsize,self.kernelsize),
                strides=(self.strides, self.strides),
                padding=self.padding,
                activation = act,
                use_bias=self.use_bias,
                input_shape=input_shape)])
            optimizer = None 
            if self.backprop:
                optimizer = eval('tf.compat.v1.train.%s' % self.opt)
            model.compile(loss='mse', optimizer=optimizer)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, target)).batch(GLOBAL_BATCH_SIZE)

                
        # Fit model and record training time
        tf.keras.backend.clear_session()
        time_callback = TimeHistory()
        model.fit(train_dataset, epochs=self.iterations_benchmark, callbacks=[time_callback], verbose=0)

        times = time_callback.times
        times = times[1:len(times)]  # remove time of the first epoch (overhead)
        
        return np.mean(times), np.median(times)
       
