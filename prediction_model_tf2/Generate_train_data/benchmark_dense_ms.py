"""Benchmark fully connected layer / matrix multiplication"""

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
        

class dense_layer(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self,
                 dim_input,
                 dim_output,
                 batchsize,
                 precision,
                 activation_fct,
                 optimizer,
                 strategy,
                 iterations_warmup,
                 iterations_benchmark,
                 backprop):
        """Initialize gemm

        Args:
            dim_input: Size of input vector / number of input features (int)
            dim_output: Size of output vector / number of output features (int)
            batchsize: (int)
            precision: bit depth (16/32/64)
            optimizer: (string)
        """
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.batchsize = batchsize
        self.precision = precision
        self.activation_fct = activation_fct
        self.opt = optimizer
        self.strategy = strategy
        self.iterations_warmup = iterations_warmup
        self.iterations_benchmark = iterations_benchmark
        self.backprop = backprop

        
    def run_benchmark(self):
        """create and run a tensorflow mirrorStrategy

        Returns:
            timeUsed: time used per iteration in ms
        """
        datatype = eval('tf.float%d' %(self.precision))
        
        GLOBAL_BATCH_SIZE = self.batchsize * self.strategy.num_replicas_in_sync
        
        VecIn = tf.Variable(tf.ones(dtype=datatype,
                            shape=[self.batchsize, self.dim_input]))

        target = tf.Variable(tf.ones([self.batchsize, self.dim_output], dtype=datatype))

        with self.strategy.scope():
            act = eval(self.activation_fct)
            model = tf.keras.Sequential([tf.keras.layers.Dense(
                units=self.dim_output,
                kernel_initializer=tf.compat.v1.ones_initializer(),
                activation = act)])
            optimizer = None
            if self.backprop:
                optimizer = eval('tf.compat.v1.train.%s' % self.opt)
            model.compile(loss='mse', optimizer=optimizer)
       
        train_dataset = tf.data.Dataset.from_tensor_slices((VecIn, target)).batch(GLOBAL_BATCH_SIZE)

        # Fit model and record training time
        tf.keras.backend.clear_session()
        time_callback = TimeHistory()
        model.fit(train_dataset, epochs=self.iterations_benchmark, callbacks=[time_callback], verbose=0)

        times = time_callback.times
        times = times[1:len(times)]  # remove time of the first epoch (overhead)
        return np.mean(times), np.median(times)
