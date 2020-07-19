"""Benchmark convolution"""

import tensorflow as tf
import numpy as np
import time


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


    def run_benchmark_mirrorStrat(self):
        """create and run a tensorflow mirrorStrategy

        Returns:
            timeUsed: time used per iteration in ms
        """
        datatype = eval('tf.float%d' %(self.precision))
        act = eval(self.activation_fct)
        if self.padding == 'SAME':
            target_size = np.ceil(np.float(self.matsize)/self.strides)
        else:
            target_size = np.ceil(np.float((self.matsize-(self.kernelsize-1)))/self.strides)
        target_size = int(target_size)
        
        GLOBAL_BATCH_SIZE = self.batchsize
        train_images = tf.Variable(tf.ones([
                        self.batchsize,
                        self.matsize,
                        self.matsize,
                        self.channels_in],
                    dtype=datatype))

        target = tf.Variable(tf.ones([
                        self.batchsize,
                        target_size,
                        target_size,
                        self.channels_out],
                    dtype=datatype))


        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, target)).batch(GLOBAL_BATCH_SIZE)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

        def create_model():
            model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=self.channels_out,
                                                                kernel_size=[self.kernelsize,self.kernelsize],
                                                                strides=(self.strides, self.strides),
                                                                padding=self.padding,
                                                                activation = act,
                                                                use_bias=self.use_bias)
                                         ])
            return model

        with self.strategy.scope():
            model = create_model()
            optimizer = None
            if self.backprop:
                optimizer = eval('tf.compat.v1.train.%s' % self.opt)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def train_step(inputs):
            images, target = inputs
            t = time.time()
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss =  tf.reduce_mean( tf.square( predictions - target))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            timeUsed = (time.time()-t) *1000
            return timeUsed

        def train_step_noBP(inputs):
            images, target = inputs
            t = time.time()
            predictions = model(images)#, training=True)
            timeUsed = (time.time()-t) *1000
            return timeUsed

            
        @tf.function
        def distributed_train_step(dataset_inputs):
            timeUsed = self.strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, timeUsed,
                                 axis=None)

        @tf.function
        def distributed_train_step_noBP(dataset_inputs):
            timeUsed = self.strategy.experimental_run_v2(train_step_noBP, args=(dataset_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, timeUsed,
                                 axis=None)


        def backprop_iteration():
            meanTime = 0.0
            for x in train_dist_dataset:
                meanTime += distributed_train_step(x)
            return meanTime

        def no_backprop_iteration():
            meanTime = 0.0
            for x in train_dist_dataset:
                meanTime += distributed_train_step_noBP(x)
            return meanTime

        timeUsed = 0.0
        if self.backprop:
            # Warm-up run
            for _ in range(self.iterations_warmup):
                timeUsed = backprop_iteration()
            # Benchmark run
            timeUsed = 0.0
            for _ in range(self.iterations_benchmark):
                timeUsed += backprop_iteration()
        else:
            # Warm-up run
            for _ in range(self.iterations_warmup):
                timeUsed = no_backprop_iteration()
            # Benchmark run
            timeUsed = 0.0
            t = time.time()
            for _ in range(self.iterations_benchmark):
                timeUsed += no_backprop_iteration()
        timeUsed /= self.iterations_benchmark
            
        return timeUsed
