"""Benchmark fully connected layer / matrix multiplication"""

import tensorflow as tf
import numpy as np
import time


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

    
    def run_benchmark_mirrorStrat(self):
        """create and run a tensorflow mirrorStrategy

        Returns:
            timeUsed: time used per iteration in ms
        """
        datatype = eval('tf.float%d' %(self.precision))
        act = eval(self.activation_fct)
        
        GLOBAL_BATCH_SIZE = self.batchsize
        
        VecIn = tf.Variable(tf.ones(dtype=datatype,
                            shape=[self.batchsize,self.dim_input]))

        target = tf.Variable(
                            tf.ones([
                                    self.batchsize,
                                    self.dim_output],
                            dtype=datatype))


        train_dataset = tf.data.Dataset.from_tensor_slices((VecIn, target)).batch(GLOBAL_BATCH_SIZE)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

        def create_model():
            model = tf.keras.Sequential([tf.keras.layers.Dense(units=self.dim_output,
                            kernel_initializer=tf.compat.v1.ones_initializer(),
                            activation = act)
                                         ])
            return model

        with self.strategy.scope():
            model = create_model()
            optimizer = None
            if self.backprop:
                optimizer = eval('tf.compat.v1.train.%s' % self.opt)

        def train_step(inputs):
            VecIn, target = inputs
            t = time.time()
            with tf.GradientTape() as tape:
                predictions = model(VecIn, training=True)
                loss =  tf.reduce_mean( tf.square( predictions - target))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            timeUsed = (time.time()-t) *1000
            
            return timeUsed

        def train_step_noBP(inputs):
            VecIn, target = inputs
            t = time.time()
            predictions = model(VecIn)
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
