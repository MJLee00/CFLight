import glob
import logging
import shutil
import subprocess
import time

import numpy
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import matplotlib.pyplot as plt
import os
import scipy.stats as ss
import math
import pandas as pd



class Qnetwork():
    def __init__(self, h_size, action_num):
        # The network recieves a state from the sumo, flattened into an array.
        # It then resizes it and processes it through three convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 3600, 2], dtype=tf.float32)
        # self.legal_actions = tf.placeholder(shape=[None, action_num], dtype=tf.float32)

        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 60, 60, 2])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[4, 4], stride=[2, 2],
                                 padding='VALID', activation_fn=self.relu, biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                 activation_fn=self.relu, biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=128, kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                 activation_fn=self.relu, biases_initializer=None)

        # It is split into Value and Advantage
        self.stream = slim.flatten(self.conv3)
        self.stream0 = slim.fully_connected(self.stream, 128, activation_fn=self.relu)

        self.streamA = self.stream0
        self.streamV = self.stream0

        self.streamA0 = slim.fully_connected(self.streamA, h_size, activation_fn=self.relu)
        self.streamV0 = slim.fully_connected(self.streamV, h_size, activation_fn=self.relu)

        xavier_init = tf.contrib.layers.xavier_initializer()
        action_num = np.int32(action_num)
        self.AW = tf.Variable(xavier_init([h_size, action_num]))
        self.VW = tf.Variable(xavier_init([h_size, 1]))
        self.Advantage = tf.matmul(self.streamA0, self.AW)
        self.Advantage = tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))\
                         /(tf.math.reduce_std(self.Advantage, axis=1, keepdims=True) + 1e-5)
        self.Value = tf.matmul(self.streamV0, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        # The final Q value is the addition of the Q value and penelized value for illegal actions
        # self.Qout = tf.add(self.Qout0, self.legal_actions)
        # The predicted action
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the mean square error between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, np.int32(action_num), dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        #         self.Q = tf.reduce_sum(self.Qout, axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

    def relu(self, x, alpha=0.01, max_value=None):
        '''ReLU.

        alpha: slope of negative section.
        '''
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                                 tf.cast(max_value, dtype=tf.float32))
        x -= tf.constant(alpha, dtype=tf.float32) * negative_part
        return x
