"""
@author: Xiao Peng
"""

import numpy as np


def mean_squared_error(prediction, exact):
    if type(prediction) is np.ndarray:
        return np.mean(np.square(prediction - exact))
    return tf.reduce_mean(tf.square(prediction - exact))


class Cnn_net(object):
    def __init__(self, *inputs, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])