import torch
import numpy as np
import scipy.io
import time
import sys
from Net import mean_squared_error, relative_error, Navier_Stoeks_3D, Gradient_Velocity_NS_3D, \
    Pinn_net


class PINN(object):
    def __int__(self, t_data, x_data, y_data, z_data, t_eqns, x_eqns, y_eqns, z_eqns,
                layers, batch_size, Rey):
        # define often
        self.layers = layers
        self.batch_size = batch_size
        self.Rey = Rey
        # define data
        [self.t_data, self.x_data, self.y_data, self.z_data] = [t_data, x_data, y_data, z_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]
