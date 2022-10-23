import torch
import numpy as np
import scipy.io
import time
import sys
import torch.nn as nn
from Net import mean_squared_error, relative_error, Navier_Stoeks_3D, Gradient_Velocity_NS_3D, \
    Pinn_net


class PINN(nn.Module):
    def __int__(self, t_data, x_data, y_data, z_data,
                t_eqns, x_eqns, y_eqns, z_eqns,
                layers, batch_size, Rey):
        # define often
        self.layers = layers
        self.batch_size = batch_size
        self.Rey = Rey

        # define data
        [self.t_data, self.x_data, self.y_data, self.z_data] = [t_data, x_data, y_data, z_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]

        self.net_cuvp = Pinn_net(self.t_data, self.x_data, self.y_data, layers=self.layers)

        [self.u_data_prediction,
         self.v_data_prediction,
         self.w_data_prediction,
         self.p_data_prediction] = self.net_cuvp(self.t_data,
                                                 self.x_data,
                                                 self.y_data,
                                                 self.z_data,
                                                 layers=self.layers)
        [self.u_eqns_prediction,
         self.v_eqns_prediction,
         self.w_eqns_prediction,
         self.p_eqns_prediction] = self.net_cuvp(self.t_eqns,
                                                 self.x_eqns,
                                                 self.y_eqns,
                                                 self.z_eqns,
                                                 layers=self.layers)
        [self.error1_eqns_prediciton,
         self.error2_eqns_prediciton,
         self.error3_eqns_prediction,
         self.error4_eqns_prediction] = Navier_Stoeks_3D(self.u_eqns_prediction,
                                                         self.v_eqns_prediction,
                                                         self.w_eqns_prediction,
                                                         self.p_eqns_prediction,
                                                         self.Rey)

        self.loss = mean_squared_error(self.error1_eqns_prediciton, 0.0) + \
                    mean_squared_error(self.error2_eqns_prediciton, 0.0) + \
                    mean_squared_error(self.error3_eqns_prediciton, 0.0) + \
                    mean_squared_error(self.error4_eqns_prediciton, 0.0)
