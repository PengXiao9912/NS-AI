"""
@author: Xiao Peng
"""

import numpy as np


# 均方误差 范数
def mean_squared_error(prediction, exact):
    if type(prediction) is np.ndarray:
        return np.mean(np.square(prediction - exact))
    return tf.reduce_mean(tf.square(prediction - exact))


# 相对误差
def relative_error(prediction, exact):
    if type(prediction) is np.ndarray:
        return np.sqrt(np.mean(np.square(prediction - exact)) / np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(
        tf.reduce_mean(tf.square(prediction - exact)) / tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))


# 前馈梯度传递
def forward_gradients(y, x):
    dummy = tf.ones_like(y)  # 每个x的权重均为1
    g = tf.gradients(y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    y_x = tf.gradients(g, dummy, colocate_gradients_with_ops=True)[0]
    return y_x


def Navier_Stoeks_3D(u, v, w, p, t, x, y, z, Rey):
    Total = tf.contact([u, v, w, p], 1)
    Total_t = forward_gradients(Total, t)
    Total_x = forward_gradients(Total, x)
    Total_y = forward_gradients(Total, y)
    Total_z = forward_gradients(Total, z)
    Total_xx = forward_gradients(Total_x, x)
    Total_yy = forward_gradients(Total_y, y)
    Total_zz = forward_gradients(Total_z, z)

    u = Total[:, 0:1]
    v = Total[:, 1:2]
    w = Total[:, 2:3]
    p = Total[:, 3:4]

    u_t = Total_t[:, 0:1]
    v_t = Total_t[:, 1:2]
    w_t = Total_t[:, 2:3]
    p_t = Total_t[:, 3:4]

    u_x = Total_x[:, 0:1]
    v_x = Total_x[:, 1:2]
    w_x = Total_x[:, 2:3]
    p_x = Total_x[:, 3:4]

    u_y = Total_y[:, 0:1]
    v_y = Total_y[:, 1:2]
    w_y = Total_y[:, 2:3]
    p_y = Total_y[:, 3:4]

    u_z = Total_z[:, 0:1]
    v_z = Total_z[:, 1:2]
    w_z = Total_z[:, 2:3]
    p_z = Total_z[:, 3:4]

    u_xx = Total_xx[:, 0:1]
    v_xx = Total_xx[:, 1:2]
    w_xx = Total_xx[:, 2:3]

    u_yy = Total_yy[:, 0:1]
    v_yy = Total_yy[:, 1:2]
    w_yy = Total_yy[:, 2:3]

    u_zz = Total_zz[:, 0:1]
    v_zz = Total_zz[:, 1:2]
    w_zz = Total_zz[:, 2:3]

    error1 =
    error2 =
    error3 =
    error4 =

    class Cnn_net(object):
        def __init__(self, *inputs, layers):
            self.layers = layers
            self.num_layers = len(self.layers)
            if len(inputs) == 0:
                in_dim = self.layers[0]
                self.X_mean = np.zeros([1, in_dim])
                self.X_std = np.ones([1, in_dim])
