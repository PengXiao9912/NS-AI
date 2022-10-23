"""
@author: Xiao Peng
"""

import numpy as np
from torch.autograd import Variable
import torch


# 均方误差 范数
def mean_squared_error(prediction, exact):
    if type(prediction) is np.ndarray:
        return np.mean(np.square(prediction - exact))
    return torch.mean(torch.square(prediction - exact))


# 相对误差
def relative_error(prediction, exact):
    if type(prediction) is np.ndarray:
        return np.sqrt(np.mean(np.square(prediction - exact)) / np.mean(np.square(exact - np.mean(exact))))
    return torch.sqrt(
        torch.mean(torch.square(prediction - exact)) / torch.mean(torch.square(exact - torch.mean(exact))))
    # torch.sqrt导函数定义域(0,无穷大


# sdasd @


# 前馈梯度传递-未改
def forward_gradients(y, x):
    x.requires_gradients = True
    dummy = torch.ones_like(y)  # 每个x的权重均为1
    g = tf.gradients(y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    y_x = tf.gradients(g, dummy, colocate_gradients_with_ops=True)[0]
    return y_x


def Navier_Stoeks_3D(u, v, w, p, t, x, y, z, Rey):
    Total = torch.cat([u, v, w, p], 1)
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


def Gradient_Velocity_NS_3D(u, v, w, x, y, z):
    Total = torch.cat([u, v, w], 1)
    Total_x = forward_gradients(Total, x)
    Total_y = forward_gradients(Total, y)
    Total_z = forward_gradients(Total, z)
    u_x = Total_x[:, 0:1]
    v_x = Total_x[:, 1:2]
    w_x = Total_x[:, 2:3]

    u_y = Total_y[:, 0:1]
    v_y = Total_y[:, 1:2]
    w_y = Total_y[:, 2:3]

    u_z = Total_z[:, 0:1]
    v_z = Total_z[:, 1:2]
    w_z = Total_z[:, 2:3]
    return [u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z]


class Pinn_net(object):
    def __init__(self, *inputs, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)  # 列扩充
            self.X_mean = X.mean(0, keepdims=True)  # 计算每个数据集的平均值
            self.X_std = X.std(0, keepdims=True)  # 计算每个数据集的标准差
        self.weights = []
        self.biases = []
        self.alpha = []
        for L in range(0, self.num_layers - 1):
            in_dim = self.layers[1]
            out_dim = self.layers[L + 1]
            w = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            a = np.ones([1, out_dim])
            self.weights.append(tf.Variable(w, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.alpha.append(tf.Variable(a, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):
        Differ = (torch.cat(inputs, 1) - self.X_mean) / self.X_std
        for L in range(0, self.num_layers - 1):
            w = self.weights[L]
            b = self.biases[L]
            a = self.alpha[L]
            W = w / tf.norm(w, axis=0, keepdims=True)
            Differ = tf.matmul(Differ, W)
            Differ = a * Differ + b
            if L < self.num_layers - 2:
                Differ = Differ * tf.sigmoid(Differ)
        Final = tf.split(Differ, num_or_size_splits=Differ.shape[1], axis=1)
        return Final


class Cnn_net(object):
    def __init__(self, *inputs, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
