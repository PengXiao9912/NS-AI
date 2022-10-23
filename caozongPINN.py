import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim
from torch.autograd import Variable
import numpy as np
from matplotlib import cm


# 模型搭建
class Net(nn.Module):
    def __init__(self, NN):  # NL n个l（线性，全连接）隐藏层， NN 输入数据的维数， 128 256
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()

        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer1 = nn.Linear(NN, int(NN / 2))  ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优
        self.hidden_layer2 = nn.Linear(int(NN / 2), int(NN / 2))  ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优
        self.output_layer = nn.Linear(int(NN / 2), 1)

    def forward(self, x):  # 一种特殊的方法 __call__() 回调
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out_final = self.output_layer(out)
        return out_final


X_u = 1  # 直航阻力
Y_u = 1  # 直航阻力
N_u = 1  # 直航阻力矩


# X方向水动力系数
def pde_e1(v, r, net_xh):
    x_h = net_xh(x)
    # 网络得到的数据
    v_2 = torch.square(v)
    v_3 = torch.mul(v_2, v)
    v_r = torch.mul(v, r)
    r_2 = torch.square(r)
    r_3 = torch.mul(r, r_2)
    v_rr = torch.mul(v, r_2)
    vv_r = torch.mul(v_2, r)

    # 求水动力导数
    x_v = torch.autograd.grad(x_h, v, grad_outputs=torch.ones_like(net(x)),
                              create_graph=True, allow_unused=True)[0]
    x_r = torch.autograd.grad(x_h, r, grad_outputs=torch.ones_like(net(x)),
                              create_graph=True, allow_unused=True)[0]
    x_vv = torch.autograd.grad(x_h, v_2, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    x_vr = torch.autograd.grad(x_h, v_r, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    x_rr = torch.autograd.grad(x_h, r_2, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    x_vvv = torch.autograd.grad(x_h, v_3, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    x_vvr = torch.autograd.grad(x_h, vv_r, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    x_vrr = torch.autograd.grad(x_h, v_rr, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    x_rrr = torch.autograd.grad(x_h, r_3, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]

    e1 = X_u + torch.mul(x_v, v) + torch.mul(x_r, r) + torch.mul(x_vv, v_2) + \
         torch.mul(x_rr, r_2) + torch.mul(x_vr, v_r) + torch.mul(x_vvv, v_3) + \
         torch.mul(x_rrr, r_3) + torch.mul(x_vrr, v_rr) + torch.mul(x_vvr, vv_r)

    return e1  # 公式X_H


# Y方向水动力系数
def pde_e2(v, r, net_yh):
    y_h = net_yh(x)
    # 网络得到的数据
    v_2 = torch.square(v)
    v_3 = torch.mul(v_2, v)
    v_r = torch.mul(v, r)
    r_2 = torch.square(r)
    r_3 = torch.mul(r, r_2)
    v_rr = torch.mul(v, r_2)
    vv_r = torch.mul(v_2, r)

    # 求水动力导数
    y_v = torch.autograd.grad(y_h, v, grad_outputs=torch.ones_like(net(x)),
                              create_graph=True, allow_unused=True)[0]
    y_r = torch.autograd.grad(y_h, r, grad_outputs=torch.ones_like(net(x)),
                              create_graph=True, allow_unused=True)[0]
    y_vv = torch.autograd.grad(y_h, v_2, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    y_vr = torch.autograd.grad(y_h, v_r, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    y_rr = torch.autograd.grad(y_h, r_2, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    y_vvv = torch.autograd.grad(y_h, v_3, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    y_vvr = torch.autograd.grad(y_h, vv_r, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    y_vrr = torch.autograd.grad(y_h, v_rr, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    y_rrr = torch.autograd.grad(y_h, r_3, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]

    e2 = Y_u + torch.mul(y_v, v) + torch.mul(y_r, r) + torch.mul(y_vv, v_2) + \
         torch.mul(y_rr, r_2) + torch.mul(y_vr, v_r) + torch.mul(y_vvv, v_3) + \
         torch.mul(y_rrr, r_3) + torch.mul(y_vrr, v_rr) + torch.mul(y_vvr, vv_r)

    return e2  # 公式Y_H


# N方向水动力系数
def pde_e3(v, r, net_nh):
    n_h = net_nh(x)
    # 网络得到的数据
    v_2 = torch.square(v)
    v_3 = torch.mul(v_2, v)
    v_r = torch.mul(v, r)
    r_2 = torch.square(r)
    r_3 = torch.mul(r, r_2)
    v_rr = torch.mul(v, r_2)
    vv_r = torch.mul(v_2, r)

    # 求水动力导数
    n_v = torch.autograd.grad(n_h, v, grad_outputs=torch.ones_like(net(x)),
                              create_graph=True, allow_unused=True)[0]
    n_r = torch.autograd.grad(n_h, r, grad_outputs=torch.ones_like(net(x)),
                              create_graph=True, allow_unused=True)[0]
    n_vv = torch.autograd.grad(n_h, v_2, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    n_vr = torch.autograd.grad(n_h, v_r, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    n_rr = torch.autograd.grad(n_h, r_2, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]
    n_vvv = torch.autograd.grad(n_h, v_3, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    n_vvr = torch.autograd.grad(n_h, vv_r, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    n_vrr = torch.autograd.grad(n_h, v_rr, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]
    n_rrr = torch.autograd.grad(n_h, r_3, grad_outputs=torch.ones_like(net(x)),
                                create_graph=True, allow_unused=True)[0]

    e3 = X_u + torch.mul(n_v, v) + torch.mul(n_r, r) + torch.mul(n_vv, v_2) + \
         torch.mul(n_rr, r_2) + torch.mul(n_vr, v_r) + torch.mul(n_vvv, v_3) + \
         torch.mul(n_rrr, r_3) + torch.mul(n_vrr, v_rr) + torch.mul(n_vvr, vv_r)

    return e3  # 公式N_H


net = Net(30)
mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# 初始化 常量
t_bc_zeros = np.zeros((2000, 1))
x_in_pos_one = np.ones((2000, 1))
x_in_neg_one = -np.ones((2000, 1))
u_in_zeros = np.zeros((2000, 1))

iterations = 50000
for epoch in range(iterations):
    optimizer.zero_grad()  # 梯度归0

    # 求边界条件的误差
    # 初始化变量
    t_in_var = np.random.uniform(low=0, high=1.0, size=(2000, 1))
    x_bc_var = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
    u_bc_sin = -np.sin(np.pi * x_bc_var)

    # 将数据转化为torch可用的
    pt_x_bc_var = Variable(torch.from_numpy(x_bc_var).float(), requires_grad=False)
    pt_t_bc_zeros = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False)
    pt_u_bc_sin = Variable(torch.from_numpy(u_bc_sin).float(), requires_grad=False)
    pt_x_in_pos_one = Variable(torch.from_numpy(x_in_pos_one).float(), requires_grad=False)
    pt_x_in_neg_one = Variable(torch.from_numpy(x_in_neg_one).float(), requires_grad=False)
    pt_t_in_var = Variable(torch.from_numpy(t_in_var).float(), requires_grad=False)
    pt_u_in_zeros = Variable(torch.from_numpy(u_in_zeros).float(), requires_grad=False)

    # 求边界条件的损失
    net_bc_out = net(torch.cat([pt_t_bc_zeros, pt_x_bc_var], 1))
    # u(x,t)的输出
    mse_u_2 = mse_cost_function(net_bc_out, pt_u_bc_sin)
    # e = u(x,t)-(-sin(pi*x))  公式（2）

    net_bc_inr = net(torch.cat([pt_t_in_var, pt_x_in_pos_one], 1))
    # 0=u(t,1) 公式（3)
    net_bc_inl = net(torch.cat([pt_t_in_var, pt_x_in_neg_one], 1))
    # 0=u(t,-1) 公式（4）

    mse_u_3 = mse_cost_function(net_bc_inr, pt_u_in_zeros)
    # e = 0-u(t,1) 公式(3)
    mse_u_4 = mse_cost_function(net_bc_inl, pt_u_in_zeros)
    # e = 0-u(t,-1) 公式（4）

    # 求PDE函数式的误差
    # 初始化变量
    x_collocation = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))
    all_zeros = np.zeros((2000, 1))
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False)

    # 将变量x,t带入公式（1）
    f_out = pde(torch.cat([pt_t_collocation, pt_x_collocation], 1), net)  # output of f(x,t) 公式（1）
    mse_f_1 = mse_cost_function(f_out, pt_all_zeros)

    # 将误差(损失)累加起来
    loss = mse_f_1 + mse_u_2 + mse_u_3 + mse_u_4

    loss.backward()  # 反向传播
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)

## 画图 ##
t = np.linspace(0, 1, 100)
x = np.linspace(-1, 1, 256)
ms_t, ms_x = np.meshgrid(t, x)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
pt_u0 = net(torch.cat([pt_t, pt_x], 1))
u = pt_u0.data.cpu().numpy()

pt_u0 = u.reshape(256, 100)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([-1, 1])
ax.plot_surface(ms_t, ms_x, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
plt.savefig('D:\Preddata.png')
plt.close(fig)
