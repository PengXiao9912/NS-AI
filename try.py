import torch
import numpy as np


@torch.enable_grad()
def fwd_backward(y, x):
    dummy = torch.ones_like(y)
    dummy.requires_grad = True
    torch.sum(y).backward()
    g = x.grad * dummy
    torch.sum(g).backward()
    y_x = dummy.grad
    return y_x


if __name__ == '__main__':
    ''' a = torch.tensor([[2., 2.], [2., 2.]])
    b = torch.full((2, 2), 3, dtype=torch.float)
    a.requires_grad = True
    b.requires_grad = True
    c = a * b
    print(c)
    grad = fwd_backward(c, a)
    print(grad)'''
    # x_in_neg_one = -np.ones((2000, 1))
    # print(x_in_neg_one)
    x = torch.Tensor([[1, 2], [3, 4]])
    y = torch.square(x)
    z = torch.mul(x, y)
    print(y)
    print(z)
