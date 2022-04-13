import torch


@torch.enable_grad()
def fwd_backward(y, x):
    x.requires_grad = True
    dummy = torch.ones_like(y)
    dummy.requires_grad = True
    torch.sum(y).backward()
    g = x.grad * dummy
    torch.sum(g).backward()
    y_x = dummy.grad
    return y_x


if __name__ == '__main__':
    a = torch.tensor([[2., 2.], [2., 2.]])
    # b = torch.full((2, 2), 3, dtype=torch.float)
    a.requires_grad = True
    # b.requires_grad = True
    # c = a * b
    c = torch.tensor([[6., 6.], [6., 6.]])
    c.requires_grad = True
    print(c)
    grad = fwd_backward(c, a)
    print(grad)
