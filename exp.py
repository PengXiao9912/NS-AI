import torch
import torch.nn as nn


class PinnNet(nn.Module):
    def __init__(self, *inputs, layers):
        super(PinnNet, self).__init__()
        self.layers = layers
        self.num_layers = len(self.layers)
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = torch.zeros([1, in_dim])
            self.X_std = torch.ones([1, in_dim])
        else:
            X = torch.cat(inputs, dim=1)  # 列扩充
            self.X_std, self.X_mean = torch.std_mean(X, dim=1, keepdim=True)  # 计算每个数据集的标准差平均值

        self.weights = []
        self.biases = []
        self.alpha = []

        for L in range(0, self.num_layers - 1):
            in_dim = self.layers[1]
            out_dim = self.layers[L + 1]
            w = torch.normal(mean=0, std=1, size=(in_dim, out_dim))
            b = torch.zeros([1, out_dim])
            a = torch.ones([1, out_dim])

            # w = np.random.normal(size=[in_dim, out_dim])
            # b = np.zeros([1, out_dim])
            # a = np.ones([1, out_dim])

            self.weights.append(torch.tensor(w, dtype=torch.float32, requires_grad=True))
            self.biases.append(torch.tensor(b, dtype=torch.float32, requires_grad=True))
            self.alpha.append(torch.tensor(a, dtype=torch.float32, requires_grad=True))

    def forward(self, *inputs):
        differ = (torch.cat(inputs, dim=1) - self.X_mean) / self.X_std
        for layer in range(0, self.num_layers):  # TODO original (self.num_layers - 1)
            w = self.weights[layer]
            b = self.biases[layer]
            a = self.alpha[layer]
            W = w / torch.norm(w, dim=0, keepdim=True)
            differ = differ @ W  # or torch.matmul
            differ = a * differ + b
            if layer < self.num_layers - 2:
                differ = differ * torch.sigmoid(differ)

        final = torch.split(differ, split_size_or_sections=1, dim=1)
        return final


if __name__ == '__main__':
    layers = [4] + 10 * [5 * 50] + [5]
    x = [torch.rand(3, 5)] * 4
    net = PinnNet(*x, layers=layers)
    x = net(*x)
    print(x)
