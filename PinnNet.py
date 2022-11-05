import torch.nn as nn
import scipy.io
from typing import Sequence, Optional
from mmengine.model import BaseModule
from mmengine.config import ConfigDict, Config
from mmengine.registry import MODELS

MODELS.register_module(name='ReLU', module=nn.ReLU)
MODELS.register_module(name='Linear', module=nn.Linear)


# class PinnNetOriginal(nn.Module):
#     def __init__(self):
#         super(PinnNetOriginal, self).__init__()
#         # self.layers = layers  # TODO layers=[7,20,20,20,20,2]
#         # 定义网络层结构与激活函数
#         self.hidden_layer1 = nn.Sequential(
#             nn.Linear(in_features=4, out_features=20, bias=True),
#             nn.ReLU())
#         self.hidden_layer2 = nn.Sequential(
#             nn.Linear(in_features=20, out_features=20, bias=True),
#             nn.ReLU())
#         self.hidden_layer3 = nn.Sequential(
#             nn.Linear(in_features=20, out_features=20, bias=True),
#             nn.ReLU())
#         self.hidden_layer4 = nn.Sequential(
#             nn.Linear(in_features=20, out_features=20, bias=True),
#             nn.ReLU())
#         self.output_layer = nn.Sequential(
#             nn.Linear(in_features=20, out_features=2, bias=True),
#             nn.ReLU())
#         # self.num_layers = len(self.layers)
#         # for L in range(1, self.num_layers):
#         # layer_name = f'self.layer{L + 1}'
#         # self.add_module(layer_name, nn.Linear(self.layers[L], self.layers[L + 1], bias=True))
#
#     def forward(self, total_input):
#         hidden_out1 = self.hidden_layer1(total_input)
#         hidden_out2 = self.hidden_layer2(hidden_out1)
#         hidden_out3 = self.hidden_layer3(hidden_out2)
#         hidden_out4 = self.hidden_layer4(hidden_out3)
#         output = self.output_layer(hidden_out4)
#         return output


# class PinnNetV1(nn.Module):
#     def __init__(self,
#                  channels: Sequence[int] = (4, 20, 20, 20, 20, 2),
#                  act: nn.Module = nn.ReLU,
#                  last_act: nn.Module = None,
#                  bias=True):
#         super(PinnNetV1, self).__init__()
#         self.layers = []
#         # 循环定义网络层架构
#         for i, channel in enumerate(channels):
#             layer = nn.Linear(channels[i], channels[i + 1], bias=bias)
#             layer_name = f'layer_{i}'
#             self.layers.append(layer_name)
#             self.add_module(layer_name, layer)
#         #定义激活函数
#         self.act = act()
#         self.last_act = last_act() if last_act is not None else last_act
#
#         self._layer_num = len(self.layers)
#
#     def forward(self, x):
#         for i, layer_name in enumerate(self.layers):
#             layer = getattr(self, layer_name)
#             x = layer(x)
#
#             if self.last_act is not None and i == (self.layer_num - 1):
#                 x = self.last_act(x)
#             else:
#                 x = self.act(x)
#
#         return x
#
#     @property
#     def layer_num(self):
#         return self._layer_num


@MODELS.register_module()
class LinearModule(BaseModule):
    def __init__(self,
                 linear_cfg: ConfigDict,
                 bn_cfg: Optional[ConfigDict] = None,
                 act_cfg: Optional[ConfigDict] = None,
                 init_cfg=None):
        super(LinearModule, self).__init__(init_cfg)
        if act_cfg is None:
            act_cfg = dict(type='Relu')

        self.linear = MODELS.build(linear_cfg)
        self.act = MODELS.build(act_cfg)

        if bn_cfg is not None:
            self.bn = MODELS.build(bn_cfg)

    def forward(self, x):
        x = self.linear(x)

        if hasattr(self, 'bn'):
            x = self.bn(x)

        x = self.act(x)
        return x


@MODELS.register_module()
class PinnNet(BaseModule):
    def __init__(self,
                 linear_module_cfgs: Sequence[ConfigDict],
                 init_cfg=None):
        super(PinnNet, self).__init__(init_cfg)
        self.layers = []
        # 通过枚举法、根据每一网络层的cfg进行搭建
        for i, lm_cfg in enumerate(linear_module_cfgs):
            layer = MODELS.build(lm_cfg)
            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    cfg_path = 'pinn_cfg.py'
    cfg = Config.fromfile(cfg_path)

    model = MODELS.build(cfg.model)
    print(model)

################################################################
    # load data and data normalization
###############################################################




    # batch_size = 1000
    # learning_rate = 0.001
    # epoch = 50
    # data = scipy.io.loadmat('../Data/DirectSailingMovement2D.mat')
    # t = data['t']
    # v_i = data['v_i']
    # v_j = data['v_j']
    # v_k = data['v_k']
    # p_i = data['p_i']
    # p_j = data['p_j']
