# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from .resample import UpSample1d, DownSample1d
import torch
import torch.nn.functional as F



class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    '''
    抗锯齿非线性的工作原理是沿时间维度对信号进行 2 倍上采样，应用 Snake 激活，然后对信号进行 2 倍下采样，
    这是受奈奎斯特-香农采样定理 (Shannon，1949) 启发的常见做法。每个上采样和下采样操作都伴随着低通滤波器，
    该低通滤波器使用带有 Kaiser 窗的加窗 sinc 滤波器
    '''
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x



class Activation1d_MAB(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

        self.weights = nn.Parameter(torch.randn(2))

    # x: [B,C,T]
    '''
    抗锯齿非线性的工作原理是沿时间维度对信号进行 2 倍上采样，应用 Snake 激活，然后对信号进行 2 倍下采样，
    这是受奈奎斯特-香农采样定理 (Shannon，1949) 启发的常见做法。每个上采样和下采样操作都伴随着低通滤波器，
    该低通滤波器使用带有 Kaiser 窗的加窗 sinc 滤波器
    '''
    def forward(self, x):
        weights = F.softmax(self.weights, dim=0)
        x = self.upsample(x)
        snake_activation = self.act(x)
        relu_activation = F.relu(x)
        weighted_sum = weights[0] * snake_activation + weights[1] * relu_activation
        x = self.downsample(weighted_sum)

        return x


class CustomModuleWithResidual(nn.Module):
    def __init__(self):
        super(CustomModuleWithResidual, self).__init__()
        self.weights = nn.Parameter(torch.randn(2))
        # 可选：如果需要调整x的维度以匹配加权激活结果的维度
        # self.adjust_dim = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # 注意：这里的in_channels和out_channels需要根据实际情况设置

    def forward(self, x):
        weights = F.softmax(self.weights, dim=0)
        snake_activation = torch.sigmoid(x)  # 假设的激活函数
        relu_activation = F.relu(x)
        weighted_sum = weights[0] * snake_activation + weights[1] * relu_activation

        # 如果x的维度需要调整
        # x_adjusted = self.adjust_dim(x)
        # result = x_adjusted + weighted_sum

        # 不需要调整维度的情况
        result = x + weighted_sum
        return result
