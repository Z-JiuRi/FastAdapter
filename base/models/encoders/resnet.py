"""See README.md for English documentation."""


import torch.nn as nn
from collections import OrderedDict


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
        ]))


class CsiResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(channels, channels, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv2", ConvBN(channels, channels, 3)),
        ]))
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNetCsiEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, width=16,
                 blocks=(1, 1, 1)):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % 4 == 0 and nc % 4 == 0
        self.stem = nn.Sequential(OrderedDict([
            ("conv", ConvBN(channel, width, 3)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.stage1 = self._make_stage(width, width, blocks[0], stride=1)
        self.stage2 = self._make_stage(width, 2 * width, blocks[1], stride=2)
        self.stage3 = self._make_stage(2 * width, 4 * width, blocks[2],
                                       stride=2)
        feature_dim = 4 * width * (nt // 4) * (nc // 4)
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        if stride != 1 or in_channels != out_channels:
            layers.extend([
                ("downsample", ConvBN(in_channels, out_channels, 3,
                                      stride=stride)),
                ("downsample_relu", nn.LeakyReLU(negative_slope=0.3,
                                                 inplace=True)),
            ])
        for idx in range(num_blocks):
            layers.append((f"res{idx + 1}", CsiResidualBlock(out_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return self.fc(out.flatten(1))
