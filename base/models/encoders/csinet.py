"""See README.md for English documentation."""


import torch.nn as nn
from collections import OrderedDict


class CsiNetEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32):
        super().__init__()
        input_dim = channel * nt * nc
        code_dim = input_dim // reduction
        self.features = nn.Sequential(OrderedDict([
            ("conv3x3", nn.Conv2d(channel, channel, 3, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(channel)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.fc = nn.Linear(input_dim, code_dim)

    def forward(self, x):
        out = self.features(x)
        return self.fc(out.flatten(1))
