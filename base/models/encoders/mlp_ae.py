"""See README.md for English documentation."""


import torch.nn as nn
from collections import OrderedDict


class MLPAEEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, hidden=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        hidden = hidden or min(4096, input_dim)
        self.net = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(input_dim, hidden)),
            ("norm", nn.LayerNorm(hidden)),
            ("gelu", nn.GELU()),
            ("fc2", nn.Linear(hidden, input_dim // reduction)),
        ]))

    def forward(self, x):
        return self.net(x)
