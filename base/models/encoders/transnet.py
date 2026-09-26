"""See README.md for English documentation."""


import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransNetEncoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        code_dim = input_dim // reduction
        self.feature_shape = (input_dim // d_model, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, code_dim)

    def forward(self, x):
        batch_size = x.size(0)
        memory = self.encoder(x.view(batch_size, self.feature_shape[0],
                                     self.feature_shape[1]))
        return self.fc(memory.flatten(1))
