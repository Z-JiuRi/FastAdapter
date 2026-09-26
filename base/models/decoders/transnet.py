"""See README.md for English documentation."""


import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class TransNetDecoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.feature_shape = (input_dim // d_model, d_model)
        self.fc_decoder = nn.Linear(input_dim // reduction, input_dim)
        decoder_layer = TransformerDecoderLayer(d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2, norm=nn.LayerNorm(d_model))

    def forward(self, code):
        batch_size = code.size(0)
        memory = self.fc_decoder(code).view(batch_size, self.feature_shape[0],
                                            self.feature_shape[1])
        out = self.decoder(memory, memory)
        out = out.view(batch_size, self.channel, self.nt, self.nc)
        return out
