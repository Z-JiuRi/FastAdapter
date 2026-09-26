"""Universal CSI encoder-decoder model factory.

Every encoder maps ``(N, channel, nt, nc)`` to ``(N, code_dim)`` and the
TransNet decoder maps the codeword back to the original CSI tensor shape.
"""


import torch
import torch.nn as nn

from .decoders import TransNetDecoder
from .encoders import AttentionCNNEncoder, CLNetEncoder
from .encoders import CRNetEncoder, CsiNetEncoder, MLPAEEncoder
from .encoders import ResNetCsiEncoder
from .encoders import TransNetEncoder

__all__ = [
    "universal_csi",
    "UniversalCSIModel",
    "build_encoder",
    "build_decoder",
    "select_init_strategy",
    "CsiNetEncoder",
    "CRNetEncoder",
    "CLNetEncoder",
    "TransNetEncoder",
    "ResNetCsiEncoder",
    "AttentionCNNEncoder",
    "MLPAEEncoder",
    "TransNetDecoder",
]


def select_init_strategy(encoder_name, decoder_name):
    encoder_name = encoder_name.lower()
    decoder_name = decoder_name.lower()
    if encoder_name == "transnet" and decoder_name == "transnet":
        return "transnet"
    return "typed"


class UniversalCSIModel(nn.Module):
    def __init__(self, encoder, decoder, init_strategy="typed"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_strategy = init_strategy
        self._reset_parameters(init_strategy)
        if hasattr(self.decoder, "reset_refinement_output"):
            self.decoder.reset_refinement_output()

    def _reset_parameters(self, strategy):
        if strategy == "transnet":
            self._reset_transnet_parameters()
        elif strategy == "typed":
            self._reset_typed_parameters()
        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    def _reset_transnet_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _reset_typed_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                continue

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                                    nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, a=0.3,
                                         nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                    nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        code = self.encode(x)
        return self.decoder(code)

    def encode(self, x):
        return self.encoder(x)



def build_encoder(name, reduction, d_model=64, channel=2, nt=32, nc=32,
                  dim_feedforward=None):
    name = name.lower()
    if name == "csinet":
        return CsiNetEncoder(reduction, channel, nt, nc)
    if name == "crnet":
        return CRNetEncoder(reduction, channel, nt, nc)
    if name == "clnet":
        return CLNetEncoder(reduction, channel, nt, nc)
    if name == "transnet":
        return TransNetEncoder(reduction, d_model, channel, nt, nc,
                               dim_feedforward)
    if name == "resnet":
        return ResNetCsiEncoder(reduction, channel, nt, nc)
    if name == "attention_cnn":
        return AttentionCNNEncoder(reduction, channel, nt, nc)
    if name == "mlp_ae":
        return MLPAEEncoder(reduction, channel, nt, nc)
    raise ValueError(f"Unknown encoder: {name}")


def build_decoder(name, reduction, d_model=64, channel=2, nt=32, nc=32,
                  dim_feedforward=None, hidden=16, num_blocks=2):
    name = name.lower()
    if name == "transnet":
        return TransNetDecoder(reduction, d_model, channel, nt, nc,
                               dim_feedforward)
    raise ValueError(f"Unknown decoder: {name}")


def universal_csi(encoder_name="transnet", reduction=4, d_model=64,
                  channel=2, nt=32, nc=32, dim_feedforward=None,
                  decoder_name="transnet", hidden=16, num_blocks=2):
    encoder = build_encoder(encoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward)
    decoder = build_decoder(decoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward, hidden=hidden,
                            num_blocks=num_blocks)
    init_strategy = select_init_strategy(encoder_name, decoder_name)
    return UniversalCSIModel(encoder, decoder, init_strategy)
