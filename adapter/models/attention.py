import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *  # noqa: F401,F403
from .paper import *  # noqa: F401,F403

class AffineResidualMLPAttentionMapper(AffineResidualMLPMapper):
    """Residual MLP followed by a delta-aware codeword attention correction."""

    def __init__(self, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 learnable_residual_gate=False, gate_max=0.5,
                 gate_mode="block", final_gate_max=1.0,
                 final_gate_init=1.0, adaptive_gate_hidden=128,
                 attention_dim=32, attention_heads=4,
                 attention_dropout=0.0, attention_scale=0.1,
                 attention_input="value_delta",
                 attention_use_position=True):
        super().__init__(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max, gate_mode=gate_mode,
            final_gate_max=final_gate_max,
            final_gate_init=final_gate_init,
            adaptive_gate_hidden=adaptive_gate_hidden)
        if attention_scale < 0:
            raise ValueError("attention_scale must be non-negative")
        self.code_attention = CodewordDeltaSelfAttention(
            weight.size(0), attention_dim=attention_dim,
            num_heads=attention_heads, dropout=attention_dropout,
            input_mode=attention_input,
            use_position=attention_use_position)
        self.attention_scale = attention_scale
        self.attention_input = attention_input
        self.register_buffer("_mlp_delta_ratio", torch.tensor(0.0))
        self.register_buffer("_attention_delta_ratio", torch.tensor(0.0))

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)

        mlp_delta = x - z0
        attention_delta = self.attention_scale * self.code_attention(
            z0, mlp_delta)
        x = x + attention_delta
        x = self.final_norm(x)
        x = self.final_gate(z0, x - z0)
        with torch.no_grad():
            z0_norm = z0.norm().clamp_min(1e-8)
            self._mlp_delta_ratio.fill_(mlp_delta.norm() / z0_norm)
            self._attention_delta_ratio.fill_(
                attention_delta.norm() / z0_norm)
            self._delta_ratio.fill_((x - z0).norm() / z0_norm)
            self._last_z0 = z0.detach()
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = super().get_metrics()
        values.update({
            "adapter/mlp_delta_ratio": float(self._mlp_delta_ratio.cpu()),
            "adapter/attention_delta_ratio": float(
                self._attention_delta_ratio.cpu()),
            "adapter/attention_dim": float(
                self.code_attention.attention_dim),
            "adapter/attention_heads": float(
                self.code_attention.num_heads),
            "adapter/attention_scale": float(self.attention_scale),
            "adapter/attention_use_position": float(
                self.code_attention.use_position),
        })
        return values


class AffineFiLMResidualMLPMapper(nn.Module):
    """Affine alignment plus residual MLP blocks with static FiLM modulation."""

    def __init__(self, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(
                dim, hidden_dim, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.register_buffer("_delta_ratio", torch.tensor(0.0))

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        with torch.no_grad():
            self._delta_ratio.fill_((x - z0).norm() / z0.norm().clamp_min(1e-8))
        return x

    @torch.no_grad()
    def get_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        values = {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
        for idx, block in enumerate(self.blocks):
            values.update(block.film_metrics(f"adapter/block{idx}"))
        return values


class AffineMultiScaleResidualMLPMapper(nn.Module):
    """Affine alignment plus full+bottleneck residual blocks."""

    def __init__(self, weight, bias, hidden_dim=1024, bottleneck_dim=128,
                 num_blocks=4, dropout=0.0, residual_scale=0.1,
                 use_block_norm=True, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            MultiScaleResidualBlock(
                dim, hidden_dim, bottleneck_dim=bottleneck_dim,
                dropout=dropout, residual_scale=residual_scale,
                use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.register_buffer("_delta_ratio", torch.tensor(0.0))

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        with torch.no_grad():
            self._delta_ratio.fill_((x - z0).norm() / z0.norm().clamp_min(1e-8))
        return x

    @torch.no_grad()
    def get_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        return {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }


class AffineLowRankResidualMapper(nn.Module):
    def __init__(self, weight, bias, rank=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, use_block_norm=True, use_final_norm=False,
                 train_affine=False, learnable_residual_gate=False,
                 gate_max=0.5):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            LowRankResidualBlock(
                dim, rank=rank, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm,
                learnable_gate=learnable_residual_gate,
                gate_max=gate_max)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.register_buffer("_delta_ratio", torch.tensor(0.0))

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        with torch.no_grad():
            self._delta_ratio.fill_((x - z0).norm() / z0.norm().clamp_min(1e-8))
        return x

    @torch.no_grad()
    def get_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        values = {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
        means = []
        maxes = []
        for idx, block in enumerate(self.blocks):
            metrics = block.gate_metrics(f"adapter/block{idx}")
            values.update(metrics)
            if metrics:
                means.append(metrics[f"adapter/block{idx}/gate_mean"])
                maxes.append(metrics[f"adapter/block{idx}/gate_max"])
        if means:
            values["adapter/gate_mean_avg"] = float(sum(means) / len(means))
            values["adapter/gate_max_max"] = float(max(maxes))
        return values

