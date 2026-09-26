import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *  # noqa: F401,F403

class AffineResidualMLPMapper(nn.Module):
    """Closed-form affine alignment followed by identity-initialized residual MLP.

    Data flow:
        source_code -> source @ W + b -> z0
        z0 -> residual blocks -> mapped_code

    W and b are registered as buffers by default, so training only updates the
    residual MLP. This keeps the model initialized exactly at the offline
    affine solution.
    """

    def __init__(self, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 learnable_residual_gate=False, gate_max=0.5,
                 gate_mode="block", final_gate_max=1.0,
                 final_gate_init=1.0, adaptive_gate_hidden=128):
        super().__init__()
        if weight.ndim != 2 or weight.size(0) != weight.size(1):
            raise ValueError(f"weight must be square 2D, got {tuple(weight.shape)}")
        if bias.ndim != 1 or bias.size(0) != weight.size(1):
            raise ValueError(
                f"bias must be ({weight.size(1)},), got {tuple(bias.shape)}")
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, hidden_dim, dropout, residual_scale,
                          use_norm=use_block_norm,
                          learnable_gate=learnable_residual_gate and gate_mode == "block",
                          gate_max=gate_max)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        if gate_mode == "block":
            final_mode = "none"
        elif gate_mode in ("none", "final_static", "final_adaptive",
                           "final_unbounded"):
            final_mode = gate_mode
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        self.final_gate = FinalGate(
            dim, mode=final_mode, gate_max=final_gate_max,
            gate_init=final_gate_init, adaptive_hidden=adaptive_gate_hidden)
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self._last_z0 = None

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.final_gate(z0, x - z0)
        with torch.no_grad():
            self._delta_ratio.fill_((x - z0).norm() / z0.norm().clamp_min(1e-8))
            self._last_z0 = z0.detach()
        return x

    def gate_regularization(self):
        return self.final_gate.regularization()

    @torch.no_grad()
    def get_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        return {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        } | self._gate_metrics() | self.final_gate.metrics(self._last_z0)

    @torch.no_grad()
    def _gate_metrics(self):
        values = {}
        means = []
        maxes = []
        for idx, block in enumerate(self.blocks):
            block_metrics = block.gate_metrics(f"adapter/block{idx}")
            values.update(block_metrics)
            if block_metrics:
                means.append(block_metrics[f"adapter/block{idx}/gate_mean"])
                maxes.append(block_metrics[f"adapter/block{idx}/gate_max"])
        if means:
            values["adapter/gate_mean_avg"] = float(sum(means) / len(means))
            values["adapter/gate_max_max"] = float(max(maxes))
        return values

