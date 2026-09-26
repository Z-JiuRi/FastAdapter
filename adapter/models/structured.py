import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *  # noqa: F401,F403

class BottleneckResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.0,
                 residual_scale=0.1, use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        init_zero_linear(self.up)

    def forward(self, x):
        delta = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + self.residual_scale * delta


class AffineBottleneckResidualMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, bottleneck_dim=128, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.blocks = nn.ModuleList([
            BottleneckResidualBlock(
                dim, bottleneck_dim=bottleneck_dim, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.bottleneck_dim = bottleneck_dim

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/bottleneck_dim"] = float(self.bottleneck_dim)
        return values


class GroupGatedResidualBlock(nn.Module):
    def __init__(self, dim, num_groups=16, group_hidden=64, dropout=0.0,
                 residual_scale=0.1, gate_init=0.5, use_norm=True,
                 gate_hidden=64):
        super().__init__()
        self.num_groups, self.group_dim = split_groups(dim, num_groups)
        self.residual_scale = residual_scale
        self.gate_init = gate_init
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.value = nn.Sequential(
            nn.Linear(self.group_dim, group_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(group_hidden, self.group_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_groups),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.value[0].weight)
        nn.init.zeros_(self.value[0].bias)
        init_zero_linear(self.value[3])
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        init_zero_linear(self.gate[2])
        init = min(max(self.gate_init, 1e-6), 1.0 - 1e-6)
        with torch.no_grad():
            self.gate[2].bias.fill_(float(torch.logit(torch.tensor(init))))

    def forward(self, x):
        n = x.size(0)
        x_norm = self.norm(x)
        grouped = x_norm.view(n, self.num_groups, self.group_dim)
        delta = self.value(grouped).view(n, -1)
        gate = torch.sigmoid(self.gate(x_norm)).view(n, self.num_groups, 1)
        delta = delta.view(n, self.num_groups, self.group_dim) * gate
        self._last_gate = gate.detach()
        return x + self.residual_scale * delta.view(n, -1)

    @torch.no_grad()
    def gate_metrics(self, prefix):
        if not hasattr(self, "_last_gate"):
            return {}
        gate = self._last_gate.float().cpu()
        return {
            f"{prefix}/gate_mean": float(gate.mean()),
            f"{prefix}/gate_std": float(gate.std()),
            f"{prefix}/gate_min": float(gate.min()),
            f"{prefix}/gate_max": float(gate.max()),
        }


class AffineGroupGatedMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, num_groups=16, group_hidden=64,
                 gate_hidden=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, gate_init=0.5, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_groups = num_groups
        self.blocks = nn.ModuleList([
            GroupGatedResidualBlock(
                dim, num_groups=num_groups, group_hidden=group_hidden,
                gate_hidden=gate_hidden, dropout=dropout,
                residual_scale=residual_scale, gate_init=gate_init,
                use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_groups"] = float(self.num_groups)
        means = []
        for idx, block in enumerate(self.blocks):
            metrics = block.gate_metrics(f"adapter/block{idx}")
            values.update(metrics)
            if metrics:
                means.append(metrics[f"adapter/block{idx}/gate_mean"])
        if means:
            values["adapter/group_gate_mean_avg"] = float(sum(means) / len(means))
        return values


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, token_dim, token_hidden=64,
                 channel_hidden=64, dropout=0.0, residual_scale=0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, num_tokens),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.token_mlp[0].weight)
        nn.init.zeros_(self.token_mlp[0].bias)
        init_zero_linear(self.token_mlp[3])
        nn.init.xavier_uniform_(self.channel_mlp[0].weight)
        nn.init.zeros_(self.channel_mlp[0].bias)
        init_zero_linear(self.channel_mlp[3])

    def forward(self, x):
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + self.residual_scale * y
        y = self.channel_mlp(self.channel_norm(x))
        return x + self.residual_scale * y


class AffineTokenMixerMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, num_tokens=16, token_hidden=64,
                 channel_hidden=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_tokens, self.token_dim = split_groups(dim, num_tokens)
        self.blocks = nn.ModuleList([
            MixerBlock(
                self.num_tokens, self.token_dim, token_hidden=token_hidden,
                channel_hidden=channel_hidden, dropout=dropout,
                residual_scale=residual_scale)
            for _ in range(num_blocks)
        ])
        self.final_norm = (
            nn.LayerNorm(self.token_dim) if use_final_norm else nn.Identity())

    def forward(self, source):
        z0 = self.start(source)
        x = z0.view(z0.size(0), self.num_tokens, self.token_dim)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x).reshape(z0.size(0), -1)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_tokens"] = float(self.num_tokens)
        values["adapter/token_dim"] = float(self.token_dim)
        return values


class AffineTinyTransformerMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, num_tokens=16, num_heads=2,
                 transformer_ffn_dim=128, num_blocks=2, dropout=0.0,
                 residual_scale=0.1, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_tokens, self.token_dim = split_groups(dim, num_tokens)
        if self.token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by heads={num_heads}")
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True)
        self.norm = nn.LayerNorm(self.token_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.out = nn.Linear(dim, dim)
        self.residual_scale = residual_scale
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        init_zero_linear(self.out)

    def forward(self, source):
        z0 = self.start(source)
        tokens = z0.view(z0.size(0), self.num_tokens, self.token_dim)
        features = self.encoder(self.norm(tokens)).reshape(z0.size(0), -1)
        x = z0 + self.residual_scale * self.out(features)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_tokens"] = float(self.num_tokens)
        values["adapter/token_dim"] = float(self.token_dim)
        return values


class ExpertBottleneck(nn.Module):
    def __init__(self, dim, bottleneck_dim=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        init_zero_linear(self.net[3])

    def forward(self, x):
        return self.net(x)


class MoEBottleneckBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim=64, num_experts=4,
                 gate_hidden=64, dropout=0.0, residual_scale=0.1,
                 use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.experts = nn.ModuleList([
            ExpertBottleneck(dim, bottleneck_dim=bottleneck_dim, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.zeros_(self.router[0].bias)
        nn.init.zeros_(self.router[2].weight)
        nn.init.zeros_(self.router[2].bias)

    def forward(self, x):
        x_norm = self.norm(x)
        weights = torch.softmax(self.router(x_norm), dim=-1)
        expert_outputs = torch.stack(
            [expert(x_norm) for expert in self.experts], dim=1)
        delta = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        self._last_router = weights.detach()
        return x + self.residual_scale * delta

    @torch.no_grad()
    def router_metrics(self, prefix):
        if not hasattr(self, "_last_router"):
            return {}
        router = self._last_router.float().cpu()
        entropy = -(router * router.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return {
            f"{prefix}/router_entropy": float(entropy),
            f"{prefix}/router_max_prob": float(router.max(dim=-1).values.mean()),
        }


class AffineMoEBottleneckMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, bottleneck_dim=64, num_experts=4,
                 gate_hidden=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_experts = num_experts
        self.blocks = nn.ModuleList([
            MoEBottleneckBlock(
                dim, bottleneck_dim=bottleneck_dim, num_experts=num_experts,
                gate_hidden=gate_hidden, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_experts"] = float(self.num_experts)
        entropies = []
        for idx, block in enumerate(self.blocks):
            metrics = block.router_metrics(f"adapter/block{idx}")
            values.update(metrics)
            if metrics:
                entropies.append(metrics[f"adapter/block{idx}/router_entropy"])
        if entropies:
            values["adapter/router_entropy_avg"] = float(
                sum(entropies) / len(entropies))
        return values


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=128, mask_even=True, scale=0.1):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("AffineCouplingLayer requires an even dim")
        self.dim = dim
        self.half = dim // 2
        self.mask_even = mask_even
        self.scale = scale
        self.net = nn.Sequential(
            nn.LayerNorm(self.half),
            nn.Linear(self.half, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.half * 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        init_zero_linear(self.net[3])

    def forward(self, x):
        a, b = x[:, :self.half], x[:, self.half:]
        if self.mask_even:
            cond, target = a, b
        else:
            cond, target = b, a
        log_s, t = self.net(cond).chunk(2, dim=-1)
        log_s = self.scale * torch.tanh(log_s)
        target = target * torch.exp(log_s) + t
        if self.mask_even:
            return torch.cat([cond, target], dim=-1)
        return torch.cat([target, cond], dim=-1)


class AffineCouplingFlowMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, flow_hidden_dim=128, num_blocks=4,
                 residual_scale=0.1, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.layers = nn.ModuleList([
            AffineCouplingLayer(
                dim, hidden_dim=flow_hidden_dim, mask_even=(idx % 2 == 0),
                scale=residual_scale)
            for idx in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        return self._base_metrics()

