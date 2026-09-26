import torch
import torch.nn as nn
import torch.nn.functional as F


def init_zero_linear(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_identity_projection(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    with torch.no_grad():
        diag = min(linear.weight.size(0), linear.weight.size(1))
        linear.weight[:diag, :diag].copy_(torch.eye(
            diag, dtype=linear.weight.dtype, device=linear.weight.device))


def split_groups(dim, num_groups):
    if dim % num_groups != 0:
        raise ValueError(f"dim={dim} must be divisible by num_groups={num_groups}")
    return num_groups, dim // num_groups


def build_activation(name):
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in ("identity", "none", "linear"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class AffineStartMixin:
    def _init_affine(self, weight, bias, train_affine):
        if weight.ndim != 2 or weight.size(0) != weight.size(1):
            raise ValueError(f"weight must be square 2D, got {tuple(weight.shape)}")
        if bias.ndim != 1 or bias.size(0) != weight.size(1):
            raise ValueError(
                f"bias must be ({weight.size(1)},), got {tuple(bias.shape)}")
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.register_buffer("_delta_ratio", torch.tensor(0.0))

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def _update_delta_ratio(self, z0, out):
        with torch.no_grad():
            self._delta_ratio.fill_(
                (out - z0).norm() / z0.norm().clamp_min(1e-8))

    @torch.no_grad()
    def _base_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        return {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, residual_scale=0.1,
                 use_norm=True, learnable_gate=False, gate_max=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = residual_scale
        self.learnable_gate = learnable_gate
        self.gate_max = gate_max
        if learnable_gate:
            if gate_max <= 0:
                raise ValueError("gate_max must be positive when learnable_gate=True")
            init = min(max(residual_scale / gate_max, 1e-6), 1.0 - 1e-6)
            raw = torch.logit(torch.full((dim,), init))
            self.raw_gate = nn.Parameter(raw)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def gate(self):
        if not self.learnable_gate:
            return self.residual_scale
        return self.gate_max * torch.sigmoid(self.raw_gate)

    def forward(self, x):
        return x + self.gate() * self.net(self.norm(x))

    @torch.no_grad()
    def gate_metrics(self, prefix):
        if not self.learnable_gate:
            return {}
        gate = self.gate().detach().float().cpu()
        return {
            f"{prefix}/gate_mean": float(gate.mean()),
            f"{prefix}/gate_std": float(gate.std()),
            f"{prefix}/gate_min": float(gate.min()),
            f"{prefix}/gate_max": float(gate.max()),
        }


class FiLMResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, residual_scale=0.1,
                 use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        residual = self.net(self.norm(x))
        residual = residual * self.gamma.view(1, -1) + self.beta.view(1, -1)
        return x + self.residual_scale * residual

    @torch.no_grad()
    def film_metrics(self, prefix):
        gamma = self.gamma.detach().float().cpu()
        beta = self.beta.detach().float().cpu()
        return {
            f"{prefix}/gamma_mean": float(gamma.mean()),
            f"{prefix}/gamma_std": float(gamma.std(unbiased=False)),
            f"{prefix}/gamma_min": float(gamma.min()),
            f"{prefix}/gamma_max": float(gamma.max()),
            f"{prefix}/beta_mean": float(beta.mean()),
            f"{prefix}/beta_std": float(beta.std(unbiased=False)),
            f"{prefix}/beta_norm": float(beta.norm()),
        }


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, bottleneck_dim=128, dropout=0.0,
                 residual_scale=0.1, use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.full = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.lowrank = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.full[0].weight)
        nn.init.zeros_(self.full[0].bias)
        nn.init.zeros_(self.full[3].weight)
        nn.init.zeros_(self.full[3].bias)
        nn.init.xavier_uniform_(self.lowrank[0].weight)
        nn.init.zeros_(self.lowrank[0].bias)
        nn.init.zeros_(self.lowrank[3].weight)
        nn.init.zeros_(self.lowrank[3].bias)

    def forward(self, x):
        h = self.norm(x)
        return x + self.residual_scale * (self.full(h) + self.lowrank(h))


class FinalGate(nn.Module):
    def __init__(self, dim, mode="none", gate_max=1.0, gate_init=1.0,
                 adaptive_hidden=128):
        super().__init__()
        self.mode = mode
        self.gate_max = gate_max
        self.gate_init = gate_init
        self._last_gate = None
        if mode == "none":
            return
        if mode == "final_unbounded":
            self.gate = nn.Parameter(torch.full((dim,), float(gate_init)))
            return
        if gate_max <= 0:
            raise ValueError("gate_max must be positive")
        init = min(max(gate_init / gate_max, 1e-6), 1.0 - 1e-6)
        raw_init = torch.logit(torch.full((dim,), init))
        if mode == "final_static":
            self.raw_gate = nn.Parameter(raw_init)
        elif mode == "final_adaptive":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, adaptive_hidden),
                nn.GELU(),
                nn.Linear(adaptive_hidden, dim),
            )
            nn.init.xavier_uniform_(self.net[1].weight)
            nn.init.zeros_(self.net[1].bias)
            nn.init.zeros_(self.net[3].weight)
            with torch.no_grad():
                self.net[3].bias.copy_(raw_init)
        else:
            raise ValueError(f"Unknown final gate mode: {mode}")

    def forward(self, z0, delta):
        if self.mode == "none":
            return z0 + delta
        if self.mode == "final_unbounded":
            gate = self.gate.view(1, -1)
            self._last_gate = gate
            return z0 + gate * delta
        if self.mode == "final_static":
            gate = self.gate_max * torch.sigmoid(self.raw_gate).view(1, -1)
        else:
            gate = self.gate_max * torch.sigmoid(self.net(z0))
        self._last_gate = gate
        return z0 + gate * delta

    def regularization(self):
        if self.mode == "none":
            return None
        if self.mode == "final_unbounded":
            return self.gate.abs().mean()
        if self.mode == "final_static":
            return (self.gate_max * torch.sigmoid(self.raw_gate)).mean()
        if self._last_gate is None:
            return None
        return self._last_gate.mean()

    @torch.no_grad()
    def metrics(self, z0=None):
        if self.mode == "none":
            return {}
        if self.mode == "final_unbounded":
            gate = self.gate.detach()
        if self.mode == "final_static":
            gate = self.gate_max * torch.sigmoid(self.raw_gate.detach())
        elif self.mode == "final_adaptive":
            if z0 is None:
                return {}
            gate = self.gate_max * torch.sigmoid(self.net(z0).detach())
        gate = gate.float().cpu()
        return {
            "adapter/final_gate_mean": float(gate.mean()),
            "adapter/final_gate_std": float(gate.std()),
            "adapter/final_gate_min": float(gate.min()),
            "adapter/final_gate_max": float(gate.max()),
        }


class LowRankResidualBlock(nn.Module):
    def __init__(self, dim, rank=64, dropout=0.0, residual_scale=0.1,
                 use_norm=True, learnable_gate=False, gate_max=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.down = nn.Linear(dim, rank)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, dim)
        self.residual_scale = residual_scale
        self.learnable_gate = learnable_gate
        self.gate_max = gate_max
        if learnable_gate:
            if gate_max <= 0:
                raise ValueError("gate_max must be positive when learnable_gate=True")
            init = min(max(residual_scale / gate_max, 1e-6), 1.0 - 1e-6)
            self.raw_gate = nn.Parameter(torch.logit(torch.full((dim,), init)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def gate(self):
        if not self.learnable_gate:
            return self.residual_scale
        return self.gate_max * torch.sigmoid(self.raw_gate)

    def forward(self, x):
        residual = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + self.gate() * residual

    @torch.no_grad()
    def gate_metrics(self, prefix):
        if not self.learnable_gate:
            return {}
        gate = self.gate().detach().float().cpu()
        return {
            f"{prefix}/gate_mean": float(gate.mean()),
            f"{prefix}/gate_std": float(gate.std()),
            f"{prefix}/gate_min": float(gate.min()),
            f"{prefix}/gate_max": float(gate.max()),
        }


class CodewordDeltaSelfAttention(nn.Module):
    """Self-attention over scalar codeword positions.

    Each position remains one token. The token feature is built from the
    affine-aligned value, the residual MLP proposed change, or both. This
    models sample-dependent relations between code dimensions without a
    Transformer feed-forward block.
    """

    def __init__(self, code_dim, attention_dim=32, num_heads=4,
                 dropout=0.0, input_mode="value_delta",
                 use_position=True):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if num_heads <= 0 or attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim={attention_dim} must be divisible by "
                f"num_heads={num_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("attention dropout must be in [0, 1)")
        if input_mode not in ("value", "delta", "value_delta"):
            raise ValueError(f"Unknown attention input mode: {input_mode}")

        input_dim = 2 if input_mode == "value_delta" else 1
        self.code_dim = code_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.dropout = dropout
        self.input_mode = input_mode
        self.use_position = use_position

        self.value_norm = nn.LayerNorm(code_dim)
        self.delta_norm = nn.LayerNorm(code_dim)
        self.input_proj = nn.Linear(input_dim, attention_dim)
        if use_position:
            self.position_embedding = nn.Parameter(
                torch.empty(1, code_dim, attention_dim))
        else:
            self.register_buffer(
                "position_embedding",
                torch.zeros(1, code_dim, attention_dim),
                persistent=False)
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.qkv = nn.Linear(attention_dim, 3 * attention_dim)
        self.context_proj = nn.Linear(attention_dim, attention_dim)
        self.output_norm = nn.LayerNorm(attention_dim)
        self.output_proj = nn.Linear(attention_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        if self.use_position:
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        init_zero_linear(self.output_proj)

    def _build_tokens(self, value, delta):
        value = self.value_norm(value)
        delta = self.delta_norm(delta)
        if self.input_mode == "value":
            token_input = value.unsqueeze(-1)
        elif self.input_mode == "delta":
            token_input = delta.unsqueeze(-1)
        else:
            token_input = torch.stack((value, delta), dim=-1)
        return self.input_proj(token_input) + self.position_embedding

    def forward(self, value, delta):
        tokens = self._build_tokens(value, delta)
        batch_size, num_tokens, _ = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value_features = qkv.unbind(0)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value_features,
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(
            batch_size, num_tokens, self.attention_dim)
        context = self.context_proj(context)
        return self.output_proj(self.output_norm(context)).squeeze(-1)

