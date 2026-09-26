import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *  # noqa: F401,F403

class AffineIterativeResidualMapper(nn.Module, AffineStartMixin):
    """Affine start + multi-step residual refine (shared or unshared).

    Motivated by per-sample latent refinement through a frozen decoder: a
    small residual network applied repeatedly can approximate learned gradient
    steps in code space while keeping parameter count well below the full AE.
    """

    def __init__(self, weight, bias, hidden_dim=512, num_iters=8,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 share_weights=True, use_step_embed=True):
        super().__init__()
        if num_iters < 1:
            raise ValueError(f"num_iters must be >= 1, got {num_iters}")
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_iters = int(num_iters)
        self.share_weights = bool(share_weights)
        self.use_step_embed = bool(use_step_embed)
        self.residual_scale = residual_scale
        self.hidden_dim = hidden_dim
        if self.use_step_embed:
            self.step_embed = nn.Embedding(self.num_iters, dim)
            nn.init.zeros_(self.step_embed.weight)
        else:
            self.step_embed = None
        if self.share_weights:
            self.block = ResidualBlock(
                dim, hidden_dim, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm,
                learnable_gate=False)
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                ResidualBlock(
                    dim, hidden_dim, dropout=dropout,
                    residual_scale=residual_scale, use_norm=use_block_norm,
                    learnable_gate=False)
                for _ in range(self.num_iters)
            ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for step in range(self.num_iters):
            if self.step_embed is not None:
                x = x + self.step_embed.weight[step].view(1, -1)
            if self.share_weights:
                x = self.block(x)
            else:
                x = self.blocks[step](x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_iters"] = float(self.num_iters)
        values["adapter/share_weights"] = float(self.share_weights)
        values["adapter/use_step_embed"] = float(self.use_step_embed)
        values["adapter/hidden_dim"] = float(self.hidden_dim)
        return values


class AffineSensWeightedResidualMapper(nn.Module, AffineStartMixin):
    """Affine + residual MLP with a learnable diagonal sensitivity gate on delta.

    Projects residual updates through a soft non-negative diagonal so the model
    can emphasize decoder-sensitive code dimensions without unfreezing decoder.
    """

    def __init__(self, weight, bias, hidden_dim=512, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 sens_init=1.0):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                dim, hidden_dim, dropout=dropout,
                residual_scale=1.0, use_norm=use_block_norm,
                learnable_gate=False)
            for _ in range(num_blocks)
        ])
        # softplus(raw) keeps gate > 0; init near sens_init
        init = max(float(sens_init), 1e-4)
        raw = torch.log(torch.expm1(torch.tensor(init)))
        self.raw_sens = nn.Parameter(torch.full((dim,), float(raw)))
        self.residual_scale = residual_scale
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def sensitivity(self):
        return F.softplus(self.raw_sens) + 1e-4

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        delta = x - z0
        sens = self.sensitivity().view(1, -1)
        # re-center sensitivity to mean 1 so scale is controlled by residual_scale
        sens = sens / sens.mean().clamp_min(1e-6)
        out = z0 + self.residual_scale * sens * delta
        out = self.final_norm(out)
        self._update_delta_ratio(z0, out)
        return out

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        sens = self.sensitivity().detach().float().cpu()
        values["adapter/sens_mean"] = float(sens.mean())
        values["adapter/sens_std"] = float(sens.std(unbiased=False))
        values["adapter/sens_min"] = float(sens.min())
        values["adapter/sens_max"] = float(sens.max())
        return values


class AffineWholeResidualMLPMapper(nn.Module, AffineStartMixin):
    """Affine alignment plus one identity-initialized whole residual MLP."""

    def __init__(self, weight, bias, hidden_dims=None, dropout=0.0,
                 residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 activation="gelu"):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        hidden_dims = list(hidden_dims or [512, 512])
        layers = []
        in_dim = dim
        self.norm = nn.LayerNorm(dim) if use_block_norm else nn.Identity()
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, int(hidden_dim)),
                build_activation(activation),
                nn.Dropout(dropout),
            ])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.residual_scale = residual_scale
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.hidden_dims = hidden_dims
        self.reset_parameters()

    def reset_parameters(self):
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        init_zero_linear(linear_layers[-1])

    def forward(self, source):
        z0 = self.start(source)
        delta = self.net(self.norm(z0))
        x = z0 + self.residual_scale * delta
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/whole_mlp_depth"] = float(len(self.hidden_dims) + 1)
        return values


class LegacyMLPAdapterMapper(nn.Module):
    """Exact single-wide residual MLP used by the historical CodeAdapter.

    The residual branch is deliberately kept identical to
    ``models.adapters.MLPAdapter``:

        LayerNorm -> Linear(dim, hidden) -> GELU -> Linear(hidden, dim)
        output = input + residual_scale * branch(input)

    The second linear layer is zero-initialized, so this starts as identity.
    ``use_affine_alignment`` optionally prepends the offline least-squares
    alignment.  This lets one experiment isolate no alignment, frozen affine
    buffer, and trainable affine without changing the MLP itself.
    """

    def __init__(self, weight, bias, hidden_dim=2048, dropout=0.0,
                 residual_scale=1.0, use_affine_alignment=False,
                 train_affine=False):
        super().__init__()
        if weight.ndim != 2 or weight.size(0) != weight.size(1):
            raise ValueError(f"weight must be square 2D, got {tuple(weight.shape)}")
        if bias.ndim != 1 or bias.size(0) != weight.size(1):
            raise ValueError(
                f"bias must be ({weight.size(1)},), got {tuple(bias.shape)}")
        if train_affine and not use_affine_alignment:
            raise ValueError("train_affine requires use_affine_alignment=True")

        dim = weight.size(0)
        self.use_affine_alignment = bool(use_affine_alignment)
        self.residual_scale = float(residual_scale)
        if self.use_affine_alignment:
            if train_affine:
                self.alignment_weight = nn.Parameter(weight.clone())
                self.alignment_bias = nn.Parameter(bias.clone())
            else:
                self.register_buffer("alignment_weight", weight.clone())
                self.register_buffer("alignment_bias", bias.clone())

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        # Match models/adapters/mlp_adapter.py exactly.
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_uniform_(self.net[0].weight, a=0.3,
                                 nonlinearity="leaky_relu")
        nn.init.zeros_(self.net[0].bias)
        init_zero_linear(self.net[3])

    def start(self, source):
        if not self.use_affine_alignment:
            return source
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        out = z0 + self.residual_scale * self.net(self.norm(z0))
        with torch.no_grad():
            self._delta_ratio.fill_(
                (out - z0).norm() / z0.norm().clamp_min(1e-8))
        return out

    @torch.no_grad()
    def get_metrics(self):
        values = {
            "adapter/use_affine_alignment": float(self.use_affine_alignment),
            "adapter/train_affine": float(
                self.use_affine_alignment
                and isinstance(self.alignment_weight, nn.Parameter)),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
        if self.use_affine_alignment:
            values.update({
                "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
                "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            })
        return values


class AffineWholeDirectMLPMapper(nn.Module, AffineStartMixin):
    """Affine alignment followed by one identity-initialized whole MLP."""

    def __init__(self, weight, bias, hidden_dims=None, dropout=0.0,
                 use_block_norm=False, use_final_norm=False,
                 train_affine=False, activation="identity"):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        hidden_dims = list(hidden_dims or [512, 512])
        layers = []
        in_dim = dim
        self.norm = nn.LayerNorm(dim) if use_block_norm else nn.Identity()
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, int(hidden_dim)),
                build_activation(activation),
                nn.Dropout(dropout),
            ])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.hidden_dims = hidden_dims
        self.reset_parameters()

    def reset_parameters(self):
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            init_identity_projection(layer)
        init_identity_projection(linear_layers[-1])

    def forward(self, source):
        z0 = self.start(source)
        x = self.net(self.norm(z0))
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/whole_mlp_depth"] = float(len(self.hidden_dims) + 1)
        return values


class AffineLinearStackMapper(nn.Module):
    """Affine alignment followed by identity-initialized linear stack.

    This is mainly a baseline. Without nonlinear activations, the whole mapper
    is still affine, so it should not improve over the closed-form affine fit.
    """

    def __init__(self, weight, bias, num_layers=1, train_affine=False):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, source):
        x = source.matmul(self.alignment_weight) + self.alignment_bias
        for layer in self.layers:
            x = layer(x)
        return x


class DirectMLPMapper(nn.Module):
    """Non-residual source-to-target MLP baseline."""

    def __init__(self, dim, hidden_dim=1024, num_layers=4, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, source):
        return self.net(source)

