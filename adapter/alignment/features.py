from .dependencies import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403

def unit_columns(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=0, keepdim=True)
    return centered / centered.norm(dim=0, keepdim=True).clamp_min(1.0e-8)


def weight_signatures(
    state: Mapping[str, torch.Tensor],
    device: torch.device,
    bias_scale: float = 0.1,
) -> list[torch.Tensor]:
    output = []
    for block in range(4):
        prefix = f"blocks.{block}."
        w1 = unit_rows(state[prefix + "net.0.weight"].to(device))
        b1 = state[prefix + "net.0.bias"].to(device).unsqueeze(1)
        w2 = unit_rows(state[prefix + "net.3.weight"].to(device).t())
        output.append(torch.cat([w1, bias_scale * b1, w2], dim=1))
    return output


def load_indexed_codes(
    record: TaskRecord,
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not record.train_code_path.is_file():
        raise FileNotFoundError(f"missing train codewords: {record.train_code_path}")
    codes = torch.load(
        record.train_code_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not torch.is_tensor(codes) or codes.ndim != 2 or codes.shape[1] != 512:
        raise ValueError(
            f"{record.train_code_path}: expected codewords (N,512), got "
            f"{type(codes)} shape={getattr(codes, 'shape', None)}"
        )
    full_shared_dataset = (
        len(indices) > 0
        and int(indices[0]) == 0
        and int(indices[-1]) == len(indices) - 1
    )
    if full_shared_dataset and len(codes) != len(indices):
        raise ValueError(
            f"{record.train_code_path}: sample count={len(codes)} differs from "
            f"reference train sample count={len(indices)}; row-wise activation "
            "alignment requires identical datasets and ordering"
        )
    if int(indices.max()) >= len(codes):
        raise ValueError(
            f"{record.train_code_path}: sample count={len(codes)} does not cover "
            f"shared maximum index={int(indices.max())}"
        )
    if (
        full_shared_dataset and len(indices) == len(codes)
    ):
        selected = codes
    else:
        selected = codes.index_select(0, indices)
    return selected.float().to(device, non_blocking=True)


def load_code_shape(record: TaskRecord) -> tuple[int, int]:
    if not record.train_code_path.is_file():
        raise FileNotFoundError(f"missing train codewords: {record.train_code_path}")
    codes = torch.load(
        record.train_code_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not torch.is_tensor(codes) or codes.ndim != 2 or codes.shape[1] != 512:
        raise ValueError(
            f"{record.train_code_path}: expected codewords (N,512), got "
            f"{type(codes)} shape={getattr(codes, 'shape', None)}"
        )
    return int(codes.shape[0]), int(codes.shape[1])


def shared_activation_indices(
    sample_count: int,
    requested_samples: int,
    seed: int,
) -> torch.Tensor:
    if sample_count <= 0:
        raise ValueError("activation sample count must be positive")
    if requested_samples < 0:
        raise ValueError("--activation-samples must be non-negative")
    if requested_samples == 0 or requested_samples >= sample_count:
        return torch.arange(sample_count, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(sample_count, generator=generator)[:requested_samples]
    return indices.sort().values


def post_gelu_activations(
    state: Mapping[str, torch.Tensor], codes: torch.Tensor
) -> list[torch.Tensor]:
    value = codes.matmul(state["alignment_weight"]) + state["alignment_bias"]
    output = []
    for block in range(4):
        prefix = f"blocks.{block}."
        hidden = F.layer_norm(
            value,
            (value.shape[-1],),
            state[prefix + "norm.weight"],
            state[prefix + "norm.bias"],
            1.0e-5,
        )
        hidden = F.gelu(F.linear(
            hidden,
            state[prefix + "net.0.weight"],
            state[prefix + "net.0.bias"],
        ))
        output.append(hidden)
        delta = F.linear(
            hidden,
            state[prefix + "net.3.weight"],
            state[prefix + "net.3.bias"],
        )
        value = value + 0.4 * delta
    return output


def task_features(
    record: TaskRecord,
    device: torch.device,
    cost_type: str,
    activation_indices: torch.Tensor | None,
    bias_scale: float = 0.1,
) -> tuple[
    OrderedDict[str, torch.Tensor],
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
]:
    cpu_state = load_state(record)
    device_state = to_device_state(cpu_state, device)
    signatures = (
        weight_signatures(cpu_state, device, bias_scale)
        if cost_type in {"weight", "hybrid"}
        else None
    )
    activations = None
    if cost_type in {"activation", "hybrid"}:
        assert activation_indices is not None
        codes = load_indexed_codes(record, activation_indices, device)
        activations = post_gelu_activations(device_state, codes)
    return device_state, signatures, activations


def cost_matrix(
    signature: torch.Tensor | None,
    reference_signature: torch.Tensor | None,
    activation: torch.Tensor | None,
    reference_activation: torch.Tensor | None,
    weight_ratio: float,
    activation_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    weight_cost = None
    activation_cost = None
    if signature is not None:
        assert reference_signature is not None
        weight_cost = 1.0 - unit_rows(signature).matmul(
            unit_rows(reference_signature).t()
        )
    if activation is not None:
        assert reference_activation is not None
        activation_cost = 1.0 - unit_columns(activation).t().matmul(
            unit_columns(reference_activation)
        )
    if weight_cost is None:
        assert activation_cost is not None
        combined = activation_ratio * activation_cost
    elif activation_cost is None:
        combined = weight_ratio * weight_cost
    else:
        combined = weight_ratio * weight_cost + activation_ratio * activation_cost
    return combined, weight_cost, activation_cost

