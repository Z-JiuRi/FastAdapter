from .dependencies import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .features import *  # noqa: F401,F403

def solve_block(
    signature: torch.Tensor | None,
    reference_signature: torch.Tensor | None,
    activation: torch.Tensor | None,
    reference_activation: torch.Tensor | None,
    weight_ratio: float,
    activation_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    combined, weight_cost, activation_cost = cost_matrix(
        signature,
        reference_signature,
        activation,
        reference_activation,
        weight_ratio,
        activation_ratio,
    )
    rows, columns = linear_sum_assignment(combined.detach().float().cpu().numpy())
    order_np = np.empty_like(columns)
    order_np[columns] = rows
    order = torch.from_numpy(order_np).long()
    order_device = order.to(combined.device)
    reference_axis = torch.arange(combined.shape[1], device=combined.device)
    matched = combined[order_device, reference_axis]
    identity = combined.diagonal()
    nearest_two = torch.topk(combined, k=2, dim=0, largest=False).values

    metrics = {
        "identity_cost": float(identity.mean()),
        "matched_cost": float(matched.mean()),
        "improvement": float(identity.mean() - matched.mean()),
        "changed_ratio": float((order != torch.arange(len(order))).float().mean()),
        "local_margin": float((nearest_two[1] - nearest_two[0]).mean()),
    }
    if weight_cost is not None:
        metrics["matched_weight_cost"] = float(
            weight_cost[order_device, reference_axis].mean()
        )
    if activation_cost is not None:
        metrics["matched_activation_cost"] = float(
            activation_cost[order_device, reference_axis].mean()
        )
    return order, metrics


def align_task_features(
    signatures: list[torch.Tensor] | None,
    reference_signatures: list[torch.Tensor] | None,
    activations: list[torch.Tensor] | None,
    reference_activations: list[torch.Tensor] | None,
    weight_ratio: float,
    activation_ratio: float,
) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
    orders = []
    metrics = []
    for block in range(4):
        order, block_metrics = solve_block(
            signatures[block] if signatures is not None else None,
            reference_signatures[block] if reference_signatures is not None else None,
            activations[block] if activations is not None else None,
            reference_activations[block] if reference_activations is not None else None,
            weight_ratio,
            activation_ratio,
        )
        orders.append(order)
        metrics.append(block_metrics)
    return orders, metrics


def apply_orders(
    state: Mapping[str, torch.Tensor], orders: list[torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    aligned = OrderedDict((key, value) for key, value in state.items())
    for block, order_cpu in enumerate(orders):
        prefix = f"blocks.{block}."
        order = order_cpu.to(state[prefix + "net.0.weight"].device)
        aligned[prefix + "net.0.weight"] = state[prefix + "net.0.weight"][order].contiguous()
        aligned[prefix + "net.0.bias"] = state[prefix + "net.0.bias"][order].contiguous()
        aligned[prefix + "net.3.weight"] = state[prefix + "net.3.weight"][:, order].contiguous()
    return aligned


def save_aligned_state(
    state: Mapping[str, torch.Tensor],
    destination: Path,
) -> None:
    cpu_state = OrderedDict(
        (key, value.detach().cpu().contiguous()) for key, value in state.items()
    )
    torch.save(cpu_state, destination)


def verify_function(
    original: Mapping[str, torch.Tensor],
    aligned: Mapping[str, torch.Tensor],
    probes: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        before = functional_adapter(probes, original, residual_scale=0.4)
        after = functional_adapter(probes, aligned, residual_scale=0.4)
        difference = (after - before).float()
        max_abs = float(difference.abs().max())
        rel_mse = float(
            difference.square().mean()
            / before.float().square().mean().clamp_min(1.0e-12)
        )
    return max_abs, rel_mse


def reference_hash(
    signatures: list[torch.Tensor] | None,
    activations: list[torch.Tensor] | None,
) -> str:
    digest = hashlib.sha256()
    for group in (signatures, activations):
        if group is None:
            continue
        for value in group:
            tensor = value.detach().float().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def mean_metric(block_metrics: list[dict[str, float]], key: str) -> float:
    values = [metrics[key] for metrics in block_metrics if key in metrics]
    return sum(values) / len(values) if values else float("nan")

