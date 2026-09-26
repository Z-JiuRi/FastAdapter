from .dependencies import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403

@torch.no_grad()
def code_metrics(pred, target):
    err = pred - target
    mse = err.pow(2).mean()
    nmse = 10.0 * torch.log10(
        err.pow(2).sum() / target.pow(2).sum().clamp_min(1e-12))
    cos = F.cosine_similarity(pred, target, dim=1).mean()
    return {
        "code_mse": float(mse.cpu()),
        "code_nmse": float(nmse.cpu()),
        "code_cos": float(cos.cpu()),
    }


def estimate_decoder_jacobian_sensitivity(
        decoder, probe_codes, device, n_hutchinson=8, max_samples=2048,
        batch_size=256):
    """Hutchinson estimate of diag(J^T J) for y=D_t(z), normalized to mean 1."""
    decoder.eval()
    codes = probe_codes[:max_samples].to(device)
    if codes.numel() == 0:
        raise ValueError("probe_codes is empty")
    dim = codes.size(1)
    sens = torch.zeros(dim, device=device, dtype=torch.float64)
    total = 0
    for start in range(0, codes.size(0), batch_size):
        z = codes[start:start + batch_size].detach().requires_grad_(True)
        y = decoder(z)
        for _ in range(n_hutchinson):
            v = torch.randn_like(y)
            grad_z = torch.autograd.grad(
                (y * v).sum(), z, retain_graph=True)[0]
            sens += grad_z.detach().double().pow(2).sum(dim=0)
        total += z.size(0) * n_hutchinson
    sens = sens / max(total, 1)
    sens = sens / sens.mean().clamp_min(1e-12)
    return sens.float().cpu()


def estimate_decoder_fc_sensitivity(decoder):
    if not hasattr(decoder, "fc_decoder"):
        raise AttributeError(
            "fc_decoder sensitivity requires decoder.fc_decoder")
    with torch.no_grad():
        sensitivity = decoder.fc_decoder.weight.detach().float().pow(2).sum(dim=0)
        sensitivity = sensitivity / sensitivity.mean().clamp_min(1e-12)
    return sensitivity.cpu()


def build_code_loss_weight(target_code, args, residual_code=None):
    if args.code_loss_type not in (
            "clipped_std_mse",
            "clipped_var_mse",
            "clipped_power_mse",
            "clipped_residual_std_mse",
            "decoder_sensitivity_mse",
            "decoder_jac_residual_mse"):
        return None, None
    if args.code_loss_type in (
            "decoder_sensitivity_mse", "decoder_jac_residual_mse"):
        if not hasattr(args, "_decoder_sensitivity_weight"):
            raise ValueError(
                f"{args.code_loss_type} requires _decoder_sensitivity_weight")
        weight = args._decoder_sensitivity_weight.float()
        scale = weight
        if args.code_loss_type == "decoder_jac_residual_mse":
            if residual_code is None:
                raise ValueError(
                    "residual_code is required for decoder_jac_residual_mse")
            # Emphasize dims that are both decoder-sensitive and hard to map
            # (large residual std after affine).
            resid_std = residual_code.float().std(dim=0).clamp_min(
                args.std_weight_eps)
            resid_factor = resid_std / resid_std.mean().clamp_min(1e-12)
            weight = weight * resid_factor
            scale = weight
    elif args.code_loss_type == "clipped_residual_std_mse":
        if residual_code is None:
            raise ValueError("residual_code is required for clipped_residual_std_mse")
        scale = residual_code.float().std(dim=0).clamp_min(args.std_weight_eps)
        weight = scale.mean() / scale
    elif args.code_loss_type == "clipped_var_mse":
        scale = target_code.float().var(dim=0, unbiased=False).clamp_min(
            args.std_weight_eps)
        weight = 1.0 / scale
    elif args.code_loss_type == "clipped_power_mse":
        scale = target_code.float().pow(2).mean(dim=0).clamp_min(
            args.std_weight_eps)
        weight = 1.0 / scale
    else:
        scale = target_code.float().std(dim=0).clamp_min(args.std_weight_eps)
        weight = 1.0 / scale
    power = float(getattr(args, "sensitivity_power", 1.0) or 1.0)
    if power != 1.0 and args.code_loss_type in (
            "decoder_sensitivity_mse", "decoder_jac_residual_mse"):
        weight = weight.clamp_min(args.std_weight_eps).pow(power)
        weight = weight / weight.mean().clamp_min(1e-12)
    weight = weight.clamp(args.std_weight_min, args.std_weight_max)
    return weight.contiguous(), scale.contiguous()


def compute_code_loss(pred, target, code_loss_type="mse", std_weight=None):
    sqerr = (pred - target).pow(2)
    raw_mse = sqerr.mean()
    if code_loss_type == "mse":
        return raw_mse, raw_mse
    if code_loss_type in (
            "clipped_std_mse",
            "clipped_var_mse",
            "clipped_power_mse",
            "clipped_residual_std_mse",
            "decoder_sensitivity_mse",
            "decoder_jac_residual_mse"):
        if std_weight is None:
            raise ValueError(f"std_weight is required for {code_loss_type}")
        weighted = sqerr * std_weight.view(1, -1)
        return weighted.mean(), raw_mse
    raise ValueError(f"Unknown code_loss_type: {code_loss_type}")


def get_start_code(model, source):
    if not hasattr(model, "start"):
        return None
    return model.start(source)


def init_delta_totals(device):
    return {
        "delta_target_cos": 0.0,
        "delta_z0_cos": 0.0,
        "n": 0,
        "delta_sq": torch.tensor(0.0, device=device),
        "target_residual_sq": torch.tensor(0.0, device=device),
        "z0_sq": torch.tensor(0.0, device=device),
        "delta_sum": torch.tensor(0.0, device=device),
        "delta_sumsq": torch.tensor(0.0, device=device),
        "delta_numel": 0,
        "delta_small": torch.tensor(0.0, device=device),
    }


@torch.no_grad()
def update_delta_totals(totals, mapped, target, z0, small_eps=1e-4):
    if z0 is None:
        return
    mapped = mapped.detach()
    target = target.detach()
    z0 = z0.detach()
    delta = mapped - z0
    target_residual = target - z0
    n = mapped.size(0)
    totals["delta_target_cos"] += float(
        F.cosine_similarity(delta, target_residual, dim=1).mean().cpu()) * n
    totals["delta_z0_cos"] += float(
        F.cosine_similarity(delta, z0, dim=1).mean().cpu()) * n
    totals["n"] += n
    totals["delta_sq"] += delta.pow(2).sum()
    totals["target_residual_sq"] += target_residual.pow(2).sum()
    totals["z0_sq"] += z0.pow(2).sum()
    totals["delta_sum"] += delta.sum()
    totals["delta_sumsq"] += delta.pow(2).sum()
    totals["delta_numel"] += delta.numel()
    totals["delta_small"] += (delta.abs() < small_eps).float().sum()


def finalize_delta_metrics(totals):
    if totals["n"] == 0:
        return {}
    delta_numel = max(totals["delta_numel"], 1)
    delta_mean = totals["delta_sum"] / delta_numel
    delta_var = totals["delta_sumsq"] / delta_numel - delta_mean.pow(2)
    return {
        "delta_target_cos": totals["delta_target_cos"] / totals["n"],
        "delta_z0_cos": totals["delta_z0_cos"] / totals["n"],
        "residual_coverage": float(torch.sqrt(
            totals["delta_sq"] /
            totals["target_residual_sq"].clamp_min(1e-12)).cpu()),
        "delta_ratio_from_z0": float(torch.sqrt(
            totals["delta_sq"] / totals["z0_sq"].clamp_min(1e-12)).cpu()),
        "delta_mean": float(delta_mean.cpu()),
        "delta_std": float(torch.sqrt(delta_var.clamp_min(0.0)).cpu()),
        "delta_small_frac": float((totals["delta_small"] / delta_numel).cpu()),
    }


@torch.no_grad()
def collect_parameter_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        key = name.replace(".", "/")
        data = param.detach().float()
        stats[f"{key}/mean"] = float(data.mean().cpu())
        stats[f"{key}/std"] = float(data.std(unbiased=False).cpu())
        stats[f"{key}/min"] = float(data.min().cpu())
        stats[f"{key}/max"] = float(data.max().cpu())
        stats[f"{key}/norm"] = float(data.norm().cpu())
        if param.grad is not None:
            grad = param.grad.detach().float()
            grad_norm = grad.norm()
            stats[f"{key}/grad_norm"] = float(grad_norm.cpu())
            stats[f"{key}/update_ratio"] = float(
                (grad_norm / data.norm().clamp_min(1e-12)).cpu())
    return stats


def summarize_parameter_stats(stats):
    groups = {}
    for key, value in stats.items():
        if not key.endswith("/norm"):
            continue
        group = key.split("/", 1)[0]
        groups.setdefault(group, 0.0)
        groups[group] += value
    return {f"{group}/norm_sum": value for group, value in sorted(groups.items())}


def loss_gradient_diagnostics(losses, parameters):
    """Return weighted-loss gradient norms/cosines without changing .grad."""
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = {}
    metrics = {}
    for name, loss in losses.items():
        if loss is None or not loss.requires_grad:
            continue
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = grads
        norm_sq = sum(
            grad.detach().double().pow(2).sum()
            for grad in grads if grad is not None)
        metrics[f"grad_norm_{name}"] = float(torch.sqrt(norm_sq).cpu())
    names = sorted(gradients)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            dot = sum(
                grad_left.detach().double().mul(grad_right.detach().double()).sum()
                for grad_left, grad_right in zip(
                    gradients[left], gradients[right])
                if grad_left is not None and grad_right is not None)
            denominator = (
                metrics[f"grad_norm_{left}"] * metrics[f"grad_norm_{right}"])
            cosine = float(dot.cpu()) / max(denominator, 1e-30)
            metrics[f"grad_cos_{left}_{right}"] = cosine
    return metrics
