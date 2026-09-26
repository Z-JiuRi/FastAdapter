from .dependencies import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403
from .optimization import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403

def train_epoch(model, loader, decoder, device, optimizer, scheduler,
                lambda_code, lambda_recon, lambda_feature=0.0,
                code_loss_type="mse", std_weight=None, gate_l1=0.0,
                target_encoder=None, lambda_encoder_consistency=0.0,
                encoder_consistency_target="mapped",
                code_noise_std=0.0, lambda_delta_norm=0.0, ema=None,
                teacher_code=None, lambda_teacher_code=0.0,
                fisher_basis=None, fisher_weight=None, lambda_fisher=0.0,
                gradient_diagnostics=False):
    model.train()
    decoder.eval()
    total = {
        "loss": 0.0,
        "code_loss": 0.0,
        "code_mse": 0.0,
        "feature_mse": 0.0,
        "teacher_code_mse": 0.0,
        "fisher_mse": 0.0,
        "encoder_consistency_mse": 0.0,
        "delta_norm_mse": 0.0,
        "gate_l1": 0.0,
        "recon_mse": 0.0,
        "cos": 0.0,
        "n": 0,
    }
    code_err = torch.tensor(0.0, device=device)
    code_power = torch.tensor(0.0, device=device)
    recon_err = torch.tensor(0.0, device=device)
    recon_power = torch.tensor(0.0, device=device)
    delta_totals = init_delta_totals(device)
    gradient_metrics = {}
    for batch_index, (source, target, csi, indices) in enumerate(loader):
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        csi = csi.to(device, non_blocking=True)
        source_input = source
        if code_noise_std > 0:
            noise_scale = source.detach().std(dim=0, unbiased=False).clamp_min(1e-6)
            source_input = source + code_noise_std * noise_scale * torch.randn_like(source)
        mapped = model(source_input)
        with torch.no_grad():
            z0_diag = get_start_code(model, source_input)
            update_delta_totals(delta_totals, mapped, target, z0_diag)
        code_loss, code_mse = compute_code_loss(
            mapped, target, code_loss_type, std_weight)
        teacher_code_loss = mapped.new_tensor(0.0)
        if lambda_teacher_code > 0:
            if teacher_code is None:
                raise ValueError("teacher_code is required when lambda_teacher_code > 0")
            teacher = teacher_code[indices].to(device, non_blocking=True)
            teacher_code_loss = F.mse_loss(mapped, teacher)
        fisher_loss = mapped.new_tensor(0.0)
        if lambda_fisher > 0:
            if fisher_basis is None:
                raise ValueError("fisher_basis is required when lambda_fisher > 0")
            coefficients = (mapped - target).matmul(fisher_basis)
            if fisher_weight is not None:
                coefficients = coefficients * fisher_weight.sqrt().view(1, -1)
            fisher_loss = coefficients.pow(2).mean()
        feature_loss = mapped.new_tensor(0.0)
        if lambda_feature > 0:
            if not hasattr(decoder, "fc_decoder"):
                raise AttributeError("decoder has no fc_decoder for feature loss")
            feature_loss = F.mse_loss(
                decoder.fc_decoder(mapped),
                decoder.fc_decoder(target).detach())
        gate_reg = mapped.new_tensor(0.0)
        if gate_l1 > 0 and hasattr(model, "gate_regularization"):
            reg = model.gate_regularization()
            if reg is not None:
                gate_reg = reg
        delta_norm_loss = mapped.new_tensor(0.0)
        if lambda_delta_norm > 0:
            z0 = get_start_code(model, source_input)
            delta_norm_loss = F.mse_loss(mapped, z0)
        need_decoder_grad = lambda_recon > 0 or lambda_encoder_consistency > 0
        encoder_consistency_loss = mapped.new_tensor(0.0)
        if need_decoder_grad:
            recon = decoder(mapped)
            recon_loss = F.mse_loss(recon, csi)
            if lambda_encoder_consistency > 0:
                if target_encoder is None:
                    raise ValueError(
                        "target_encoder is required when lambda_encoder_consistency > 0")
                reencoded = target_encoder(recon)
                if encoder_consistency_target == "mapped":
                    enc_target = mapped
                elif encoder_consistency_target == "target":
                    enc_target = target
                else:
                    raise ValueError(
                        f"Unknown encoder_consistency_target: "
                        f"{encoder_consistency_target}")
                encoder_consistency_loss = F.mse_loss(reencoded, enc_target)
            loss = (
                lambda_code * code_loss
                + lambda_teacher_code * teacher_code_loss
                + lambda_fisher * fisher_loss
                + lambda_feature * feature_loss
                + lambda_recon * recon_loss
                + lambda_encoder_consistency * encoder_consistency_loss
                + lambda_delta_norm * delta_norm_loss
                + gate_l1 * gate_reg)
        else:
            with torch.no_grad():
                recon = decoder(mapped.detach())
                recon_loss = F.mse_loss(recon, csi)
            loss = (
                lambda_code * code_loss
                + lambda_teacher_code * teacher_code_loss
                + lambda_fisher * fisher_loss
                + lambda_feature * feature_loss
                + lambda_delta_norm * delta_norm_loss
                + gate_l1 * gate_reg)
        if gradient_diagnostics and batch_index == 0:
            diagnostic_losses = {
                "code": lambda_code * code_loss if lambda_code > 0 else None,
                "recon": lambda_recon * recon_loss
                if lambda_recon > 0 else None,
                "encoder": lambda_encoder_consistency * encoder_consistency_loss
                if lambda_encoder_consistency > 0 else None,
                "fisher": lambda_fisher * fisher_loss
                if lambda_fisher > 0 else None,
            }
            gradient_metrics = loss_gradient_diagnostics(
                diagnostic_losses, model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if scheduler is not None:
            scheduler.step()
        n = source.size(0)
        cerr = mapped.detach() - target
        rerr = recon.detach() - csi
        code_err += cerr.pow(2).sum()
        code_power += target.pow(2).sum()
        recon_err += rerr.pow(2).sum()
        recon_power += csi.pow(2).sum()
        total["loss"] += float(loss.detach().cpu()) * n
        total["code_loss"] += float(code_loss.detach().cpu()) * n
        total["code_mse"] += float(code_mse.detach().cpu()) * n
        total["feature_mse"] += float(feature_loss.detach().cpu()) * n
        total["teacher_code_mse"] += float(teacher_code_loss.detach().cpu()) * n
        total["fisher_mse"] += float(fisher_loss.detach().cpu()) * n
        total["encoder_consistency_mse"] += float(
            encoder_consistency_loss.detach().cpu()) * n
        total["delta_norm_mse"] += float(delta_norm_loss.detach().cpu()) * n
        total["gate_l1"] += float(gate_reg.detach().cpu()) * n
        total["recon_mse"] += float(recon_loss.detach().cpu()) * n
        total["cos"] += float(
            F.cosine_similarity(mapped.detach(), target, dim=1).mean().cpu()) * n
        total["n"] += n
    metrics = {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}
    metrics["code_nmse"] = float((10.0 * torch.log10(
        code_err / code_power.clamp_min(1e-12))).cpu())
    metrics["code_cos"] = metrics.pop("cos")
    metrics["decoder_nmse"] = float((10.0 * torch.log10(
        recon_err / recon_power.clamp_min(1e-12))).cpu())
    metrics["decoder_mse"] = metrics["recon_mse"]
    metrics.update(finalize_delta_metrics(delta_totals))
    metrics.update(gradient_metrics)
    return metrics


@torch.no_grad()
def evaluate(model, loader, decoder, device, code_loss_type="mse",
             std_weight=None, target_encoder=None,
             encoder_consistency_target="mapped",
             fisher_basis=None, fisher_eigenvalues=None):
    model.eval()
    decoder.eval()
    code_err = torch.tensor(0.0, device=device)
    code_power = torch.tensor(0.0, device=device)
    recon_err = torch.tensor(0.0, device=device)
    recon_power = torch.tensor(0.0, device=device)
    code_loss_sum = 0.0
    code_mse_sum = 0.0
    feature_mse_sum = 0.0
    encoder_consistency_sum = 0.0
    recon_mse_sum = 0.0
    cos_sum = 0.0
    n_total = 0
    delta_totals = init_delta_totals(device)
    z0_recon_err = torch.tensor(0.0, device=device)
    target_recon_err = torch.tensor(0.0, device=device)
    decoder_delta_err = torch.tensor(0.0, device=device)
    decoder_delta_code = torch.tensor(0.0, device=device)
    reencoded_target_mse_sum = 0.0
    reencoded_mapped_mse_sum = 0.0
    reencoded_target_cos_sum = 0.0
    reencoded_mapped_cos_sum = 0.0
    fisher_sq_sum = None
    if fisher_basis is not None:
        fisher_sq_sum = torch.zeros(
            fisher_basis.size(1), device=device, dtype=torch.float64)
    for source, target, csi, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        csi = csi.to(device, non_blocking=True)
        mapped = model(source)
        z0 = get_start_code(model, source)
        recon = decoder(mapped)
        update_delta_totals(delta_totals, mapped, target, z0)
        if z0 is not None:
            z0_recon = decoder(z0)
            target_recon = decoder(target)
            z0_recon_err += (z0_recon - csi).pow(2).sum()
            target_recon_err += (target_recon - csi).pow(2).sum()
            decoder_delta_err += (recon - z0_recon).pow(2).sum()
            decoder_delta_code += (mapped - z0).pow(2).sum()
        if hasattr(decoder, "fc_decoder"):
            feature_mse = F.mse_loss(
                decoder.fc_decoder(mapped),
                decoder.fc_decoder(target))
        else:
            feature_mse = mapped.new_tensor(0.0)
        if target_encoder is not None:
            reencoded = target_encoder(recon)
            reencoded_target_mse = F.mse_loss(reencoded, target)
            reencoded_mapped_mse = F.mse_loss(reencoded, mapped)
            reencoded_target_cos = F.cosine_similarity(
                reencoded, target, dim=1).mean()
            reencoded_mapped_cos = F.cosine_similarity(
                reencoded, mapped, dim=1).mean()
            if encoder_consistency_target == "mapped":
                enc_target = mapped
            elif encoder_consistency_target == "target":
                enc_target = target
            else:
                raise ValueError(
                    f"Unknown encoder_consistency_target: "
                    f"{encoder_consistency_target}")
            encoder_consistency_mse = F.mse_loss(reencoded, enc_target)
        else:
            encoder_consistency_mse = mapped.new_tensor(0.0)
            reencoded_target_mse = mapped.new_tensor(0.0)
            reencoded_mapped_mse = mapped.new_tensor(0.0)
            reencoded_target_cos = mapped.new_tensor(0.0)
            reencoded_mapped_cos = mapped.new_tensor(0.0)
        n = source.size(0)
        cerr = mapped - target
        if fisher_sq_sum is not None:
            fisher_coeff = cerr.matmul(fisher_basis)
            fisher_sq_sum += fisher_coeff.double().pow(2).sum(dim=0)
        rerr = recon - csi
        code_loss, code_mse = compute_code_loss(
            mapped, target, code_loss_type, std_weight)
        code_err += cerr.pow(2).sum()
        code_power += target.pow(2).sum()
        recon_err += rerr.pow(2).sum()
        recon_power += csi.pow(2).sum()
        code_loss_sum += float(code_loss.cpu()) * n
        code_mse_sum += float(code_mse.cpu()) * n
        feature_mse_sum += float(feature_mse.cpu()) * n
        encoder_consistency_sum += float(encoder_consistency_mse.cpu()) * n
        reencoded_target_mse_sum += float(reencoded_target_mse.cpu()) * n
        reencoded_mapped_mse_sum += float(reencoded_mapped_mse.cpu()) * n
        reencoded_target_cos_sum += float(reencoded_target_cos.cpu()) * n
        reencoded_mapped_cos_sum += float(reencoded_mapped_cos.cpu()) * n
        recon_mse_sum += float(rerr.pow(2).mean().cpu()) * n
        cos_sum += float(F.cosine_similarity(mapped, target, dim=1).mean().cpu()) * n
        n_total += n
    metrics = {
        "code_loss": code_loss_sum / max(n_total, 1),
        "code_mse": code_mse_sum / max(n_total, 1),
        "feature_mse": feature_mse_sum / max(n_total, 1),
        "encoder_consistency_mse": encoder_consistency_sum / max(n_total, 1),
        "reencoded_target_mse": reencoded_target_mse_sum / max(n_total, 1),
        "reencoded_mapped_mse": reencoded_mapped_mse_sum / max(n_total, 1),
        "reencoded_target_cos": reencoded_target_cos_sum / max(n_total, 1),
        "reencoded_mapped_cos": reencoded_mapped_cos_sum / max(n_total, 1),
        "code_nmse": float((10.0 * torch.log10(
            code_err / code_power.clamp_min(1e-12))).cpu()),
        "code_cos": cos_sum / max(n_total, 1),
        "decoder_mse": recon_mse_sum / max(n_total, 1),
        "decoder_nmse": float((10.0 * torch.log10(
            recon_err / recon_power.clamp_min(1e-12))).cpu()),
        "n": n_total,
    }
    metrics.update(finalize_delta_metrics(delta_totals))
    if fisher_sq_sum is not None:
        dim = fisher_sq_sum.numel()
        total_projected = fisher_sq_sum.sum().clamp_min(1e-30)
        if fisher_eigenvalues is None:
            normalized_eigenvalues = torch.ones_like(fisher_sq_sum)
        else:
            normalized_eigenvalues = fisher_eigenvalues[:dim].to(
                device=device, dtype=torch.float64).clamp_min(0.0)
            normalized_eigenvalues /= normalized_eigenvalues.mean().clamp_min(1e-30)
        weighted_by_dim = fisher_sq_sum * normalized_eigenvalues
        total_weighted = weighted_by_dim.sum().clamp_min(1e-30)
        metrics["fisher_full_mse"] = float(
            (total_projected / max(n_total * dim, 1)).cpu())
        metrics["fisher_full_weighted_mse"] = float(
            (total_weighted / max(n_total * dim, 1)).cpu())
        boundaries = [0, 16, 32, 64, 128, 256, 384, dim]
        boundaries = sorted(set(min(max(value, 0), dim) for value in boundaries))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            prefix = f"fisher_band_{start:03d}_{end:03d}"
            band_sq = fisher_sq_sum[start:end].sum()
            band_weighted = weighted_by_dim[start:end].sum()
            metrics[f"{prefix}_mse"] = float(
                (band_sq / max(n_total * (end - start), 1)).cpu())
            metrics[f"{prefix}_energy_frac"] = float(
                (band_sq / total_projected).cpu())
            metrics[f"{prefix}_impact_frac"] = float(
                (band_weighted / total_weighted).cpu())
        for rank in (64, 128, 256, 384):
            if rank > dim:
                continue
            top_sq = fisher_sq_sum[:rank].sum()
            top_weighted = weighted_by_dim[:rank].sum()
            metrics[f"fisher_top{rank}_mse"] = float(
                (top_sq / max(n_total * rank, 1)).cpu())
            metrics[f"fisher_top{rank}_energy_frac"] = float(
                (top_sq / total_projected).cpu())
            metrics[f"fisher_top{rank}_impact_frac"] = float(
                (top_weighted / total_weighted).cpu())
    if delta_totals["n"] > 0:
        metrics["z0_decoder_nmse"] = float((10.0 * torch.log10(
            z0_recon_err / recon_power.clamp_min(1e-12))).cpu())
        metrics["target_decoder_nmse"] = float((10.0 * torch.log10(
            target_recon_err / recon_power.clamp_min(1e-12))).cpu())
        metrics["decoder_sensitivity"] = float(torch.sqrt(
            decoder_delta_err / decoder_delta_code.clamp_min(1e-12)).cpu())
    return metrics


@torch.no_grad()
def export_mapped_code(model, dataset, device, output_path, batch_size, workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    model.eval()
    chunks = []
    indices = []
    for source, _, _, idx in loader:
        source = source.to(device, non_blocking=True)
        chunks.append(model(source).cpu())
        indices.append(idx.cpu())
    mapped = torch.cat(chunks, dim=0)
    idx = torch.cat(indices, dim=0).long()
    if not torch.equal(idx, torch.arange(idx.numel())):
        aligned = torch.empty_like(mapped)
        aligned[idx] = mapped
        mapped = aligned
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, output_path)
    return tuple(mapped.shape)


def log_metrics(writer, prefix, metrics, step):
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters()
                    if param.requires_grad)
    return total, trainable


def add_text_json(writer, tag, payload, step=0):
    text = "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"
    writer.add_text(tag, text, step)

