from .dependencies import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403
from .model_io import *  # noqa: F401,F403
from .optimization import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .engine import *  # noqa: F401,F403
from .cli import *  # noqa: F401,F403

def run_training(args, preloaded_data=None):
    if args.stage1_epochs < 0 or args.stage1_epochs >= args.epochs:
        if args.stage1_epochs != 0:
            raise ValueError("stage1_epochs must satisfy 0 <= stage1_epochs < epochs")
    for name in (
            "stage2_affine_freeze_epochs",
            "stage2_recon_warmup_epochs",
            "stage2_encoder_delay_epochs",
            "stage2_encoder_warmup_epochs",
            "stage2_noise_decay_epochs"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
    if args.ema_start_epoch <= 0:
        raise ValueError("ema_start_epoch must be positive")
    if args.ema_update_every <= 0:
        raise ValueError("ema_update_every must be positive")
    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    codeword_dir = exp_dir / "codewords"
    tensorboard_dir = exp_dir / "tensorboard"
    for path in (checkpoint_dir, codeword_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    (exp_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8")
    writer = SummaryWriter(str(tensorboard_dir))
    add_text_json(writer, "config/args", vars(args))

    set_seed(args.seed)
    device = resolve_device(args.gpu, args.cpu)
    log_experiment_header(args, exp_dir=exp_dir, target_logger=logger)
    logger.info("=> Device: %s", device)

    decoder, decoder_cfg = load_decoder(args, device)
    if args.code_loss_type in (
            "decoder_sensitivity_mse", "decoder_jac_residual_mse"):
        source = getattr(args, "sensitivity_source", "jacobian")
        if source == "fc_decoder":
            sensitivity = estimate_decoder_fc_sensitivity(decoder)
            logger.info(
                "=> Decoder sensitivity from fc_decoder: min=%.4f max=%.4f std=%.4f",
                float(sensitivity.min()), float(sensitivity.max()),
                float(sensitivity.std()))
        else:
            # Probe with target train codes after they are loaded below if needed;
            # use a temporary load of target train codes for estimation.
            probe_path = None
            if args.target_train_code:
                probe_path = args.target_train_code
            elif args.target_exp:
                probe_path = str(
                    Path(args.target_exp) / "codewords" / "train_code.pt")
            if probe_path is None or not Path(probe_path).exists():
                raise FileNotFoundError(
                    "jacobian sensitivity needs target train codes "
                    f"(tried {probe_path})")
            probe = load_code(
                probe_path, max_samples=args.sensitivity_probe_samples)
            sensitivity = estimate_decoder_jacobian_sensitivity(
                decoder,
                probe,
                device,
                n_hutchinson=args.sensitivity_hutchinson,
                max_samples=args.sensitivity_probe_samples,
            )
            logger.info(
                "=> Decoder jacobian sensitivity (Hutchinson=%d, n=%d): "
                "min=%.4f max=%.4f std=%.4f",
                args.sensitivity_hutchinson,
                min(probe.size(0), args.sensitivity_probe_samples),
                float(sensitivity.min()), float(sensitivity.max()),
                float(sensitivity.std()))
        args._decoder_sensitivity_weight = sensitivity
        torch.save(
            {
                "sensitivity": sensitivity,
                "source": source,
                "power": args.sensitivity_power,
            },
            exp_dir / "decoder_sensitivity.pt")
    target_encoder = None
    if (args.lambda_encoder_consistency > 0
            or args.stage1_lambda_encoder_consistency > 0):
        target_encoder = load_target_encoder(args, device)
        logger.info(
            "=> Loaded frozen target encoder for consistency loss: "
            "lambda=%s target=%s",
            args.lambda_encoder_consistency,
            args.encoder_consistency_target)
    channel, nt, nc = decoder_cfg["channel"], decoder_cfg["nt"], decoder_cfg["nc"]

    datasets = {}
    for split in ("train", "val", "test"):
        if preloaded_data is None:
            max_samples = args.max_train_samples if split == "train" else args.max_eval_samples
            source = load_code(split_paths(args, "source", split), max_samples)
            target = load_code(split_paths(args, "target", split), max_samples)
            csi_path = getattr(args, f"{split}_csi")
            csi = load_csi(csi_path, channel, nt, nc, max_samples)
        else:
            source, target, csi = preloaded_data[split]
        datasets[split] = CodeCsiDataset(source, target, csi)
        logger.info(
            "=> %s dataset: source=%s target=%s csi=%s n=%d",
            split,
            tuple(source.shape),
            tuple(target.shape),
            tuple(csi.shape),
            len(datasets[split]))
        writer.add_scalar(f"data/{split}_samples", len(datasets[split]), 0)
        writer.add_scalar(f"data/{split}_code_dim", source.size(1), 0)

    if args.affine_fit_splits == "train":
        affine_splits = ("train",)
    elif args.affine_fit_splits == "train_val_test":
        affine_splits = ("train", "val", "test")
    else:
        raise ValueError(f"Unknown affine_fit_splits={args.affine_fit_splits}")
    affine_source = torch.cat(
        [datasets[split].source for split in affine_splits], dim=0)
    affine_target = torch.cat(
        [datasets[split].target for split in affine_splits], dim=0)
    logger.info(
        "=> Fitting affine alignment on splits=%s n=%d ridge=%s%s",
        ",".join(affine_splits), len(affine_source), args.align_ridge,
        " [ORACLE: val/test target codes used]"
        if args.affine_fit_splits == "train_val_test" else "")
    weight, bias = fit_affine(
        affine_source, affine_target, ridge=args.align_ridge)
    torch.save({"weight": weight, "bias": bias}, exp_dir / "affine_alignment.pt")
    writer.add_scalar("affine/fit_samples", len(affine_source), 0)
    writer.add_text(
        "affine/fit_splits", ",".join(affine_splits), 0)
    writer.add_scalar("affine/weight_norm", float(weight.norm().item()), 0)
    writer.add_scalar("affine/bias_norm", float(bias.norm().item()), 0)
    writer.add_scalar("affine/weight_mean", float(weight.mean().item()), 0)
    writer.add_scalar("affine/weight_std", float(weight.std().item()), 0)
    writer.add_scalar("affine/bias_mean", float(bias.mean().item()), 0)
    writer.add_scalar("affine/bias_std", float(bias.std().item()), 0)
    mapper = build_mapper(
        args.mapper_type,
        weight,
        bias,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        use_block_norm=not args.no_block_norm,
        use_final_norm=args.use_final_norm,
        train_affine=args.train_affine,
        learnable_residual_gate=args.learnable_residual_gate,
        gate_max=args.gate_max,
        gate_mode=args.gate_mode,
        final_gate_max=args.final_gate_max,
        final_gate_init=args.final_gate_init,
        adaptive_gate_hidden=args.adaptive_gate_hidden,
        bottleneck_dim=args.bottleneck_dim,
        num_groups=args.num_groups,
        group_hidden=args.group_hidden,
        gate_hidden=args.gate_hidden,
        gate_init=args.gate_init,
        num_tokens=args.num_tokens,
        token_hidden=args.token_hidden,
        channel_hidden=args.channel_hidden,
        num_heads=args.num_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
        attention_scale=args.attention_scale,
        attention_input=args.attention_input,
        attention_use_position=args.attention_use_position,
        num_experts=args.num_experts,
        flow_hidden_dim=args.flow_hidden_dim,
        whole_mlp_dims=args.whole_mlp_dims,
        whole_mlp_activation=args.whole_mlp_activation,
        use_affine_alignment=not args.no_affine_alignment,
        lowrank_rank=args.lowrank_rank).to(device)
    if args.init_mapper_checkpoint:
        checkpoint = torch.load(
            args.init_mapper_checkpoint, weights_only=False, map_location="cpu")
        if args.init_mapper_use_ema:
            if "ema" not in checkpoint or "shadow" not in checkpoint["ema"]:
                raise RuntimeError(
                    "--init_mapper_use_ema requested but checkpoint has no ema.shadow")
            state_dict = checkpoint["ema"]["shadow"]
            logger.info(
                "=> Loading mapper initialization from EMA shadow: %s",
                args.init_mapper_checkpoint)
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)
            logger.info(
                "=> Loading mapper initialization from state_dict: %s",
                args.init_mapper_checkpoint)
        missing, unexpected = mapper.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"mapper checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if args.train_last_blocks > 0:
        if not hasattr(mapper, "blocks"):
            raise AttributeError("--train_last_blocks requires mapper.blocks")
        if args.train_last_blocks > len(mapper.blocks):
            raise ValueError(
                f"train_last_blocks={args.train_last_blocks} exceeds "
                f"num_blocks={len(mapper.blocks)}")
        for parameter in mapper.parameters():
            parameter.requires_grad_(False)
        for block in mapper.blocks[-args.train_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        logger.info(
            "=> Frozen mapper except last %d/%d residual blocks",
            args.train_last_blocks, len(mapper.blocks))
    log_parameter_table(mapper, logger)
    total_params, trainable_params = count_parameters(mapper)
    writer.add_scalar("model/total_params", total_params, 0)
    writer.add_scalar("model/trainable_params", trainable_params, 0)
    writer.add_scalar("model/frozen_params", total_params - trainable_params, 0)
    add_text_json(writer, "config/model", {
        "mapper_type": args.mapper_type,
        "hidden_dim": args.hidden_dim,
        "lowrank_rank": args.lowrank_rank,
        "bottleneck_dim": args.bottleneck_dim,
        "num_groups": args.num_groups,
        "group_hidden": args.group_hidden,
        "gate_hidden": args.gate_hidden,
        "gate_init": args.gate_init,
        "num_tokens": args.num_tokens,
        "token_hidden": args.token_hidden,
        "channel_hidden": args.channel_hidden,
        "num_heads": args.num_heads,
        "transformer_ffn_dim": args.transformer_ffn_dim,
        "attention_dim": args.attention_dim,
        "attention_heads": args.attention_heads,
        "attention_dropout": args.attention_dropout,
        "attention_scale": args.attention_scale,
        "attention_input": args.attention_input,
        "attention_use_position": args.attention_use_position,
        "num_experts": args.num_experts,
        "flow_hidden_dim": args.flow_hidden_dim,
        "whole_mlp_dims": args.whole_mlp_dims,
        "whole_mlp_activation": args.whole_mlp_activation,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "residual_scale": args.residual_scale,
        "learnable_residual_gate": args.learnable_residual_gate,
        "gate_max": args.gate_max,
        "gate_mode": args.gate_mode,
        "final_gate_max": args.final_gate_max,
        "final_gate_init": args.final_gate_init,
        "adaptive_gate_hidden": args.adaptive_gate_hidden,
        "gate_l1": args.gate_l1,
        "use_block_norm": not args.no_block_norm,
        "use_final_norm": args.use_final_norm,
        "train_affine": args.train_affine,
        "use_affine_alignment": not args.no_affine_alignment,
        "affine_fit_splits": args.affine_fit_splits,
        "affine_fit_samples": len(affine_source),
        "lambda_encoder_consistency": args.lambda_encoder_consistency,
        "encoder_consistency_target": args.encoder_consistency_target,
        "lambda_delta_norm": args.lambda_delta_norm,
        "code_noise_std": args.code_noise_std,
        "total_params": total_params,
        "trainable_params": trainable_params,
    })

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda"),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda"),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda"),
    }
    eval_loaders = {
        key: DataLoader(
            Subset(dataset, range(min(len(dataset), args.max_eval_samples)))
            if args.max_eval_samples and key != "train" else dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda")
        for key, dataset in datasets.items()
    }

    train_affine_start = datasets["train"].source.matmul(weight) + bias
    train_residual = datasets["train"].target - train_affine_start
    std_weight_cpu, loss_scale = build_code_loss_weight(
        datasets["train"].target, args, train_residual)
    std_weight = std_weight_cpu.to(device) if std_weight_cpu is not None else None
    teacher_train_code = load_optional_code(
        args.teacher_train_code, args.max_train_samples)
    if args.lambda_teacher_code > 0:
        if teacher_train_code is None:
            raise ValueError(
                "--teacher_train_code is required when lambda_teacher_code > 0")
        if teacher_train_code.shape != datasets["train"].target.shape:
            raise ValueError(
                "teacher_train_code shape mismatch: "
                f"{tuple(teacher_train_code.shape)} vs "
                f"{tuple(datasets['train'].target.shape)}")
        logger.info("=> Loaded teacher train code: %s",
                    tuple(teacher_train_code.shape))
    fisher_basis = None
    fisher_weight = None
    fisher_eval_basis = None
    fisher_eval_eigenvalues = None
    if args.lambda_fisher > 0 and not args.fisher_basis_path:
        raise ValueError(
            "--fisher_basis_path is required when lambda_fisher > 0")
    if args.fisher_basis_path:
        fisher_state = torch.load(
            args.fisher_basis_path, weights_only=True, map_location="cpu")
        if not isinstance(fisher_state, dict) or "eigenvectors" not in fisher_state:
            raise ValueError(
                "fisher basis file must contain eigenvectors and eigenvalues")
        eigenvectors = fisher_state["eigenvectors"].float()
        eigenvalues = fisher_state.get("eigenvalues")
        fisher_eval_basis = eigenvectors.contiguous().to(device)
        if eigenvalues is not None:
            fisher_eval_eigenvalues = eigenvalues.float().contiguous().to(device)
        rank = args.fisher_rank or eigenvectors.size(1)
        if rank <= 0 or rank > eigenvectors.size(1):
            raise ValueError(
                f"invalid fisher_rank={rank} for {tuple(eigenvectors.shape)}")
        if args.lambda_fisher > 0:
            fisher_basis = eigenvectors[:, :rank].contiguous().to(device)
        if (args.lambda_fisher > 0 and eigenvalues is not None
                and args.fisher_weight_power != 0):
            fisher_weight = eigenvalues[:rank].float().clamp_min(
                args.std_weight_eps)
            fisher_weight = fisher_weight / fisher_weight.mean().clamp_min(1e-12)
            fisher_weight = fisher_weight.pow(args.fisher_weight_power)
            fisher_weight = fisher_weight / fisher_weight.mean().clamp_min(1e-12)
            fisher_weight = fisher_weight.clamp(max=args.fisher_weight_max).to(device)
        logger.info(
            "=> Fisher diagnostics/loss: path=%s eval_rank=%d train_rank=%d "
            "lambda=%s power=%s weight=[%.4f, %.4f]",
            args.fisher_basis_path, eigenvectors.size(1), rank, args.lambda_fisher,
            args.fisher_weight_power,
            float(fisher_weight.min()) if fisher_weight is not None else 1.0,
            float(fisher_weight.max()) if fisher_weight is not None else 1.0)
    if std_weight_cpu is not None:
        torch.save({
            "std_weight": std_weight_cpu,
            "loss_scale": loss_scale,
            "code_loss_type": args.code_loss_type,
            "std_weight_min": args.std_weight_min,
            "std_weight_max": args.std_weight_max,
            "std_weight_eps": args.std_weight_eps,
        }, exp_dir / "code_loss_weight.pt")
        weight_stats = {
            "mean": float(std_weight_cpu.mean()),
            "std": float(std_weight_cpu.std()),
            "min": float(std_weight_cpu.min()),
            "max": float(std_weight_cpu.max()),
            "scale_mean": float(loss_scale.mean()),
            "scale_min": float(loss_scale.min()),
            "scale_max": float(loss_scale.max()),
        }
        for key, value in weight_stats.items():
            writer.add_scalar(f"code_loss_weight/{key}", value, 0)
        logger.info("=> Code loss weight stats: %s", weight_stats)

    optimizer = build_optimizer(mapper, args.lr, args.weight_decay)
    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(
            mapper, decay=args.ema_decay,
            update_every=args.ema_update_every)
        logger.info(
            "=> EMA enabled: decay=%s start_epoch=%d update_every=%d",
            args.ema_decay, args.ema_start_epoch, args.ema_update_every)
    first_stage_epochs = args.stage1_epochs or args.epochs
    scheduler = build_scheduler(
        optimizer, args.scheduler, first_stage_epochs, len(loaders["train"]),
        args.eta_min)
    logger.info(
        "=> Optimizer: AdamW lr=%s weight_decay=%s scheduler=%s eta_min=%s",
        args.lr, args.weight_decay, args.scheduler, args.eta_min)
    logger.info(
        "=> Loss: type=%s lambda_code=%s lambda_feature=%s lambda_recon=%s "
        "lambda_encoder_consistency=%s encoder_consistency_target=%s "
        "lambda_delta_norm=%s lambda_teacher_code=%s code_noise_std=%s",
        args.code_loss_type, args.lambda_code, args.lambda_feature,
        args.lambda_recon, args.lambda_encoder_consistency,
        args.encoder_consistency_target, args.lambda_delta_norm,
        args.lambda_teacher_code,
        args.code_noise_std)
    writer.add_scalar("loss_weights/code_loss_type_is_weighted",
                      float(args.code_loss_type != "mse"), 0)
    writer.add_scalar("loss_weights/lambda_code", args.lambda_code, 0)
    writer.add_scalar("loss_weights/lambda_feature", args.lambda_feature, 0)
    writer.add_scalar("loss_weights/lambda_recon", args.lambda_recon, 0)
    writer.add_scalar(
        "loss_weights/lambda_encoder_consistency",
        args.lambda_encoder_consistency,
        0)
    writer.add_scalar("loss_weights/gate_l1", args.gate_l1, 0)
    writer.add_scalar("loss_weights/lambda_delta_norm", args.lambda_delta_norm, 0)
    writer.add_scalar("loss_weights/lambda_teacher_code", args.lambda_teacher_code, 0)
    writer.add_scalar("loss_weights/lambda_fisher", args.lambda_fisher, 0)
    writer.add_scalar("regularization/code_noise_std", args.code_noise_std, 0)

    best = {
        "val_code_mse": {"metric": math.inf, "epoch": 0},
        "val_decoder_nmse": {"metric": math.inf, "epoch": 0},
    }
    history = []

    def save_checkpoint(name, epoch, metrics):
        payload = {
            "epoch": epoch,
            "state_dict": mapper.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()
        torch.save(payload, checkpoint_dir / name)

    for epoch in range(1, args.epochs + 1):
        if args.stage1_epochs > 0 and epoch == args.stage1_epochs + 1:
            stage1_path = checkpoint_dir / "best_code_mse.pth"
            if not stage1_path.exists():
                raise RuntimeError(
                    "two-stage training requires a stage-1 best_code_mse "
                    "checkpoint; check eval_every and stage1_epochs")
            checkpoint = torch.load(
                stage1_path, weights_only=True, map_location=device)
            mapper.load_state_dict(checkpoint["state_dict"])
            if ema is not None:
                ema.reset(mapper)
            stage1_best = dict(best["val_code_mse"])
            torch.save(checkpoint, checkpoint_dir / "stage1_best_code_mse.pth")
            best["stage1_val_code_mse"] = stage1_best
            best["val_decoder_nmse"] = {"metric": math.inf, "epoch": 0}

            freeze_affine = args.stage2_affine_freeze_epochs > 0
            for name, param in mapper.named_parameters():
                if name in ("alignment_weight", "alignment_bias"):
                    param.requires_grad_(not freeze_affine)
            stage2_lr = args.stage2_lr or args.lr
            optimizer = build_optimizer(
                mapper, stage2_lr, args.weight_decay,
                affine_lr_multiplier=args.stage2_affine_lr_multiplier)
            scheduler = build_scheduler(
                optimizer, args.scheduler,
                args.epochs - args.stage1_epochs,
                len(loaders["train"]), args.eta_min)
            logger.info(
                "=> Stage 2 starts from stage-1 best code checkpoint: "
                "epoch=%d code_mse=%.6e lr=%s affine_lr_multiplier=%s "
                "affine_frozen=%s",
                checkpoint["epoch"], stage1_best["metric"], stage2_lr,
                args.stage2_affine_lr_multiplier, freeze_affine)

        unfreeze_epoch = (
            args.stage1_epochs + args.stage2_affine_freeze_epochs + 1)
        if (args.stage1_epochs > 0
                and args.stage2_affine_freeze_epochs > 0
                and epoch == unfreeze_epoch):
            for name, param in mapper.named_parameters():
                if name in ("alignment_weight", "alignment_bias"):
                    param.requires_grad_(True)
            stage2_lr = args.stage2_lr or args.lr
            optimizer = build_optimizer(
                mapper, stage2_lr, args.weight_decay,
                affine_lr_multiplier=args.stage2_affine_lr_multiplier)
            scheduler = build_scheduler(
                optimizer, args.scheduler, args.epochs - epoch + 1,
                len(loaders["train"]), args.eta_min)
            logger.info(
                "=> Stage 2 affine parameters unfrozen at epoch %d; "
                "affine lr=%s", epoch,
                stage2_lr * args.stage2_affine_lr_multiplier)

        stage_values = stage_hyperparameters(args, epoch)
        if ema is not None and epoch == args.ema_start_epoch:
            ema.reset(mapper)
        active_ema = (
            ema if ema is not None and epoch >= args.ema_start_epoch else None)
        train_metrics = train_epoch(
            mapper,
            loaders["train"],
            decoder,
            device,
            optimizer,
            scheduler,
            args.lambda_code,
            stage_values["lambda_recon"],
            args.lambda_feature,
            args.code_loss_type,
            std_weight,
            args.gate_l1,
            target_encoder,
            stage_values["lambda_encoder_consistency"],
            args.encoder_consistency_target,
            stage_values["code_noise_std"],
            args.lambda_delta_norm,
            active_ema,
            teacher_train_code,
            args.lambda_teacher_code,
            fisher_basis,
            fisher_weight,
            args.lambda_fisher,
            args.gradient_diagnostics_every > 0
            and epoch % args.gradient_diagnostics_every == 0)
        record = {"epoch": epoch, "lr": scheduler.get_lr()[0],
                  "stage": stage_values,
                  "eval_weights": "ema" if active_ema is not None else "raw",
                  "train": train_metrics}
        log_metrics(writer, "train", train_metrics, epoch)
        writer.add_scalar("train/lr", scheduler.get_lr()[0], epoch)
        writer.add_scalar("schedule/stage", stage_values["stage"], epoch)
        writer.add_scalar(
            "schedule/lambda_recon", stage_values["lambda_recon"], epoch)
        writer.add_scalar(
            "schedule/lambda_encoder_consistency",
            stage_values["lambda_encoder_consistency"], epoch)
        writer.add_scalar(
            "schedule/code_noise_std", stage_values["code_noise_std"], epoch)

        if args.eval_every and epoch % args.eval_every == 0:
            if active_ema is not None:
                active_ema.apply(mapper)
            for split in ("val", "test"):
                metrics = evaluate(
                    mapper,
                    eval_loaders[split],
                    decoder,
                    device,
                    args.code_loss_type,
                    std_weight,
                    target_encoder,
                    args.encoder_consistency_target,
                    fisher_eval_basis,
                    fisher_eval_eigenvalues)
                record[split] = metrics
                log_metrics(writer, split, metrics, epoch)
                logger.info(
                    "Epoch [%d/%d] %s code_loss=%.6e code_mse=%.6e code_nmse=%.3fdB "
                    "cos=%.6f decoder_mse=%.6e decoder_nmse=%.3fdB n=%d",
                    epoch, args.epochs, split, metrics["code_loss"],
                    metrics["code_mse"], metrics["code_nmse"], metrics["code_cos"],
                    metrics["decoder_mse"], metrics["decoder_nmse"],
                    metrics["n"])
                logger.info(
                    "Epoch [%d/%d] %s_metrics=%s",
                    epoch,
                    args.epochs,
                    split,
                    json.dumps(metrics, sort_keys=True))
            val_metrics = record["val"]
            param_stats = collect_parameter_stats(mapper)
            param_summary = summarize_parameter_stats(param_stats)
            record["param_stats"] = param_stats
            record["param_summary"] = param_summary
            log_metrics(writer, "params", param_stats, epoch)
            log_metrics(writer, "param_summary", param_summary, epoch)
            logger.info(
                "Epoch [%d/%d] param_summary=%s",
                epoch,
                args.epochs,
                json.dumps(param_summary, sort_keys=True))
            logger.info(
                "Epoch [%d/%d] param_stats=%s",
                epoch,
                args.epochs,
                json.dumps(param_stats, sort_keys=True))
            if val_metrics["code_mse"] < best["val_code_mse"]["metric"]:
                best["val_code_mse"] = {
                    "metric": val_metrics["code_mse"],
                    "epoch": epoch,
                    "metrics": val_metrics,
                }
                save_checkpoint("best_code_mse.pth", epoch, val_metrics)
                writer.add_scalar(
                    "best/val_code_mse", val_metrics["code_mse"], epoch)
                writer.add_scalar(
                    "best/val_code_nmse", val_metrics["code_nmse"], epoch)
            decoder_selection_enabled = (
                args.stage1_epochs <= 0 or epoch > args.stage1_epochs)
            if (decoder_selection_enabled
                    and val_metrics["decoder_nmse"]
                    < best["val_decoder_nmse"]["metric"]):
                best["val_decoder_nmse"] = {
                    "metric": val_metrics["decoder_nmse"],
                    "epoch": epoch,
                    "metrics": val_metrics,
                }
                save_checkpoint("best_decoder_nmse.pth", epoch, val_metrics)
                writer.add_scalar(
                    "best/val_decoder_nmse",
                    val_metrics["decoder_nmse"],
                    epoch)
                writer.add_scalar(
                    "best/val_decoder_mse",
                    val_metrics["decoder_mse"],
                    epoch)
            if active_ema is not None:
                active_ema.restore(mapper)

        if hasattr(mapper, "get_metrics"):
            adapter_metrics = mapper.get_metrics()
            record["adapter"] = adapter_metrics
            log_metrics(writer, "adapter", adapter_metrics, epoch)

        history.append(record)
        adapter_msg = ""
        if record.get("adapter"):
            keep = [
                "adapter/delta_ratio",
                "adapter/gate_mean_avg",
                "adapter/gate_max_max",
            ]
            adapter_msg = " " + " ".join(
                f"{key.split('/')[-1]}={record['adapter'][key]:.6e}"
                for key in keep
                if key in record["adapter"])
        logger.info(
            "Epoch [%d/%d] lr=%.6e train_loss=%.6e train_code_loss=%.6e "
            "train_code_mse=%.6e train_code_nmse=%.3fdB train_cos=%.6f "
            "train_recon_mse=%.6e train_decoder_nmse=%.3fdB "
            "delta_target_cos=%.6f residual_coverage=%.6f%s",
            epoch, args.epochs, scheduler.get_lr()[0],
            train_metrics["loss"], train_metrics["code_loss"],
            train_metrics["code_mse"], train_metrics["code_nmse"],
            train_metrics["code_cos"], train_metrics["recon_mse"],
            train_metrics["decoder_nmse"],
            train_metrics.get("delta_target_cos", 0.0),
            train_metrics.get("residual_coverage", 0.0),
            adapter_msg)
        logger.info(
            "Epoch [%d/%d] train_metrics=%s",
            epoch,
            args.epochs,
            json.dumps(train_metrics, sort_keys=True))

    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps(best, indent=2))
    add_text_json(writer, "result/best", best, args.epochs)
    if math.isfinite(best["val_code_mse"]["metric"]):
        writer.add_scalar(
            "result/best_val_code_mse",
            best["val_code_mse"]["metric"],
            best["val_code_mse"]["epoch"])
    if math.isfinite(best["val_decoder_nmse"]["metric"]):
        writer.add_scalar(
            "result/best_val_decoder_nmse",
            best["val_decoder_nmse"]["metric"],
            best["val_decoder_nmse"]["epoch"])
    save_checkpoint("last.pth", args.epochs, history[-1] if history else {})

    if args.export_codewords:
        ckpt_path = checkpoint_dir / "best_decoder_nmse.pth"
        if not ckpt_path.exists():
            ckpt_path = checkpoint_dir / "best_code_mse.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
            mapper.load_state_dict(ckpt["state_dict"])
            logger.info("=> Exporting mapped codewords from %s", ckpt_path)
        for split in ("train", "val", "test"):
            shape = export_mapped_code(
                mapper,
                datasets[split],
                device,
                codeword_dir / f"{split}_mapped_code.pt",
                args.batch_size,
                args.workers)
            logger.info("=> Saved %s mapped codewords %s", split, shape)

    writer.flush()
    writer.close()
    logger.info("=> Best val_code_mse: %s", best["val_code_mse"])
    logger.info("=> Best val_decoder_nmse: %s", best["val_decoder_nmse"])

