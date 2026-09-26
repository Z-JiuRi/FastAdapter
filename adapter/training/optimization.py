from .dependencies import *  # noqa: F401,F403

def build_optimizer(model, lr, weight_decay, affine_lr_multiplier=1.0):
    groups = {
        "main_decay": [],
        "main_no_decay": [],
        "affine_decay": [],
        "affine_no_decay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_affine = name in ("alignment_weight", "alignment_bias")
        no_decay = param.ndim == 1 or name.endswith(".bias")
        prefix = "affine" if is_affine else "main"
        suffix = "no_decay" if no_decay else "decay"
        groups[f"{prefix}_{suffix}"].append(param)
    param_groups = []
    for name, params in groups.items():
        if not params:
            continue
        group_lr = lr * affine_lr_multiplier if name.startswith("affine") else lr
        group_decay = 0.0 if name.endswith("no_decay") else weight_decay
        param_groups.append({
            "params": params,
            "lr": group_lr,
            "initial_lr": group_lr,
            "weight_decay": group_decay,
            "group_name": name,
        })
    return torch.optim.AdamW(param_groups, lr=lr)


def linear_warmup(epoch, start_epoch, warmup_epochs, target):
    if target <= 0 or epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return target
    progress = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
    return target * progress


def stage_hyperparameters(args, epoch):
    if args.stage1_epochs <= 0 or epoch <= args.stage1_epochs:
        return {
            "stage": 1 if args.stage1_epochs > 0 else 0,
            "lambda_recon": (
                args.stage1_lambda_recon if args.stage1_epochs > 0
                else args.lambda_recon),
            "lambda_encoder_consistency": (
                args.stage1_lambda_encoder_consistency
                if args.stage1_epochs > 0
                else args.lambda_encoder_consistency),
            "code_noise_std": (
                args.stage1_code_noise_std if args.stage1_epochs > 0
                else args.code_noise_std),
        }

    stage2_epoch = epoch - args.stage1_epochs
    recon = linear_warmup(
        stage2_epoch, 1, args.stage2_recon_warmup_epochs,
        args.lambda_recon)
    encoder = linear_warmup(
        stage2_epoch, args.stage2_encoder_delay_epochs + 1,
        args.stage2_encoder_warmup_epochs,
        args.lambda_encoder_consistency)
    noise = args.code_noise_std
    if args.stage2_noise_decay_epochs > 0:
        stage2_epochs = args.epochs - args.stage1_epochs
        decay_start = max(1, stage2_epochs - args.stage2_noise_decay_epochs + 1)
        if stage2_epoch >= decay_start:
            remaining = stage2_epochs - stage2_epoch
            noise *= max(0.0, remaining / args.stage2_noise_decay_epochs)
    return {
        "stage": 2,
        "lambda_recon": recon,
        "lambda_encoder_consistency": encoder,
        "code_noise_std": noise,
    }


def build_scheduler(optimizer, name, epochs, steps_per_epoch, eta_min):
    if name == "const":
        return FakeLR(optimizer)
    if name == "cosine":
        total_steps = max(1, epochs * steps_per_epoch)
        return WarmUpCosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            T_warmup=0.1 * total_steps,
            eta_min=eta_min)
    raise ValueError(f"Unknown scheduler: {name}")


class ModelEMA:
    def __init__(self, model, decay=0.999, update_every=1):
        if not 0.0 < decay < 1.0:
            raise ValueError("ema_decay must satisfy 0 < decay < 1")
        if update_every <= 0:
            raise ValueError("ema_update_every must be positive")
        self.decay = decay
        self.update_every = update_every
        self.num_updates = 0
        self.shadow = {}
        self.backup = None
        self.reset(model)

    @torch.no_grad()
    def reset(self, model):
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        if self.num_updates % self.update_every:
            return
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if value.is_floating_point():
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    @torch.no_grad()
    def apply(self, model):
        if self.backup is not None:
            raise RuntimeError("EMA weights are already applied")
        self.backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)

    @torch.no_grad()
    def restore(self, model):
        if self.backup is None:
            raise RuntimeError("EMA weights are not applied")
        model.load_state_dict(self.backup)
        self.backup = None

    def state_dict(self):
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

