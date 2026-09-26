import torch
import os
import logging
import json
import warnings
from pathlib import Path

from omegaconf import OmegaConf
import sys

def load_config(config_path: str, cli_args = None,):
    """See README.md for English documentation."""
    # English documentation is provided in README.md.
    config = OmegaConf.load(config_path)
    
    # English documentation is provided in README.md.
    if cli_args is None:
        cli_args = sys.argv[1:]
    
    if not cli_args:
        return config
    
    # English documentation is provided in README.md.
    cli_conf = OmegaConf.from_cli(cli_args)
    config = OmegaConf.merge(config, cli_conf)
    
    return config

def setup_logger(exp_dir: str | Path, append: bool = False):
    """See README.md for English documentation."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(levelname).1s %(asctime)s %(filename)s:%(lineno)-4d] %(message)s",
        datefmt="%m.%d/%H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        exp_dir / "run.log", mode="a" if append else "w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.propagate = False
    warnings_logger.addHandler(stream_handler)
    warnings_logger.addHandler(file_handler)
    return root_logger


def log_runtime_context(target_logger, cfg, exp_dir: str | Path) -> None:
    """See README.md for English documentation."""
    target_logger.info("=> Experiment directory: %s", exp_dir)
    target_logger.info("=> Runtime Context:")
    target_logger.info("   cwd: %s", os.getcwd())
    target_logger.info("   pid: %s", os.getpid())
    target_logger.info("   command: %s", " ".join(sys.argv))
    target_logger.info("   python: %s", sys.version.replace("\n", " "))
    target_logger.info("   pytorch: %s", torch.__version__)
    target_logger.info("   cuda_available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        target_logger.info("   cuda_version: %s", torch.version.cuda)
        target_logger.info("   CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    target_logger.info(
        "=> Config:\n%s",
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2, ensure_ascii=False),
    )

def create_exp_dirs(path: str):
    """See README.md for English documentation."""
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / "logs").mkdir(parents=True, exist_ok=True)
    (Path(path) / "tensorboard").mkdir(parents=True, exist_ok=True)
    (Path(path) / "ckpts").mkdir(parents=True, exist_ok=True)
    (Path(path) / "results" / "hist").mkdir(parents=True, exist_ok=True)
    (Path(path) / "results" / "heatmap").mkdir(parents=True, exist_ok=True)
    (Path(path) / "results" / "diff").mkdir(parents=True, exist_ok=True)
    return Path(path)

def compute_model_parameters(model):
    """See README.md for English documentation."""
    return sum(p.numel() for p in model.parameters())

def get_grad_norm(model):
    """See README.md for English documentation."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def zscore(x, mean=None, std=None):
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / std

def inv_zscore(x, mean, std):
    return x * std + mean

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
