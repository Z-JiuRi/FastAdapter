import argparse
from pathlib import Path
from omegaconf import OmegaConf

from core.trainer import Trainer
from core.inferencer import Inferencer
from utils.distributed import cleanup_distributed, init_distributed
from utils.preflight import validate_runtime_before_experiment
from utils.threading import configure_torch_threads


SUBMIT_ROOT = Path(__file__).resolve().parents[1]


def resolve_runtime_paths(cfg):
    fields = (
        "data.task_root", "cache.stats_path", "exp_dir",
        "inference.checkpoint_path", "inference.cond_path", "inference.output_dir",
        "eval.decoder_args", "eval.decoder_checkpoint", "eval.csi_path",
        "eval.csi_val_path",
    )
    for field in fields:
        value = OmegaConf.select(cfg, field)
        if value in (None, ""):
            continue
        path = Path(str(value))
        if not path.is_absolute():
            OmegaConf.update(cfg, field, str((SUBMIT_ROOT / path).resolve()))
    return cfg

def main():
    thread_state = configure_torch_threads()
    dist_state = init_distributed()
    parser = argparse.ArgumentParser(
        description='Codeword-conditioned Adapter diffusion generation')
    parser.add_argument('--config', type=str, default='', help="")
    parser.add_argument('--mode', type=str, default='', help="")
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Override a configuration value; may be repeated.',
    )
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    cfg = OmegaConf.load(config_path)
    base_path = cfg.pop("_base_", None)
    if base_path:
        base_path = Path(str(base_path))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        cfg = OmegaConf.merge(OmegaConf.load(base_path), cfg)
    if args.override:
        invalid_overrides = [item for item in args.override if '=' not in item]
        if invalid_overrides:
            parser.error(
                '--override must use KEY=VALUE format: '
                + ', '.join(invalid_overrides)
            )
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    if dist_state["enabled"]:
        OmegaConf.update(cfg, "data.device", f"cuda:{dist_state['local_rank']}", merge=False)
    cfg = resolve_runtime_paths(cfg)
    validate_runtime_before_experiment([cfg], [str(config_path)])

    try:
        if args.mode == 'train':
            trainer = Trainer(cfg, distributed=bool(dist_state["enabled"]))
            if trainer.is_main:
                import logging
                logging.getLogger(__name__).info(
                    "=> CPU threads: torch_num_threads=%d torch_num_interop_threads=%d",
                    thread_state["torch_num_threads"],
                    thread_state["torch_num_interop_threads"],
                )
            trainer.train()
        elif args.mode in ('infer', 'inference'):
            inferencer = Inferencer(cfg, force_eval=False)
            inferencer.inference()
        elif args.mode == 'eval':
            inferencer = Inferencer(cfg, force_eval=True)
            inferencer.inference()
        else:
            raise ValueError("--mode must be train, infer, or eval")
    finally:
        cleanup_distributed()

if __name__ == '__main__':
    main()
