from .dependencies import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403

def initialize_worker(torch_num_threads: int) -> None:
    """Apply the shared intra-op limit when each task thread starts."""
    torch.set_num_threads(torch_num_threads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze hidden-neuron permutation alignment and save logs, tables, "
            "statistics, and plots"
        )
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="single-device mode compute device (default: cpu)",
    )
    parser.add_argument(
        "--cost-type",
        choices=("weight", "activation", "hybrid"),
        default="activation",
    )
    parser.add_argument("--weight-ratio", type=float, default=0.3)
    parser.add_argument("--activation-ratio", type=float, default=0.7)
    parser.add_argument("--bias-scale", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--activation-samples",
        type=int,
        default=4096,
        help="shared train.pt rows for activation cost; 0 uses all rows",
    )
    parser.add_argument(
        "--activation-sample-seed",
        type=int,
        default=20260716,
        help="seed for shared activation row sampling",
    )
    parser.add_argument(
        "--feature-cache",
        choices=("auto", "on", "off"),
        default="auto",
        help="cache raw task features after first computation",
    )
    parser.add_argument(
        "--reference-task",
        default=None,
        help="train task such as transnet/03397; default is first sorted train task",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=(
            "Adapter data root. Accepts either a dataset container or a "
            "concrete task root containing train, val, and test directories."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"dataset name under --data-root when using the new layout (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help=f"scenario name under --data-root when using the new layout (default: {DEFAULT_SCENARIO})",
    )
    parser.add_argument(
        "--splits",
        type=parse_split_list,
        default=list(SPLIT_ORDER),
        help="comma-separated splits to scan; empty selected splits are skipped",
    )
    parser.add_argument(
        "--adapter-name",
        default="adapter.pth",
        help="source compact Adapter filename inside each task directory",
    )
    parser.add_argument(
        "--save-aligned",
        action="store_true",
        help="save the final permuted Adapter state beside each source Adapter",
    )
    parser.add_argument(
        "--aligned-name",
        default="aligned_adapter.pth",
        help="output filename used with --save-aligned",
    )
    parser.add_argument("--probe-samples", type=int, default=16)
    parser.add_argument("--probe-seed", type=int, default=20260716)
    parser.add_argument("--function-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="write logs/tables/statistics but skip PNG/PDF figures",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="task threads in single-device mode; ignored by the GPU pool",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="process-wide PyTorch intra-op threads used by every worker",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="comma-separated GPU ids, for example 0,1,2",
    )
    parser.add_argument(
        "--gpu-tasks",
        default=None,
        help="per-GPU concurrent task limits, for example 2,2,4",
    )
    return parser.parse_args()


def parse_gpu_pool(args: argparse.Namespace) -> tuple[list[torch.device], list[int]] | None:
    if args.gpu_ids is None and args.gpu_tasks is None:
        return None
    if args.gpu_ids is None or args.gpu_tasks is None:
        raise ValueError("--gpu-ids and --gpu-tasks must be specified together")
    try:
        gpu_ids = [int(value.strip()) for value in args.gpu_ids.split(",")]
        gpu_tasks = [int(value.strip()) for value in args.gpu_tasks.split(",")]
    except ValueError as error:
        raise ValueError("GPU ids/tasks must be comma-separated integers") from error
    if not gpu_ids or len(gpu_ids) != len(gpu_tasks):
        raise ValueError("--gpu-ids and --gpu-tasks must have equal non-zero lengths")
    if len(set(gpu_ids)) != len(gpu_ids) or any(value < 0 for value in gpu_ids):
        raise ValueError("GPU ids must be unique non-negative integers")
    if any(value <= 0 for value in gpu_tasks):
        raise ValueError("every --gpu-tasks value must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pool requested but CUDA is unavailable")
    gpu_count = torch.cuda.device_count()
    if any(value >= gpu_count for value in gpu_ids):
        raise ValueError(f"GPU id out of range; visible CUDA device count={gpu_count}")
    return [torch.device(f"cuda:{value}") for value in gpu_ids], gpu_tasks


def validate_args(args: argparse.Namespace) -> tuple[float, float]:
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.probe_samples <= 0:
        raise ValueError("--probe-samples must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.torch_num_threads <= 0:
        raise ValueError("--torch-num-threads must be positive")
    if args.activation_samples < 0:
        raise ValueError("--activation-samples must be non-negative")
    if args.bias_scale < 0.0:
        raise ValueError("--bias-scale must be non-negative")
    if "/" in args.adapter_name or "\\" in args.adapter_name:
        raise ValueError("--adapter-name must be a filename, not a path")
    if "/" in args.aligned_name or "\\" in args.aligned_name:
        raise ValueError("--aligned-name must be a filename, not a path")
    if args.save_aligned and args.adapter_name == args.aligned_name:
        raise ValueError("--adapter-name and --aligned-name must differ when saving")
    if args.cost_type == "weight":
        weight_ratio, activation_ratio = 1.0, 0.0
    elif args.cost_type == "activation":
        weight_ratio, activation_ratio = 0.0, 1.0
    else:
        weight_ratio = args.weight_ratio
        activation_ratio = args.activation_ratio
    if weight_ratio < 0.0 or activation_ratio < 0.0:
        raise ValueError("matching ratios must be non-negative")
    if weight_ratio + activation_ratio <= 0.0:
        raise ValueError("at least one active matching ratio must be positive")
    total = weight_ratio + activation_ratio
    weight_ratio /= total
    activation_ratio /= total
    return weight_ratio, activation_ratio
