from .dependencies import *  # noqa: F401,F403

class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class TaskRecord:
    split: str
    name: str
    directory: Path
    adapter_name: str = "adapter.pth"

    @property
    def adapter_path(self) -> Path:
        return self.directory / self.adapter_name

    @property
    def train_code_path(self) -> Path:
        return self.directory / "train.pt"


@dataclass
class FeatureBundle:
    state: OrderedDict[str, torch.Tensor]
    signatures: list[torch.Tensor] | None
    activations: list[torch.Tensor] | None


def resolve_task_root(data_root: Path, dataset: str, scenario: str) -> Path:
    """Resolve either a legacy task root or the new data/dataset/scenario root."""
    if (data_root / "train").is_dir() or (data_root / "val").is_dir() or (
        data_root / "test"
    ).is_dir():
        return data_root
    return data_root / dataset / scenario


def discover_tasks(
    split: str,
    limit: int,
    task_root: Path = DEFAULT_TASK_ROOT,
    adapter_name: str = "adapter.pth",
) -> list[TaskRecord]:
    split_root = task_root / split
    paths = sorted(split_root.glob(f"*/*/{adapter_name}"))
    records = [
        TaskRecord(
            split=split,
            name=path.parent.relative_to(split_root).as_posix(),
            directory=path.parent,
            adapter_name=adapter_name,
        )
        for path in paths
    ]
    return records[:limit] if limit > 0 else records


def parse_split_list(text: str) -> list[str]:
    splits = [value.strip() for value in text.split(",") if value.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("expected at least one split")
    invalid = sorted(set(splits) - set(SPLIT_ORDER))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown split(s): {','.join(invalid)}; choices: {','.join(SPLIT_ORDER)}"
        )
    return list(dict.fromkeys(splits))


def row_splits(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["split"] for row in rows}
    return [split for split in SPLIT_ORDER if split in present]


def load_state(record: TaskRecord) -> OrderedDict[str, torch.Tensor]:
    payload = torch.load(record.adapter_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{record.adapter_path}: compact Adapter must be a mapping")
    payload_keys = set(payload)
    expected_keys = set(EXPECTED_SHAPES)
    unexpected_keys = payload_keys - expected_keys
    unexpected_state_keys = sorted(
        key for key in unexpected_keys if not str(key).startswith("_")
    )
    if expected_keys - payload_keys or unexpected_state_keys:
        missing = sorted(set(EXPECTED_SHAPES) - set(payload))
        raise ValueError(
            f"{record.adapter_path}: expected compact 26-tensor state; "
            f"missing={missing}, unexpected={unexpected_state_keys}"
        )
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, shape in EXPECTED_SHAPES.items():
        value = payload[key]
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{record.adapter_path}: {key} is not a floating tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{record.adapter_path}: {key} shape={tuple(value.shape)}, expected={shape}"
            )
        value = value.detach().float().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"{record.adapter_path}: {key} contains NaN or Inf")
        state[key] = value
    return state


def to_device_state(
    state: Mapping[str, torch.Tensor], device: torch.device
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key, value.to(device, non_blocking=True)) for key, value in state.items()
    )


def unit_rows(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=1, keepdim=True).clamp_min(1.0e-8)

