from .dependencies import *  # noqa: F401,F403

def parse_int_list(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item) for item in text.split(",") if item.strip()]


class CodeCsiDataset(Dataset):
    def __init__(self, source_code, target_code, csi):
        if source_code.ndim != 2 or target_code.ndim != 2:
            raise ValueError("source_code and target_code must be 2D tensors")
        if source_code.shape != target_code.shape:
            raise ValueError(
                f"source/target code shape mismatch: "
                f"{tuple(source_code.shape)} vs {tuple(target_code.shape)}")
        if csi.ndim != 4:
            raise ValueError(f"csi must be 4D, got {tuple(csi.shape)}")
        n = min(source_code.size(0), target_code.size(0), csi.size(0))
        self.source = source_code[:n].contiguous()
        self.target = target_code[:n].contiguous()
        self.csi = csi[:n].contiguous()
        self.indices = torch.arange(n, dtype=torch.long)

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx], self.csi[idx], self.indices[idx]


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_code(path, max_samples=0):
    code = torch.load(path, weights_only=True, map_location="cpu").float()
    if code.ndim != 2:
        raise ValueError(f"{path}: expected 2D code tensor, got {tuple(code.shape)}")
    if max_samples and code.size(0) > max_samples:
        code = code[:max_samples].contiguous()
    return code


def load_optional_code(path, max_samples=0):
    if not path:
        return None
    return load_code(path, max_samples)


def load_csi(path, channel=2, nt=32, nc=32, max_samples=0):
    data = torch.load(path, weights_only=True, map_location="cpu").float()
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(
            f"{path}: expected (N,{channel},{nt},{nc}), got {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data


def fit_affine(source, target, ridge=1.0):
    dim = source.size(1)
    src = source.to(torch.float64)
    tgt = target.to(torch.float64)
    ones = torch.ones(src.size(0), 1, dtype=src.dtype)
    aug = torch.cat([src, ones], dim=1)
    reg = ridge * torch.eye(dim + 1, dtype=src.dtype)
    reg[-1, -1] = 0.0
    solution = torch.linalg.solve(aug.t().matmul(aug) + reg, aug.t().matmul(tgt))
    return solution[:-1].float().contiguous(), solution[-1].float().contiguous()

