from .dependencies import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403

def load_main_models_package():
    package_name = "adapter_main_models"
    spec = importlib.util.spec_from_file_location(
        package_name,
        BASE_ROOT / "models" / "__init__.py",
        submodule_search_locations=[str(BASE_ROOT / "models")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def clean_state_dict(checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    for key in list(state_dict.keys()):
        if key.endswith("total_ops") or key.endswith("total_params"):
            del state_dict[key]
    return state_dict


def load_decoder(args, device):
    cfg = {}
    if args.decoder_args_json:
        cfg = json.loads(Path(args.decoder_args_json).read_text())
    main_models = load_main_models_package()
    decoder_name = cfg.get("decoder", args.decoder)
    cr = cfg.get("cr", args.cr)
    d_model = cfg.get("d_model", args.d_model)
    channel = cfg.get("channel", args.channel)
    nt = cfg.get("nt", args.nt)
    nc = cfg.get("nc", args.nc)
    dim_feedforward = cfg.get("dim_feedforward", args.dim_feedforward)
    hidden = cfg.get("hidden", args.hidden)
    num_blocks = cfg.get("num_blocks", args.decoder_num_blocks)
    model = main_models.universal_csi(
        encoder_name="transnet",
        decoder_name=decoder_name,
        reduction=cr,
        d_model=d_model,
        channel=channel,
        nt=nt,
        nc=nc,
        dim_feedforward=dim_feedforward,
        hidden=hidden,
        num_blocks=num_blocks)
    state_dict = clean_state_dict(args.decoder_checkpoint)
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        decoder_state = state_dict
    missing, unexpected = model.decoder.load_state_dict(decoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"decoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    decoder = model.decoder.to(device).eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder, {"channel": channel, "nt": nt, "nc": nc}


def load_target_encoder(args, device):
    cfg = {}
    args_json = args.encoder_args_json or args.decoder_args_json
    if args_json:
        cfg = json.loads(Path(args_json).read_text())
    main_models = load_main_models_package()
    # E_0 is the fixed reference encoder, independent of the source encoder.
    encoder_name = cfg.get("encoder", "transnet")
    decoder_name = cfg.get("decoder", args.decoder)
    cr = cfg.get("cr", args.cr)
    d_model = cfg.get("d_model", args.d_model)
    channel = cfg.get("channel", args.channel)
    nt = cfg.get("nt", args.nt)
    nc = cfg.get("nc", args.nc)
    dim_feedforward = cfg.get("dim_feedforward", args.dim_feedforward)
    hidden = cfg.get("hidden", args.hidden)
    num_blocks = cfg.get("num_blocks", args.decoder_num_blocks)
    model = main_models.universal_csi(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        reduction=cr,
        d_model=d_model,
        channel=channel,
        nt=nt,
        nc=nc,
        dim_feedforward=dim_feedforward,
        hidden=hidden,
        num_blocks=num_blocks)
    checkpoint = args.encoder_checkpoint or args.decoder_checkpoint
    state_dict = clean_state_dict(checkpoint)
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise RuntimeError(f"encoder checkpoint has no encoder.* weights: {checkpoint}")
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    encoder = model.encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder
