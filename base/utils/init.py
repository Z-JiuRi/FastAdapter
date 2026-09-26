import os
import random
import thop
import torch

from models import universal_csi
from utils import logger, line_seg, log_parameter_table

__all__ = ["seed_everything", "init_device", "init_model", "show_parameter"]


def seed_everything(seed):
    logger.info(f"Random seed set to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_parameter(model):
    log_parameter_table(model, logger)


def _load_clean_state_dict(checkpoint_path):
    assert os.path.isfile(checkpoint_path)
    state_dict = torch.load(checkpoint_path,
                            weights_only=True,
                            map_location=torch.device('cpu'))['state_dict']
    # Strip thop profiling keys (total_ops, total_params) that were saved
    # during training but don't exist in the freshly-created model.
    for key in list(state_dict.keys()):
        if key.endswith('total_ops') or key.endswith('total_params'):
            del state_dict[key]
    return state_dict


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        seed_everything(seed)

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        logger.info("Running on GPU %d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    model = universal_csi(encoder_name=args.encoder,
                          decoder_name=args.decoder,
                          reduction=args.cr,
                          d_model=args.d_model,
                          channel=args.channel,
                          nt=args.nt,
                          nc=args.nc,
                          dim_feedforward=args.dim_feedforward,
                          hidden=args.hidden,
                          num_blocks=args.num_blocks)

    if args.pretrained is not None:
        state_dict = _load_clean_state_dict(args.pretrained)
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    H_a = torch.randn([1, args.channel, args.nt, args.nc])
    try:
        flops, params = thop.profile(model, inputs=(H_a,), verbose=False)
        flops, params = thop.clever_format([flops, params], "%.4e")
    except Exception as exc:
        flops, params = "unavailable", "unavailable"
        logger.warning(f"=> Model profiling skipped: {exc}")

    # Model info logging
    model_name = "UniversalCSI"
    logger.info(f'=> Model Name: {model_name} '
                f'[pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}; '
                f'encoder={args.encoder}; '
                f'decoder={args.decoder}; '
                f'input shape=({args.channel}, {args.nt}, {args.nc}); '
                f'input dim={args.channel * args.nt * args.nc}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'\n{line_seg}\n{model}\n{line_seg}\n')

    show_parameter(model)

    return model
