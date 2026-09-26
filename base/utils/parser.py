import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--train_path', type=str, metavar='PATH', required=True,
                    help='path to training data')
parser.add_argument('--val_path', type=str, metavar='PATH', required=True,
                    help='path to validation data')
parser.add_argument('--test_path', type=str, metavar='PATH', required=True,
                    help='path to test data')
parser.add_argument('-b', '--batch_size', type=int, required=True, metavar='N',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu_affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Multi-experiment batch arguments (comma-separated lists)
parser.add_argument('--exp_root', default='COST2100/in/base', type=str,
                    help='experiment root used by multi-experiment mode')
parser.add_argument('--torch_num_threads', default=0, type=int,
                    help='intra-op CPU threads per multi-experiment worker; '
                         'use <=0 to keep PyTorch default')
parser.add_argument('--seed_list', default=None, type=str,
                    help='comma-separated seed list for multi-experiment mode')
parser.add_argument('--encoder_list', default=None, type=str,
                    help='comma-separated encoder list for multi-experiment mode')
parser.add_argument('--decoder_list', default=None, type=str,
                    help='comma-separated decoder list for multi-experiment mode')
parser.add_argument('--gpu_list', default=None, type=str,
                    help='comma-separated GPU id list for multi-experiment mode')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--encoder', type=str, default='transnet',
                    choices=['csinet', 'crnet', 'clnet', 'transnet',
                             'resnet', 'attention_cnn', 'mlp_ae'],
                    help='encoder backbone to use')
parser.add_argument('--decoder', type=str, default='transnet',
                    choices=['transnet'],
                    help='decoder backbone to use')
parser.add_argument('--exp_name', metavar='NAME', type=str, default='exp_1',
                    help='experiment name; outputs are saved under ./exps/NAME')
parser.add_argument('--channel', type=int, default=2,
                    help='number of channels in the CSI tensor')
parser.add_argument('--nt', type=int, default=32,
                    help='number of antennas in the CSI tensor')
parser.add_argument('--nc', type=int, default=32,
                    help='number of delay/frequency bins in the CSI tensor')
parser.add_argument('-d', '--d_model', type=int, default=64, metavar='N',
                    help='number of Transformer feature dimension')
parser.add_argument('--dim_feedforward', type=int, default=2048,
                    help='hidden dimension of Transformer feed-forward layers')
parser.add_argument('--hidden', type=int, default=16,
                    help='internal channel count in CNN refinement head (decoder=hybrid)')
parser.add_argument('--num_blocks', type=int, default=2,
                    help='number of ConvResidualBlock in CNN refinement head (decoder=hybrid)')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('--lr_init', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight decay for AdamW')

args = parser.parse_args()

# ----- Multi-experiment mode (list args provided) -----
if args.seed_list is not None:
    _seeds = [int(x.strip()) for x in args.seed_list.split(",")]
    _encoders = [x.strip() for x in args.encoder_list.split(",")]
    _decoders = [x.strip() for x in args.decoder_list.split(",")]
    _gpus = [int(x.strip()) for x in args.gpu_list.split(",")]
    if not (len(_seeds) == len(_encoders) == len(_decoders) == len(_gpus)):
        raise ValueError(
            "seed_list, encoder_list, decoder_list, gpu_list must have "
            f"equal lengths, got {len(_seeds)}, {len(_encoders)}, "
            f"{len(_decoders)}, {len(_gpus)}")
    args.experiments = []
    for s, e, d, g in zip(_seeds, _encoders, _decoders, _gpus):
        args.experiments.append({
            "seed": s, "encoder": e, "decoder": d, "gpu": g,
            "exp_name": f"{args.exp_root}/seed{s}/{e}_{d}"
        })
else:
    args.experiments = None
