from .dependencies import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403

def split_paths(args, prefix, split):
    root = getattr(args, f"{prefix}_exp")
    explicit = getattr(args, f"{prefix}_{split}_code")
    if explicit:
        return explicit
    if not root:
        raise ValueError(f"Need --{prefix}_exp or --{prefix}_{split}_code")
    return str(Path(root) / "codewords" / f"{split}_code.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_exp", default=None)
    parser.add_argument("--target_exp", default=None)
    for prefix in ("source", "target"):
        for split in ("train", "val", "test"):
            parser.add_argument(f"--{prefix}_{split}_code", default=None)
    parser.add_argument("--train_csi", required=True)
    parser.add_argument("--val_csi", required=True)
    parser.add_argument("--test_csi", required=True)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--decoder_args_json", default=None)
    parser.add_argument("--encoder_checkpoint", default=None)
    parser.add_argument("--encoder_args_json", default=None)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--mapper_type", default="affine_residual_mlp",
                        choices=[
                            "affine_residual_mlp",
                            "affine_residual_mlp_attention",
                            "affine_iterative_residual",
                            "affine_iterative_residual_unshared",
                            "affine_sens_weighted_residual",
                            "affine_film_residual_mlp",
                            "affine_multiscale_residual_mlp",
                            "affine_lowrank_residual",
                            "affine_bottleneck_residual",
                            "affine_group_gated",
                            "affine_token_mixer",
                            "affine_tiny_transformer",
                            "affine_moe_bottleneck",
                            "affine_coupling_flow",
                            "affine_whole_residual_mlp",
                            "affine_whole_direct_mlp",
                            "legacy_mlp_adapter",
                            "affine_linear",
                            "direct_mlp",
                        ])
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lowrank_rank", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--num_groups", type=int, default=16)
    parser.add_argument("--group_hidden", type=int, default=64)
    parser.add_argument("--gate_hidden", type=int, default=64)
    parser.add_argument("--gate_init", type=float, default=0.5)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--token_hidden", type=int, default=64)
    parser.add_argument("--channel_hidden", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--transformer_ffn_dim", type=int, default=128)
    parser.add_argument("--attention_dim", type=int, default=32)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--attention_scale", type=float, default=0.1)
    parser.add_argument(
        "--attention_input",
        default="value_delta",
        choices=["value", "delta", "value_delta"])
    parser.add_argument(
        "--attention_use_position",
        action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=128)
    parser.add_argument("--whole_mlp_dims", type=parse_int_list, default=None)
    parser.add_argument("--whole_mlp_activation", default="gelu")
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_scale", type=float, default=0.4)
    parser.add_argument("--learnable_residual_gate", action="store_true")
    parser.add_argument("--gate_max", type=float, default=0.5)
    parser.add_argument(
        "--gate_mode",
        default="none",
        choices=[
            "block",
            "none",
            "final_static",
            "final_adaptive",
            "final_unbounded",
        ])
    parser.add_argument("--final_gate_max", type=float, default=1.0)
    parser.add_argument("--final_gate_init", type=float, default=1.0)
    parser.add_argument("--adaptive_gate_hidden", type=int, default=128)
    parser.add_argument("--gate_l1", type=float, default=0.0)
    parser.add_argument("--no_block_norm", action="store_true")
    parser.add_argument("--use_final_norm", action="store_true")
    parser.add_argument("--train_affine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no_affine_alignment", action="store_true",
        help=("For legacy_mlp_adapter, bypass the fitted affine alignment and "
              "apply the residual MLP directly to source codes."))
    parser.add_argument("--align_ridge", type=float, default=1.0)
    parser.add_argument(
        "--affine_fit_splits",
        choices=["train", "train_val_test"],
        default="train",
        help=("Code splits used only to fit the initial affine alignment. "
              "train_val_test is an oracle diagnostic because it consumes "
              "validation and test target codes."))
    parser.add_argument("--lambda_code", type=float, default=0.0)
    parser.add_argument("--lambda_recon", type=float, default=1000.0)
    parser.add_argument("--lambda_feature", type=float, default=0.0)
    parser.add_argument("--lambda_encoder_consistency", type=float, default=2.0)
    parser.add_argument("--lambda_delta_norm", type=float, default=0.0)
    parser.add_argument("--lambda_teacher_code", type=float, default=1.0)
    parser.add_argument("--teacher_train_code", default=None)
    parser.add_argument("--lambda_fisher", type=float, default=0.0)
    parser.add_argument("--fisher_basis_path", default=None)
    parser.add_argument("--fisher_rank", type=int, default=0)
    parser.add_argument("--fisher_weight_power", type=float, default=0.5)
    parser.add_argument("--fisher_weight_max", type=float, default=4.0)
    parser.add_argument(
        "--gradient_diagnostics_every", type=int, default=0,
        help="Log first-batch weighted loss gradient norms/cosines every N epochs")
    parser.add_argument(
        "--train_last_blocks", type=int, default=0,
        help="Freeze the mapper except for its last N residual blocks; 0 trains all")
    parser.add_argument("--code_noise_std", type=float, default=0.0)
    parser.add_argument("--stage1_epochs", type=int, default=0)
    parser.add_argument("--stage1_code_noise_std", type=float, default=0.0)
    parser.add_argument("--stage1_lambda_recon", type=float, default=0.0)
    parser.add_argument(
        "--stage1_lambda_encoder_consistency", type=float, default=0.0)
    parser.add_argument("--stage2_lr", type=float, default=None)
    parser.add_argument("--stage2_affine_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--stage2_affine_freeze_epochs", type=int, default=0)
    parser.add_argument("--stage2_recon_warmup_epochs", type=int, default=0)
    parser.add_argument("--stage2_encoder_delay_epochs", type=int, default=0)
    parser.add_argument("--stage2_encoder_warmup_epochs", type=int, default=0)
    parser.add_argument("--stage2_noise_decay_epochs", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--ema_start_epoch", type=int, default=1)
    parser.add_argument("--ema_update_every", type=int, default=1)
    parser.add_argument("--init_mapper_checkpoint", default=None)
    parser.add_argument("--init_mapper_use_ema", action="store_true")
    parser.add_argument(
        "--encoder_consistency_target",
        default="target",
        choices=["mapped", "target"])
    parser.add_argument("--code_loss_type", default="mse",
                        choices=[
                            "mse",
                            "clipped_std_mse",
                            "clipped_var_mse",
                            "clipped_power_mse",
                            "clipped_residual_std_mse",
                            "decoder_sensitivity_mse",
                            "decoder_jac_residual_mse",
                        ])
    parser.add_argument(
        "--sensitivity_source",
        default="jacobian",
        choices=["jacobian", "fc_decoder"],
        help="How to build decoder sensitivity weights for "
             "decoder_sensitivity_mse / decoder_jac_residual_mse")
    parser.add_argument(
        "--sensitivity_power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized sensitivity before clipping")
    parser.add_argument(
        "--sensitivity_hutchinson",
        type=int,
        default=8,
        help="Hutchinson probes for jacobian sensitivity")
    parser.add_argument(
        "--sensitivity_probe_samples",
        type=int,
        default=2048,
        help="Number of target codes used to estimate jacobian sensitivity")
    parser.add_argument("--std_weight_min", type=float, default=0.25)
    parser.add_argument("--std_weight_max", type=float, default=4.0)
    parser.add_argument("--std_weight_eps", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["const", "cosine"], default="cosine")
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--export_codewords", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--decoder", default="transnet")
    parser.add_argument("--encoder", default="transnet")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--decoder_num_blocks", type=int, default=2)
    return parser.parse_args()
