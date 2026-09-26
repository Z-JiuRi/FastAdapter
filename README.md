# FastAdapter CSI Code Release

This directory contains the source code for the FastAdapter pipeline. The
paper method is the default in the entry points below. Local
configuration files, datasets, checkpoints, experiment outputs, logs, plots,
and machine-specific paths are intentionally excluded.
## Repository layout

```text
codes/
├── base/       # Plain CSI encoder -> TransNet decoder training
├── adapter/    # Target refinement, Adapter training, and ParamRealign
├── diffusion/  # Gram--Raw conditioning and Adapter diffusion generation
├── scripts/    # Portable shell entry points for all pipeline stages
└── README.md
```

The three stages are deliberately separated:

```text
CSI samples
  -> base: train encoder/decoder pairs and export codewords
  -> adapter: train codeword-space Adapters for heterogeneous encoders
  -> diffusion: align/tokenize Adapter parameters and train conditional diffusion
```

The Adapter stage is internally structured as follows:

```text
adapter/
├── models/
│   ├── blocks.py          # Shared affine, normalization, residual, and gate blocks
│   ├── paper.py           # AffineResidualMLPMapper used by the paper path
│   ├── attention.py       # Existing attention-related mapper variants
│   ├── structured.py      # Existing structured mapper variants
│   ├── iterative.py       # Existing iterative and whole-code variants
│   └── factory.py         # Mapper name-to-class construction
├── training/
│   ├── data.py            # Codeword/CSI datasets and affine fitting
│   ├── model_io.py        # Frozen encoder/decoder and checkpoint loading
│   ├── optimization.py    # Optimizer, scheduler, stage schedule, and EMA
│   ├── losses.py          # Code losses, sensitivity, and diagnostics
│   ├── engine.py          # Epoch training, evaluation, metrics, and export
│   ├── cli.py             # Command-line arguments
│   └── pipeline.py        # Complete Adapter orchestration
├── alignment/
│   ├── io.py              # Compact state and task-tree loading
│   ├── features.py        # Weight and activation signatures
│   ├── matching.py        # Assignment, permutation, and equivalence checks
│   ├── reporting.py       # Tables, statistics, and optional plots
│   ├── cli.py             # ParamRealign arguments and validation
│   └── pipeline.py        # Complete reference/alignment orchestration
├── scripts/                    # Target-code refinement entry point
├── train_adapter.py            # Backward-compatible training entry point
└── param_realign.py             # Backward-compatible alignment entry point
```

All commands below use placeholders such as `PATH_TO_TRAIN_PT`. Replace them
with paths for the local machine. No absolute user path is required by the
packaged code.

## Method scope

The release follows the FastAdapter pipeline represented by the following
source components:

- Seven heterogeneous CSI encoders: `csinet`, `crnet`, `clnet`, `transnet`,
  `resnet`, `attention_cnn`, and `mlp_ae`.
- One fixed `transnet` decoder interface for base reconstruction and downstream
  Adapter evaluation.
- A code-space Adapter initialized by ridge affine alignment and followed by
  four residual LayerNorm--MLP blocks.
- A compact 26-tensor Adapter state: affine weight/bias plus six tensors for
  each of four residual blocks.
- ParamRealign hidden-neuron permutation alignment before parameter modeling.
- Per-task Gram--Raw conditions, parameter tokenization, task-level contrastive
  context alignment, epsilon-prediction diffusion, classifier-free guidance,
  and DDIM sampling.

The original Adapter source contains reusable training diagnostics and loss
switches. For the paper path, use the explicit arguments shown below:
`affine_residual_mlp`, four blocks, hidden size 512, residual scale 0.4,
no final gate, and epsilon prediction in diffusion.

## Requirements

Use Python 3.10 or newer. The core packages are:

```text
torch
tensorboard
numpy
scipy
matplotlib
thop
omegaconf
```

`mamba_ssm` is optional and is required only when an existing diffusion
configuration selects the Mamba context implementation. The Transformer and
GRU context implementations do not require it.

## Shell scripts

The `scripts/` directory provides portable wrappers for the complete pipeline.
They contain no dataset paths, checkpoints, usernames, or local configuration
defaults. Supply paths through environment variables. Any arguments appended
to a script command are forwarded to its Python entry point, except for
`build_diffusion_inputs.sh`, which runs two fixed preparation commands.

Available scripts:

- `base_train.sh`: train one of the seven base encoders with the TransNet
  decoder and export all codeword splits.
- `base_evaluate.sh`: evaluate a complete base checkpoint and export its
  codewords.
- `refine_codes.sh`: generate frozen-decoder refined code targets.
- `train_adapter.sh`: run the paper Adapter path with the fixed 26-tensor
  architecture arguments.
- `strip_adapters.sh`: validate or compact full EMA Adapter checkpoints.
- `align_adapters.sh`: run ParamRealign and optionally save aligned states.
- `build_diffusion_inputs.sh`: build per-task Gram--Raw tensors and
  training-only Adapter statistics.
- `train_context_alignment.sh`: train the contrastive context alignment model.
- `train_diffusion.sh`: run single-process diffusion training.
- `train_diffusion_distributed.sh`: run multi-GPU diffusion training through
  `torchrun`.
- `infer_diffusion.sh`: generate compact Adapter parameters.
- `evaluate_diffusion.sh`: generate and evaluate Adapters with the frozen base
  decoder.
- `common.sh`: shared path resolution and required-variable validation.

All scripts use `python3` by default. Set `PYTHON_BIN` to select another Python
executable. Base and Adapter scripts accept `GPU=<id>` or `CPU=1`.

Example base run:

```bash
TRAIN_PATH=PATH_TO_TRAIN_PT \
VAL_PATH=PATH_TO_VAL_PT \
TEST_PATH=PATH_TO_TEST_PT \
ENCODER=transnet \
EXP_NAME=base/transnet_seed42 \
GPU=0 \
bash scripts/base_train.sh
```

Example Adapter run:

```bash
SOURCE_TRAIN_CODE=PATH_TO_SOURCE_TRAIN_CODE \
SOURCE_VAL_CODE=PATH_TO_SOURCE_VAL_CODE \
SOURCE_TEST_CODE=PATH_TO_SOURCE_TEST_CODE \
TARGET_TRAIN_CODE=PATH_TO_TARGET_TRAIN_CODE \
TARGET_VAL_CODE=PATH_TO_TARGET_VAL_CODE \
TARGET_TEST_CODE=PATH_TO_TARGET_TEST_CODE \
TEACHER_TRAIN_CODE=PATH_TO_REFINED_TRAIN_CODE \
TRAIN_PATH=PATH_TO_TRAIN_PT \
VAL_PATH=PATH_TO_VAL_PT \
TEST_PATH=PATH_TO_TEST_PT \
DECODER_CHECKPOINT=PATH_TO_TARGET_BASE_CHECKPOINT \
ENCODER_CHECKPOINT=PATH_TO_TARGET_BASE_CHECKPOINT \
OUTPUT_DIR=PATH_TO_ADAPTER_OUTPUT \
GPU=0 \
bash scripts/train_adapter.sh
```

`strip_adapters.sh` defaults to the safe dry-run mode. Set `DRY_RUN=0` only
after reviewing the validation output:

```bash
TASK_ROOT=PATH_TO_TASK_ROOT DRY_RUN=1 bash scripts/strip_adapters.sh
TASK_ROOT=PATH_TO_TASK_ROOT DRY_RUN=0 bash scripts/strip_adapters.sh
```

Example diffusion preparation and training:

```bash
TASK_ROOT=PATH_TO_TASK_ROOT \
STATS_PATH=PATH_TO_STATS_PT \
bash scripts/build_diffusion_inputs.sh

CONFIG_PATH=PATH_TO_EXISTING_DIFFUSION_CONFIG \
TASK_ROOT=PATH_TO_TASK_ROOT \
STATS_PATH=PATH_TO_STATS_PT \
OUTPUT_DIR=PATH_TO_DIFFUSION_OUTPUT \
bash scripts/train_diffusion.sh
```

The CSI tensors are expected to be floating-point PyTorch files. The default
single-sample shape is `(2, 32, 32)`. A two-dimensional tensor is reshaped to
that sample shape by the base data loader. With compression ratio denominator
`cr`, the code dimension is:

```text
code_dim = channel * nt * nc // cr
```

For the default dimensions and `cr=4`, `code_dim=512`.

## 1. Base encoder/decoder training

The `base/` folder is the plain end-to-end path only:

```text
CSI -> selected encoder -> codeword -> TransNet decoder -> reconstructed CSI
```

It does not include an Adapter module, partial encoder/decoder loading,
teacher-code distillation, or code regularization. Training uses reconstruction
MSE. Evaluation reports aggregate NMSE:

```text
NMSE = 10 * log10(sum(error^2) / sum(target^2))
```

Run from `codes/base`:

```bash
cd codes/base

python3 main.py \
  --exp_name base/transnet_seed42 \
  --train_path PATH_TO_TRAIN_PT \
  --val_path PATH_TO_VAL_PT \
  --test_path PATH_TO_TEST_PT \
  --epochs 400 \
  --batch_size 200 \
  --workers 0 \
  --cr 4 \
  --encoder transnet \
  --decoder transnet \
  --channel 2 \
  --nt 32 \
  --nc 32 \
  --d_model 64 \
  --dim_feedforward 2048 \
  --scheduler cosine \
  --lr_init 2e-4 \
  --seed 42 \
  --gpu 0
```

Replace `transnet` with any of the seven encoder names listed above to train
the heterogeneous base models against the same decoder architecture. Each run
writes checkpoints and exported training codewords below its runtime
`exps/<exp_name>/` directory. These generated files are not part of this
source-only release.

Evaluate a complete base checkpoint:

```bash
python3 main.py \
  --evaluate \
  --pretrained PATH_TO_BASE_CHECKPOINT \
  --exp_name evaluation/base_model \
  --train_path PATH_TO_TRAIN_PT \
  --val_path PATH_TO_VAL_PT \
  --test_path PATH_TO_TEST_PT \
  --batch_size 200 \
  --workers 0 \
  --cr 4 \
  --encoder transnet \
  --decoder transnet
```

Important files:

- `base/main.py`: training, evaluation, checkpointing, and codeword export.
- `base/models/UniversalCSI.py`: encoder/decoder factory and fixed interfaces.
- `base/models/encoders/`: the seven selected original encoders.
- `base/models/decoders/transnet.py`: the fixed decoder implementation.
- `base/dataloader/dataloader.py`: `.pt` tensor loading and reshaping.
- `base/utils/solver.py`: plain reconstruction training and NMSE evaluation.

## 2. Target-code refinement

The Adapter stage consumes codewords from a source encoder and a target model.
The optional refinement script preserves the original offline optimization
flow: initialize a target-space code, optimize it through the frozen target
decoder, and save refined code targets for Adapter supervision.

Run from `codes`:

```bash
cd codes

python3 adapter/scripts/generate_latent_refined_codes.py \
  --source_exp PATH_TO_SOURCE_BASE_EXPERIMENT \
  --target_exp PATH_TO_TARGET_BASE_EXPERIMENT \
  --train_csi PATH_TO_TRAIN_PT \
  --val_csi PATH_TO_VAL_PT \
  --test_csi PATH_TO_TEST_PT \
  --output_dir PATH_TO_REFINED_CODE_OUTPUT \
  --steps 20 \
  --lr 0.01 \
  --batch_size 256 \
  --align_ridge 1.0 \
  --init_mode reencode \
  --loss_target source_recon \
  --gpu 0
```

The source and target experiment directories must contain the base runtime
layout used by `base/main.py`, including `args.json`,
`checkpoints/best_nmse.pth`, and split codewords. The refinement output contains
`train_refined_code.pt`, `val_refined_code.pt`, and
`test_refined_code.pt` when all splits are selected.

## 3. Adapter training

`adapter/train_adapter.py` is the stable standalone entry point and delegates
to the structured `adapter/training/` package. The package loads precomputed
source and target codewords, fits the affine start, freezes the selected
decoder, trains the mapper, evaluates CSI reconstruction, and can export mapped
codewords.

For the 26-tensor paper Adapter, run from `codes` with the method-defining
arguments shown explicitly:

```bash
python3 adapter/train_adapter.py \
  --source_train_code PATH_TO_SOURCE_TRAIN_CODE \
  --source_val_code PATH_TO_SOURCE_VAL_CODE \
  --source_test_code PATH_TO_SOURCE_TEST_CODE \
  --target_train_code PATH_TO_TARGET_TRAIN_CODE \
  --target_val_code PATH_TO_TARGET_VAL_CODE \
  --target_test_code PATH_TO_TARGET_TEST_CODE \
  --teacher_train_code PATH_TO_REFINED_TRAIN_CODE \
  --train_csi PATH_TO_TRAIN_PT \
  --val_csi PATH_TO_VAL_PT \
  --test_csi PATH_TO_TEST_PT \
  --decoder_checkpoint PATH_TO_TARGET_BASE_CHECKPOINT \
  --encoder_checkpoint PATH_TO_TARGET_BASE_CHECKPOINT \
  --exp_dir PATH_TO_ADAPTER_OUTPUT \
  --mapper_type affine_residual_mlp \
  --hidden_dim 512 \
  --num_blocks 4 \
  --residual_scale 0.4 \
  --gate_mode none \
  --dropout 0 \
  --align_ridge 1.0 \
  --affine_fit_splits train \
  --train_affine \
  --lambda_code 0.0 \
  --lambda_teacher_code 1.0 \
  --lambda_recon 1000.0 \
  --lambda_encoder_consistency 2.0 \
  --encoder_consistency_target target \
  --epochs 100 \
  --batch_size 4096 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --ema_decay 0.999 \
  --export_codewords \
  --encoder transnet \
  --decoder transnet \
  --cr 4 \
  --gpu 0
```

The refined training tensor supplies the codeword-alignment target. The
reference encoder checkpoint supplies the re-encoding loss. Keep
`--affine_fit_splits train`; the `train_val_test` option is an oracle diagnostic.

Important files:

- `adapter/models/paper.py`: the original `AffineResidualMLPMapper`; the paper
  path selects it through `affine_residual_mlp`.
- `adapter/models/factory.py`: preserves mapper construction compatibility.
- `adapter/training/pipeline.py`: complete fitting, optimization, evaluation,
  EMA, checkpoint, and export orchestration.
- `adapter/training/engine.py`: train/evaluate epochs and codeword export.
- `adapter/train_adapter.py`: compatibility entry point; existing commands are
  unchanged.
- `adapter/scripts/generate_latent_refined_codes.py`: frozen-decoder target-code
  refinement.
- `adapter/functional.py`: differentiable functional execution of compact
  Adapter states.

Before ParamRealign or diffusion tokenization, convert the selected EMA
checkpoint to the compact 26-tensor state. The existing conversion utility
performs an atomic in-place replacement, so first place the selected best
checkpoint at each task's `adapter.pth` path and retain a backup outside the
task tree:

```bash
python3 adapter/strip_adapter_checkpoints.py \
  --data-root PATH_TO_TASK_ROOT \
  --adapter-name adapter.pth \
  --dry-run

python3 adapter/strip_adapter_checkpoints.py \
  --data-root PATH_TO_TASK_ROOT \
  --adapter-name adapter.pth \
  --workers 4
```

The first command validates every checkpoint without writing. The second
extracts `ema.shadow`, removes the runtime-only `_delta_ratio` buffer, validates
all 26 shapes and hashes, writes `adapter_meta.json`, and atomically replaces
the full checkpoint with the compact state.

## 4. ParamRealign

Adapter parameter tensors contain hidden-neuron permutation symmetries.
`adapter/param_realign.py` delegates to the structured `adapter/alignment/`
package while preserving the existing command. It builds a reference only from
training tasks, aligns all splits to the frozen reference, and checks
functional equivalence before optionally saving aligned states.

The task tree is expected to follow this shape:

```text
TASK_ROOT/
├── train/<encoder>/<seed>/adapter.pth
├── val/<encoder>/<seed>/adapter.pth
└── test/<encoder>/<seed>/adapter.pth
```

Inspect the exact options first:

```bash
python3 adapter/param_realign.py --help
```

Then run the paper alignment path, supplying the local task root and an output
directory. Use `--save-aligned` to write `aligned_adapter.pth` beside each raw
Adapter only after the functional-equivalence checks pass:

```bash
python3 adapter/param_realign.py \
  --data-root PATH_TO_TASK_ROOT \
  --output-dir PATH_TO_ALIGNMENT_REPORTS \
  --adapter-name adapter.pth \
  --aligned-name aligned_adapter.pth \
  --cost-type activation \
  --iterations 10 \
  --save-aligned
```

This command writes runtime reports and aligned weights; none are bundled in
`codes`.

## 5. Build Gram--Raw diffusion inputs

The diffusion stage expects each task to contain a compact Adapter state and a
per-task sampled codeword matrix. First create the matched raw-probe and Gram
condition tensors:

```bash
cd codes/diffusion

python3 cache/build_gram_raw_probe_condition.py \
  --task-root PATH_TO_TASK_ROOT \
  --source-file train.pt \
  --indices-output-file support_indices.pt \
  --raw-output-file raw_probe_K128.pt \
  --gram-output-file gram_K128.pt \
  --k 128 \
  --seed 2026
```

Each task receives its own deterministic random sample of 128 distinct rows
from that encoder's training-codeword matrix. The task-local index file makes
reruns reproducible; Gram and Raw share the sampled rows within each task.
Use `--overwrite` to rebuild conditions made by the earlier shared-index procedure.
After changing support sampling, rebuild the condition tensors for every split
and retrain contrastive alignment and diffusion; old checkpoints correspond to
different inputs.

Next build normalization statistics from training Adapters only:

```bash
python3 cache/build_stats.py \
  --task-root PATH_TO_TASK_ROOT \
  --output PATH_TO_STATS_PT \
  --alignment aligned \
  --token-size 512
```

`build_stats.py` records the 26-tensor manifest, token masks, structure IDs,
per-parameter normalization statistics, and a fingerprint used by training and
inference validation.

## 6. Contrastive context alignment

The `diffusion/alignment/` package uses task-level symmetric InfoNCE, positive
cosine alignment, and position-residual cosine alignment. It trains the codeword-condition
context generator before diffusion, so condition representations remain tied
to the matching Adapter parameters.

No YAML configuration is included in this source-only release. Use an existing
local configuration file and override paths from the command line:

```bash
python3 alignment/train.py \
  --config PATH_TO_EXISTING_ALIGNMENT_CONFIG \
  --override data.task_root=PATH_TO_TASK_ROOT \
  --override cache.stats_path=PATH_TO_STATS_PT \
  --override alignment.exp_dir=PATH_TO_ALIGNMENT_OUTPUT \
  --override alignment.level=task
```

The configuration must explicitly set `alignment.task_temperature`,
`alignment.task_cosine_weight`, and `alignment.position_residual_weight`.
Both auxiliary weights must be positive for the paper objective.
The configuration consumed by this entry point must also define the data fields,
Adapter token structure, condition encoder, token-context generator, alignment
model, optimizer, scheduler, and loss weights used by the original code.
The cross-attention forward path now follows the equation in the paper, so
previous context-generator and diffusion checkpoints should be retrained.
The paper shell entry points select a bidirectional Transformer sequence model;
the optional Mamba and prefix paths are separate architecture experiments.

## 7. Diffusion training

`diffusion/main.py` loads an external OmegaConf YAML file. Configuration files
were excluded by design, but every setting can be overridden with repeated
`--override KEY=VALUE` arguments.

The paper path requires at least these semantic settings in the external
configuration:

- aligned 26-tensor Adapter states and the training-only statistics file;
- Gram condition plus its matched raw support-codeword condition;
- the pretrained contrastive token-context generator checkpoint;
- `diffusion.prediction_type=eps`;
- the chosen beta schedule, diffusion step count, and Min-SNR setting;
- classifier-free condition dropout during training;
- tensor-balanced parameter-token loss;
- frozen target TransNet decoder information for functional evaluation.

Run training:

```bash
python3 main.py \
  --config PATH_TO_EXISTING_DIFFUSION_CONFIG \
  --mode train \
  --override data.task_root=PATH_TO_TASK_ROOT \
  --override cache.stats_path=PATH_TO_STATS_PT \
  --override exp_dir=PATH_TO_DIFFUSION_OUTPUT \
  --override diffusion.prediction_type=eps
```

For distributed training, launch the same entry point through the local
PyTorch distributed runner and preserve the same overrides.

Important files:

- `diffusion/data/adapter_tokenizer.py`: compact-state validation,
  normalization, tokenization, and differentiable detokenization.
- `diffusion/models/condition_context.py`: Gram--Raw condition encoding,
  structural embeddings, and token-context generation.
- `diffusion/models/denoiser.py`: parameter-token denoisers.
- `diffusion/models/ddpm.py`: forward diffusion, epsilon loss, CFG, DDPM, and
  DDIM sampling.
- `diffusion/core/trainer.py`: optimization, checkpointing, validation, and
  functional reconstruction losses.
- `diffusion/core/inferencer.py`: conditional sampling, Adapter reconstruction,
  and target-decoder evaluation.

## 8. Inference and evaluation

Use the same external diffusion configuration and override the checkpoint,
condition, and output paths:

```bash
python3 main.py \
  --config PATH_TO_EXISTING_DIFFUSION_CONFIG \
  --mode infer \
  --override cache.stats_path=PATH_TO_STATS_PT \
  --override inference.checkpoint_path=PATH_TO_DIFFUSION_CHECKPOINT \
  --override inference.cond_path=PATH_TO_CONDITION_PT \
  --override inference.output_dir=PATH_TO_GENERATED_ADAPTERS \
  --override diffusion.prediction_type=eps
```

Use `--mode eval` to additionally run the configured frozen decoder evaluation:

```bash
python3 main.py \
  --config PATH_TO_EXISTING_DIFFUSION_CONFIG \
  --mode eval \
  --override cache.stats_path=PATH_TO_STATS_PT \
  --override inference.checkpoint_path=PATH_TO_DIFFUSION_CHECKPOINT \
  --override eval.decoder_args=PATH_TO_BASE_ARGS_JSON \
  --override eval.decoder_checkpoint=PATH_TO_BASE_CHECKPOINT \
  --override eval.csi_path=PATH_TO_TEST_PT \
  --override data.query_codeword_file=test.pt
```

The test codewords in each task directory must correspond, in row order, to
the independent test CSI tensor. The `evaluate_diffusion.sh` wrapper requires
`TEST_PATH` and defaults its query codeword filename to `test.pt`.

Sampling supports classifier-free guidance and DDIM through the corresponding
existing configuration fields under `inference` and `diffusion`. Generated
parameter tokens are denormalized and reconstructed into the original compact
Adapter state dictionary before functional evaluation.

## Reproducibility and safety notes

- Fit affine alignment, ParamRealign references, and parameter statistics on
  training tasks only.
- Keep CSI dimensions, compression ratio, encoder name, decoder name, and
  checkpoint architecture consistent across all stages.
- Sample support indices independently for every task while preserving its
  within-task Raw/Gram row correspondence.
- Do not place datasets, checkpoints, generated Adapters, logs, TensorBoard
  events, or local configuration files inside this source package.
- Run `python3 -m compileall -q codes` from the project root after source
  changes to verify Python syntax.
- Run `python3 -m unittest discover -s codes/tests -v` for the paper-method
  regression checks (requires PyTorch and OmegaConf).
