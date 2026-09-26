#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var CONFIG_PATH
require_var TASK_ROOT
require_var STATS_PATH
require_var OUTPUT_DIR

cd "${SUBMIT_DIR}/diffusion"
"${PYTHON_BIN}" main.py \
    --config "${CONFIG_PATH}" \
    --mode train \
    --override "data.task_root=${TASK_ROOT}" \
    --override "cache.stats_path=${STATS_PATH}" \
    --override "exp_dir=${OUTPUT_DIR}" \
    --override data.codeword_file=gram_K128.pt \
    --override data.raw_codeword_file=raw_probe_K128.pt \
    --override condition_encoder.type=gram_raw_probe \
    --override ablation.alignment=aligned \
    --override train.epochs=1000 \
    --override lr_scheduler.max_lr=0.0001 \
    --override lr_scheduler.type=cosine_warmup \
    --override lr_scheduler.warmup_ratio=0.05 \
    --override diffusion.prediction_type=eps \
    --override token_context.condition_injection=cross_attention \
    --override token_context.implementation=transformer \
    "$@"
