#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var CONFIG_PATH
require_var STATS_PATH
require_var DIFFUSION_CHECKPOINT
require_var CONDITION_PATH
require_var OUTPUT_DIR

cd "${SUBMIT_DIR}/diffusion"
"${PYTHON_BIN}" main.py \
    --config "${CONFIG_PATH}" \
    --mode infer \
    --override "cache.stats_path=${STATS_PATH}" \
    --override "inference.checkpoint_path=${DIFFUSION_CHECKPOINT}" \
    --override "inference.cond_path=${CONDITION_PATH}" \
    --override "inference.output_dir=${OUTPUT_DIR}" \
    --override data.codeword_file=gram_K128.pt \
    --override data.raw_codeword_file=raw_probe_K128.pt \
    --override condition_encoder.type=gram_raw_probe \
    --override ablation.alignment=aligned \
    --override diffusion.prediction_type=eps \
    --override token_context.condition_injection=cross_attention \
    --override token_context.implementation=transformer \
    --override inference.use_ddim=true \
    --override inference.ddim_steps=50 \
    --override inference.eta=0.0 \
    "$@"
