#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var CONFIG_PATH
require_var TASK_ROOT
require_var STATS_PATH
require_var OUTPUT_DIR

cd "${SUBMIT_DIR}/diffusion"
"${PYTHON_BIN}" alignment/train.py \
    --config "${CONFIG_PATH}" \
    --override "data.task_root=${TASK_ROOT}" \
    --override "cache.stats_path=${STATS_PATH}" \
    --override "alignment.exp_dir=${OUTPUT_DIR}" \
    --override data.codeword_file=gram_K128.pt \
    --override data.raw_codeword_file=raw_probe_K128.pt \
    --override condition_encoder.type=gram_raw_probe \
    --override ablation.alignment=aligned \
    --override token_context.condition_injection=cross_attention \
    --override token_context.implementation=transformer \
    --override alignment.level=task \
    "$@"
