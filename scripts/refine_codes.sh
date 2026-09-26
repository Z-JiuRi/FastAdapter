#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var SOURCE_EXP
require_var TARGET_EXP
require_var TRAIN_PATH
require_var VAL_PATH
require_var TEST_PATH
require_var OUTPUT_DIR

"${PYTHON_BIN}" "${SUBMIT_DIR}/adapter/scripts/generate_latent_refined_codes.py" \
    --source_exp "${SOURCE_EXP}" \
    --target_exp "${TARGET_EXP}" \
    --train_csi "${TRAIN_PATH}" \
    --val_csi "${VAL_PATH}" \
    --test_csi "${TEST_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --steps "${REFINE_STEPS:-20}" \
    --lr "${REFINE_LR:-0.01}" \
    --batch_size "${BATCH_SIZE:-256}" \
    --align_ridge "${ALIGN_RIDGE:-1.0}" \
    --init_mode "${INIT_MODE:-reencode}" \
    --loss_target "${LOSS_TARGET:-source_recon}" \
    --gpu "${GPU:-0}" \
    "$@"
