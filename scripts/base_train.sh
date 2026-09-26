#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var TRAIN_PATH
require_var VAL_PATH
require_var TEST_PATH

device_args=()
while IFS= read -r item; do
    [[ -n "${item}" ]] && device_args+=("${item}")
done < <(optional_gpu_args)

cd "${SUBMIT_DIR}/base"
"${PYTHON_BIN}" main.py \
    --exp_name "${EXP_NAME:-base/transnet_seed42}" \
    --train_path "${TRAIN_PATH}" \
    --val_path "${VAL_PATH}" \
    --test_path "${TEST_PATH}" \
    --epochs "${EPOCHS:-400}" \
    --batch_size "${BATCH_SIZE:-200}" \
    --workers "${WORKERS:-0}" \
    --cr "${CR:-4}" \
    --encoder "${ENCODER:-transnet}" \
    --decoder transnet \
    --channel "${CHANNEL:-2}" \
    --nt "${NT:-32}" \
    --nc "${NC:-32}" \
    --d_model "${D_MODEL:-64}" \
    --dim_feedforward "${DIM_FEEDFORWARD:-2048}" \
    --scheduler "${SCHEDULER:-cosine}" \
    --lr_init "${LR:-2e-4}" \
    --weight_decay "${WEIGHT_DECAY:-1e-3}" \
    --seed "${SEED:-42}" \
    "${device_args[@]+"${device_args[@]}"}" \
    "$@"
