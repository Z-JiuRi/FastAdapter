#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var TRAIN_PATH
require_var VAL_PATH
require_var TEST_PATH
require_var PRETRAINED

device_args=()
while IFS= read -r item; do
    [[ -n "${item}" ]] && device_args+=("${item}")
done < <(optional_gpu_args)

cd "${SUBMIT_DIR}/base"
"${PYTHON_BIN}" main.py \
    --evaluate \
    --pretrained "${PRETRAINED}" \
    --exp_name "${EXP_NAME:-evaluation/base_model}" \
    --train_path "${TRAIN_PATH}" \
    --val_path "${VAL_PATH}" \
    --test_path "${TEST_PATH}" \
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
    "${device_args[@]+"${device_args[@]}"}" \
    "$@"
