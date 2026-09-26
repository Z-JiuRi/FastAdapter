#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for name in \
    SOURCE_TRAIN_CODE SOURCE_VAL_CODE SOURCE_TEST_CODE \
    TARGET_TRAIN_CODE TARGET_VAL_CODE TARGET_TEST_CODE \
    TRAIN_PATH VAL_PATH TEST_PATH DECODER_CHECKPOINT \
    TEACHER_TRAIN_CODE OUTPUT_DIR; do
    require_var "${name}"
done

device_args=()
while IFS= read -r item; do
    [[ -n "${item}" ]] && device_args+=("${item}")
done < <(optional_gpu_args)

model_args=()
if [[ -n "${DECODER_ARGS_JSON:-}" ]]; then
    model_args+=(--decoder_args_json "${DECODER_ARGS_JSON}")
fi
if [[ -n "${ENCODER_ARGS_JSON:-}" ]]; then
    model_args+=(--encoder_args_json "${ENCODER_ARGS_JSON}")
fi

"${PYTHON_BIN}" "${SUBMIT_DIR}/adapter/train_adapter.py" \
    --source_train_code "${SOURCE_TRAIN_CODE}" \
    --source_val_code "${SOURCE_VAL_CODE}" \
    --source_test_code "${SOURCE_TEST_CODE}" \
    --target_train_code "${TARGET_TRAIN_CODE}" \
    --target_val_code "${TARGET_VAL_CODE}" \
    --target_test_code "${TARGET_TEST_CODE}" \
    --train_csi "${TRAIN_PATH}" \
    --val_csi "${VAL_PATH}" \
    --test_csi "${TEST_PATH}" \
    --decoder_checkpoint "${DECODER_CHECKPOINT}" \
    --encoder_checkpoint "${ENCODER_CHECKPOINT:-$DECODER_CHECKPOINT}" \
    --teacher_train_code "${TEACHER_TRAIN_CODE}" \
    --exp_dir "${OUTPUT_DIR}" \
    --mapper_type affine_residual_mlp \
    --hidden_dim 512 \
    --num_blocks 4 \
    --residual_scale 0.4 \
    --gate_mode none \
    --dropout 0 \
    --align_ridge "${ALIGN_RIDGE:-1.0}" \
    --affine_fit_splits train \
    --train_affine \
    --lambda_code "${LAMBDA_CODE:-0.0}" \
    --lambda_teacher_code "${LAMBDA_TEACHER_CODE:-1.0}" \
    --lambda_recon "${LAMBDA_RECON:-1000.0}" \
    --lambda_encoder_consistency "${LAMBDA_ENCODER_CONSISTENCY:-2.0}" \
    --encoder_consistency_target target \
    --epochs "${EPOCHS:-100}" \
    --batch_size "${BATCH_SIZE:-4096}" \
    --workers "${WORKERS:-0}" \
    --lr "${LR:-1e-3}" \
    --weight_decay "${WEIGHT_DECAY:-1e-4}" \
    --scheduler "${SCHEDULER:-cosine}" \
    --ema_decay "${EMA_DECAY:-0.999}" \
    --export_codewords \
    --encoder "${ENCODER:-transnet}" \
    --decoder transnet \
    --cr "${CR:-4}" \
    --channel "${CHANNEL:-2}" \
    --nt "${NT:-32}" \
    --nc "${NC:-32}" \
    --d_model "${D_MODEL:-64}" \
    --dim_feedforward "${DIM_FEEDFORWARD:-2048}" \
    --hidden "${HIDDEN:-16}" \
    --decoder_num_blocks "${DECODER_NUM_BLOCKS:-2}" \
    "${model_args[@]+"${model_args[@]}"}" \
    "${device_args[@]+"${device_args[@]}"}" \
    "$@"
