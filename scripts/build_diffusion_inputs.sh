#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var TASK_ROOT
require_var STATS_PATH

cd "${SUBMIT_DIR}/diffusion"
"${PYTHON_BIN}" cache/build_gram_raw_probe_condition.py \
    --task-root "${TASK_ROOT}" \
    --source-file "${SOURCE_FILE:-train.pt}" \
    --indices-output-file "${INDICES_OUTPUT_FILE:-support_indices.pt}" \
    --raw-output-file "${RAW_OUTPUT_FILE:-raw_probe_K128.pt}" \
    --gram-output-file "${GRAM_OUTPUT_FILE:-gram_K128.pt}" \
    --k "${PROBE_COUNT:-128}" \
    --seed "${SEED:-2026}" \
    "$@"

"${PYTHON_BIN}" cache/build_stats.py \
    --task-root "${TASK_ROOT}" \
    --output "${STATS_PATH}" \
    --alignment "${ALIGNMENT:-aligned}" \
    --token-size "${TOKEN_SIZE:-512}"
