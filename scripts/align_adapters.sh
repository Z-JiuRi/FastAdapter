#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var TASK_ROOT
require_var OUTPUT_DIR

save_args=()
if [[ "${SAVE_ALIGNED:-1}" == "1" ]]; then
    save_args+=(--save-aligned)
fi

"${PYTHON_BIN}" "${SUBMIT_DIR}/adapter/param_realign.py" \
    --data-root "${TASK_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --adapter-name "${ADAPTER_NAME:-adapter.pth}" \
    --aligned-name "${ALIGNED_NAME:-aligned_adapter.pth}" \
    --cost-type "${COST_TYPE:-activation}" \
    --iterations "${ITERATIONS:-10}" \
    --workers "${WORKERS:-4}" \
    --torch-num-threads "${TORCH_NUM_THREADS:-1}" \
    "${save_args[@]+"${save_args[@]}"}" \
    "$@"
