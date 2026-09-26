#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var TASK_ROOT

args=(
    --data-root "${TASK_ROOT}"
    --adapter-name "${ADAPTER_NAME:-adapter.pth}"
    --workers "${WORKERS:-4}"
)
if [[ "${DRY_RUN:-1}" == "1" ]]; then
    args+=(--dry-run)
fi

"${PYTHON_BIN}" "${SUBMIT_DIR}/adapter/strip_adapter_checkpoints.py" \
    "${args[@]}" \
    "$@"
