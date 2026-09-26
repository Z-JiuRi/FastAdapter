#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Required environment variable is not set: ${name}" >&2
        exit 2
    fi
}

optional_gpu_args() {
    if [[ "${CPU:-0}" == "1" ]]; then
        printf '%s\n' "--cpu"
    elif [[ -n "${GPU:-}" ]]; then
        printf '%s\n' "--gpu" "${GPU}"
    fi
}
