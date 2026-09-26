from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _has_c_compiler() -> bool:
    cc = os.environ.get("CC")
    if cc and (shutil.which(cc) or Path(cc).is_file()):
        return True
    candidates = (
        "gcc",
        "cc",
        "/usr/bin/gcc",
        "/usr/bin/cc",
    )
    return any(shutil.which(path) or Path(path).is_file() for path in candidates)


def check_mamba_runtime(timeout: int = 90) -> tuple[bool, str]:
    """See README.md for English documentation."""
    if not _has_c_compiler():
        return False, "no C compiler is available"
    code = "from mamba_ssm import Mamba; print('mamba_import_ok')"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return False, "mamba_ssm import timed out"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        tail = " | ".join(detail[-8:])
        return False, tail or "mamba_ssm import failed"
    return True, "mamba_ssm is importable"


def validate_runtime_before_experiment(configs: list[Any], names: list[str] | None = None) -> None:
    """See README.md for English documentation."""
    requires_mamba = [
        index for index, cfg in enumerate(configs)
        if str(cfg.token_context.get("implementation", "transformer")) == "mamba"
    ]
    if not requires_mamba:
        return
    ok, reason = check_mamba_runtime()
    if ok:
        return
    labels = names or [str(index) for index in range(len(configs))]
    failed = ", ".join(labels[index] for index in requires_mamba)
    raise RuntimeError(
        "The following configurations require "
        "token_context.implementation=mamba, but the mamba_ssm/Triton "
        "runtime is unavailable. Startup stopped before creating experiment "
        f"directories. Configurations: {failed}. Reason: {reason}"
    )
