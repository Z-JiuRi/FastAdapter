from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from torch import nn


logger = logging.getLogger(__name__)
line_seg = "=" * 50


def seed_everything(seed: int = 42) -> None:
    """See README.md for English documentation."""
    logger.info("=> Random seed set to %d", seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_parameter(model: nn.Module, target_logger=None) -> None:
    """See README.md for English documentation."""
    target_logger = target_logger or logger
    parameters = [
        (name, str(parameter.requires_grad), str(tuple(parameter.shape)), f"{parameter.numel():,}")
        for name, parameter in model.named_parameters()
    ]
    if not parameters:
        target_logger.info("=> Parameter Table: no parameters")
        return

    name_width = max(65, max(len(name) for name, _, _, _ in parameters))
    shape_width = max(len(shape) for _, _, shape, _ in parameters)
    count_width = max(len(count) for _, _, _, count in parameters)
    row_format = "{:<{name_width}} {:<8} {:>{shape_width}} {:>{count_width}}"
    lines = [
        row_format.format(
            "name", "grad", "shape", "numel",
            name_width=name_width, shape_width=shape_width, count_width=count_width,
        ),
        row_format.format(
            "-" * 4, "-" * 4, "-" * 5, "-" * 5,
            name_width=name_width, shape_width=shape_width, count_width=count_width,
        ),
    ]
    lines.extend(
        row_format.format(
            name, requires_grad, shape, count,
            name_width=name_width, shape_width=shape_width, count_width=count_width,
        )
        for name, requires_grad, shape, count in parameters
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lines.append(line_seg)
    lines.append(f"total={total:,}, trainable={trainable:,}, frozen={total - trainable:,}")
    target_logger.info("\n%s", "\n".join(lines))
