import logging
import os
import json
import sys
import warnings
import traceback

import torch

line_seg = ''.join(['*'] * 10)
logger = logging.getLogger("transnet")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def setup_logging(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(levelname).1s %(asctime)s %(filename)s:%(lineno)-4d] %(message)s',
        datefmt='%m.%d/%H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(exp_dir, "run.log"), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Capture warnings
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(stream_handler)
    warnings_logger.addHandler(file_handler)

    # Capture unhandled exceptions
    sys.excepthook = handle_exception

    return logger


def _json_default(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return str(obj)


def log_args(args, title="Arguments", target_logger=None):
    target_logger = target_logger or logger
    if hasattr(args, "__dict__"):
        args = vars(args)
    target_logger.info(
        "=> %s:\n%s",
        title,
        json.dumps(args, indent=2, sort_keys=True, default=_json_default))


def log_runtime_context(target_logger=None):
    target_logger = target_logger or logger
    target_logger.info("=> Runtime Context:")
    target_logger.info("   cwd: %s", os.getcwd())
    target_logger.info("   pid: %s", os.getpid())
    target_logger.info("   command: %s", " ".join(sys.argv))
    target_logger.info("   python: %s", sys.version.replace("\n", " "))
    target_logger.info("   pytorch: %s", torch.__version__)
    target_logger.info("   cuda_available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        target_logger.info("   cuda_version: %s", torch.version.cuda)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        target_logger.info("   CUDA_VISIBLE_DEVICES: %s", visible)
        if torch.cuda.is_initialized():
            try:
                target_logger.info(
                    "   current_device: %s (%s)",
                    torch.cuda.current_device(),
                    torch.cuda.get_device_name(torch.cuda.current_device()))
            except Exception as exc:
                target_logger.info("   current_device: unavailable (%s)", exc)
        else:
            target_logger.info(
                "   current_device: not initialized yet")


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters()
                    if param.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def log_parameter_table(model, target_logger=None):
    target_logger = target_logger or logger
    params = [
        (name, str(param.requires_grad), str(tuple(param.shape)),
         f"{param.numel():,}")
        for name, param in model.named_parameters()
    ]
    if not params:
        target_logger.info("=> Parameter Table: no parameters")
        return
    shape_width = max(len(shape) for _, _, shape, _ in params)
    count_width = max(len(count) for _, _, _, count in params)
    fmt = "{:<65} {:<8} {:>{shape_width}} {:>{count_width}}"
    lines = [
        fmt.format(
            "name",
            "grad",
            "shape",
            "numel",
            shape_width=shape_width,
            count_width=count_width),
        fmt.format(
            "-" * 4,
            "-" * 4,
            "-" * 5,
            "-" * 5,
            shape_width=shape_width,
            count_width=count_width),
    ]
    lines.extend([
        fmt.format(
            name,
            requires_grad,
            shape,
            count,
            shape_width=shape_width,
            count_width=count_width)
        for name, requires_grad, shape, count in params
    ])
    total, trainable, frozen = count_parameters(model)
    lines.append(line_seg)
    lines.append(
        f"total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
    target_logger.info("\n" + "\n".join(lines))


def log_experiment_header(args, exp_dir=None, target_logger=None):
    target_logger = target_logger or logger
    if exp_dir is not None:
        target_logger.info("=> Experiment directory: %s", exp_dir)
    log_runtime_context(target_logger)
    log_args(args, target_logger=target_logger)
