import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from utils.parser import args
from utils import (
    logger,
    setup_logging,
    Trainer,
    Tester,
    log_experiment_header,
)
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from utils.init import _load_clean_state_dict
from dataloader import MyDataLoader

import logging


def load_checkpoint_ignoring_thop_stats(model, checkpoint_path):
    result = model.load_state_dict(
        _load_clean_state_dict(checkpoint_path),
        strict=False)
    ignored_suffixes = ("total_ops", "total_params")
    missing_keys = [
        key for key in result.missing_keys
        if not key.endswith(ignored_suffixes)
    ]
    unexpected_keys = [
        key for key in result.unexpected_keys
        if not key.endswith(ignored_suffixes)
    ]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict mismatch after ignoring THOP stats: "
            f"missing={missing_keys}, unexpected={unexpected_keys}")
    return result


def train_worker(cfg, train_tensor, val_tensor, test_tensor):
    """Run a single experiment inside a spawned multiprocessing worker.
    
    cfg  keys: seed, encoder, decoder, gpu, exp_name
    train_tensor / val_tensor / test_tensor : preloaded shared-memory tensors.
    """
    import copy
    worker_args = copy.deepcopy(args)
    worker_args.experiments = None
    for k, v in cfg.items():
        setattr(worker_args, k, v)

    if worker_args.torch_num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(worker_args.torch_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(worker_args.torch_num_threads)
        torch.set_num_threads(worker_args.torch_num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    exp_dir = os.path.join(os.getcwd(), "exps", worker_args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    tensorboard_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    setup_logging(exp_dir)
    logger.info(f"[Worker seed={worker_args.seed} enc={worker_args.encoder}] started")

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(worker_args), f, indent=2, sort_keys=True)

    device, pin_memory = init_device(
        worker_args.seed, worker_args.cpu,
        worker_args.gpu, worker_args.cpu_affinity)
    log_experiment_header(worker_args, exp_dir=exp_dir, target_logger=logger)
    logger.info(f'=> Checkpoint directory: {checkpoint_dir}')
    logger.info(f'=> TensorBoard directory: {tensorboard_dir}')

    data_builder = MyDataLoader(
        train_path=worker_args.train_path,
        val_path=worker_args.val_path,
        test_path=worker_args.test_path,
        batch_size=worker_args.batch_size,
        num_workers=0,
        pin_memory=True,
        channel=worker_args.channel,
        nt=worker_args.nt,
        nc=worker_args.nc,
        return_indices=True,
        train_tensor=train_tensor,
        val_tensor=val_tensor,
        test_tensor=test_tensor)
    train_loader, val_loader, test_loader = data_builder()
    logger.info("=> Dataset sizes: train=%d, val=%d, test=%d",
                len(train_loader.dataset), len(val_loader.dataset),
                len(test_loader.dataset))
    logger.info("=> Loader config: batch_size=%d, workers=0, pin_memory=%s",
                worker_args.batch_size, True)

    model = init_model(worker_args)
    model.to(device)
    criterion = nn.MSELoss().to(device)

    if worker_args.evaluate:
        Tester(model, device, criterion)(test_loader)
        trainer = Trainer(model=model, device=device, optimizer=None,
                          criterion=criterion, scheduler=None,
                          save_path=checkpoint_dir,
                          tensorboard_dir=tensorboard_dir)
        trainer.save_codewords(
            train_loader,
            os.path.join(exp_dir, "codewords", "train_code.pt"))
        trainer.save_codewords(
            val_loader,
            os.path.join(exp_dir, "codewords", "val_code.pt"))
        trainer.save_codewords(
            test_loader,
            os.path.join(exp_dir, "codewords", "test_code.pt"))
        return

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    if not decay_params and not no_decay_params:
        raise ValueError("No trainable parameters found for optimizer")
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": worker_args.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=worker_args.lr_init)
    logger.info("=> Optimizer: AdamW lr=%s weight_decay=%s",
                worker_args.lr_init, worker_args.weight_decay)

    if worker_args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            T_max=worker_args.epochs * len(train_loader),
            T_warmup=0.1 * worker_args.epochs * len(train_loader),
            eta_min=5e-5)

    trainer = Trainer(model=model, device=device, optimizer=optimizer,
                      criterion=criterion, scheduler=scheduler,
                      resume=worker_args.resume,
                      save_path=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir,
                      # The multi-stage workflow consumes best_nmse.pth.  Keep
                      # the normal 10-epoch cadence, but evaluate the final
                      # epoch for short smoke runs as well.
                      test_freq=max(1, min(10, worker_args.epochs)),
                      test_every_epoch=False)
    trainer.loop(worker_args.epochs, train_loader, val_loader, test_loader)

    best_checkpoint = os.path.join(checkpoint_dir, "best_nmse.pth")
    if os.path.isfile(best_checkpoint):
        logger.info(f'=> Loading best checkpoint before codeword export: {best_checkpoint}')
        _ = load_checkpoint_ignoring_thop_stats(model, best_checkpoint)
    else:
        logger.warning('=> No best_nmse.pth found before codeword export; using current state')
    trainer.save_codewords(
        train_loader,
        os.path.join(exp_dir, "codewords", "train_code.pt"))
    trainer.save_codewords(
        val_loader,
        os.path.join(exp_dir, "codewords", "val_code.pt"))
    trainer.save_codewords(
        test_loader,
        os.path.join(exp_dir, "codewords", "test_code.pt"))

    loss, nmse = Tester(model, device, criterion)(test_loader)
    logger.info(f'\n=! Final test loss: {loss:.4e}\n         test NMSE: {nmse:.4e}\n')


def main():
    # ----- Basic logging (console) - always available -----
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _console_handler = logging.StreamHandler(sys.stdout)
        _console_handler.setFormatter(logging.Formatter(
            fmt='%(levelname).1s %(asctime)s %(filename)s:%(lineno)-4d] %(message)s',
            datefmt='%m.%d/%H:%M:%S'))
        logger.addHandler(_console_handler)
        logging.captureWarnings(True)

    # ----- Multi-experiment mode -----
    if args.experiments is not None:
        logger.info("=> Multi-experiment mode: %d experiments", len(args.experiments))
        for cfg in args.experiments:
            logger.info("   seed=%s enc=%s dec=%s gpu=%s",
                        cfg['seed'], cfg['encoder'], cfg['decoder'], cfg['gpu'])
        # Load data once into shared memory
        def _load_shared_tensor(path):
            data = torch.load(path, weights_only=True, map_location="cpu")
            data = data.to(torch.float32)
            expected = (args.channel, args.nt, args.nc)
            if data.ndim == 2:
                data = data.view(-1, *expected)
            if data.ndim != 4 or tuple(data.shape[1:]) != expected:
                raise ValueError(f"{path}: expected (N,{expected}), got {tuple(data.shape)}")
            data.share_memory_()
            return data
        logger.info("=> Loading shared data ...")
        train_data = _load_shared_tensor(args.train_path)
        val_data = _load_shared_tensor(args.val_path)
        test_data = _load_shared_tensor(args.test_path)
        logger.info("=> Shared data: train=%s val=%s test=%s",
                     train_data.shape, val_data.shape, test_data.shape)

        ctx = mp.get_context("spawn")
        processes = []
        for i, cfg in enumerate(args.experiments):
            p = ctx.Process(target=train_worker,
                            args=(cfg, train_data, val_data, test_data),
                            name=f"w{i}_s{cfg['seed']}_{cfg['encoder']}")
            p.start()
            processes.append(p)
            time.sleep(2)
        for p in processes:
            p.join()
        logger.info("=> All workers finished.")
        return

    exp_dir = os.path.join(os.getcwd(), "exps", args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    tensorboard_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    setup_logging(exp_dir)

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Set CUDA_VISIBLE_DEVICES before logging CUDA runtime details.
    device, pin_memory = init_device(
        args.seed,
        args.cpu,
        args.gpu,
        args.cpu_affinity)
    log_experiment_header(args, exp_dir=exp_dir, target_logger=logger)
    logger.info(f'=> Checkpoint directory: {checkpoint_dir}')
    logger.info(f'=> TensorBoard directory: {tensorboard_dir}')
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Create the data loader

    data_builder = MyDataLoader(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        channel=args.channel,
        nt=args.nt,
        nc=args.nc,
        return_indices=True)
    train_loader, val_loader, test_loader = data_builder()
    logger.info(
        "=> Dataset sizes: train=%d, val=%d, test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset))
    logger.info(
        "=> Loader config: batch_size=%d, workers=%d, pin_memory=%s",
        args.batch_size,
        args.workers,
        pin_memory)

    # Define model

    model = init_model(args)
    model.to(device)
    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        trainer = Trainer(model=model,
                          device=device,
                          optimizer=None,
                          criterion=criterion,
                          scheduler=None,
                          save_path=checkpoint_dir,
                          tensorboard_dir=tensorboard_dir)
        trainer.save_codewords(
            train_loader,
            os.path.join(exp_dir, "codewords", "train_code.pt"))
        trainer.save_codewords(
            val_loader,
            os.path.join(exp_dir, "codewords", "val_code.pt"))
        trainer.save_codewords(
            test_loader,
            os.path.join(exp_dir, "codewords", "test_code.pt"))
        return

    # Define optimizer and scheduler

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if not decay_params and not no_decay_params:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr_init,
    )
    logger.info(
        "=> Optimizer: AdamW lr=%s weight_decay=%s "
        "decay_params=%d no_decay_params=%d",
        args.lr_init,
        args.weight_decay,
        sum(param.numel() for param in decay_params),
        sum(param.numel() for param in no_decay_params))

    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
        logger.info("=> Scheduler: const")

    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=0.1 * args.epochs * len(train_loader),
                                            eta_min=5e-5)
        logger.info(
            "=> Scheduler: cosine T_max=%s T_warmup=%s eta_min=%s",
            args.epochs * len(train_loader),
            0.1 * args.epochs * len(train_loader),
            5e-5)

    # Define the training pipeline

    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume,
                      save_path=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir,
                      test_freq=max(1, min(10, args.epochs)),
                      test_every_epoch=False)

    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader)

    best_checkpoint = os.path.join(checkpoint_dir, "best_nmse.pth")
    if os.path.isfile(best_checkpoint):
        logger.info(f'=> Loading best checkpoint before codeword export: '
                    f'{best_checkpoint}')
        result = load_checkpoint_ignoring_thop_stats(model, best_checkpoint)
        if result.missing_keys or result.unexpected_keys:
            logger.info(
                '=> Ignored THOP statistic keys while loading best checkpoint: '
                f'missing={len(result.missing_keys)}, '
                f'unexpected={len(result.unexpected_keys)}')
    else:
        logger.warning('=> No best_nmse.pth found before codeword export; '
                       'using current model state')

    trainer.save_codewords(
        train_loader,
        os.path.join(exp_dir, "codewords", "train_code.pt"))
    trainer.save_codewords(
        val_loader,
        os.path.join(exp_dir, "codewords", "val_code.pt"))
    trainer.save_codewords(
        test_loader,
        os.path.join(exp_dir, "codewords", "test_code.pt"))

    # Final testing
    loss, nmse = Tester(model, device, criterion)(test_loader)
    logger.info(f'\n=! Final test loss: {loss:.4e}'
                f'\n         test NMSE: {nmse:.4e}\n')

    # Create images for loss and nmse



if __name__ == "__main__":
    main()
