import os
import time

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from utils import logger
from utils.statics import AverageMeter, evaluator, nmse_from_sums

__all__ = ["Trainer", "Tester"]


class Trainer:
    """Training pipeline for the encoder-decoder architecture."""

    def __init__(self, model, device, optimizer, criterion, scheduler,
                 resume=None, save_path="./checkpoints", tensorboard_dir=None,
                 print_freq=20, val_freq=10, test_freq=10,
                 test_every_epoch=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.resume_file = resume
        self.save_path = save_path
        self.tensorboard_dir = tensorboard_dir
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq
        self.test_every_epoch = test_every_epoch
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_nmse = {"nmse": None, "epoch": None}
        self.last_train_metrics = {}
        self.last_val_metrics = {}
        self.last_test_metrics = {}
        self.tester = Tester(model, device, criterion, print_freq)
        if self.tensorboard_dir is None:
            self.tensorboard_dir = os.path.join(
                "exps", "default", "tensorboard")
        self.vision = SummaryWriter(log_dir=self.tensorboard_dir)

    def loop(self, epochs, train_loader, val_loader, test_loader):
        """Run training, validation, evaluation, and checkpoint saving."""
        self.all_epoch = epochs
        self._resume()
        for epoch in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = epoch
            self.train_loss = self.train(train_loader)
            if epoch % self.val_freq == 0:
                self.val_loss = self.val(val_loader)
            if self.test_every_epoch or epoch % self.test_freq == 0:
                self.test_loss, nmse = self.test(test_loader)
                self.vision.add_scalar("test/loss", self.test_loss, epoch)
                self.vision.add_scalar("test/nmse", nmse, epoch)
                self.vision.add_scalar("test/train_loss", self.train_loss,
                                       epoch)
            else:
                nmse = None
            self._loop_postprocessing(nmse)

    def train(self, train_loader):
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=False)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter("Iter loss")
        iter_time = AverageMeter("Iter time")
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0].to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            if (batch_idx + 1) % self.print_freq == 0:
                lr = self.scheduler.get_lr()[0]
                logger.info(
                    "Epoch: [%d/%d][%d/%d] lr: %.2e | loss: %.4e | "
                    "time: %.3f",
                    self.cur_epoch, self.all_epoch, batch_idx + 1,
                    len(data_loader), lr, iter_loss.avg, iter_time.avg)
                self.vision.add_scalar("every/lr", lr, self.cur_epoch)
                self.vision.add_scalar("every/loss", iter_loss.avg,
                                       self.cur_epoch)

        mode = "Train" if self.model.training else "Val"
        logger.info("=> %s loss: %.4e\n", mode, iter_loss.avg)
        metrics = {"loss": self._as_float(iter_loss.avg)}
        if self.model.training:
            self.last_train_metrics = metrics
        else:
            self.last_val_metrics = metrics
        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning("No path to save checkpoints.")
            return
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        if self.resume_file is None:
            return
        if not os.path.isfile(self.resume_file):
            raise FileNotFoundError(self.resume_file)
        logger.info("=> loading checkpoint %s", self.resume_file)
        checkpoint = torch.load(
            self.resume_file, weights_only=True, map_location=self.device)
        self.cur_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_nmse = checkpoint.get(
            "best_nmse", {"nmse": None, "epoch": None})
        self.cur_epoch += 1
        logger.info("=> successfully loaded checkpoint %s from epoch %d\n",
                    self.resume_file, checkpoint["epoch"])

    def _loop_postprocessing(self, nmse):
        if isinstance(nmse, torch.Tensor):
            nmse = float(nmse.detach().cpu())
        state = {
            "epoch": self.cur_epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_nmse": self.best_nmse,
        }
        if nmse is not None and (
                self.best_nmse["nmse"] is None
                or self.best_nmse["nmse"] > nmse):
            self.best_nmse = {"nmse": nmse, "epoch": self.cur_epoch}
            state["best_nmse"] = self.best_nmse
            self._save(state, "best_nmse.pth")
        if self.best_nmse["nmse"] is not None:
            logger.info("\n=! Best NMSE: %.4e (epoch=%d)\n",
                        self.best_nmse["nmse"],
                        self.best_nmse["epoch"])
            self.vision.add_scalar(
                "best/mse", self.best_nmse["nmse"],
                self.best_nmse["epoch"])

    @staticmethod
    def _as_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def save_codewords(self, data_loader, output_path):
        if output_path is None:
            logger.warning("No path to save codewords.")
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        codewords = []
        sample_indices = []
        with torch.no_grad():
            for batch in data_loader:
                sparse_gt = batch[0].to(self.device)
                indices = batch[1] if len(batch) > 1 else None
                codewords.append(self.model.encode(sparse_gt).cpu())
                if indices is not None:
                    sample_indices.append(indices.cpu())

        codewords_tensor = torch.cat(codewords, dim=0)
        index_aligned = bool(sample_indices)
        if index_aligned:
            indices_tensor = torch.cat(sample_indices).long()
            if indices_tensor.numel() != codewords_tensor.size(0):
                raise ValueError(
                    "The number of codewords does not match the number of "
                    "indices.")
            expected = torch.arange(indices_tensor.numel(), dtype=torch.long)
            if not torch.equal(torch.sort(indices_tensor).values, expected):
                raise ValueError(
                    "Codeword indices must cover every sample exactly once.")
            aligned_codewords = torch.empty_like(codewords_tensor)
            aligned_codewords[indices_tensor] = codewords_tensor
            codewords_tensor = aligned_codewords

        torch.save(codewords_tensor, output_path)
        order = "index-aligned" if index_aligned else "loader-order"
        logger.info("=> Saved %s codewords %s to %s", order,
                    tuple(codewords_tensor.shape), output_path)


class Tester:
    """Evaluate reconstruction loss and aggregate NMSE."""

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        self.last_metrics = {}

    def __call__(self, test_data, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, nmse = self._iteration(test_data)
        if verbose:
            logger.info("\n=> Test result:\nloss: %.4e    NMSE: %.4e\n",
                        loss, nmse)
        return loss, nmse

    def _iteration(self, data_loader):
        iter_loss = AverageMeter("Iter loss")
        iter_time = AverageMeter("Iter time")
        total_error = torch.tensor(0.0, device=self.device)
        total_power = torch.tensor(0.0, device=self.device)
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0].to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)
            error_sum, power_sum = evaluator(sparse_pred, sparse_gt)
            total_error += error_sum
            total_power += power_sum
            nmse = nmse_from_sums(total_error, total_power)
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(
                    "[%d/%d] loss: %.4e | NMSE: %.4e | time: %.3f",
                    batch_idx + 1, len(data_loader), iter_loss.avg, nmse,
                    iter_time.avg)

        nmse = nmse_from_sums(total_error, total_power)
        self.last_metrics = {
            "aggregate": {
                "loss": self._as_float(iter_loss.avg),
                "nmse": self._as_float(nmse),
            }
        }
        logger.info("=> Test NMSE: %.4e\n", nmse)
        return iter_loss.avg, nmse

    @staticmethod
    def _as_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)
