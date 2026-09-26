import torch

__all__ = ['AverageMeter', 'evaluator', 'nmse_from_sums']


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluator(sparse_pred, sparse_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor
         Computes error and signal power sums for global NMSE.
    """

    with torch.no_grad():
        # # De-centralize
        # sparse_gt = sparse_gt - 0.5
        # sparse_pred = sparse_pred - 0.5

        # Calculate the NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        error_sum = mse.sum()
        power_sum = power_gt.sum()

        return error_sum, power_sum


def nmse_from_sums(error_sum, power_sum):
    with torch.no_grad():
        return 10 * torch.log10(error_sum / power_sum.clamp_min(1e-12))
