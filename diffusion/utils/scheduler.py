import math
import torch
import bisect
from torch.optim.lr_scheduler import _LRScheduler
from typing import List

class ConstantLR(_LRScheduler):
    """See README.md for English documentation."""
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # English documentation is provided in README.md.
        return self.base_lrs


class CosineAnnealingLR(_LRScheduler):
    """See README.md for English documentation."""
    def __init__(self, optimizer, max_steps, eta_min=0, last_epoch=-1):
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        
        # English documentation is provided in README.md.
        step = min(self.last_epoch, self.max_steps)
        
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * step / self.max_steps)) / 2
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmup(_LRScheduler):
    """See README.md for English documentation."""
    def __init__(self, optimizer, max_steps, warmup_ratio=0.0, start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.start_lr = start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # English documentation is provided in README.md.
        if step < self.warmup_steps:
            # English documentation is provided in README.md.
            progress = step / max(1, self.warmup_steps) 
            return [
                self.start_lr + (base_lr - self.start_lr) * progress
                for base_lr in self.base_lrs
            ]
        
        # English documentation is provided in README.md.
        else:
            # English documentation is provided in README.md.
            cos_step = step - self.warmup_steps
            cos_total = self.max_steps - self.warmup_steps
            cos_total = max(1, cos_total)
            
            # English documentation is provided in README.md.
            if cos_step > cos_total:
                return [self.eta_min for _ in self.base_lrs]

            progress = cos_step / cos_total
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class CosineAnnealingRestartAtPoints(_LRScheduler):
    """See README.md for English documentation."""
    def __init__(self, optimizer, restart_points, max_steps, decay_ratio=1.0, eta_min=0, last_epoch=-1):
        """See README.md for English documentation."""
        self.max_steps = max_steps
        self.decay_ratio = decay_ratio
        self.eta_min = eta_min
        
        # English documentation is provided in README.md.
        points = [0.0] + sorted(restart_points) + [1.0]
        # English documentation is provided in README.md.
        self.milestones = sorted(list(set([int(p * max_steps) for p in points])))
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        
        # English documentation is provided in README.md.
        if current_step >= self.milestones[-1]:
             idx = len(self.milestones) - 2
        else:
             # English documentation is provided in README.md.
             idx = bisect.bisect_right(self.milestones, current_step) - 1
             # English documentation is provided in README.md.
             idx = max(0, idx)
            
        # English documentation is provided in README.md.
        start_step = self.milestones[idx]
        end_step = self.milestones[idx + 1]
        
        # English documentation is provided in README.md.
        segment_len = end_step - start_step
        if segment_len <= 0:
            progress = 1.0
        else:
            progress = (current_step - start_step) / segment_len
            
        # English documentation is provided in README.md.
        current_decay = self.decay_ratio ** idx

        # English documentation is provided in README.md.
        return [
            self.eta_min + (base_lr * current_decay - self.eta_min) *
            (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def get_lr_scheduler(optimizer, **kwargs):
    """See README.md for English documentation."""
    scheduler_type = kwargs.get("type", "const").lower()
    
    # English documentation is provided in README.md.
    max_steps = kwargs.get("max_steps", 100)
    eta_min = kwargs.get("eta_min", 0.0)
    
    if scheduler_type == "const":
        return ConstantLR(optimizer)
        
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, 
            max_steps=max_steps, 
            eta_min=eta_min
        )
        
    elif scheduler_type == "cosine_warmup":
        warmup_ratio = kwargs.get("warmup_ratio", 0.0)
        start_lr = kwargs.get("start_lr", 0.0)
        return CosineAnnealingWarmup(
            optimizer, 
            max_steps=max_steps, 
            warmup_ratio=warmup_ratio, 
            start_lr=start_lr, 
            eta_min=eta_min
        )
    
    elif scheduler_type == "cosine_restart":
        restart_points = kwargs.get("restart_points", [0.5])
        decay_ratio = kwargs.get("decay_ratio", 1.0)
        return CosineAnnealingRestartAtPoints(
            optimizer, 
            restart_points=restart_points, 
            max_steps=max_steps, 
            decay_ratio=decay_ratio, 
            eta_min=eta_min
        )
    
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")