"""Warmup + cosine LR scheduler for storm-surge training.

The returned LambdaLR must be stepped once per optimizer step.
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer,
    *,
    num_epochs: int,
    optimizer_steps_per_epoch: int,
    warmup_ratio: float,
    min_lr_ratio: float,
):
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError("warmup_ratio must satisfy 0.0 <= warmup_ratio < 1.0.")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError("min_lr_ratio must satisfy 0.0 <= min_lr_ratio <= 1.0.")

    total_steps = max(1, num_epochs * optimizer_steps_per_epoch)
    warmup_steps = int(warmup_ratio * total_steps)
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
