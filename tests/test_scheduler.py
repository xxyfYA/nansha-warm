import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from scheduler import build_scheduler  # noqa: E402


def _make_optimizer(lr: float = 1.0):
    param = torch.zeros(1, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


def _make_cosine(
    num_epochs=10,
    steps_per_epoch=100,
    warmup_ratio=0.05,
    min_lr_ratio=0.01,
    base_lr=1.0,
):
    optimizer = _make_optimizer(lr=base_lr)
    scheduler = build_scheduler(
        optimizer,
        num_epochs=num_epochs,
        optimizer_steps_per_epoch=steps_per_epoch,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )
    return optimizer, scheduler


def test_build_cosine_returns_lambdalr():
    _, scheduler = _make_cosine()
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_cosine_initial_lr_is_first_warmup_factor():
    # total=1000, warmup=50 -> initial factor = 1/50
    optimizer, scheduler = _make_cosine()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0 / 50)


def test_cosine_end_of_warmup_reaches_base_lr():
    # After 49 steps, last_epoch=49, factor = 50/50 = 1.0
    optimizer, scheduler = _make_cosine()
    for _ in range(49):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_cosine_final_step_reaches_min_lr_ratio():
    # total=1000 -> last_epoch=999 after 999 steps
    optimizer, scheduler = _make_cosine()
    for _ in range(999):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01, abs=1e-3)


def test_cosine_warmup_is_monotonically_increasing():
    optimizer, scheduler = _make_cosine()
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(49):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert all(lrs[i + 1] >= lrs[i] for i in range(len(lrs) - 1))


def test_cosine_decay_is_monotonically_decreasing():
    optimizer, scheduler = _make_cosine()
    # Advance through warmup
    for _ in range(50):
        scheduler.step()
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(949):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert all(lrs[i + 1] <= lrs[i] for i in range(len(lrs) - 1))


def test_cosine_with_zero_warmup_starts_at_base_lr():
    optimizer, scheduler = _make_cosine(warmup_ratio=0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_cosine_rejects_warmup_ratio_equal_one():
    optimizer = _make_optimizer(lr=1.0)
    with pytest.raises(ValueError, match="warmup_ratio"):
        build_scheduler(
            optimizer,
            num_epochs=10,
            optimizer_steps_per_epoch=100,
            warmup_ratio=1.0,
            min_lr_ratio=0.01,
        )


def test_cosine_rejects_negative_warmup_ratio():
    optimizer = _make_optimizer(lr=1.0)
    with pytest.raises(ValueError, match="warmup_ratio"):
        build_scheduler(
            optimizer,
            num_epochs=10,
            optimizer_steps_per_epoch=100,
            warmup_ratio=-0.1,
            min_lr_ratio=0.01,
        )


def test_cosine_rejects_min_lr_ratio_greater_than_one():
    optimizer = _make_optimizer(lr=1.0)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        build_scheduler(
            optimizer,
            num_epochs=10,
            optimizer_steps_per_epoch=100,
            warmup_ratio=0.1,
            min_lr_ratio=1.1,
        )
