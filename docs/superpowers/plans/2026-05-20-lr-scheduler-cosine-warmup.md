# LR Scheduler: Warmup + Cosine Decay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-switchable warmup + cosine decay LR scheduler alongside the existing StepLR.

**Architecture:** A new factory `build_scheduler` in `model/scheduler.py` returns `(scheduler, step_per_batch)`. StepLR keeps per-epoch semantics; cosine uses `LambdaLR` and steps after each `optimizer.step()`. `train.py` gains a `step_per_batch` flag to route the step call.

**Tech Stack:** PyTorch (`torch.optim.lr_scheduler.StepLR`, `LambdaLR`), pytest.

**Spec:** [docs/superpowers/specs/2026-05-20-lr-scheduler-cosine-warmup-design.md](../specs/2026-05-20-lr-scheduler-cosine-warmup-design.md)

---

## File Structure

- **Create** `model/scheduler.py` — `build_scheduler(name, optimizer, **kwargs) -> (scheduler, step_per_batch)`
- **Create** `tests/test_scheduler.py` — unit tests for both branches
- **Modify** `model/main.py` — add 3 CONFIG fields, replace StepLR construction, pass `step_per_batch` to `train_model`
- **Modify** `model/train.py` — add `step_per_batch: bool = False` param, route `scheduler.step()` to per-step vs per-epoch, log `train/lr_step`

---

### Task 1: Create `model/scheduler.py` with StepLR branch

**Files:**
- Create: `model/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scheduler.py`:

```python
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from scheduler import build_scheduler  # noqa: E402


def _make_optimizer(lr: float = 1.0):
    param = torch.zeros(1, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


def test_build_steplr_returns_steplr_and_step_per_batch_false():
    optimizer = _make_optimizer(lr=1.0)
    scheduler, step_per_batch = build_scheduler(
        "steplr",
        optimizer,
        num_epochs=10,
        optimizer_steps_per_epoch=100,
        lr_step_size=3,
        lr_gamma=0.5,
        warmup_ratio=0.05,
        min_lr_ratio=0.01,
    )
    assert step_per_batch is False
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)


def test_steplr_applies_gamma_after_step_size_epochs():
    optimizer = _make_optimizer(lr=1.0)
    scheduler, _ = build_scheduler(
        "steplr",
        optimizer,
        num_epochs=10,
        optimizer_steps_per_epoch=100,
        lr_step_size=3,
        lr_gamma=0.5,
        warmup_ratio=0.05,
        min_lr_ratio=0.01,
    )
    for _ in range(3):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scheduler'`

- [ ] **Step 3: Create `model/scheduler.py` with StepLR branch**

Create `model/scheduler.py`:

```python
"""LR scheduler factory for storm-surge training.

Returns (scheduler, step_per_batch). step_per_batch=True means the caller
must invoke scheduler.step() after every optimizer.step(); False means
once per epoch.
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR, StepLR


def build_scheduler(
    name: str,
    optimizer,
    *,
    num_epochs: int,
    optimizer_steps_per_epoch: int,
    lr_step_size: int,
    lr_gamma: float,
    warmup_ratio: float,
    min_lr_ratio: float,
):
    if name == "steplr":
        return StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma), False

    raise ValueError(f"Unknown scheduler: {name!r}. Use 'steplr' or 'cosine'.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scheduler.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add model/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): add build_scheduler factory with steplr branch"
```

---

### Task 2: Add cosine branch with warmup + decay

**Files:**
- Modify: `model/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Append failing tests for cosine**

Append to `tests/test_scheduler.py`:

```python
def _make_cosine(num_epochs=10, steps_per_epoch=100, warmup_ratio=0.05, min_lr_ratio=0.01, base_lr=1.0):
    optimizer = _make_optimizer(lr=base_lr)
    scheduler, step_per_batch = build_scheduler(
        "cosine",
        optimizer,
        num_epochs=num_epochs,
        optimizer_steps_per_epoch=steps_per_epoch,
        lr_step_size=50,
        lr_gamma=0.5,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )
    return optimizer, scheduler, step_per_batch


def test_build_cosine_returns_lambdalr_and_step_per_batch_true():
    optimizer, scheduler, step_per_batch = _make_cosine()
    assert step_per_batch is True
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_cosine_initial_lr_is_first_warmup_factor():
    # total=1000, warmup=50 -> initial factor = 1/50
    optimizer, scheduler, _ = _make_cosine()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0 / 50)


def test_cosine_end_of_warmup_reaches_base_lr():
    # After 49 steps, last_epoch=49, factor = 50/50 = 1.0
    optimizer, scheduler, _ = _make_cosine()
    for _ in range(49):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_cosine_final_step_reaches_min_lr_ratio():
    # total=1000 -> last_epoch=999 after 999 steps
    optimizer, scheduler, _ = _make_cosine()
    for _ in range(999):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01, abs=1e-3)


def test_cosine_warmup_is_monotonically_increasing():
    optimizer, scheduler, _ = _make_cosine()
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(49):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert all(lrs[i + 1] >= lrs[i] for i in range(len(lrs) - 1))


def test_cosine_decay_is_monotonically_decreasing():
    optimizer, scheduler, _ = _make_cosine()
    # Advance through warmup
    for _ in range(50):
        scheduler.step()
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(949):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert all(lrs[i + 1] <= lrs[i] for i in range(len(lrs) - 1))


def test_cosine_with_zero_warmup_starts_at_base_lr():
    optimizer, scheduler, _ = _make_cosine(warmup_ratio=0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scheduler.py -v`
Expected: 2 pass (from Task 1), 7 fail with `ValueError: Unknown scheduler: 'cosine'`.

- [ ] **Step 3: Add cosine branch to `model/scheduler.py`**

Edit `model/scheduler.py`. Replace the body of `build_scheduler` so the cosine branch is added before the final `raise`:

```python
def build_scheduler(
    name: str,
    optimizer,
    *,
    num_epochs: int,
    optimizer_steps_per_epoch: int,
    lr_step_size: int,
    lr_gamma: float,
    warmup_ratio: float,
    min_lr_ratio: float,
):
    if name == "steplr":
        return StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma), False

    if name == "cosine":
        total_steps = max(1, num_epochs * optimizer_steps_per_epoch)
        warmup_steps = int(warmup_ratio * total_steps)
        decay_steps = max(1, total_steps - warmup_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / decay_steps
            progress = min(1.0, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda), True

    raise ValueError(f"Unknown scheduler: {name!r}. Use 'steplr' or 'cosine'.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scheduler.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add model/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): add cosine decay with linear warmup"
```

---

### Task 3: Add parameter validation

**Files:**
- Modify: `model/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Append failing validation tests**

Append to `tests/test_scheduler.py`:

```python
def test_unknown_scheduler_name_raises():
    optimizer = _make_optimizer()
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(
            "rmsprop",
            optimizer,
            num_epochs=1,
            optimizer_steps_per_epoch=1,
            lr_step_size=1,
            lr_gamma=0.5,
            warmup_ratio=0.05,
            min_lr_ratio=0.01,
        )


def test_cosine_rejects_warmup_ratio_out_of_range():
    optimizer = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_ratio"):
        build_scheduler(
            "cosine",
            optimizer,
            num_epochs=10,
            optimizer_steps_per_epoch=100,
            lr_step_size=50,
            lr_gamma=0.5,
            warmup_ratio=1.0,
            min_lr_ratio=0.01,
        )


def test_cosine_rejects_negative_warmup_ratio():
    optimizer = _make_optimizer()
    with pytest.raises(ValueError, match="warmup_ratio"):
        build_scheduler(
            "cosine",
            optimizer,
            num_epochs=10,
            optimizer_steps_per_epoch=100,
            lr_step_size=50,
            lr_gamma=0.5,
            warmup_ratio=-0.1,
            min_lr_ratio=0.01,
        )


def test_cosine_rejects_min_lr_ratio_out_of_range():
    optimizer = _make_optimizer()
    with pytest.raises(ValueError, match="min_lr_ratio"):
        build_scheduler(
            "cosine",
            optimizer,
            num_epochs=10,
            optimizer_steps_per_epoch=100,
            lr_step_size=50,
            lr_gamma=0.5,
            warmup_ratio=0.05,
            min_lr_ratio=1.5,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scheduler.py -v`
Expected: `test_unknown_scheduler_name_raises` passes (already raises); the 3 cosine-validation tests fail because no range check is implemented yet.

- [ ] **Step 3: Add validation to cosine branch**

Edit `model/scheduler.py`. Insert two validation lines at the very start of the `if name == "cosine":` block, before computing `total_steps`:

```python
    if name == "cosine":
        if not (0.0 <= warmup_ratio < 1.0):
            raise ValueError(f"warmup_ratio must be in [0,1), got {warmup_ratio}")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0,1], got {min_lr_ratio}")

        total_steps = max(1, num_epochs * optimizer_steps_per_epoch)
        # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scheduler.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add model/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): validate cosine warmup/min_lr ratios"
```

---

### Task 4: Wire scheduler into `model/main.py`

**Files:**
- Modify: `model/main.py`

- [ ] **Step 1: Add CONFIG fields**

Edit `model/main.py`. In the `CONFIG` dict (around [main.py:73-77](../../../model/main.py#L73)), add three keys near the existing `lr_step_size` / `lr_gamma`:

Find:
```python
    "lr_step_size": 50,
    "lr_gamma": 0.5,
```

Replace with:
```python
    "lr_step_size": 50,
    "lr_gamma": 0.5,
    "scheduler": "steplr",
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.01,
```

- [ ] **Step 2: Replace the StepLR import with build_scheduler**

Find the line ([main.py:27](../../../model/main.py#L27)):
```python
from torch.optim.lr_scheduler import StepLR
```

Replace with:
```python
from scheduler import build_scheduler
```

- [ ] **Step 3: Replace the StepLR construction block**

Find ([main.py:314-322](../../../model/main.py#L314)):
```python
        scheduler = StepLR(
            optimizer,
            step_size=CONFIG["lr_step_size"],
            gamma=CONFIG["lr_gamma"],
        )
        rank0_print(
            dist_ctx,
            f"[main] StepLR: step_size={CONFIG['lr_step_size']}, gamma={CONFIG['lr_gamma']}",
        )
```

Replace with:
```python
        scheduler, step_per_batch = build_scheduler(
            CONFIG["scheduler"],
            optimizer,
            num_epochs=CONFIG["num_epochs"],
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            lr_step_size=CONFIG["lr_step_size"],
            lr_gamma=CONFIG["lr_gamma"],
            warmup_ratio=CONFIG["warmup_ratio"],
            min_lr_ratio=CONFIG["min_lr_ratio"],
        )
        if CONFIG["scheduler"] == "steplr":
            rank0_print(
                dist_ctx,
                f"[main] StepLR: step_size={CONFIG['lr_step_size']}, gamma={CONFIG['lr_gamma']}",
            )
        else:
            total_steps = CONFIG["num_epochs"] * optimizer_steps_per_epoch
            warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
            rank0_print(
                dist_ctx,
                f"[main] Cosine: total_steps={total_steps}, warmup_steps={warmup_steps}, "
                f"min_lr={CONFIG['lr'] * CONFIG['min_lr_ratio']:.2e}",
            )
```

- [ ] **Step 4: Pass `step_per_batch` to `train_model`**

Find the `train_model(...)` call ([main.py:345-362](../../../model/main.py#L345)). Add `step_per_batch=step_per_batch,` next to the existing `scheduler=scheduler,` argument:

```python
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            num_epochs=CONFIG["num_epochs"],
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            step_per_batch=step_per_batch,
            coords_2d_device=coords_2d_device,
            writer=writer,
            grad_clip=CONFIG["grad_clip"],
            loss_type=CONFIG["loss_type"],
            state_channels=state_channels,
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
        )
```

- [ ] **Step 5: Smoke-import to ensure no syntax error**

Run:
```bash
python -c "import sys; sys.path.insert(0, 'model'); import main"
```
Expected: no output, no traceback.

- [ ] **Step 6: Commit**

```bash
git add model/main.py
git commit -m "feat(main): wire build_scheduler with cosine/warmup config"
```

---

### Task 5: Route `scheduler.step()` per batch vs per epoch in `train.py`

**Files:**
- Modify: `model/train.py`
- Modify: `tests/test_train.py` (add coverage)

- [ ] **Step 1: Write a failing test that asserts per-batch scheduler stepping**

Append to `tests/test_train.py`:

```python
from train import train_model  # noqa: E402


class _StepCountingScheduler:
    """Records how many times .step() was called."""

    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def test_train_model_steps_scheduler_per_batch_when_flag_true(tmp_path):
    num_channels = 3
    bundle_size = 2
    B = 1
    N = 4
    model = _build(num_channels=num_channels, bundle_size=bundle_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = _StepCountingScheduler()
    batches = [_make_batch(num_channels, bundle_size, B=B, N=N) for _ in range(3)]
    loader = _FixedLoader(batches)
    coords = torch.zeros(N, 2)
    train_model(
        model=model,
        train_loader=loader,
        test_loader=loader,
        num_epochs=1,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        step_per_batch=True,
        coords_2d_device=coords,
        writer=None,
        state_channels=("u", "v", "h"),
        grad_clip=None,
        checkpoint_path=str(tmp_path / "ck.pt"),
        accum_steps=1,
    )
    # 3 micro-batches with accum_steps=1 -> 3 optimizer steps -> 3 scheduler.step() calls
    assert scheduler.step_calls == 3


def test_train_model_steps_scheduler_per_epoch_when_flag_false(tmp_path):
    num_channels = 3
    bundle_size = 2
    B = 1
    N = 4
    model = _build(num_channels=num_channels, bundle_size=bundle_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = _StepCountingScheduler()
    batches = [_make_batch(num_channels, bundle_size, B=B, N=N) for _ in range(3)]
    loader = _FixedLoader(batches)
    coords = torch.zeros(N, 2)
    train_model(
        model=model,
        train_loader=loader,
        test_loader=loader,
        num_epochs=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        step_per_batch=False,
        coords_2d_device=coords,
        writer=None,
        state_channels=("u", "v", "h"),
        grad_clip=None,
        checkpoint_path=str(tmp_path / "ck.pt"),
        accum_steps=1,
    )
    # 2 epochs -> 2 epoch-end scheduler.step() calls
    assert scheduler.step_calls == 2
```

If `_make_batch` / `_build` / `_FixedLoader` are not yet imported at the top of `test_train.py`, they are defined within the file already; the appended tests use them directly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v -k "scheduler"`
Expected: both new tests fail with `TypeError: train_model() got an unexpected keyword argument 'step_per_batch'`.

- [ ] **Step 3: Add `step_per_batch` parameter and per-step branch**

Edit `model/train.py`. Update the signature at [train.py:164-181](../../../model/train.py#L164):

Find:
```python
def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    device,
    optimizer,
    scheduler,
    coords_2d_device,
    writer,
    state_channels,
    grad_clip=None,
    loss_type: str = "rel_l2",
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
):
```

Replace with:
```python
def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    device,
    optimizer,
    scheduler,
    coords_2d_device,
    writer,
    state_channels,
    grad_clip=None,
    loss_type: str = "rel_l2",
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
    step_per_batch: bool = False,
):
```

- [ ] **Step 4: Insert per-batch `scheduler.step()` and per-step LR logging**

Find the block at [train.py:239-247](../../../model/train.py#L239):
```python
            if should_sync:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if is_rank0(dist_ctx) and writer is not None:
                    writer.add_scalar("train/loss_step", loss_unscaled, global_step)
                global_step += 1
```

Replace with:
```python
            if should_sync:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if step_per_batch and scheduler is not None:
                    scheduler.step()

                if is_rank0(dist_ctx) and writer is not None:
                    writer.add_scalar("train/loss_step", loss_unscaled, global_step)
                    writer.add_scalar(
                        "train/lr_step",
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                global_step += 1
```

- [ ] **Step 5: Guard the epoch-end `scheduler.step()` with the flag**

Find ([train.py:252-253](../../../model/train.py#L252)):
```python
        if scheduler is not None:
            scheduler.step()
```

Replace with:
```python
        if scheduler is not None and not step_per_batch:
            scheduler.step()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_train.py tests/test_scheduler.py -v`
Expected: all previously-passing tests still pass, plus the 2 new train tests pass.

- [ ] **Step 7: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat(train): route scheduler.step per batch vs per epoch"
```

---

### Task 6: End-to-end integration check

**Files:**
- Read-only verification — no code changes.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass (existing tests + 13 new scheduler tests + 2 new train tests).

- [ ] **Step 2: Smoke-import the full training entrypoint**

Run:
```bash
python -c "import sys; sys.path.insert(0, 'model'); import main; print(main.CONFIG['scheduler'], main.CONFIG['warmup_ratio'], main.CONFIG['min_lr_ratio'])"
```
Expected: `steplr 0.05 0.01`

- [ ] **Step 3: Verify cosine path constructs without crashing**

Run:
```bash
python <<'PY'
import sys, torch
sys.path.insert(0, "model")
from scheduler import build_scheduler

opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=2e-3)
sched, spb = build_scheduler(
    "cosine", opt,
    num_epochs=200, optimizer_steps_per_epoch=100,
    lr_step_size=50, lr_gamma=0.5,
    warmup_ratio=0.05, min_lr_ratio=0.01,
)
print("step_per_batch=", spb)
print("initial lr=", opt.param_groups[0]["lr"])
for _ in range(1000):
    sched.step()
print("after 1000 steps lr=", opt.param_groups[0]["lr"])
for _ in range(19000):
    sched.step()
print("after 20000 steps lr=", opt.param_groups[0]["lr"])
PY
```

Expected: `step_per_batch= True`, initial LR ≈ `2e-3 / 1000 = 2e-6`, intermediate LR somewhere between `2e-6` and `2e-3`, final LR ≈ `2e-5` (= `2e-3 * 0.01`).

- [ ] **Step 4: No commit needed (verification only).**

---

## Self-Review Notes

- **Spec coverage:** CONFIG fields (Task 4), scheduler.py factory (Tasks 1–3), main.py wiring (Task 4), train.py routing + LR logging (Task 5), validation errors (Task 3), DDP/accum_steps correctness verified by design (steps follow optimizer.step, which is rank-synchronous). End-to-end smoke (Task 6).
- **Placeholder scan:** all code blocks complete; no TBDs.
- **Type consistency:** `build_scheduler` signature, returned tuple shape, and `step_per_batch` flag are identical across Tasks 1, 2, 3, 4, 5.
