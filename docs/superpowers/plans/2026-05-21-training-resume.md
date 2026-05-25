# Training Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow interrupting training and continuing on a different machine by persisting model / optimizer / scheduler / EMA / best-loss / TB run-tag at each epoch end and reloading them at startup.

**Architecture:** Each epoch end, rank0 writes a `last_<name>.pt` payload (atomic via `.tmp` + `os.replace`) alongside the existing `best_<name>.pt`. `main.py` adds two CONFIG fields (`resume_from`, `resume_epoch`) and, when `resume_from` is set, loads the checkpoint before training begins and overrides `run_tag` so the same TensorBoard directory is reused. `train_model` gains 5 new kwargs (`start_epoch`, `best_loss_init`, `resume_ema_shadow`, `last_checkpoint_path`, `run_tag`) — all default-safe so existing callers stay unchanged.

**Tech Stack:** PyTorch (`torch.save` / `torch.load`, `state_dict()`), `os.replace` for atomic rename, pytest.

**Spec:** [docs/superpowers/specs/2026-05-21-training-resume-design.md](../specs/2026-05-21-training-resume-design.md)

---

## File Structure

- **Modify** `model/train.py` — add 5 kwargs to `train_model`, use them to seed epoch/global_step/best_loss/EMA shadow, and write `last.pt` each epoch end.
- **Modify** `model/main.py` — add `resume_from` and `resume_epoch` CONFIG fields, import `unwrap_model`, add a load block before `SummaryWriter` creation, pass new kwargs to `train_model`.
- **Modify** `tests/test_train.py` — add tests for `start_epoch`, `best_loss_init`, `resume_ema_shadow`, `last_checkpoint_path` save, atomic write.

No new files. No new dependencies.

---

### Task 1: Add `start_epoch` and `best_loss_init` params to `train_model`

These two together control "how many epochs to skip" and "what previous best loss to compare against."

**Files:**
- Modify: `model/train.py:205-244`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train.py`:

```python
def test_train_model_start_epoch_skips_completed_epochs():
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-2)
    scheduler = _CountingScheduler()
    coords = torch.rand(6, 2)

    train_model(
        model,
        _FixedLoader(train_batches),
        _FixedLoader(val_batches),
        num_epochs=5,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        start_epoch=3,
    )

    # num_epochs=5, start_epoch=3 -> 2 epochs run -> 2 epoch-end scheduler steps
    assert scheduler.step_calls == 2


def test_train_model_best_loss_init_threshold_blocks_best_save(tmp_path):
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-2)
    scheduler = _CountingScheduler()
    coords = torch.rand(6, 2)
    checkpoint_path = tmp_path / "best.pt"

    # best_loss_init = -1.0 makes "current < best" impossible -> no best.pt written
    train_model(
        model,
        _FixedLoader(train_batches),
        _FixedLoader(val_batches),
        num_epochs=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        best_loss_init=-1.0,
        checkpoint_path=str(checkpoint_path),
    )

    assert not checkpoint_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py::test_train_model_start_epoch_skips_completed_epochs tests/test_train.py::test_train_model_best_loss_init_threshold_blocks_best_save -v`

Expected: both fail with `TypeError: train_model() got an unexpected keyword argument 'start_epoch'` (or `best_loss_init`).

- [ ] **Step 3: Add params to `train_model` signature**

Edit `model/train.py`. Find the signature at [model/train.py:205-225](../../../model/train.py#L205):

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
    channel_weights: dict[str, float] | None = None,
    ema_decay: float | None = None,
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
    step_per_batch: bool = False,
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
    channel_weights: dict[str, float] | None = None,
    ema_decay: float | None = None,
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
    step_per_batch: bool = False,
    start_epoch: int = 0,
    best_loss_init: float = float("inf"),
):
```

- [ ] **Step 4: Use new params in init and loop range**

Find at [model/train.py:240-244](../../../model/train.py#L240):

```python
    global_step = 0
    best_loss = float("inf")
    x_in_base = coords_2d_device.to(device, non_blocking=True).unsqueeze(0)

    for epoch in range(num_epochs):
```

Replace with:

```python
    global_step = 0
    best_loss = best_loss_init
    x_in_base = coords_2d_device.to(device, non_blocking=True).unsqueeze(0)

    for epoch in range(start_epoch, num_epochs):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`

Expected: all existing tests still pass + 2 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat(train): add start_epoch and best_loss_init for resume"
```

---

### Task 2: Add `resume_ema_shadow` param to `train_model`

Allow seeding EMA shadow weights with a saved state instead of `deepcopy(live)` from scratch.

**Files:**
- Modify: `model/train.py:236-238`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_train.py`:

```python
def test_train_model_loads_resume_ema_shadow():
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-2)
    scheduler = _CountingScheduler()
    coords = torch.rand(6, 2)

    sentinel_state = {
        name: torch.full_like(param, 7.5)
        for name, param in model.state_dict().items()
    }

    # Spy on ExponentialMovingAverage: track what state_dict gets loaded into shadow.
    import train as train_mod
    loaded_states: list[dict] = []
    real_cls = train_mod.ExponentialMovingAverage

    class _TrackEMA(real_cls):
        def __init__(self, model_, decay):
            super().__init__(model_, decay)
            orig_load = self.shadow.load_state_dict

            def _track_load(state):
                loaded_states.append({k: v.detach().clone() for k, v in state.items()})
                return orig_load(state)

            self.shadow.load_state_dict = _track_load

    train_mod.ExponentialMovingAverage = _TrackEMA
    try:
        train_model(
            model,
            _FixedLoader(train_batches),
            _FixedLoader(val_batches),
            num_epochs=0,             # no epochs run -> EMA only initialized + loaded
            device=torch.device("cpu"),
            optimizer=optimizer,
            scheduler=scheduler,
            coords_2d_device=coords,
            writer=None,
            state_channels=(0, 1, 2),
            ema_decay=0.5,
            resume_ema_shadow=sentinel_state,
        )
    finally:
        train_mod.ExponentialMovingAverage = real_cls

    assert len(loaded_states) == 1
    loaded = loaded_states[0]
    assert loaded.keys() == sentinel_state.keys()
    for name in sentinel_state:
        assert torch.allclose(loaded[name], sentinel_state[name])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_train_model_loads_resume_ema_shadow -v`

Expected: fail with `TypeError: train_model() got an unexpected keyword argument 'resume_ema_shadow'`.

- [ ] **Step 3: Add `resume_ema_shadow` param + load logic**

Edit `model/train.py`. Add to signature (after `best_loss_init`):

```python
    resume_ema_shadow: dict | None = None,
```

Find at [model/train.py:236-238](../../../model/train.py#L236):

```python
    ema = None
    if ema_decay is not None:
        ema = ExponentialMovingAverage(model, decay=ema_decay)
```

Replace with:

```python
    ema = None
    if ema_decay is not None:
        ema = ExponentialMovingAverage(model, decay=ema_decay)
        if resume_ema_shadow is not None:
            ema.shadow.load_state_dict(resume_ema_shadow)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`

Expected: all existing tests still pass + new test passes.

- [ ] **Step 5: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat(train): seed EMA shadow from saved state on resume"
```

---

### Task 3: Add `last_checkpoint_path` + `run_tag` params and save the resume bundle each epoch

**Files:**
- Modify: `model/train.py` (imports + signature + epoch-end block + global_step init)
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train.py`:

```python
def test_train_model_writes_last_checkpoint_with_expected_payload(tmp_path):
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-2)
    # Use a real StepLR so scheduler.state_dict() works.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    coords = torch.rand(6, 2)
    last_path = tmp_path / "last.pt"

    train_model(
        model,
        _FixedLoader(train_batches),
        _FixedLoader(val_batches),
        num_epochs=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        ema_decay=0.5,
        last_checkpoint_path=str(last_path),
        run_tag="GeoFNO_test_run",
    )

    assert last_path.exists()
    assert not (tmp_path / "last.pt.tmp").exists()  # atomic rename completed

    payload = torch.load(last_path, map_location="cpu")
    assert set(payload.keys()) == {
        "epoch", "best_loss", "model", "optimizer",
        "scheduler", "ema_shadow", "run_tag",
    }
    assert payload["epoch"] == 2
    assert payload["run_tag"] == "GeoFNO_test_run"
    assert payload["ema_shadow"] is not None
    assert payload["scheduler"] is not None


def test_train_model_last_checkpoint_ema_shadow_none_when_disabled(tmp_path):
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    coords = torch.rand(6, 2)
    last_path = tmp_path / "last.pt"

    train_model(
        model,
        _FixedLoader(train_batches),
        _FixedLoader(val_batches),
        num_epochs=1,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        last_checkpoint_path=str(last_path),
        run_tag="run-A",
    )

    payload = torch.load(last_path, map_location="cpu")
    assert payload["ema_shadow"] is None


def test_train_model_skips_last_checkpoint_when_path_none(tmp_path):
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    coords = torch.rand(6, 2)

    train_model(
        model,
        _FixedLoader(train_batches),
        _FixedLoader(val_batches),
        num_epochs=1,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        last_checkpoint_path=None,
    )

    # No file should appear anywhere in tmp_path
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v -k "last_checkpoint or skips_last"`

Expected: 3 fail with `TypeError: train_model() got an unexpected keyword argument 'last_checkpoint_path'` (or `run_tag`).

- [ ] **Step 3: Add `os` import**

Edit `model/train.py`. Find the imports block at [model/train.py:1-11](../../../model/train.py#L1):

```python
"""Training loop for Geo-FNO bundle-only mode."""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm import tqdm

from temporal_utils import CHANNEL_ORDER
```

Replace with:

```python
"""Training loop for Geo-FNO bundle-only mode."""
from __future__ import annotations

import copy
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm import tqdm

from temporal_utils import CHANNEL_ORDER
```

- [ ] **Step 4: Add params to signature**

Edit `model/train.py`. Add to signature (after `resume_ema_shadow`):

```python
    last_checkpoint_path: str | None = None,
    run_tag: str | None = None,
```

The final signature block should now read:

```python
    accum_steps: int = 1,
    step_per_batch: bool = False,
    start_epoch: int = 0,
    best_loss_init: float = float("inf"),
    resume_ema_shadow: dict | None = None,
    last_checkpoint_path: str | None = None,
    run_tag: str | None = None,
):
```

- [ ] **Step 5: Recompute `global_step` to align with `start_epoch`**

Find:

```python
    global_step = 0
    best_loss = best_loss_init
```

Replace with:

```python
    global_step = start_epoch * (len(train_loader) // accum_steps)
    best_loss = best_loss_init
```

- [ ] **Step 6: Add the save block at end of epoch**

Find the best-save block + barrier at [model/train.py:356-364](../../../model/train.py#L356):

```python
        current_test_loss = test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                save_target = ema.shadow if ema is not None else unwrap_model(model)
                torch.save(save_target.state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")

        barrier_if_distributed(dist_ctx)
```

Replace with:

```python
        current_test_loss = test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                save_target = ema.shadow if ema is not None else unwrap_model(model)
                torch.save(save_target.state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")

        if is_rank0(dist_ctx) and last_checkpoint_path is not None:
            payload = {
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "ema_shadow": ema.shadow.state_dict() if ema is not None else None,
                "run_tag": run_tag,
            }
            tmp_path = last_checkpoint_path + ".tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, last_checkpoint_path)

        barrier_if_distributed(dist_ctx)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`

Expected: all existing tests still pass + 3 new tests pass.

- [ ] **Step 8: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat(train): save last.pt resume bundle each epoch with atomic write"
```

---

### Task 4: Wire `resume_from` + `resume_epoch` into `model/main.py`

**Files:**
- Modify: `model/main.py`

- [ ] **Step 1: Add CONFIG fields**

Edit `model/main.py`. Find the CONFIG dict at [model/main.py:50-87](../../../model/main.py#L50). Locate the closing brace area:

```python
    "ema_decay": 0.999,

    "channels": "uvh",
}
```

Replace with:

```python
    "ema_decay": 0.999,

    "channels": "uvh",

    "resume_from": None,
    "resume_epoch": None,
}
```

- [ ] **Step 2: Import `unwrap_model`**

Find at [model/main.py:46](../../../model/main.py#L46):

```python
from train import train_model
```

Replace with:

```python
from train import train_model, unwrap_model
```

- [ ] **Step 3: Insert the resume-load block before `SummaryWriter` creation**

Find at [model/main.py:344-350](../../../model/main.py#L344) (the section right after the scheduler print and before the rank0 TB setup):

```python
        else:
            total_steps = CONFIG["num_epochs"] * optimizer_steps_per_epoch
            warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
            rank0_print(
                dist_ctx,
                f"[main] Cosine: total_steps={total_steps}, warmup_steps={warmup_steps}, "
                f"min_lr={CONFIG['lr'] * CONFIG['min_lr_ratio']:.2e}",
            )

        if dist_ctx["is_rank0"]:
            tb_run_dir = os.path.join(CONFIG["tb_dir"], run_tag)
```

Replace with:

```python
        else:
            total_steps = CONFIG["num_epochs"] * optimizer_steps_per_epoch
            warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
            rank0_print(
                dist_ctx,
                f"[main] Cosine: total_steps={total_steps}, warmup_steps={warmup_steps}, "
                f"min_lr={CONFIG['lr'] * CONFIG['min_lr_ratio']:.2e}",
            )

        last_checkpoint_name = checkpoint_name.replace("best_", "last_", 1)
        start_epoch = 0
        best_loss_init = float("inf")
        resume_ema_shadow = None
        if CONFIG["resume_from"]:
            ckpt = torch.load(CONFIG["resume_from"], map_location=device)
            unwrap_model(model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            resume_ema_shadow = ckpt.get("ema_shadow")
            best_loss_init = ckpt["best_loss"]
            run_tag = ckpt["run_tag"]

            if CONFIG["resume_epoch"] is not None:
                start_epoch_1idx = CONFIG["resume_epoch"]
            else:
                start_epoch_1idx = ckpt["epoch"] + 1

            if start_epoch_1idx < 1:
                raise ValueError(
                    f"resume_epoch must be >= 1 (1-indexed); got {start_epoch_1idx}"
                )
            if start_epoch_1idx > CONFIG["num_epochs"]:
                raise ValueError(
                    f"resume_epoch={start_epoch_1idx} > num_epochs={CONFIG['num_epochs']}; "
                    "nothing to train"
                )
            start_epoch = start_epoch_1idx - 1

            rank0_print(
                dist_ctx,
                f"[main] resumed from {CONFIG['resume_from']} | "
                f"start_epoch={start_epoch_1idx}/{CONFIG['num_epochs']} | "
                f"best_loss={best_loss_init:.6f} | "
                f"run_tag={run_tag} | "
                f"LR(after sched load)={optimizer.param_groups[0]['lr']:.2e}",
            )

        if dist_ctx["is_rank0"]:
            tb_run_dir = os.path.join(CONFIG["tb_dir"], run_tag)
```

- [ ] **Step 4: Pass new kwargs to `train_model`**

Find the `train_model(...)` call at [model/main.py:367-387](../../../model/main.py#L367):

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
            channel_weights=CONFIG.get("loss_channel_weights"),
            ema_decay=CONFIG.get("ema_decay"),
            state_channels=state_channels,
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
        )
```

Replace with:

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
            channel_weights=CONFIG.get("loss_channel_weights"),
            ema_decay=CONFIG.get("ema_decay"),
            state_channels=state_channels,
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
            start_epoch=start_epoch,
            best_loss_init=best_loss_init,
            resume_ema_shadow=resume_ema_shadow,
            last_checkpoint_path=last_checkpoint_name,
            run_tag=run_tag,
        )
```

- [ ] **Step 5: Smoke-import `main`**

Run:

```bash
python -c "import sys; sys.path.insert(0, 'model'); import main; print(main.CONFIG['resume_from'], main.CONFIG['resume_epoch'])"
```

Expected: `None None`

- [ ] **Step 6: Commit**

```bash
git add model/main.py
git commit -m "feat(main): wire resume_from and resume_epoch with last.pt loading"
```

---

### Task 5: Round-trip integration test (Python-level)

Verify that save + load actually preserves training state across a `train_model` invocation boundary.

**Files:**
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the round-trip test**

Append to `tests/test_train.py`:

```python
def test_train_model_save_then_resume_roundtrip(tmp_path):
    """End-to-end: train 2 epochs, save last.pt, resume for 2 more.

    Verifies:
    - The resumed run runs exactly (num_epochs - start_epoch) more epochs.
    - The resumed run loads EMA shadow correctly (no NaN crash, evaluates OK).
    - The final last.pt has epoch=num_epochs (full run completed).
    """
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    coords = torch.rand(6, 2)
    last_path = tmp_path / "last.pt"

    # --- Run 1: train 2 epochs, save last.pt ---
    model_a = _TrainToyModel(in_channels, bundle_size, num_channels)
    opt_a = _CountingSGD(model_a.parameters(), lr=1e-2)
    sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.5)
    train_model(
        model_a,
        _FixedLoader(train_batches),
        _FixedLoader(val_batches),
        num_epochs=2,
        device=torch.device("cpu"),
        optimizer=opt_a,
        scheduler=sched_a,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        ema_decay=0.5,
        last_checkpoint_path=str(last_path),
        run_tag="roundtrip-run",
    )
    assert opt_a.step_calls == 2 * len(train_batches)
    assert last_path.exists()
    ckpt = torch.load(last_path, map_location="cpu")
    assert ckpt["epoch"] == 2

    # --- Run 2: resume from last.pt, train 2 more epochs ---
    model_b = _TrainToyModel(in_channels, bundle_size, num_channels)
    model_b.load_state_dict(ckpt["model"])
    opt_b = _CountingSGD(model_b.parameters(), lr=1e-2)
    opt_b.load_state_dict(ckpt["optimizer"])
    sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.5)
    sched_b.load_state_dict(ckpt["scheduler"])
    train_model(
        model_b,
        _FixedLoader(train_batches),
        _FixedLoader(val_batches),
        num_epochs=4,
        device=torch.device("cpu"),
        optimizer=opt_b,
        scheduler=sched_b,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        ema_decay=0.5,
        start_epoch=2,
        best_loss_init=ckpt["best_loss"],
        resume_ema_shadow=ckpt["ema_shadow"],
        last_checkpoint_path=str(last_path),
        run_tag=ckpt["run_tag"],
    )

    # Only 2 more epochs should have run on the second invocation
    assert opt_b.step_calls == 2 * len(train_batches)

    # last.pt should now reflect full run (epoch=4)
    ckpt2 = torch.load(last_path, map_location="cpu")
    assert ckpt2["epoch"] == 4
    assert ckpt2["run_tag"] == "roundtrip-run"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_train.py::test_train_model_save_then_resume_roundtrip -v`

Expected: PASS.

- [ ] **Step 3: Run the full test suite to catch regressions**

Run: `pytest tests/ -v`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_train.py
git commit -m "test(train): roundtrip save-then-resume preserves epoch and EMA state"
```

---

### Task 6: Manual smoke verification (no code changes)

- [ ] **Step 1: Verify `main.py` constructs the resume path correctly without a real checkpoint**

Run:

```bash
python <<'PY'
import sys
sys.path.insert(0, "model")
import main
# default config: resume_from=None
assert main.CONFIG["resume_from"] is None
assert main.CONFIG["resume_epoch"] is None
print("CONFIG defaults OK")
PY
```

Expected: `CONFIG defaults OK`.

- [ ] **Step 2: Verify the `last_<name>.pt` filename derivation**

Run:

```bash
python <<'PY'
import sys
sys.path.insert(0, "model")
from temporal_utils import build_checkpoint_name, channels_suffix, parse_channels
ch = channels_suffix(parse_channels("uvh"))
name = build_checkpoint_name(8, ch)
print("best:", name)
print("last:", name.replace("best_", "last_", 1))
PY
```

Expected:

```
best: best_geofno_b8.pt
last: last_geofno_b8.pt
```

- [ ] **Step 3: (Manual on training machine) end-to-end interrupt+resume sanity check**

Document for the user. Not executed automatically:

1. Edit CONFIG: `num_epochs=4`, `resume_from=None`.
2. Run `python model/main.py` (or `torchrun ...`); let it complete 2 epochs; Ctrl-C.
3. Confirm `last_geofno_b8.pt` exists and `runs/GeoFNO_b8_*/` contains TB events for 2 epochs.
4. Edit CONFIG: `resume_from="last_geofno_b8.pt"`, keep `num_epochs=4`, leave `resume_epoch=None`.
5. Re-run. Verify the printout includes `[main] resumed from last_geofno_b8.pt | start_epoch=3/4`.
6. Open TensorBoard and confirm `train/loss_step` and `val/rel_l2` are continuous in the original run directory (no second run created).

- [ ] **Step 4: No commit needed (verification only).**

---

## Self-Review Notes

- **Spec coverage:**
  - CONFIG `resume_from` + `resume_epoch` → Task 4 step 1
  - Atomic write with `.tmp` + `os.replace` → Task 3 step 6
  - Payload structure (7 keys) → Task 3 step 6 + Task 3 step 1 test
  - `unwrap_model` for DDP → Task 4 step 2 import + step 3 usage
  - `map_location=device` → Task 4 step 3
  - 1-indexed `resume_epoch` validation (`< 1`, `> num_epochs`) → Task 4 step 3
  - TB log dir continuity via `run_tag` override → Task 4 step 3 (overwrites `run_tag` local)
  - EMA shadow seed → Task 2
  - `start_epoch` loop range + `global_step` recompute → Task 1 + Task 3 step 5
  - Skip `last_checkpoint_path=None` → Task 3 step 6 (guarded `if`) + Task 3 step 1 third test
  - Skip `scheduler` save/load when `None` → Task 3 step 6 (`scheduler.state_dict() if scheduler is not None else None`)
  - Skip `ema_shadow` save/load when EMA disabled → Task 3 step 6 (`ema.shadow.state_dict() if ema is not None else None`)
  - Round-trip integration test → Task 5
- **Placeholder scan:** All code blocks are concrete; no TBDs. The two manual verification steps in Task 6 step 3 are documentation, not implementation steps.
- **Type consistency:** new params `start_epoch: int`, `best_loss_init: float`, `resume_ema_shadow: dict | None`, `last_checkpoint_path: str | None`, `run_tag: str | None` referenced identically in Task 1, 2, 3, 4. The payload dict has the same 7 keys across Task 3 step 1 (assertions), Task 3 step 6 (write), Task 4 step 3 (read).
