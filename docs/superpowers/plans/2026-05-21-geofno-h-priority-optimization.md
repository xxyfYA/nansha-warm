# Geo-FNO h-Priority Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stack 5 independent improvements that push h prediction quality without restructuring the model: cosine LR, width 32→48, fc1_hidden 128→256, per-channel weighted rel_l2 loss, EMA weights.

**Architecture:** Three component additions in `model/` (model.py gains `fc1_hidden` param; train.py gains `weighted_rel_l2_loss` and `ExponentialMovingAverage`; train_model gains `channel_weights` and `ema_decay` kwargs), plus `main.py` CONFIG flips. All changes are additive and backward-compatible — defaults preserve current behavior.

**Tech Stack:** PyTorch (`nn.Linear`, `torch.linalg.vector_norm`, deepcopy for EMA), pytest.

**Spec:** [docs/superpowers/specs/2026-05-21-geofno-h-priority-optimization-design.md](../specs/2026-05-21-geofno-h-priority-optimization-design.md)

---

## File Structure

- **Modify** `model/model.py` — add `fc1_hidden: int = 128` constructor param; thread to `self.fc1`, `self.fc2`
- **Modify** `model/train.py` — add `weighted_rel_l2_loss()`, `ExponentialMovingAverage` class; add `channel_weights` and `ema_decay` kwargs to `train_model`; route loss + EMA-aware evaluate + EMA-aware best checkpoint
- **Modify** `model/main.py` — CONFIG: flip `scheduler="cosine"`, set `width=48`, add `fc1_hidden=256`, add `loss_channel_weights`, add `ema_decay`; pass new params to `GeoFNO2d` and `train_model`
- **Create** `tests/test_model_fc1_hidden.py` — fc1_hidden configurability
- **Create** `tests/test_loss_weighted.py` — weighted_rel_l2_loss correctness
- **Create** `tests/test_ema.py` — ExponentialMovingAverage class
- **Modify** `tests/test_train.py` — add integration tests for `channel_weights` and `ema_decay` kwargs

---

### Task 1: Add `fc1_hidden` parameter to `GeoFNO2d`

**Files:**
- Create: `tests/test_model_fc1_hidden.py`
- Modify: `model/model.py:244-322`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_fc1_hidden.py`:

```python
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import GeoFNO2d  # noqa: E402


def _build(fc1_hidden=None, num_channels=3, bundle_size=2):
    kwargs = dict(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=num_channels + 5 * bundle_size + 8,
        out_channels=num_channels * bundle_size,
        s1=4,
        s2=4,
        num_fno_layers=1,
        num_channels=num_channels,
    )
    if fc1_hidden is not None:
        kwargs["fc1_hidden"] = fc1_hidden
    return GeoFNO2d(**kwargs)


def test_fc1_hidden_default_128():
    """Default fc1_hidden=128 keeps backward compatibility with prior checkpoints."""
    m = _build()
    assert m.fc1.out_features == 128
    assert m.fc2.in_features == 128


def test_fc1_hidden_configurable_256():
    """Override fc1_hidden=256 propagates to both fc1 and fc2."""
    m = _build(fc1_hidden=256)
    assert m.fc1.out_features == 256
    assert m.fc2.in_features == 256


def test_fc1_hidden_forward_shape_unchanged():
    """Forward pass output shape is independent of fc1_hidden."""
    torch.manual_seed(0)
    B, N, num_channels, bundle_size = 1, 6, 3, 2
    m_default = _build(num_channels=num_channels, bundle_size=bundle_size)
    m_wide = _build(fc1_hidden=64, num_channels=num_channels, bundle_size=bundle_size)
    in_channels = num_channels + 5 * bundle_size + 8
    u = torch.randn(B, N, in_channels)
    x = torch.rand(B, N, 2)
    out_default = m_default(u, x)
    out_wide = m_wide(u, x)
    assert out_default.shape == (B, bundle_size, N, num_channels)
    assert out_wide.shape == (B, bundle_size, N, num_channels)


def test_fc1_hidden_changes_param_count():
    """Increasing fc1_hidden increases parameter count."""
    m_default = _build()
    m_wide = _build(fc1_hidden=256)
    n_default = sum(p.numel() for p in m_default.parameters())
    n_wide = sum(p.numel() for p in m_wide.parameters())
    assert n_wide > n_default
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_fc1_hidden.py -v`
Expected: tests fail with `TypeError: __init__() got an unexpected keyword argument 'fc1_hidden'` (for tests passing fc1_hidden) and pass for the default test (since current default is hardcoded 128).

- [ ] **Step 3: Modify `GeoFNO2d.__init__` to accept `fc1_hidden`**

In `model/model.py`, modify `GeoFNO2d.__init__` signature (currently at line 244) to add the new keyword arg. Replace the signature block and the fc1/fc2 construction:

```python
class GeoFNO2d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width,
        in_channels,
        out_channels,
        s1=40,
        s2=40,
        num_fno_layers: int = 3,
        num_channels: int = 3,
        fc1_hidden: int = 128,
    ):
```

Then in the body (currently at line 314), replace:

```python
        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
```

with:

```python
        # Projection layers
        self.fc1 = nn.Linear(self.width, fc1_hidden)
        self.fc2 = nn.Linear(fc1_hidden, out_channels)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_fc1_hidden.py -v`
Expected: all 4 tests PASS.

Also run the full existing model tests to ensure no regression:

Run: `pytest tests/test_geofno_num_channels.py tests/test_spectralconv.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_fc1_hidden.py model/model.py
git commit -m "feat(model): make fc1 hidden dim configurable

Add fc1_hidden constructor param (default 128, backward-compatible).
Unblocks h-priority optimization plan which needs fc1_hidden=256.
"
```

---

### Task 2: Add `weighted_rel_l2_loss` to `train.py`

**Files:**
- Create: `tests/test_loss_weighted.py`
- Modify: `model/train.py` (add new function near `rel_l2_loss` at line 47)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loss_weighted.py`:

```python
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from train import _channel_rel_l2, weighted_rel_l2_loss  # noqa: E402


def _seeded_pred_target(C: int):
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 5, C)
    target = torch.randn(2, 3, 5, C)
    return pred, target


def test_uvh_full_with_default_weights():
    """channels=uvh, weights {u:1, v:1, h:5} → (rel_u + rel_v + 5*rel_h) / 7."""
    pred, target = _seeded_pred_target(3)
    rel_u = _channel_rel_l2(pred, target, 0).item()
    rel_v = _channel_rel_l2(pred, target, 1).item()
    rel_h = _channel_rel_l2(pred, target, 2).item()
    expected = (1.0 * rel_u + 1.0 * rel_v + 5.0 * rel_h) / 7.0
    actual = weighted_rel_l2_loss(
        pred, target, (0, 1, 2), {"u": 1.0, "v": 1.0, "h": 5.0}
    ).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_h_only_subset_yields_rel_h():
    """channels=h subset (state_channels=(2,)) → loss equals rel_h."""
    pred, target = _seeded_pred_target(1)
    expected = _channel_rel_l2(pred, target, 0).item()
    actual = weighted_rel_l2_loss(
        pred, target, (2,), {"u": 1.0, "v": 1.0, "h": 5.0}
    ).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_uv_subset_uniform_weights():
    """channels=uv → (rel_u + rel_v) / 2."""
    pred, target = _seeded_pred_target(2)
    rel_u = _channel_rel_l2(pred, target, 0).item()
    rel_v = _channel_rel_l2(pred, target, 1).item()
    expected = (rel_u + rel_v) / 2.0
    actual = weighted_rel_l2_loss(
        pred, target, (0, 1), {"u": 1.0, "v": 1.0, "h": 5.0}
    ).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_zero_weight_drops_channel():
    """weights {u:0, v:0, h:1} on uvh → loss equals rel_h."""
    pred, target = _seeded_pred_target(3)
    expected = _channel_rel_l2(pred, target, 2).item()
    actual = weighted_rel_l2_loss(
        pred, target, (0, 1, 2), {"u": 0.0, "v": 0.0, "h": 1.0}
    ).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_missing_key_defaults_to_one():
    """Missing dict entry falls back to weight 1.0."""
    pred, target = _seeded_pred_target(3)
    rel_u = _channel_rel_l2(pred, target, 0).item()
    rel_v = _channel_rel_l2(pred, target, 1).item()
    rel_h = _channel_rel_l2(pred, target, 2).item()
    # Only 'h' specified; u and v default to 1.0
    expected = (1.0 * rel_u + 1.0 * rel_v + 5.0 * rel_h) / 7.0
    actual = weighted_rel_l2_loss(
        pred, target, (0, 1, 2), {"h": 5.0}
    ).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_none_weights_defaults_to_uniform():
    """channel_weights=None → all channels weight 1.0 (mean of per-channel rel)."""
    pred, target = _seeded_pred_target(3)
    rel_u = _channel_rel_l2(pred, target, 0).item()
    rel_v = _channel_rel_l2(pred, target, 1).item()
    rel_h = _channel_rel_l2(pred, target, 2).item()
    expected = (rel_u + rel_v + rel_h) / 3.0
    actual = weighted_rel_l2_loss(pred, target, (0, 1, 2), None).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_all_zero_weights_raises():
    """All weights zero → ValueError, since loss is undefined."""
    pred, target = _seeded_pred_target(3)
    with pytest.raises(ValueError, match="zero total weight"):
        weighted_rel_l2_loss(
            pred, target, (0, 1, 2), {"u": 0.0, "v": 0.0, "h": 0.0}
        )


def test_returns_torch_scalar_with_grad():
    """Result is a 0-d torch tensor that participates in autograd."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 5, 3, requires_grad=True)
    target = torch.randn(2, 3, 5, 3)
    loss = weighted_rel_l2_loss(pred, target, (0, 1, 2), {"u": 1.0, "v": 1.0, "h": 5.0})
    assert loss.ndim == 0
    loss.backward()
    assert pred.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loss_weighted.py -v`
Expected: `ImportError: cannot import name 'weighted_rel_l2_loss' from 'train'`

- [ ] **Step 3: Add `weighted_rel_l2_loss` to `model/train.py`**

In `model/train.py`, add the import (modify the existing line 10 `from temporal_utils import CHANNEL_ORDER` — already present, no change needed).

Add the function right after `rel_l2_loss` (currently at line 47-53). Insert this block:

```python
def weighted_rel_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    state_channels: tuple[int, ...],
    channel_weights: dict[str, float] | None,
) -> torch.Tensor:
    """Per-channel rel-L2 loss with name-based channel weighting.

    Looks up weights by channel name (u/v/h), not by local position, so the
    function is naturally compatible with `channels` subsets:

      - state_channels=(0,1,2) + {u:1,v:1,h:5} → (rel_u + rel_v + 5*rel_h) / 7
      - state_channels=(2,)    + {u:1,v:1,h:5} → 5*rel_h / 5 = rel_h
      - state_channels=(0,1)   + any weights   → (rel_u*w_u + rel_v*w_v) / (w_u+w_v)

    Args:
        pred:            (B, T, N, C_local) — model output for selected state_channels.
        target:          same shape as pred.
        state_channels:  original (u,v,h) indices, sorted ascending unique.
        channel_weights: e.g. {"u": 1.0, "v": 1.0, "h": 5.0}. None or missing
                         entries fall back to 1.0. Weight 0.0 drops that channel.

    Returns:
        Scalar tensor: sum_c (w_c * rel_l2_c) / sum_c w_c.

    Raises:
        ValueError: if all weights for the supplied state_channels are 0.
    """
    if channel_weights is None:
        channel_weights = {}
    total_loss = pred.new_tensor(0.0)
    total_weight = 0.0
    for local_idx, original_idx in enumerate(state_channels):
        name = CHANNEL_ORDER[original_idx]
        w = float(channel_weights.get(name, 1.0))
        if w == 0.0:
            continue
        total_loss = total_loss + w * _channel_rel_l2(pred, target, local_idx)
        total_weight += w
    if total_weight <= 0.0:
        raise ValueError(
            f"weighted_rel_l2_loss: zero total weight for "
            f"state_channels={state_channels}, channel_weights={channel_weights}"
        )
    return total_loss / total_weight
```

Note: `_channel_rel_l2` already exists at train.py:71-77 — no change needed to it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loss_weighted.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_loss_weighted.py model/train.py
git commit -m "feat(train): add per-channel weighted rel_l2 loss

Name-based channel weighting (u/v/h) so the function works with any
'channels' subset. Default fallback weight 1.0 per channel; zero weight
drops a channel from the loss. Raises if all weights sum to zero.
"
```

---

### Task 3: Add `ExponentialMovingAverage` class to `train.py`

**Files:**
- Create: `tests/test_ema.py`
- Modify: `model/train.py` (add class near top)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ema.py`:

```python
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from train import ExponentialMovingAverage  # noqa: E402


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class _ModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.register_buffer("running", torch.zeros(1))


def test_ema_initial_shadow_matches_live():
    """Shadow is a deep copy of live model on construction."""
    m = _TinyModel()
    with torch.no_grad():
        m.weight.fill_(3.0)
    ema = ExponentialMovingAverage(m, decay=0.5)
    assert ema.shadow.weight.item() == 3.0
    # Mutating live does not change shadow
    with torch.no_grad():
        m.weight.fill_(99.0)
    assert ema.shadow.weight.item() == 3.0


def test_ema_decay_05_step_by_step():
    """decay=0.5: shadow = 0.5*shadow_prev + 0.5*live_new."""
    m = _TinyModel()
    ema = ExponentialMovingAverage(m, decay=0.5)
    # shadow starts at 0.0; live=1 → shadow = 0.5*0 + 0.5*1 = 0.5
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)
    assert ema.shadow.weight.item() == pytest.approx(0.5)
    # live=2 → shadow = 0.5*0.5 + 0.5*2 = 1.25
    with torch.no_grad():
        m.weight.fill_(2.0)
    ema.update(m)
    assert ema.shadow.weight.item() == pytest.approx(1.25)


def test_ema_decay_0999_close_to_live_after_many_steps():
    """decay=0.999: after 1000 updates with live=1, shadow ~ 0.632."""
    m = _TinyModel()
    ema = ExponentialMovingAverage(m, decay=0.999)
    with torch.no_grad():
        m.weight.fill_(1.0)
    for _ in range(1000):
        ema.update(m)
    # 1 - 0.999^1000 ≈ 1 - 0.368 = 0.632
    assert ema.shadow.weight.item() == pytest.approx(0.632, abs=0.01)


def test_ema_shadow_params_no_grad():
    """Shadow parameters do not participate in autograd."""
    m = _TinyModel()
    ema = ExponentialMovingAverage(m, decay=0.999)
    for p in ema.shadow.parameters():
        assert p.requires_grad is False


def test_ema_buffers_synced_on_update():
    """Live buffers are copied into shadow on each update."""
    m = _ModelWithBuffer()
    ema = ExponentialMovingAverage(m, decay=0.5)
    with torch.no_grad():
        m.running.fill_(7.0)
    ema.update(m)
    assert ema.shadow.running.item() == 7.0


def test_ema_invalid_decay_raises():
    """decay must be in [0, 1)."""
    m = _TinyModel()
    with pytest.raises(ValueError, match="EMA decay"):
        ExponentialMovingAverage(m, decay=-0.1)
    with pytest.raises(ValueError, match="EMA decay"):
        ExponentialMovingAverage(m, decay=1.0)
    with pytest.raises(ValueError, match="EMA decay"):
        ExponentialMovingAverage(m, decay=1.5)


def test_ema_update_with_ddp_wrapped_unwraps():
    """ema.update should unwrap a DDP-style model (one with .module attr)."""
    class _Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
    m = _TinyModel()
    wrapped = _Wrapped(m)
    ema = ExponentialMovingAverage(wrapped, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(wrapped)
    assert ema.shadow.weight.item() == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ema.py -v`
Expected: `ImportError: cannot import name 'ExponentialMovingAverage' from 'train'`

- [ ] **Step 3: Add `ExponentialMovingAverage` to `model/train.py`**

In `model/train.py`, add at the top (after the existing `from contextlib import nullcontext` import, line 4):

```python
import copy
```

Then insert this class definition before `class RMSELoss` (currently at line 37). The class needs `unwrap_model` which is defined later in the file (line 21-22); since it's at module scope, it's reachable from inside the methods at call time.

```python
class ExponentialMovingAverage:
    """Maintain an EMA shadow copy of a model's parameters.

    The shadow is a `copy.deepcopy` of the live model with all parameters
    detached and `requires_grad=False`. Each `update(model)` does
    `shadow = decay * shadow + (1 - decay) * live` over parameters and
    copies live buffers into shadow (so coordinate / k_x buffers stay in
    sync, even though they are constant in this codebase).

    DDP-safe: each rank constructs its own EMA from the same live params,
    so EMA stays bit-identical across ranks without all-reduce.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        live = unwrap_model(model)
        self.shadow = copy.deepcopy(live).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        live = unwrap_model(model)
        d = self.decay
        for s_param, l_param in zip(
            self.shadow.parameters(), live.parameters(), strict=True
        ):
            s_param.mul_(d).add_(l_param.detach(), alpha=1.0 - d)
        for s_buf, l_buf in zip(
            self.shadow.buffers(), live.buffers(), strict=True
        ):
            s_buf.copy_(l_buf)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ema.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ema.py model/train.py
git commit -m "feat(train): add ExponentialMovingAverage helper

Maintains a deep-copied shadow model that tracks live weights with
decay-based averaging. Each rank holds its own copy; DDP synchrony
is maintained because live params are identical across ranks.
"
```

---

### Task 4: Wire `channel_weights` kwarg into `train_model`

**Files:**
- Modify: `model/train.py` — `train_model` signature and loss block
- Modify: `tests/test_train.py` — new integration tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py` (append at the end):

```python
def test_train_model_uses_weighted_loss_when_channel_weights_provided():
    """train_model accepts channel_weights kwarg and runs without error."""
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
        num_epochs=1,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        coords_2d_device=coords,
        writer=None,
        state_channels=(0, 1, 2),
        step_per_batch=False,
        channel_weights={"u": 1.0, "v": 1.0, "h": 5.0},
    )

    # Optimizer ran on each batch
    assert optimizer.step_calls == len(train_batches)


def test_train_model_uses_flat_rel_l2_when_channel_weights_none():
    """Without channel_weights, train_model retains original rel_l2_loss path."""
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

    # Should not raise; channel_weights defaults to None
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
        step_per_batch=False,
    )
    assert optimizer.step_calls == len(train_batches)
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `pytest tests/test_train.py::test_train_model_uses_weighted_loss_when_channel_weights_provided -v`
Expected: FAIL with `TypeError: train_model() got an unexpected keyword argument 'channel_weights'`

The second new test (`test_train_model_uses_flat_rel_l2_when_channel_weights_none`) should already PASS — it only confirms the existing path still works.

- [ ] **Step 3: Add `channel_weights` param and dispatch loss**

In `model/train.py`, modify the `train_model` signature (currently at lines 164-182). Add the new kwarg at the end, keeping defaults None:

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
    channel_weights: dict[str, float] | None = None,
):
```

Then modify the loss-computation block. The current block (lines 229-232 in `train_model`) reads:

```python
                if loss_type == "rmse":
                    loss = criterion(pred_block, target_block)
                else:
                    loss = rel_l2_loss(pred_block, target_block)
```

Replace it with:

```python
                if loss_type == "rmse":
                    loss = criterion(pred_block, target_block)
                elif channel_weights is not None:
                    loss = weighted_rel_l2_loss(
                        pred_block, target_block, state_channels, channel_weights
                    )
                else:
                    loss = rel_l2_loss(pred_block, target_block)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: all tests PASS, including the two new ones and all existing scheduler tests.

- [ ] **Step 5: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat(train): dispatch weighted rel_l2 loss when channel_weights set

train_model accepts an optional channel_weights kwarg. When provided,
the per-channel weighted loss replaces the flat rel_l2_loss; when None
(default), behavior is unchanged.
"
```

---

### Task 5: Wire `ema_decay` kwarg into `train_model`

**Files:**
- Modify: `model/train.py` — `train_model` signature, EMA init, update, evaluate, checkpoint
- Modify: `tests/test_train.py` — new integration test

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py`:

```python
def test_train_model_with_ema_decay_creates_and_updates_shadow():
    """train_model with ema_decay maintains and uses an EMA shadow model."""
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    # Two batches so the optimizer takes two steps
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-1)
    scheduler = _CountingScheduler()
    coords = torch.rand(6, 2)

    # Snapshot initial live params
    init_params = [p.detach().clone() for p in model.parameters()]

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
        step_per_batch=False,
        ema_decay=0.5,
    )

    # Live params should have moved (optimizer ran)
    moved = any(
        not torch.allclose(p_init, p_now)
        for p_init, p_now in zip(init_params, model.parameters())
    )
    assert moved, "live model params should be updated by optimizer"


def test_train_model_without_ema_decay_skips_ema():
    """ema_decay=None (default) does not create an EMA shadow."""
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

    # Should run without error even though ema_decay is omitted
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
        step_per_batch=False,
    )
    assert optimizer.step_calls == len(train_batches)


def test_train_model_with_ema_saves_ema_shadow_to_checkpoint(tmp_path):
    """When ema_decay is set, the best checkpoint state_dict matches the EMA shadow, not live."""
    torch.manual_seed(0)
    num_channels = 3
    bundle_size = 1
    in_channels = num_channels + 5 * bundle_size + 8
    train_batches = [_make_batch(num_channels, bundle_size) for _ in range(2)]
    val_batches = [_make_batch(num_channels, bundle_size)]
    model = _TrainToyModel(in_channels, bundle_size, num_channels)
    optimizer = _CountingSGD(model.parameters(), lr=1e-1)
    scheduler = _CountingScheduler()
    coords = torch.rand(6, 2)
    ckpt = tmp_path / "best.pt"

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
        step_per_batch=False,
        checkpoint_path=str(ckpt),
        ema_decay=0.5,
    )

    assert ckpt.exists(), "best checkpoint should be written"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Saved should differ from current live params (since EMA != live with decay=0.5)
    live_state = model.state_dict()
    any_diff = any(
        not torch.allclose(saved[k], live_state[k])
        for k in saved.keys()
    )
    assert any_diff, "saved EMA state should differ from live state after training"
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `pytest tests/test_train.py::test_train_model_with_ema_decay_creates_and_updates_shadow -v`
Expected: FAIL with `TypeError: train_model() got an unexpected keyword argument 'ema_decay'`

- [ ] **Step 3: Add `ema_decay` param and wire EMA into the loop**

In `model/train.py`, modify the `train_model` signature again to add `ema_decay` at the end:

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
    channel_weights: dict[str, float] | None = None,
    ema_decay: float | None = None,
):
```

After the existing initialization block (after `x_in_base = ...` near line 195), add EMA initialization:

```python
    ema = None
    if ema_decay is not None:
        ema = ExponentialMovingAverage(model, decay=ema_decay)
```

In the inner training loop, find the optimizer.step() block (currently at lines 240-246 inside `if should_sync:`):

```python
            if should_sync:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if step_per_batch and scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
```

Insert an EMA update right after `optimizer.zero_grad(...)`:

```python
            if should_sync:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if step_per_batch and scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
```

Modify the evaluate_model call (currently at lines 264-271). Replace:

```python
        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
            coords_2d_device,
            state_channels=state_channels,
            dist_ctx=dist_ctx,
        )
```

with:

```python
        eval_model = ema.shadow if ema is not None else model
        test_metrics = evaluate_model(
            eval_model,
            test_loader,
            device,
            coords_2d_device,
            state_channels=state_channels,
            dist_ctx=dist_ctx,
        )
```

Modify the best-checkpoint save block (currently at lines 299-304). Replace:

```python
        current_test_loss = test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                torch.save(unwrap_model(model).state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")
```

with:

```python
        current_test_loss = test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                save_target = ema.shadow if ema is not None else unwrap_model(model)
                torch.save(save_target.state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: all tests PASS, including the three new EMA tests and all existing tests.

- [ ] **Step 5: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat(train): EMA weights for evaluation and best checkpoint

train_model accepts an optional ema_decay kwarg. When set, an EMA
shadow model is maintained alongside live training; evaluate_model
and best-checkpoint save use the shadow. ema_decay=None (default)
preserves prior behavior.
"
```

---

### Task 6: Update `main.py` CONFIG and wire new params

**Files:**
- Modify: `model/main.py` — CONFIG block, `GeoFNO2d` constructor call, `train_model` call

- [ ] **Step 1: Update CONFIG block**

In `model/main.py`, replace the CONFIG block (currently lines 50-84):

```python
CONFIG = {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "coords_path": "data/coordinates.mat",
    "norm_path": "data/normalization.mat",
    "tb_dir": "runs",

    "seed": 42,

    "bundle_size": 8,
    "batch_size": 16,
    "num_workers": 4,
    "lru_files_per_worker": 2,

    "modes": 16,
    "width": 48,
    "s1": 64,
    "s2": 64,
    "num_fno_layers": 3,
    "fc1_hidden": 256,

    "num_epochs": 200,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "lr_step_size": 50,
    "lr_gamma": 0.5,
    "scheduler": "cosine",
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.01,
    "grad_clip": 1.0,
    "accum_steps": 1,
    "loss_type": "rel_l2",
    "loss_channel_weights": {"u": 1.0, "v": 1.0, "h": 5.0},
    "ema_decay": 0.999,

    "channels": "uvh",
}
```

The three changed values are `"width": 48`, `"scheduler": "cosine"`, and the three new keys `"fc1_hidden"`, `"loss_channel_weights"`, `"ema_decay"`.

- [ ] **Step 2: Pass `fc1_hidden` to `GeoFNO2d`**

In `model/main.py`, locate the `GeoFNO2d` construction (currently lines 290-300). Replace it with:

```python
        model = GeoFNO2d(
            modes1=CONFIG["modes"],
            modes2=CONFIG["modes"],
            width=CONFIG["width"],
            in_channels=in_channels,
            out_channels=out_channels,
            s1=CONFIG["s1"],
            s2=CONFIG["s2"],
            num_fno_layers=CONFIG["num_fno_layers"],
            num_channels=num_channels,
            fc1_hidden=CONFIG["fc1_hidden"],
        ).to(device)
```

- [ ] **Step 3: Pass `channel_weights` and `ema_decay` to `train_model`**

In `model/main.py`, locate the `train_model` call (currently lines 362-380). Replace it with:

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
            channel_weights=CONFIG.get("loss_channel_weights"),
            ema_decay=CONFIG.get("ema_decay"),
        )
```

Note: `.get()` is intentional — if a future user removes the key from CONFIG, behavior falls back to None (legacy path).

- [ ] **Step 4: Smoke-import main.py to verify no syntax / import error**

Run: `python -c "import sys; sys.path.insert(0, 'model'); import main; print('OK', main.CONFIG['width'], main.CONFIG['scheduler'], main.CONFIG['fc1_hidden'])"`
Expected: `OK 48 cosine 256`

- [ ] **Step 5: Run full test suite to verify no regression**

Run: `pytest tests/ -v`
Expected: all tests PASS. Existing tests must not break because every new arg has a None / default fallback.

- [ ] **Step 6: Commit**

```bash
git add model/main.py
git commit -m "feat(main): flip CONFIG to h-priority optimization defaults

- scheduler: steplr -> cosine
- width: 32 -> 48
- fc1_hidden: 128 -> 256 (new key)
- loss_channel_weights: {u:1, v:1, h:5} (new key)
- ema_decay: 0.999 (new key)

Pass new params through to GeoFNO2d constructor and train_model call.
All legacy paths remain reachable by clearing the new CONFIG keys.
"
```

---

### Task 7: Integration smoke test on a short run

**Files:**
- No code changes; this task only runs the training script for a few iterations to verify end-to-end behavior.

- [ ] **Step 1: Verify training data manifests exist**

Run: `ls data/train/manifest.json data/val/manifest.json data/test/manifest.json`
Expected: all three exist. If any are missing, regenerate with:

```bash
python scripts/build_manifest.py data/train --bundle_size_warn 8
python scripts/build_manifest.py data/val --bundle_size_warn 8
python scripts/build_manifest.py data/test --bundle_size_warn 8
```

- [ ] **Step 2: Temporarily reduce num_epochs to 1 for smoke test**

Edit `model/main.py` CONFIG: change `"num_epochs": 200` → `"num_epochs": 1`. Do NOT commit this change.

- [ ] **Step 3: Run a 1-epoch training**

Run: `python model/main.py 2>&1 | tee /tmp/smoke.log`
Expected behavior in the log:
- `[main] device=...`, `[main] model params=...` shows ~11M (was 5.3M)
- `[main] Cosine: total_steps=..., warmup_steps=..., min_lr=2.00e-05`
- Epoch 1 finishes; `Test Rel-H` printed
- `-> Saved best model to best_geofno_b8.pt`
- No tracebacks

- [ ] **Step 4: Verify checkpoint loads back**

Run:
```bash
python -c "
import torch, sys
sys.path.insert(0, 'model')
from model import GeoFNO2d
sd = torch.load('best_geofno_b8.pt', map_location='cpu', weights_only=False)
m = GeoFNO2d(modes1=16, modes2=16, width=48, in_channels=51, out_channels=24, s1=64, s2=64, num_fno_layers=3, num_channels=3, fc1_hidden=256)
m.load_state_dict(sd)
print('OK', sum(p.numel() for p in m.parameters()))
"
```
Expected: `OK <param count around 11_000_000>`

- [ ] **Step 5: Revert num_epochs to 200**

Edit `model/main.py` CONFIG: change `"num_epochs": 1` → `"num_epochs": 200`. (The earlier task already wrote 200; this is only to undo the smoke-test edit.)

- [ ] **Step 6: Verify TensorBoard log got new scalars**

Run: `ls runs/ | tail -1`
Expected: a new `GeoFNO_b8_<timestamp>` directory.

(Optional inspection):

```bash
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
latest = sorted(os.listdir('runs'))[-1]
ea = EventAccumulator(f'runs/{latest}'); ea.Reload()
print('scalars:', ea.Tags()['scalars'])
"
```
Expected scalars include `train/loss_step`, `train/lr_step`, `val/rel_h`, `val/rel_u`, `val/rel_v`.

- [ ] **Step 7: No commit needed**

This task only validates behavior; no source change to commit. The smoke-test num_epochs flip is reverted before leaving the working tree clean.

Run: `git status`
Expected: clean tree (or only `runs/`, `best_geofno_b8.pt`, `*.txt` artifacts which are gitignored).

---

## Notes on later kickoff

After all 7 tasks are committed, the full training run is launched separately by the user (not part of this plan):

```bash
python model/main.py
# or DDP:
torchrun --nproc_per_node=4 model/main.py
```

Expected ~2x wall-clock vs prior baseline. Best checkpoint written to `best_geofno_b8.pt`. Compare validation metrics from the new TensorBoard run against the baseline run `runs/GeoFNO_b8_20260520-141017` to confirm the predicted gains (val rel_h ~0.075, step 50 wl rel_l2 ~0.16).

The autoregressive test is run separately via `model/test_all.py` once training completes (existing script, no changes).
