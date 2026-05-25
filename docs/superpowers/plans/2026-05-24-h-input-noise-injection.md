# h 通道输入噪音注入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在训练循环里对 `features[..., 0:1]`（h state 通道）注入 i.i.d. 高斯噪音，强度由 `CONFIG["noise_sigma"]` 控制；默认 0.0 等价于现行行为。

**Architecture:** 在 [model/train.py](../../../model/train.py) 新增一个纯函数 helper `_apply_h_state_noise`，由训练循环每个 micro-batch 调用一次；`train_model` 签名新增 `noise_sigma` 参数并在函数顶部做非负校验；[model/main.py](../../../model/main.py) `CONFIG` 字典新增同名键并透传。dataset、model、scheduler、EMA、DDP、`evaluate_model`、`test_all.py` 完全不动。

**Tech Stack:** PyTorch、pytest。

**关联 Spec:** [docs/superpowers/specs/2026-05-24-h-input-noise-injection-design.md](../specs/2026-05-24-h-input-noise-injection-design.md)

---

## 文件结构

- 修改：[model/train.py](../../../model/train.py)
  - 新增 `_apply_h_state_noise(features, noise_sigma)` 纯函数 helper。
  - `train_model` 签名新增 `noise_sigma: float = 0.0`，顶部新增非负校验；micro-batch 循环里调用 helper。
- 修改：[model/main.py](../../../model/main.py)
  - `CONFIG` 字典新增 `"noise_sigma": 0.0`。
  - 调用 `train_model(...)` 时透传 `noise_sigma=CONFIG["noise_sigma"]`。
- 新建：[tests/test_train_noise.py](../../../tests/test_train_noise.py)
  - 覆盖 helper 的纯函数行为（4 个用例）与 `train_model` 的参数校验（1 个用例）。

---

### Task 1: 在 `train.py` 增加 `_apply_h_state_noise` helper + helper 单测

**Files:**
- Modify: [model/train.py](../../../model/train.py)
- Create: [tests/test_train_noise.py](../../../tests/test_train_noise.py)

- [ ] **Step 1: 先写失败测试 — 创建 `tests/test_train_noise.py`，内容如下**

```python
"""Unit tests for h-channel input noise injection in train.py."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from train import _apply_h_state_noise  # noqa: E402


def _make_features(batch=2, num_nodes=4, channels=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, num_nodes, channels, generator=g)


def test_apply_h_state_noise_zero_sigma_is_noop():
    features = _make_features()
    original = features.clone()
    _apply_h_state_noise(features, noise_sigma=0.0)
    assert torch.equal(features, original)


def test_apply_h_state_noise_negative_sigma_is_noop():
    features = _make_features()
    original = features.clone()
    _apply_h_state_noise(features, noise_sigma=-0.1)
    assert torch.equal(features, original)


def test_apply_h_state_noise_only_touches_h_channel():
    features = _make_features()
    other_channels_before = features[..., 1:].clone()
    _apply_h_state_noise(features, noise_sigma=0.05)
    assert torch.equal(features[..., 1:], other_channels_before)


def test_apply_h_state_noise_magnitude_matches_sigma():
    torch.manual_seed(42)
    features = torch.zeros(64, 512, 9)
    sigma = 0.1
    _apply_h_state_noise(features, noise_sigma=sigma)
    h_after = features[..., 0:1]
    assert h_after.mean().abs().item() < 0.01
    assert abs(h_after.std().item() - sigma) < 0.01
```

- [ ] **Step 2: 运行测试，确认 ImportError**

Run: `pytest tests/test_train_noise.py -v`
Expected: collection error / ImportError，因为 `_apply_h_state_noise` 还不存在。

- [ ] **Step 3: 在 [model/train.py](../../../model/train.py) 的 `rel_l2_loss` 函数之后（约第 75 行）插入 helper**

```python
def _apply_h_state_noise(features: torch.Tensor, noise_sigma: float) -> None:
    """Add i.i.d. Gaussian noise to the h state-in channel (in-place).

    Only ``features[..., 0:1]`` is modified; other channels stay unchanged.
    When ``noise_sigma <= 0`` this is a no-op.
    """
    if noise_sigma <= 0.0:
        return
    noise = torch.randn_like(features[..., 0:1]) * noise_sigma
    features[..., 0:1].add_(noise)
```

- [ ] **Step 4: 再次运行测试，确认全部通过**

Run: `pytest tests/test_train_noise.py -v`
Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add model/train.py tests/test_train_noise.py
git commit -m "feat: add h-state noise injection helper in train.py"
```

---

### Task 2: `train_model` 签名加入 `noise_sigma` + 顶部校验 + 训练循环内调用 helper

**Files:**
- Modify: [model/train.py](../../../model/train.py)
- Modify: [tests/test_train_noise.py](../../../tests/test_train_noise.py)

- [ ] **Step 1: 在 `tests/test_train_noise.py` 文件末尾追加一个用例（先写失败测试）**

```python
from train import train_model  # noqa: E402


def test_train_model_rejects_negative_noise_sigma():
    with pytest.raises(ValueError, match="noise_sigma"):
        train_model(
            model=None,
            train_loader=None,
            test_loader=None,
            num_epochs=0,
            device=None,
            optimizer=None,
            scheduler=None,
            coords_2d_device=None,
            writer=None,
            noise_sigma=-0.1,
        )
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_train_noise.py::test_train_model_rejects_negative_noise_sigma -v`
Expected: FAIL（TypeError: unexpected keyword 'noise_sigma'）。

- [ ] **Step 3: 在 [model/train.py](../../../model/train.py) 修改 `train_model` 签名（约第 137-154 行），在 `accum_steps` 之后新增 `noise_sigma` 参数**

把当前签名

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
    grad_clip=None,
    loss_type: str = "rel_l2",
    ema_decay: float | None = None,
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
):
```

改为

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
    grad_clip=None,
    loss_type: str = "rel_l2",
    ema_decay: float | None = None,
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
    noise_sigma: float = 0.0,
):
```

- [ ] **Step 4: 在 `train_model` 顶部的 `accum_steps` 校验之后（约第 155-156 行）新增 `noise_sigma` 非负校验**

把

```python
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    if loss_type == "rmse":
```

改为

```python
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    if noise_sigma < 0.0:
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")
    if loss_type == "rmse":
```

- [ ] **Step 5: 运行新增的校验测试，确认通过**

Run: `pytest tests/test_train_noise.py::test_train_model_rejects_negative_noise_sigma -v`
Expected: PASS。

- [ ] **Step 6: 在 `train_model` 的 micro-batch 循环里，`features.to(device)` 与 `target_block.to(device)` 之后插入 helper 调用**

在 [model/train.py](../../../model/train.py) 约第 192-198 行，把

```python
        for micro_idx, (features, target_block) in enumerate(pbar):
            if micro_idx >= usable_micro_batches:
                break

            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            batch_size = features.shape[0]
```

改为

```python
        for micro_idx, (features, target_block) in enumerate(pbar):
            if micro_idx >= usable_micro_batches:
                break

            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            _apply_h_state_noise(features, noise_sigma)
            batch_size = features.shape[0]
```

- [ ] **Step 7: 跑全量训练相关测试，确认无回归**

Run: `pytest tests/test_train_noise.py tests/test_ema.py -v`
Expected: 全部通过。

- [ ] **Step 8: Commit**

```bash
git add model/train.py tests/test_train_noise.py
git commit -m "feat: wire noise_sigma into train_model with validation and loop hook"
```

---

### Task 3: `main.py` CONFIG 新增 `noise_sigma` 并透传给 `train_model`

**Files:**
- Modify: [model/main.py](../../../model/main.py)

- [ ] **Step 1: 在 [model/main.py](../../../model/main.py) `CONFIG` 字典（约第 49-80 行）的 `accum_steps` 之后、`loss_type` 之前新增一行**

把

```python
    "grad_clip": 1.0,
    "accum_steps": 1,
    "loss_type": "rel_l2",
    "ema_decay": 0.999,
}
```

改为

```python
    "grad_clip": 1.0,
    "accum_steps": 1,
    "noise_sigma": 0.0,
    "loss_type": "rel_l2",
    "ema_decay": 0.999,
}
```

- [ ] **Step 2: 在 `main()` 调用 `train_model(...)` 处（约第 345-362 行）透传 `noise_sigma`**

把

```python
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            num_epochs=CONFIG["num_epochs"],
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            coords_2d_device=coords_2d_device,
            writer=writer,
            grad_clip=CONFIG["grad_clip"],
            loss_type=CONFIG["loss_type"],
            ema_decay=CONFIG.get("ema_decay"),
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
        )
```

改为

```python
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            num_epochs=CONFIG["num_epochs"],
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            coords_2d_device=coords_2d_device,
            writer=writer,
            grad_clip=CONFIG["grad_clip"],
            loss_type=CONFIG["loss_type"],
            ema_decay=CONFIG.get("ema_decay"),
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
            noise_sigma=CONFIG["noise_sigma"],
        )
```

- [ ] **Step 3: import sanity 验证**

Run:
```bash
cd /Volumes/P7000Z/nansha && PYTHONPATH=model python -c "from main import CONFIG; print(CONFIG['noise_sigma'])"
```
Expected: 输出 `0.0`，无报错。

- [ ] **Step 4: Commit**

```bash
git add model/main.py
git commit -m "feat: expose noise_sigma in main CONFIG and pass through to train_model"
```

---

### Task 4: 全量 pytest + 最终 smoke

**Files:**（不修改任何文件，仅验收）

- [ ] **Step 1: 运行完整 pytest 套件，确认全绿**

Run: `cd /Volumes/P7000Z/nansha && pytest tests/ -v`
Expected: 所有测试通过（含新增的 5 个噪音测试和原有套件）。

- [ ] **Step 2: import sanity（与 spec 验收方式一致）**

Run:
```bash
cd /Volumes/P7000Z/nansha && PYTHONPATH=model python -c "from train import train_model, _apply_h_state_noise; print('ok')"
```
Expected: 输出 `ok`，无报错。

- [ ] **Step 3: 确认默认行为零回归 — 跑一个 0 噪音的最小 fake-data forward+backward smoke**

Run:
```bash
cd /Volumes/P7000Z/nansha && PYTHONPATH=model python - <<'PY'
import torch
from train import _apply_h_state_noise

# σ=0 严格无副作用
features = torch.randn(2, 16, 9)
snapshot = features.clone()
_apply_h_state_noise(features, noise_sigma=0.0)
assert torch.equal(features, snapshot), "σ=0 must be a no-op"

# σ>0 仅动 h 通道
features = torch.zeros(2, 16, 9)
_apply_h_state_noise(features, noise_sigma=0.05)
assert features[..., 1:].abs().sum().item() == 0.0
assert features[..., 0:1].abs().sum().item() > 0.0
print("smoke ok")
PY
```
Expected: 输出 `smoke ok`，无 assertion 失败。

- [ ] **Step 4: 最终状态确认**

Run: `git -C /Volumes/P7000Z/nansha status`
Expected: working tree clean，分支领先 origin/main 7 个 commits（spec 1 + 本计划 3）。

---

## 自检（已完成 — 用户无需复看）

**Spec 覆盖：**
- §2 表「噪音作用通道」/「噪音幅度」/「注入位置」/「目标是否加噪音」/「空间结构」/「应用频率」 → Task 1 helper 实现（i.i.d. Gaussian、仅 h、可选 σ、不动 target）。
- §2「训练期 rollout 验证」 → 明确不做，Task 4 不引入。
- §3.1 `CONFIG["noise_sigma"]` + 透传 → Task 3。
- §3.2 `train_model` 签名 + 校验 + 调用 → Task 2。
- §4 与 EMA/accum/DDP/RNG/checkpoint 的交互 → 通过不修改这些路径自然保留；Task 4 import + 全量 pytest 兜底。
- §5 推理路径不动 → 不在文件结构里出现 `test_all.py`，结构上即满足。
- §6 测试计划全部覆盖：zero/negative no-op、only h channel、magnitude、negative sigma rejected、smoke。
- §7 范围外项均未出现在任务列表。
- §8 回滚（CONFIG 设 0.0 即可） → 自然满足。

**占位符：** 无。

**类型一致：** `_apply_h_state_noise(features: torch.Tensor, noise_sigma: float) -> None` 与 `train_model(..., noise_sigma: float = 0.0)` 在所有任务中保持一致。
