# LR Scheduler: Warmup + Cosine Decay Option

Date: 2026-05-20

## 背景

`model/main.py` 当前固定使用 `StepLR`（`lr_step_size=50, lr_gamma=0.5`），在 `train.py` 的每个 epoch 末尾调用一次 `scheduler.step()`。本设计新增一个 warmup + cosine decay 选项，并通过配置参数在两种调度器之间切换。

- 现有 StepLR 入口：[model/main.py:314-322](../../../model/main.py#L314)
- 现有 step 位置：[model/train.py:252-253](../../../model/train.py#L252)

## 目标与非目标

**目标**
- 新增 cosine decay + 线性 warmup（默认 5% warmup，末期 LR = base_lr × 0.01）
- 通过 CONFIG 字段 `scheduler` 在 `"steplr"` 与 `"cosine"` 之间选择
- 保持 StepLR 现有行为（per-epoch、checkpoint 兼容）不变
- 与 DDP、`accum_steps>1` 协同工作

**非目标**
- 不引入 polynomial / plateau / 带 restart 的 cosine 等其他调度器
- 不把 StepLR 改成 per-step（保持原语义）
- 不引入绝对值 `min_lr` 参数（仅 ratio）

## 设计概览

引入一个工厂函数 `build_scheduler`，根据 `name` 返回 `(scheduler, step_per_batch: bool)`。`train.py` 增加一个 `step_per_batch` 开关：
- `step_per_batch=True`：每次 `optimizer.step()` 之后调用 `scheduler.step()`（cosine 走这条路径）
- `step_per_batch=False`：epoch 末尾调用（StepLR 走这条路径，保持现有行为）

Cosine 用 `LambdaLR` 实现（而非 `SequentialLR(LinearLR, CosineAnnealingLR)`），原因：
- 一个 lambda 即可表达 warmup + cosine，逻辑透明
- `SequentialLR` 在 DDP / state_dict 上有过历史 bug，调试成本高
- `LambdaLR` 的 state_dict 行为可靠

## 详细设计

### 1. CONFIG 新增字段（model/main.py）

```python
"scheduler": "steplr",      # "steplr" | "cosine"，默认保持现有行为
"warmup_ratio": 0.05,       # cosine 模式：warmup 步数 = warmup_ratio * 总优化步数
"min_lr_ratio": 0.01,       # cosine 模式：末期 lr = base_lr * min_lr_ratio
# 现有 lr_step_size / lr_gamma 仅在 scheduler="steplr" 时生效
```

### 2. 新文件 `model/scheduler.py`

```python
import math
from torch.optim.lr_scheduler import StepLR, LambdaLR


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
    """Build LR scheduler.

    Returns:
        (scheduler, step_per_batch): step_per_batch=True 表示
        scheduler.step() 应在每个 optimizer.step() 之后调用；
        False 表示在 epoch 末尾调用。
    """
    if name == "steplr":
        return StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma), False

    if name == "cosine":
        if not (0.0 <= warmup_ratio < 1.0):
            raise ValueError(f"warmup_ratio must be in [0,1), got {warmup_ratio}")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0,1], got {min_lr_ratio}")

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

要点：
- `LambdaLR` 的返回值是相对 `base_lr` 的倍数
- warmup 段从 `1/warmup_steps` 线性升到 `1.0`
- cosine 段从 `1.0` 降到 `min_lr_ratio`
- `progress` 截断到 1.0，防止 `total_steps` 估算误差导致越界
- `warmup_ratio=0` 时 `warmup_steps=0`，直接进入 cosine

### 3. `model/main.py` 改动

替换 `StepLR` 的 import 与构造块（[main.py:27](../../../model/main.py#L27), [main.py:314-322](../../../model/main.py#L314)）：

```python
# 顶部：删除 from torch.optim.lr_scheduler import StepLR
from scheduler import build_scheduler

# 在 optimizer_steps_per_epoch 计算之后：
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

`train_model(...)` 调用新增一个参数：

```python
train_model(
    ...
    scheduler=scheduler,
    step_per_batch=step_per_batch,
    ...
)
```

### 4. `model/train.py` 改动

`train_model` 签名新增 `step_per_batch: bool = False`（默认 False 保持旧调用兼容）。

**(a) 内层循环 `optimizer.step()` 之后**（现 [train.py:242-243](../../../model/train.py#L242) 之后）：

```python
optimizer.step()
optimizer.zero_grad(set_to_none=True)

if step_per_batch and scheduler is not None:
    scheduler.step()

if is_rank0(dist_ctx) and writer is not None:
    writer.add_scalar("train/loss_step", loss_unscaled, global_step)
    writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)
global_step += 1
```

**(b) Epoch 末尾**（现 [train.py:252-253](../../../model/train.py#L252)）：

```python
if scheduler is not None and not step_per_batch:
    scheduler.step()
```

### 5. 边界与失败处理

- `accum_steps > 1`：`scheduler.step()` 跟随 optimizer step 调用，而非 micro-batch，因此累加梯度不影响调度
- DDP：所有 rank 见到相同 `optimizer_steps_per_epoch` 与相同 lambda 初始化，per-step 调度结果一致
- 非法 `scheduler` 值：在 `build_scheduler` 中抛 `ValueError`
- `warmup_ratio` 越界、`min_lr_ratio` 越界：同上抛 `ValueError`
- StepLR 路径完全不受影响（包括 checkpoint 中的 scheduler state）

### 6. 测试计划

- 单元测试 `tests/test_scheduler.py`：
  - `scheduler="steplr"` 返回 `(StepLR, False)`，step_size/gamma 正确
  - `scheduler="cosine"` 返回 `(LambdaLR, True)`
  - cosine LR 在 `step=0` 时约等于 `base_lr / warmup_steps`
  - cosine LR 在 `step=warmup_steps-1` 时约等于 `base_lr`
  - cosine LR 在最末 step 时约等于 `base_lr * min_lr_ratio`
  - cosine LR 曲线在 warmup 段单调递增、在 cosine 段单调递减
  - 非法参数（unknown name、warmup_ratio<0、min_lr_ratio>1）抛 `ValueError`
- 集成验证：
  - 跑一个短训练（`num_epochs=2`、小数据），检查 TB 中 `train/lr_step` 曲线形状符合预期

### 7. 不做的事（YAGNI）

- 不引入绝对值 `min_lr` 参数
- 不引入 polynomial decay、cosine with restarts、reduce-on-plateau
- 不改 StepLR 为 per-step（保留语义与 checkpoint 兼容）
- 不引入对 `optimizer.param_groups` 多组不同 lr 的特殊处理（当前只有 AdamW 单组）
