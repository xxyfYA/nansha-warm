# 训练 Resume：支持中断后续训

Date: 2026-05-21

## 背景

`model/train.py` 当前只在验证集指标变好时保存 `best_<name>.pt`，且只保存 `state_dict()`（EMA 优先）。`model/main.py` 每次启动都从头训练，没有任何加载逻辑：

- best checkpoint 保存位置：[model/train.py:357-362](../../../model/train.py#L357)
- 每次启动用时间戳建新 TB run：[model/main.py:200-205](../../../model/main.py#L200)
- 调度器 step、optimizer 状态、epoch 计数器、EMA shadow 全部不持久化

用户需要在 A 机器中断训练，换到 B 机器后继续从已完成 epoch 之后接着训。同 GPU 数量场景下，要求接近 bit-exact 续训（除 RNG 状态外）。

## 目标与非目标

**目标**
- 每 epoch 末尾保存一个 `last_<name>.pt`（覆盖式，原子写）
- 通过 CONFIG 字段 `resume_from` 加载 last checkpoint，恢复 model / optimizer / scheduler / EMA shadow / best_loss / TB run 目录
- 通过 CONFIG 字段 `resume_epoch` 显式指定续训起点（1-indexed，对齐打印的 `Epoch X/N`）；未设置时 fallback 到 checkpoint 中存的 `epoch + 1`
- 续训时 TensorBoard 写回原 run 目录，曲线连续
- 现有 `best_<name>.pt` 写入逻辑不变（保留向后兼容）

**非目标**
- 不支持跨 world size 续训（用户确认同 GPU 数）
- 不保存/恢复 RNG 状态（用户选了"不校验 CONFIG"，意味不强求 bit-exact）
- 不实现按 step 保存（每 epoch 足够）
- 不保留历史 N 个 last checkpoint（覆盖式即可）
- 不校验 resume 时 CONFIG 是否与 checkpoint 一致（用户明确选择）
- 不引入 `model/checkpoint.py` 单独模块，逻辑内联在 `main.py` 与 `train.py`

## 设计概览

resume 由两个独立配置项控制：

- `resume_from`：checkpoint 文件路径，为 `None` 时从头训练
- `resume_epoch`：显式指定起始 epoch（1-indexed），为 `None` 时使用 checkpoint 中的 `epoch + 1`

加载逻辑放在 `main.py` 中 model/optimizer/scheduler 构造完成、`SummaryWriter` 创建之前。保存逻辑放在 `train.py` 的 `train_model` 中，每 epoch 末由 rank0 写入。

### Epoch 编号约定

代码内部沿用 0-indexed（`for epoch in range(num_epochs)`），打印用 `epoch + 1`（1-indexed，"Epoch X/N"）。`resume_epoch` 参数采用**用户视角的 1-indexed**，与打印对齐：

- 上次跑到 "Epoch 10/200" 崩了，checkpoint 保留到 epoch 9
  - `resume_epoch=10` → 重跑 epoch 10（打印 "Epoch 10/200"）
  - `resume_epoch=11` → 跳过 epoch 10
  - 不设 → fallback 到 `ckpt["epoch"] + 1` = 10

转换：内部 `start_epoch = resume_epoch_1idx - 1`，循环 `for epoch in range(start_epoch, num_epochs)`。

## 详细设计

### 1. CONFIG 新增字段（model/main.py）

在 [model/main.py:50-87](../../../model/main.py#L50) 的 CONFIG dict 中新增：

```python
"resume_from": None,     # checkpoint 文件路径；None 表示从头训
"resume_epoch": None,    # 显式起始 epoch（1-indexed，对齐 "Epoch X/N" 打印）
                         # None 时 fallback 到 checkpoint 中存的 epoch+1
```

### 2. Checkpoint 文件

**命名**：与 best 文件并列，命名规则一致。`build_checkpoint_name` 返回 `best_geofno[_b{N}][_ch{...}].pt`（默认 channels="uvh" 时无 `_ch` 后缀），new last 文件由 `main.py` 通过字符串替换得到：`last_checkpoint_name = checkpoint_name.replace("best_", "last_", 1)`。当前默认配置下即 `best_geofno_b8.pt` ↔ `last_geofno_b8.pt`。

**Payload 结构**（dict）：

```python
{
    "epoch": int,                       # 已完成的 epoch 数（0-indexed 的 epoch+1）
    "best_loss": float,                 # 历史最佳验证指标
    "model": state_dict,                # unwrap 后的 live 模型
    "optimizer": state_dict,
    "scheduler": state_dict | None,     # None 仅当 scheduler 自身为 None
    "ema_shadow": state_dict | None,    # ema_decay=None 时为 None
    "run_tag": str,                     # TB 日志目录的标识（用于续写同一 run）
}
```

**不存** `global_step`：续训时由 `start_epoch * optimizer_steps_per_epoch` 重新推导，确保 `resume_epoch` 跳跃时 TB step 也对应跳跃。

**原子写**：先写到 `last_checkpoint_path + ".tmp"`，再用 `os.replace()` 重命名。防止中途 kill 留下损坏文件。

### 3. `model/main.py` 改动

#### (a) 顶部：加 `unwrap_model` 的本地引用

`main.py` 当前没用到 `unwrap_model`，加载 model state 时需要。从 `train` 导入：

```python
from train import train_model, unwrap_model
```

#### (b) 在 model/optimizer/scheduler 构造之后、`SummaryWriter` 之前插入加载块

位置：[model/main.py:344](../../../model/main.py#L344) 之后（scheduler 打印之后、`if dist_ctx["is_rank0"]: tb_run_dir = ...` 之前）。

```python
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
    run_tag = ckpt["run_tag"]   # 覆盖前面用时间戳生成的 run_tag

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
```

要点：
- `map_location=device` 让 checkpoint 加载到当前 device，跨机/跨卡号都安全
- `unwrap_model` 处理 DDP wrap 后的 `.module` 嵌套
- 显式打印 LR 让用户能看到 scheduler step 与 resume_epoch 的关系（手动跳 epoch 时 LR 不会跟着跳）

#### (c) `train_model(...)` 调用追加参数

```python
train_model(
    ...
    start_epoch=start_epoch,
    best_loss_init=best_loss_init,
    resume_ema_shadow=resume_ema_shadow,
    last_checkpoint_path=last_checkpoint_name,
    run_tag=run_tag,
)
```

### 4. `model/train.py` 改动

#### (a) 新增 import

```python
import os
```

#### (b) `train_model` 签名新增 5 个参数

```python
def train_model(
    ...
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
    step_per_batch: bool = False,
    # 新增（带默认值，保持向后兼容）：
    start_epoch: int = 0,
    best_loss_init: float = float("inf"),
    resume_ema_shadow: dict | None = None,
    last_checkpoint_path: str | None = None,
    run_tag: str | None = None,
):
```

#### (c) 替换 `best_loss` 与 `global_step` 初始化

原代码（[model/train.py:240-242](../../../model/train.py#L240)）：

```python
global_step = 0
best_loss = float("inf")
```

改为：

```python
global_step = start_epoch * (len(train_loader) // accum_steps)
best_loss = best_loss_init
```

#### (d) 创建 EMA 后加载 shadow

原代码（[model/train.py:236-238](../../../model/train.py#L236)）创建 EMA 之后加：

```python
if ema is not None and resume_ema_shadow is not None:
    ema.shadow.load_state_dict(resume_ema_shadow)
```

#### (e) 替换循环起点

原代码（[model/train.py:244](../../../model/train.py#L244)）：

```python
for epoch in range(num_epochs):
```

改为：

```python
for epoch in range(start_epoch, num_epochs):
```

#### (f) 在 epoch 末（best checkpoint 写入逻辑之后、`barrier_if_distributed` 之前）插入 last checkpoint 保存

位置：[model/train.py:362](../../../model/train.py#L362)（best save 块之后）。

```python
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
```

`last_checkpoint_path=None` 时跳过——用于不传 resume 相关参数的旧调用兼容（理论上 main.py 总会传，但作为防御性默认）。

### 5. TensorBoard 复用

由于 resume 时 `run_tag` 被 checkpoint 中的值覆盖，[main.py:347](../../../model/main.py#L347) 的 `tb_run_dir = os.path.join(CONFIG["tb_dir"], run_tag)` 自动指向已存在的目录。`SummaryWriter` 默认是追加写入，event 文件叠加在同一目录。`global_step` 从 `start_epoch * steps_per_epoch` 起算与已写入数据对齐。

不需要任何额外代码。

### 6. 边界与失败处理

| 场景 | 行为 |
|------|------|
| `resume_from` 文件不存在 | `torch.load` 抛 `FileNotFoundError`，直接传播给用户 |
| `resume_epoch > num_epochs` | `main.py` 显式 `raise ValueError`，提示用户调整 num_epochs |
| `resume_epoch < 1`（含 0 与负数）| `main.py` 显式 `raise ValueError`，1-indexed 起步必须 >= 1 |
| Checkpoint 中无 `ema_shadow` 但当前启用 EMA | EMA shadow 由 `deepcopy(live)` 初始化（与冷启动等价），静默 |
| Checkpoint 中有 `ema_shadow` 但当前禁用 EMA | 静默忽略 `resume_ema_shadow`（已通过 `if ema is not None` 短路） |
| DDP 加载 | 所有 rank 各自读同一文件，`map_location=device` 安全；`unwrap_model` 处理 `.module` |
| `last.pt.tmp` 残留 | 不主动清理；下次写入会覆盖。不影响正确性 |
| Scheduler 是 `LambdaLR` | `load_state_dict` 恢复 `last_epoch` 与 `_last_lr`；lambda 闭包用新 CONFIG 重建，若 `num_epochs` 改了，cosine 曲线相应拉伸 |
| 同名 best.pt 已存在 | 不影响，best 逻辑完全独立 |

### 7. 测试计划

**单元测试** `tests/test_resume.py`（新增）：

- 构造一个简易 model + optimizer + scheduler，保存 checkpoint，重新构造同结构对象，加载 checkpoint，断言：
  - model state_dict 字段一一相等
  - optimizer state_dict 字段一一相等（特别是 AdamW 的 `exp_avg` / `exp_avg_sq`）
  - scheduler `last_epoch` 与 `_last_lr` 相等
- 原子写：手动将 `.tmp` 文件留下，验证下次 `torch.save` + `os.replace` 仍正常覆盖
- `resume_epoch` 解析：
  - 不设 `resume_epoch` → 起始 epoch = ckpt["epoch"] + 1
  - 显式设置 `resume_epoch=N` → 起始 epoch = N - 1
  - `resume_epoch=0` → 抛 ValueError
  - `resume_epoch > num_epochs` → 抛 ValueError

**集成验证**（手动）：

- 用 `num_epochs=4` 跑一次完整训练，记录 TB 曲线作为基线
- 用 `num_epochs=2` 跑半截，停止；再用 `resume_from=last_*.pt, num_epochs=4` 跑后半截
- 比较两次 TB 曲线在 epoch 2-3 之间应该接近（仅 DataLoader shuffle 顺序因为 sampler.set_epoch 是确定性的，可对齐）
- 检查 `runs/<run_tag>/` 下只有一个 run，曲线连续无断裂

### 8. 使用方式（用户视角）

**初次训练**（不变）：
```python
"resume_from": None,
```

**中断后续训**：
```python
"resume_from": "last_geofno_b8.pt",
"resume_epoch": None,    # 自动接续
```

**手动重跑某个 epoch**：
```python
"resume_from": "last_geofno_b8.pt",
"resume_epoch": 10,      # 从 Epoch 10/N 重跑
```

### 9. 不做的事（YAGNI）

- 不引入 CLI 参数（保持 CONFIG-only 风格）
- 不引入 checkpoint 历史保留（覆盖式即可）
- 不引入 RNG 状态保存
- 不引入 CONFIG diff 校验
- 不引入跨 world size 的 optimizer state reshape
- 不抽出 `model/checkpoint.py` 单独模块
