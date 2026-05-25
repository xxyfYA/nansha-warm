# Geo-FNO h-Priority Optimization

Date: 2026-05-21

## 背景

当前 Geo-FNO 模型（[model/model.py](../../../model/model.py)）以 `channels="uvh"` 训练，验证集表现：

- `val/rel_l2 = 0.215`（bundle 平均），训练 161/200 epoch 后近平台
- 训练-验证 gap ≈ 0.0015，**模型欠拟合**（无过拟合，但训练集也学不动了）
- 每次 StepLR 阶梯（epoch 50/100/150）后立刻饱和——LR 调度在主导收敛
- 自回归 rollout 漂移：step 1 wl rel_l2=0.066 → step 50=0.234（[geofno_autoregressive_results_normalized.txt](../../../geofno_autoregressive_results_normalized.txt)）

按甲方需求，代理模型的核心交付物是**水位 h** 的预报，u, v 可丢。当前 rel_h=0.111 已优于 rel_u=0.27、rel_v=0.24 一倍以上，但 u, v 在混合 rel_l2 损失里主导了梯度，h 的潜力未被榨干。

本设计在**不动模型结构**的前提下，通过 4 个相互独立的小改动榨干现有 Geo-FNO 在 h 上的性能：
- 调度器换 cosine（已有代码）
- 模型扩容 `width 32→48`、`fc1 hidden 128→256`
- Per-channel 加权 rel_l2 loss（h 权重 5x）
- EMA 权重

明确**不做** pushforward / 多步链式训练（用户先前实验提升有限）。

## 目标与非目标

**目标**
- 把 val rel_h（bundle 平均）从 0.111 压到 ~0.075（-32%）
- 把自回归 step 50 wl rel_l2 从 0.234 压到 ~0.16（-30%）
- 训练成本控制在 ~2.0x 当前（width 扩容主导）
- 改动可独立 ablate，每一项失败可单独回退
- 与现有 `channels` 子集训练、DDP、`accum_steps>1` 完全兼容

**非目标**
- 不引入 pushforward / 多步 rollout 训练（用户既有结论）
- 不改 IPHI、不改 SpectralConv2d 内部结构、不改 residual delta 形式
- 不动 dataset 接口、`bundle_size`、特征构造
- 不引入 LayerNorm / Dropout / residual skip
- 不调 `s1, s2` 网格分辨率（保持 64×64）
- 不引入数据增强 / 课程学习

## 设计概览

四个独立改动，按代码改动量从小到大排列：

| # | 改动 | 入口 | 改动量 |
|---|---|---|---|
| 1 | `scheduler="cosine"`（含 warmup） | CONFIG | 1 行 |
| 2 | `width=48`、`num_epochs=200` 保持 | CONFIG | 1 行 |
| 3 | `fc1_hidden` 128→256，提升为可配置参数 | model.py + main.py | ~8 行 |
| 4 | Per-channel 加权 rel_l2 loss | train.py + main.py | ~40 行 |
| 5 | EMA 权重 | train.py + main.py | ~50 行 |

总改动 ≈ 100 行。所有改动**向后兼容**——不提供新 CONFIG 字段时退化到当前行为。

## 详细设计

### 1. Cosine 调度（CONFIG-only）

代码已实现于 [model/scheduler.py:14-46](../../../model/scheduler.py#L14)，由 [main.py:317-326](../../../model/main.py#L317) 调用。

CONFIG 改动：

```python
"scheduler": "cosine",      # 从 "steplr" → "cosine"
# 现有默认值已经合理：warmup_ratio=0.05, min_lr_ratio=0.01
```

成本：0%。预期收益：3-8%（FNO 模型上 cosine vs StepLR 的经验值）。

### 2. width 32 → 48

CONFIG 改动：

```python
"width": 48,               # 从 32 → 48
# modes、s1、s2、num_fno_layers 保持
```

每个 SpectralConv2d 参数从 `32² × 16² × 4 = 1.05M` 涨到 `48² × 16² × 4 = 2.36M`。模型总参数 5.3M → ~11M。

GeoFNO2d 构造函数 [model/model.py:243-322](../../../model/model.py#L243) 不需改——`width` 已经是参数。`iphi` 内部 width 保持 32（独立的坐标 MLP，不需与 FNO 主干同步）。

成本：~2x（spectral 乘法是 width² 主导）。

### 3. `fc1_hidden` 128 → 256（顺便变可配置）

当前 [model/model.py:314](../../../model/model.py#L314) 硬编码：

```python
self.fc1 = nn.Linear(self.width, 128)
self.fc2 = nn.Linear(128, out_channels)
```

改为可配置（默认 128，向后兼容）：

```python
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
    fc1_hidden: int = 128,         # 新增
):
    ...
    self.fc1 = nn.Linear(self.width, fc1_hidden)
    self.fc2 = nn.Linear(fc1_hidden, out_channels)
```

CONFIG 新增：

```python
"fc1_hidden": 256,
```

main.py 在 [GeoFNO2d 构造处 main.py:290-300](../../../model/main.py#L290) 增加传参：

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
    fc1_hidden=CONFIG["fc1_hidden"],   # 新增
).to(device)
```

成本：约增加 `width × 256 + 256 × out_channels = 48 × 256 + 256 × 24 = 18.4K` 参数（相对 11M 主体可忽略）。

### 4. Per-channel 加权 rel_l2 loss

#### 4.1 设计原则

**按通道名字（u/v/h）查权重，不按位置查**——保证与 `channels` 参数完全兼容：

- `channels="uvh"` + 权重 `{u:1, v:1, h:5}` → `(rel_u + rel_v + 5·rel_h) / 7`
- `channels="h"` + 权重 `{u:1, v:1, h:5}` → 只有 h 在 state_channels 里，结果 `5·rel_h / 5 = rel_h`，自然退化
- `channels="uv"` → `(rel_u + rel_v) / 2`
- `channels="uvh"` + 权重 `{u:0, v:0, h:1}` → 等价于"只算 h"（最激进）

权重总和归一化，使损失数量级与单通道 rel_l2 相当（不影响梯度 clip、初始 LR 的有效范围）。

#### 4.2 实现位置

[model/train.py](../../../model/train.py) 已经有一个 per-channel 助手 [_channel_rel_l2 (train.py:71-77)](../../../model/train.py#L71)，evaluate_model 复用之。新增训练用的加权版本：

```python
# 文件顶部 import
from temporal_utils import CHANNEL_ORDER

def weighted_rel_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    state_channels: tuple[int, ...],
    channel_weights: dict[str, float] | None,
) -> torch.Tensor:
    """Per-channel rel-L2 with name-based channel weighting.

    Args:
        pred:           (B, T, N, C_local) — model output for selected state_channels
        target:         same shape as pred
        state_channels: original (u,v,h) indices, sorted ascending unique
        channel_weights: e.g. {"u":1.0, "v":1.0, "h":5.0}. None or missing entries
                         fall back to 1.0. Weight 0.0 drops that channel from loss.

    Returns:
        scalar loss (sum_c w_c * rel_l2_c) / sum_c w_c
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
            f"weighted_rel_l2_loss: zero total weight for state_channels={state_channels}, "
            f"channel_weights={channel_weights}"
        )
    return total_loss / total_weight
```

注意 `_channel_rel_l2` 当前的实现里 `(num / den).mean()` 已经做了 batch 平均；我们直接复用它，不做二次平均。

#### 4.3 train_model 接入

[train_model 签名 (train.py:164-182)](../../../model/train.py#L164) 增加参数 `channel_weights`：

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
    channel_weights: dict[str, float] | None = None,   # 新增
    # ...EMA 参数见 §5
):
```

[损失计算处 (train.py:229-232)](../../../model/train.py#L229) 改为：

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

**保留 `rel_l2_loss` 不变作为默认 fallback**。`channel_weights=None`（缺省）走旧路径，完全向后兼容。

#### 4.4 CONFIG 新增

[main.py CONFIG (main.py:50-84)](../../../model/main.py#L50) 加：

```python
"loss_channel_weights": {"u": 1.0, "v": 1.0, "h": 5.0},
```

main.py 在调用 train_model 时传入：

```python
train_model(
    ...
    channel_weights=CONFIG.get("loss_channel_weights"),
    ...
)
```

用 `.get()` 保证 CONFIG 不含该键时退化到 `None`，走旧 loss。

#### 4.5 兼容性验证矩阵

| `channels` | state_channels | weights | 实际 loss |
|---|---|---|---|
| `"uvh"` | (0,1,2) | `{u:1,v:1,h:5}` | `(rel_u + rel_v + 5·rel_h) / 7` |
| `"h"` | (2,) | `{u:1,v:1,h:5}` | `5·rel_h / 5 = rel_h` |
| `"uv"` | (0,1) | `{u:1,v:1,h:5}` | `(rel_u + rel_v) / 2` |
| `"uvh"` | (0,1,2) | `{u:0,v:0,h:1}` | `rel_h` |
| 任意 | 任意 | `None` | 旧 `rel_l2_loss`（flat） |
| 任意 | 任意 | `{}` | 所有通道默认 weight=1，等价 mean per-channel rel_l2 |

### 5. EMA 权重

#### 5.1 设计原则

- 维护一个独立 EMA 副本，**decay=0.999**（200 epoch × 62 step ≈ 12.4K step，半衰期约 693 步，覆盖整个训练后期）
- 每次 optimizer.step() 之后更新一次 EMA
- **evaluate 用 EMA 模型**，best checkpoint 保存 EMA 权重
- 原模型继续正常训练，EMA 不参与梯度
- DDP 下：EMA 只在 rank0 维护并保存；其他 rank 跳过

#### 5.2 实现：纯 PyTorch，不引入新依赖

[model/train.py](../../../model/train.py) 顶部新增类：

```python
import copy


class ExponentialMovingAverage:
    """Maintain an EMA shadow copy of a model's parameters.

    Buffers (e.g., k_x1/k_x2/grid in SpectralConv2d) are taken from the live
    model as-is on shadow construction and copied each update so eval matches
    the current normalization / coordinate state.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0,1), got {decay}")
        self.decay = float(decay)
        # 深拷贝同步当前权重；不参与梯度
        self.shadow = copy.deepcopy(unwrap_model(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        live = unwrap_model(model)
        d = self.decay
        for s_param, l_param in zip(self.shadow.parameters(), live.parameters(), strict=True):
            s_param.mul_(d).add_(l_param.detach(), alpha=1.0 - d)
        # 同步 buffers（grid、k_x1/k_x2 都是 register_buffer 出来的常量，但保险起见同步）
        for s_buf, l_buf in zip(self.shadow.buffers(), live.buffers(), strict=True):
            s_buf.copy_(l_buf)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}
```

#### 5.3 train_model 接入

签名再增：

```python
def train_model(
    ...
    ema_decay: float | None = None,    # None → 不启用 EMA
):
```

初始化（在 `for epoch in range(num_epochs)` 之前）。**所有 rank 都维护各自的 EMA shadow**：

```python
ema = None
if ema_decay is not None:
    ema = ExponentialMovingAverage(model, decay=ema_decay)
```

**为什么所有 rank 都维护**：evaluate_model 内部用 `all_reduce` 汇总指标，所有 rank 必须用同一个权重做 forward。如果只在 rank0 维护 EMA 而非 rank0 用 live model，汇总出的指标是混合的，无意义。

**各 rank 的 EMA 是否保持一致**：是。所有 rank 的 live model 由 DDP 同步过梯度，每次 optimizer.step() 后参数完全一致；每个 rank 各自从同一个 live model 用同一个 decay 更新自己的 EMA shadow，结果数学上等价。不需要 all_reduce EMA。

内存代价：每 rank 多一份模型副本（~44MB for 11M params fp32），可接受。

更新（紧跟 [optimizer.step() (train.py:243)](../../../model/train.py#L243) 之后）：

```python
if should_sync:
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if step_per_batch and scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None:
        ema.update(model)              # 新增
    # ... 后续 writer 写入不变
```

#### 5.4 Evaluate 用 EMA 模型

[evaluate 调用处 (train.py:264-271)](../../../model/train.py#L264) 改为：

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

所有 rank 都用各自的 EMA shadow 跑 forward，all_reduce 汇总的指标自然一致。

#### 5.5 Best checkpoint 保存

[checkpoint 保存处 (train.py:300-304)](../../../model/train.py#L300) 改为：

```python
if current_test_loss < best_loss:
    best_loss = current_test_loss
    if is_rank0(dist_ctx):
        save_target = ema.shadow if ema is not None else unwrap_model(model)
        torch.save(save_target.state_dict(), checkpoint_path)
        print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")
```

**Checkpoint 格式不变**——仍是 `state_dict`，下游 test_all.py 等无需改动，直接 load 即可。

#### 5.6 CONFIG 新增

```python
"ema_decay": 0.999,        # None / 缺省 → 不启用 EMA
```

main.py 调用：

```python
train_model(
    ...
    ema_decay=CONFIG.get("ema_decay"),
)
```

## CONFIG 变更总结

[model/main.py:50-84](../../../model/main.py#L50) 改动后：

```python
CONFIG = {
    # ...保持
    "bundle_size": 8,
    "batch_size": 16,
    "num_workers": 4,
    "lru_files_per_worker": 2,

    "modes": 16,
    "width": 48,                  # ← 32 → 48
    "s1": 64,
    "s2": 64,
    "num_fno_layers": 3,
    "fc1_hidden": 256,            # ← 新增（默认 128）

    "num_epochs": 200,            # 保持
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "lr_step_size": 50,
    "lr_gamma": 0.5,
    "scheduler": "cosine",        # ← "steplr" → "cosine"
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.01,
    "grad_clip": 1.0,
    "accum_steps": 1,
    "loss_type": "rel_l2",
    "loss_channel_weights": {"u": 1.0, "v": 1.0, "h": 5.0},   # ← 新增
    "ema_decay": 0.999,           # ← 新增

    "channels": "uvh",
}
```

## 边界与失败处理

- **`channel_weights` 总和为 0**：在 `weighted_rel_l2_loss` 中抛 `ValueError`（避免静默除零）
- **`channel_weights` 缺某通道**：`.get(name, 1.0)` 默认 1.0
- **`ema_decay ∉ [0, 1)`**：构造时抛 `ValueError`
- **DDP + EMA**：所有 rank 各自维护 EMA shadow，数学上一致
- **`channels="h"` + 权重 `{u:1, v:1, h:5}`**：自然退化为 `rel_h`（无 u, v 通道可计算），不报错
- **`channels="uvh"` + 权重 `{u:0, v:0, h:0}`**：触发"总和为 0"分支，抛错（避免误配）
- **训练 resume**：当前 train.py 不支持训练中途 resume；本设计也不引入。`best_*.pt` 保存的是 EMA shadow 的 `state_dict`（与现有格式同），可被 `test_all.py` 等下游直接 load。如未来需要 resume 训练，EMA shadow 需独立保存（YAGNI，先不做）
- **`fc1_hidden` 与现有 checkpoint 不兼容**：因为新模型 fc1, fc2 形状变了，旧 `best_geofno_b8.pt` 无法直接 load。这是预期的，需要全量重训

## 测试计划

### 单元测试 `tests/test_loss_weighted.py`（新文件）

- `channel_weights={u:1, v:1, h:5}` + `state_channels=(0,1,2)`：手算 `(rel_u + rel_v + 5·rel_h) / 7`，对比函数输出
- `channel_weights={u:1, v:1, h:5}` + `state_channels=(2,)`：等价于 `rel_h`
- `channel_weights={u:1, v:1, h:5}` + `state_channels=(0,1)`：等价于 `(rel_u + rel_v) / 2`
- `channel_weights={u:0, v:0, h:1}` + `state_channels=(0,1,2)`：等价于 `rel_h`
- `channel_weights={u:0, v:0, h:0}` + `state_channels=(0,1,2)`：抛 `ValueError`
- `channel_weights=None`：函数不会被调用（由 train_model 上游分支决定），不在此测试范围

### 单元测试 `tests/test_ema.py`（新文件）

- 构造一个 2 参数 toy model，`ema_decay=0.5`，连续 update 三次后手算对比
- EMA shadow 在 update 后 `requires_grad=False`
- EMA shadow 的 buffers 与 live model 同步（修改 live model 的 register_buffer，update 后 shadow 也更新）
- `ema_decay=-0.1` / `1.0` / `2.0` 抛 `ValueError`

### 单元测试 `tests/test_model_fc1_hidden.py`

- `GeoFNO2d(..., fc1_hidden=256)` 构造成功；`fc1.out_features=256`、`fc2.in_features=256`
- 默认 `fc1_hidden=128` 时与现有行为一致
- 前向 shape 不变（仅参数量改变）

### 集成验证

- 短训练（`num_epochs=2`，小数据集）跑完，验证：
  - TB 出现 `train/lr_step` 的 cosine warmup → decay 曲线
  - `val/rel_h` 出现且数值合理
  - best checkpoint 文件生成且 load 成功
  - 注意本机性能不足，测试复杂度要降低
- 用 `channels="h"` + 加权损失跑一个 epoch，验证退化路径不报错

### 性能基线对比

- 同种子、同硬件下，跑：
  - baseline（当前配置）+ 5 epoch
  - 新配置（width=48 + cosine + 加权 + EMA）+ 5 epoch
- 对比 `val/rel_h`、`train/loss_step` 收敛速度
- 验证训练时间比约 2.0x（width 主导）

## YAGNI

- 不引入 pushforward / 多步 rollout 训练
- 不引入 LayerNorm / Dropout / 模型 residual skip
- 不调整 `s1`、`s2`、`modes`、`num_fno_layers`
- 不重写 `rel_l2_loss`（保留作为 fallback）
- 不引入按 timestep 加权（bundle 内 8 步等权）
- 不引入 EMA 的 checkpoint resume 支持（重训时 EMA 重头预热）
- 不引入 EMA decay 的 warmup（前期低 decay、后期高 decay 的曲线）——常量 0.999 已经足够
- 不引入 swa（Stochastic Weight Averaging）作为 EMA 的替代
- 不动 IPHI 的 width 参数（保持 32，独立于 FNO 主干）
- 不引入新的优化器（AdamW 不变）
- 不引入 mixed precision（bf16/fp16）——独立优化，留作下次

## 预期效果（基于 baseline 与 FNO 文献）

| 指标 | 当前 | 方案预估 | 来源 |
|---|---|---|---|
| Val rel_h（bundle 均值） | 0.111 | ~0.075 | width 扩容 + 加权 + EMA |
| Val rel_u | 0.269 | ~0.20 | width 扩容主导（u 仍受监督） |
| Val rel_v | 0.241 | ~0.18 | width 扩容主导 |
| 单步 wl rel_l2 | 0.066 | ~0.045 | 同 rel_h 趋势 |
| 自回归 step 50 wl rel_l2 | 0.234 | ~0.16 | 单步基底变小 + 加权偏向 h |
| 自回归 step 72 wl rel_l2 | ~0.25 | ~0.18 | 同上 |
| 训练时长 | 1x | ~2.0x | width=48 |
| 参数量 | 5.3M | ~11M | width=48 + fc1_hidden=256 |

预估的不确定性主要来自加权损失的实际收益——这在不同 PDE 上变化较大。如果加权完全无效，至少有 width 扩容和 cosine 这两项的兜底（合计 +15-25%）。
