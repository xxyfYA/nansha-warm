# 移除多通道训练支持，硬编码 h-only 设计

**日期**：2026-05-24
**类型**：重构 / 代码简化
**作用域**：`model/`、`tests/`、checkpoint 命名

## 背景

模型当前在 [model/main.py](../../../model/main.py) 中通过 `CONFIG["channels"] = "h"` 配置只学习水位通道 h；底层 `dataset.py` / `train.py` / `model.py` / `temporal_utils.py` / `test_all.py` 通过 `state_channels`、`num_channels`、`parse_channels`、`channels_suffix` 等参数化抽象支持 u/v/h 任意子集训练。该抽象由 [2026-05-18-channel-subset-training-design.md](2026-05-18-channel-subset-training-design.md) 引入。

经过实验后已确认 **h-only 是最终方案**，u/v 通道的训练 / 评估路径不再使用。多通道抽象成为纯死代码，增加阅读理解和维护成本。

## 目标

将 h-only 硬编码到代码中，移除所有 `state_channels` / `num_channels` 参数化抽象，使训练 / 评估代码与"只学习 h"这一事实一致。

## 非目标

- **不改 `bundle_size`**：bundle_size 与通道正交，仍然作为 [main.py](../../../model/main.py) `CONFIG["bundle_size"]` 暴露，保留可调。
- **不改模型架构**：`GeoFNO2d` 的 `nn.Parameter` / `nn.Module` 张量结构在 `num_channels=1` 下与改造后完全一致。改造仅删除 Python 层的参数和分支。
- **不删历史 spec / plan**：[2026-05-18-channel-subset-training-design.md](2026-05-18-channel-subset-training-design.md) 等历史文档保留作为项目决策记录。
- **不动 manifest 数据格式**：[scripts/build_manifest.py](../../../scripts/build_manifest.py) 与通道无关，无需变更。

## 清理后的 API 形态

### `model/temporal_utils.py`
```python
CHANNEL_NAME = "h"  # 仅用于日志 / test_all 表头

@dataclass(frozen=True)
class TemporalConfig:
    bundle_size: int = 1
    # 删除 num_channels 字段
    @property
    def input_channels(self) -> int: ...   # 调用 input_channels_for_bundle(S)
    @property
    def out_channels(self) -> int: ...     # 调用 output_channels_for_bundle(S)

def validate_temporal_params(bundle_size: int) -> None: ...
def num_temporal_samples(num_time: int, bundle_size: int) -> int: ...

def input_channels_for_bundle(bundle_size: int) -> int:
    """C_in = 1 + 5*S + 8 = 5*S + 9"""
    return 5 * bundle_size + 9

def output_channels_for_bundle(bundle_size: int) -> int:
    """C_out = S (单通道残差)"""
    return bundle_size

def build_checkpoint_name(bundle_size: int) -> str:
    return "best_geofno.pt" if bundle_size == 1 else f"best_geofno_b{bundle_size}.pt"

def build_run_suffix(bundle_size: int) -> str:
    return "" if bundle_size == 1 else f"_b{bundle_size}"
```

**删除符号**：`CHANNEL_ORDER`、`CHANNEL_TO_INDEX`、`VALID_CHANNEL_INDEX_SETS`、`parse_channels`、`channels_suffix`

### `model/dataset.py`
```python
class StormSurgeDataset(Dataset):
    def __init__(self, path, bundle_size, btype_oh, lru_capacity: int = 1): ...

class MultiStormSurgeDataset(Dataset):
    def __init__(self, data_dir, bundle_size, btype_oh, lru_files_per_worker: int = 2): ...
```

- `_build_features(state_t, storm_window, inner_window, btype_oh)`：内部硬编码 `state_sub = state_t[..., 2:3]`
- `__getitem__` 中 target 改为 `graph[t+1 : t+S+1, :, 2:3].contiguous()`
- **删除** `_validate_state_channels`

### `model/model.py`
```python
class GeoFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels,
                 s1=40, s2=40, num_fno_layers: int = 3, fc1_hidden: int = 256):
        # 内部硬编码 self.num_channels = 1
        # self.bundle_size = out_channels
```

- 删 `num_channels` 构造参数及相关校验
- `forward` 中 `state_in = u[..., :1]`，residual reshape 为 `(B, N, S, 1)` 后 `permute(0, 2, 1, 3)`

### `model/train.py`
```python
def rel_l2_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """单通道相对 L2 loss，对 batch 取平均。"""
    diff = (pred - target).reshape(pred.size(0), -1)
    base = target.reshape(pred.size(0), -1)
    num = torch.linalg.vector_norm(diff, ord=2, dim=1)
    den = torch.linalg.vector_norm(base, ord=2, dim=1).clamp(min=eps)
    return (num / den).mean()

def train_model(model, train_loader, test_loader, num_epochs, device, optimizer,
                scheduler, coords_2d_device, writer, grad_clip=None,
                loss_type: str = "rel_l2", ema_decay=None,
                checkpoint_path: str = "best_geofno.pt",
                train_sampler=None, dist_ctx=None, accum_steps: int = 1): ...

def evaluate_model(model, test_loader, device, coords_2d_device, dist_ctx=None) -> dict:
    # 返回 {"mse", "rmse", "mae", "rel_l2"}
```

- **删除** `_validate_state_channels`、`_channel_rel_l2`、`mean_channel_rel_l2_loss`
- **删除** tensorboard `val/rel_h` 标量（单通道下与 `val/rel_l2` 等价，去除冗余）
- Epoch 打印简化为单行 Test Rel-L2 / Test RMSE 等

### `model/main.py`
```python
CONFIG = {
    # ...
    "bundle_size": 8,
    # 删除 "channels"
}
```

`main()` 中删除：
- `state_channels = parse_channels(...)` / `num_channels = ...` / `ch_suffix = ...`
- `parse_channels` / `channels_suffix` 的 import
- 所有 `num_channels=num_channels` 实参（`GeoFNO2d(...)`、`input_channels_for_bundle(...)` 等）
- 所有 `state_channels=state_channels` 实参（dataset 构造、`train_model(...)`)
- `[main] channels=... -> state_channels=...` 日志行

### `model/test_all.py`
- `parse_args` 删 `--channels` 参数
- `main()` 删 `state_channels` / `ch_suffix` / `selected_channel_names` 等变量；直接用常量 `("h",)` 或字面量 `"h"`
- `load_normalization_stats(stats_path, device)`：删 `state_channels` 参数；内部 `mean_sub = mean_full[..., 2:3]`、`std_sub = std_full[..., 2:3]`；`mean_full / std_full` **保留 3 通道**（dry mask 仍需要 water-level 物理化判断）
- `autoregressive_one_file(...)`：删 `state_channels` 参数；`real_start = graph_all[0:1, :, 2:3].to(device)`；`target_norm_sub = target_full_norm[..., 2:3]`
- `init_bucket(device)` / `compute_stats(bucket, num_nodes)` / `compute_auc(results)`：删 `num_channels` / `selected_channel_names` 参数，内部固定为单通道
- `write_results(...)`：删 `selected_channel_names`、`channels_suffix` 参数；`metric_output_path(base, space)` 不再带 `_ch*` 后缀；输出文件名为 `geofno_autoregressive_results_{physical|normalized}.txt`
- `build_features_batch` 签名不变

## 文件级修改清单

| 文件 | 操作 |
|---|---|
| [model/temporal_utils.py](../../../model/temporal_utils.py) | 删除 5 个符号，简化 `TemporalConfig` 和 4 个函数 |
| [model/dataset.py](../../../model/dataset.py) | 删 `state_channels` 参数与 `_validate_state_channels`，硬编码 `[..., 2:3]` |
| [model/model.py](../../../model/model.py) | 删 `num_channels` 参数，`forward` 切片硬编码 |
| [model/train.py](../../../model/train.py) | 新增 `rel_l2_loss`，删 3 个 helper，简化 `train_model` / `evaluate_model` 签名 |
| [model/main.py](../../../model/main.py) | 删 `channels` CONFIG 与所有派生变量 |
| [model/test_all.py](../../../model/test_all.py) | 删 `--channels`，所有 helper 签名瘦身 |
| [tests/test_geofno_num_channels.py](../../../tests/test_geofno_num_channels.py) | **整文件删除** |
| [tests/test_temporal_utils.py](../../../tests/test_temporal_utils.py) | 删除 `parse_channels` / `channels_suffix` / `num_channels` 参数化用例，更新公式断言 |
| [tests/test_dataset.py](../../../tests/test_dataset.py) | 删 `state_channels` 参数化用例，更新 target shape 断言 |
| [tests/test_test_all_helpers.py](../../../tests/test_test_all_helpers.py) | 同步更新调用签名 |

## 公式断言更新

新公式：
- `input_channels_for_bundle(1) = 14`
- `input_channels_for_bundle(8) = 49`
- `input_channels_for_bundle(24) = 129`
- `output_channels_for_bundle(S) = S`

## Checkpoint 迁移

- 默认 checkpoint 名变为 `best_geofno_b8.pt`（bundle_size=8 时）
- 训练前**手动** `mv best_geofno_b8_chh.pt best_geofno_b8.pt`
- 新旧模型参数张量形状一致，`load_state_dict` 应严格通过

## 实施顺序

按依赖自底向上：

1. `temporal_utils.py`
2. `dataset.py`
3. `model.py`
4. `train.py`
5. `main.py`
6. `test_all.py`
7. `tests/` 同步修正
8. 手动重命名 checkpoint（用户操作）

每步完成后做 `python -c "import <module>"` 静态检查，不留 import error。

## 验证策略

本机算力差且数据不在本机，**本机只跑离线验证**；训练 / DDP / test_all 冒烟由用户在服务器上手动跑。

### 本机验证
- `pytest tests/ -v` 全绿（synthetic 数据驱动，无外部依赖）
- 静态 import 检查：`python -c "import temporal_utils, dataset, model, train, test_all, main"` 无 import 错误
- 文档扫描：`grep -rn "state_channels\|--channels\|num_channels\|parse_channels\|channels_suffix\|CHANNEL_ORDER" AGENTS.md claude.md docs/` 中运行手册类文档已清理，仅历史 spec / plan 保留

### 服务器端验证（用户手动执行）
- **训练冒烟**：`CONFIG["num_epochs"]=1` 跑 `python model/main.py` 或 `torchrun --nproc_per_node=N model/main.py`，确认能跑通 1 epoch 且 tensorboard 标量含 `val/rel_l2`、不含 `val/rel_h`
- **test_all 冒烟**：`python model/test_all.py --num_files 1 --max_rollout 8`，输出文件名为 `geofno_autoregressive_results_{physical|normalized}.txt`，表格仅含 `wl` 一行
- **Checkpoint 迁移**：`mv best_geofno_b8_chh.pt best_geofno_b8.pt`（如旧 checkpoint 还需要复用）

## 风险

| 风险 | 缓解 |
|---|---|
| Checkpoint 张量形状漂移 | 改造不触碰任何 `nn.Parameter` / `nn.Module` 结构（仅删 Python 层参数与分支）；服务器端跑通训练即间接验证 `load_state_dict` 可加载 |
| Dry mask 退化（test_all） | 保留 `load_event_file` 读 3 通道 graph、`mean_full / std_full` 三通道版本；仅 `state_t` / `target` 切到 `[..., 2:3]` |
| 历史 tensorboard / 结果文件混淆 | 不动旧 `runs/` 与 `geofno_autoregressive_results_chh_*.txt`；新 run 用新 `run_tag`（`GeoFNO_b8_<ts>`） |
| `AGENTS.md` / `claude.md` / docs 文档引用 `--channels` | 简化完后扫一遍这些 markdown，删除或加注"历史"标签 |

## 历史关联

- [2026-05-18-channel-subset-training-design.md](2026-05-18-channel-subset-training-design.md) — 引入本设计移除的多通道抽象。保留作为决策历史。
- [2026-05-21-geofno-h-priority-optimization-design.md](2026-05-21-geofno-h-priority-optimization-design.md) — h 优先优化，与本简化方向一致。
