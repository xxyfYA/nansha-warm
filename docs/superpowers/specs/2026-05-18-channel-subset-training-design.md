# 通道子集训练设计

**日期**: 2026-05-18
**作者**: brainstorming session

## 背景

当前 Geo-FNO 实现把 graph 状态的三个通道（u 流速、v 流速、h 水位）作为一个整体训练：输入特征包含 `state_t (N, 3)`，模型输出 `(B, S, N, 3)` 残差。这使得无法做以下场景：

- **消融实验**：评估 u/v 信息对水位预测的贡献
- **专家模型**：训练只预测水位 h 的更小、更专注的模型
- **任意子集组合**：例如只训 `(u, h)` 看横向流速对水位的耦合作用

需要一个参数化机制，让训练入口、checkpoint、自回归测试都能按"任意通道子集"端到端运行。

## 目标

1. 新增字符串参数 `channels`，支持 `u/v/h` 任意子集组合（`"u"`、`"v"`、`"h"`、`"uv"`、`"uh"`、`"vh"`、`"uvh"` 共 7 种），大小写不敏感，去重
2. 端到端裁剪：dataset 输出、模型残差基底、模型输出维度、loss 报告通道、checkpoint 命名、自回归测试均按子集走
3. `storm_boundary`、`inner_boundary`、`btype_oh`（外部驱动 + 静态拓扑）始终全量输入，不受 `channels` 影响
4. 默认 `channels="uvh"` 时行为与当前完全等价（向后兼容），checkpoint 命名不带额外后缀
5. test_all.py 的 dry-grid mask 始终用真实 h 构建，与是否预测 h 无关
6. 同时清理掉训练噪声相关代码（项目不再使用）

## 非目标

- 不引入"输入通道与输出通道分别配置"的能力（不需要"输入 uvh、只预测 h"这种异构组合）
- 不引入 per-channel loss 加权（loss 在被选通道全空间上等权）
- 不动 Geo-FNO 模型主体（IPHI、FNO 块、谱卷积维持原样），只参数化 `in_channels`/`out_channels`/`num_channels`
- 不为旧 checkpoint 提供加载迁移（旧 ckpt 是 3 通道，对应新参数 `channels="uvh"`，路径不变即可）

## 通道规范

- 排列顺序固定为 `(u, v, h)`，解析后规范化为升序 tuple，例如 `"hv"` → `(1, 2)`
- 字符串只允许包含 `u/v/h`，否则报错
- `num_channels = len(state_channels)` 在 1–3 之间
- 数据切片之后，被选通道在前 K 维按 `(u, v, h)` 原顺序排列（即 `graph[..., state_channels]`）

## 涉及模块改动

### 1. `model/temporal_utils.py`

**新增**：

```python
CHANNEL_ORDER = ("u", "v", "h")
CHANNEL_TO_INDEX = {"u": 0, "v": 1, "h": 2}

def parse_channels(spec: str) -> tuple[int, ...]:
    """ "h"→(2,), "uvh"→(0,1,2), "vh"→(1,2). 大小写不敏感, 去重, 升序. """

def channels_suffix(indices: tuple[int, ...]) -> str:
    """ (0,1,2)→"" (向后兼容); 其它→"_ch" + 顺序拼接, 如 (2,)→"_chh" """
```

**修改签名**：

- `input_channels_for_bundle(bundle_size, num_channels=3) -> int`
  - 返回 `num_channels + 5*bundle_size + 8`
  - 推导：`state(K) + storm(3*(S+1)) + inner(2*(S+1)) + btype(3) = K + 5*S + 8`
- `output_channels_for_bundle(bundle_size, num_channels=3) -> int`
  - 返回 `num_channels * bundle_size`
- `build_checkpoint_name(bundle_size, channels_suffix="") -> str`
  - 移除 `noise_suffix` 参数
  - 默认 `bundle_size=1` 时返回 `f"best_geofno{channels_suffix}.pt"`
  - 否则 `f"best_geofno_b{bundle_size}{channels_suffix}.pt"`
- `build_run_suffix(bundle_size, channels_suffix="") -> str`
  - 移除 `noise_suffix` 参数
  - 默认 `bundle_size=1` 时返回 `channels_suffix`
  - 否则 `f"_b{bundle_size}{channels_suffix}"`

`TemporalConfig` dataclass 增加 `num_channels: int = 3`，对应属性同步更新。

### 2. `model/dataset.py`

**`_build_features`** 签名加 `state_channels: tuple[int, ...]`：

```python
def _build_features(state_t, storm_window, inner_window, btype_oh, state_channels):
    state_t_sub = state_t[..., list(state_channels)]
    storm_flat = storm_window.permute(1, 0, 2).reshape(N, -1)
    inner_flat = inner_window.permute(1, 0, 2).reshape(N, -1)
    return torch.cat([state_t_sub, storm_flat, inner_flat, btype_oh], dim=-1).contiguous()
```

**`StormSurgeDataset` / `MultiStormSurgeDataset`**：

- `__init__` 新增 `state_channels: tuple[int, ...] = (0, 1, 2)` 参数并保存
- `__getitem__`：
  - 调用 `_build_features(..., self.state_channels)`
  - `target = graph[idx+1 : idx+S+1][..., list(self.state_channels)].contiguous()`

### 3. `model/model.py`

`GeoFNO2d.__init__` 增加 `num_channels: int = 3`：

- 校验 `out_channels % num_channels == 0`（替代原硬编码 `% 3`）
- `self.bundle_size = out_channels // num_channels`
- `self.num_channels = num_channels`

`GeoFNO2d.forward`：

- 校验 `if u.size(-1) < self.num_channels: raise ...`
- `state_in = u[..., :self.num_channels]`
- `delta = delta_flat.view(B, N, self.bundle_size, self.num_channels)`
- 输出形状 `(B, bundle_size, N, num_channels)`
- docstring 更新："前 K 通道必须是当前归一化状态（按 `(u,v,h)` 中 state_channels 选定项的顺序排列）"

### 4. `model/train.py`

**删除**：

- `make_uvh_noise_std_tensor`
- `add_uvh_training_noise`
- `train_model` 中所有 `add_noise / uvh_noise_std / noise_t` 相关代码

**修改**：

`evaluate_model(model, test_loader, device, coords_2d_device, state_channels, dist_ctx=None)`：

- 用 `CHANNEL_ORDER = ("u","v","h")` 按 `state_channels` 动态构建 per-channel 累加项
- `_channel_rel_l2(pred_block, target_block, ch_idx_in_tensor)`：`ch_idx_in_tensor` 是裁剪后 tensor 内的 0..K-1 索引
- `reduce_sums` 累加列表长度根据 K 动态变化
- 返回 dict 包含 `mse / rmse / mae / rel_l2`，加上 `f"rel_{CHANNEL_ORDER[c]}"` for each `c in state_channels`

`train_model(...)` 签名删除 `add_noise / uvh_noise_std`，增加 `state_channels: tuple[int, ...]`：

- evaluate 调用传 `state_channels`
- tensorboard scalar：循环 `state_channels` 写入 `f"val/rel_{CHANNEL_ORDER[c]}"`，只写存在的键
- print：循环拼接 `Test Rel-{X}: {value:.6f}` 部分
- best-checkpoint 逻辑不变（仍按 `rel_l2` 选择）

### 5. `model/main.py`

**CONFIG 变更**：

- 删除 `"add_noise": False`、`"uvh_noise_std": [...]`
- 新增 `"channels": "uvh"`（默认全 3 通道）

**删除**：

- `format_noise_value`、`build_noise_run_suffix`、所有 `noise_suffix` 相关代码路径
- rank0 print 中 noise 摘要那一行

**新增/修改**：

```python
state_channels = parse_channels(CONFIG["channels"])
num_channels = len(state_channels)
ch_suffix = channels_suffix(state_channels)

in_channels = input_channels_for_bundle(CONFIG["bundle_size"], num_channels)
out_channels = output_channels_for_bundle(CONFIG["bundle_size"], num_channels)
checkpoint_name = build_checkpoint_name(CONFIG["bundle_size"], ch_suffix)
run_tag = "GeoFNO" + build_run_suffix(CONFIG["bundle_size"], ch_suffix) + "_" + timestamp
```

- 数据集构造传 `state_channels=state_channels`
- 模型构造传 `num_channels=num_channels`
- `train_model` 调用传 `state_channels=state_channels`
- rank0 print 加 `channels=..., state_channels=..., num_channels=...` 摘要

### 6. `model/test_all.py`

**新增**：

- CLI 参数 `--channels`，默认 `"uvh"`
- `state_channels = parse_channels(args.channels)`、`num_channels = len(state_channels)`、`ch_suffix = channels_suffix(state_channels)`
- 默认 checkpoint 经 `build_checkpoint_name(args.bundle_size, ch_suffix)`

**`load_normalization_stats(stats_path, device, state_channels=None)`**：

- 全量保留 `mean_full, std_full`（形状 `(1,1,3)`）
- 如果 `state_channels is not None`，额外返回 `mean_sub, std_sub`（形状 `(1,1,K)`）
- 调用处获取两份：`mean_sub, std_sub, mean_full, std_full = load_normalization_stats(args.norm, device, state_channels)`

**`load_event_file` 不变**（始终返回完整 `(T, N, 3)` graph）

**`autoregressive_one_file`** 签名加 `state_channels`、`mean_full`、`std_full`：

- `current_state = graph_all[batch_starts][..., list(state_channels)].to(device)`（K 通道）
- 自回归循环里：
  - `features = build_features_batch(current_state, storm_window, inner_window, btype_oh_device)`（state 已是 K 通道）
  - `pred_block = model(features, x_in)` → `(B, S, N, K)`
  - 对每个 `bundle_step`：
    - `target_norm_sub = graph_all[target_indices][..., list(state_channels)].to(device)` 用于 metric
    - `target_full = graph_all[target_indices].to(device)` 仅用于 dry mask 的 h 通道
    - 在两个 metric_space 上：
      - `metric_space="physical"`: `pred_metric = denormalize(pred_norm, mean_sub, std_sub)`、`target_metric = denormalize(target_norm_sub, mean_sub, std_sub)`
      - `metric_space="normalized"`: `pred_metric = pred_norm`、`target_metric = target_norm_sub`
      - `diff = pred_metric - target_metric`
      - `diff = apply_dry_grid_error_mask(diff, target_full, mean_full, std_full)`（mask 永远基于真实 h，用 `mean_full/std_full` 反归一化判断）
      - 累加到 K 维 bucket
  - `current_state = pred_block[:, -1]`

**`apply_dry_grid_error_mask(diff, target_full_norm, mean_full, std_full)`**：

- 始终用 `target_full_norm[..., 2]` 和 `mean_full[..., 2:3]`、`std_full[..., 2:3]` 反归一化得到物理水位
- `dry_mask = target_wl < DRY_WATER_LEVEL_THRESHOLD`
- `diff.masked_fill(dry_mask.unsqueeze(-1), 0.0)` —— 注意 diff 最后一维是 K，mask broadcast 自然成立

**`init_bucket(device, num_channels)`**：所有 tensor 维度从 3 改为 `num_channels`

**`compute_stats`、`compute_auc`、`write_results`、终端 summary**：

- `CHANNEL_NAMES` 全局常量保留（仍是 `("u","v","h")`）
- 这些函数都接受 `state_channels` 参数，循环只覆盖被选通道
- 写文件的列、AUC 表的行只展示被选通道
- `metric_output_path(base_path, metric_space, channels_suffix)`：stem 末尾加 `channels_suffix`，例如：
  - `geofno_autoregressive_results_chh_physical.txt`
  - `channels="uvh"` 时 stem 不变，保持向后兼容

## 输入/输出形状汇总

| 量 | uvh (K=3) | uv (K=2) | h (K=1) |
|---|---|---|---|
| dataset features | `(N, 5S+11)` | `(N, 5S+10)` | `(N, 5S+9)` |
| dataset target | `(S, N, 3)` | `(S, N, 2)` | `(S, N, 1)` |
| model out_channels | `3S` | `2S` | `S` |
| model output | `(B, S, N, 3)` | `(B, S, N, 2)` | `(B, S, N, 1)` |

## 命名约定

| 场景 | checkpoint 名 | run tag 前缀 | test 输出文件 stem 后缀 |
|---|---|---|---|
| `channels="uvh"`, `bundle=8` | `best_geofno_b8.pt` | `GeoFNO_b8_<ts>` | `_physical` / `_normalized` |
| `channels="h"`, `bundle=8` | `best_geofno_b8_chh.pt` | `GeoFNO_b8_chh_<ts>` | `_chh_physical` / `_chh_normalized` |
| `channels="uv"`, `bundle=1` | `best_geofno_chuv.pt` | `GeoFNO_chuv_<ts>` | `_chuv_physical` / `_chuv_normalized` |

## 测试与验证

实现完成后，跑通以下三种 smoke：

1. `channels="uvh"`：训练 + 测试结果应与改动前数值等价（关键回归测试）
2. `channels="h"`：单通道训练能跑通，checkpoint 名带 `_chh`，test 报告里只有 `wl` 一行
3. `channels="uv"`：双通道训练能跑通；test 自回归 dry mask 仍生效（用真实 h）

unit test 层面（若 `tests/` 已有结构）：

- `parse_channels` 的合法/非法输入边界
- `channels_suffix` 的 7 种组合输出
- `input_channels_for_bundle` 在 K=1/2/3 下的值
- dataset `__getitem__` 在 K=1/2/3 下输出形状
- model forward 在 K=1/2/3 下输出形状

## 风险与缓解

- **旧 checkpoint 不兼容新的 `num_channels=K` 路径**：默认 `channels="uvh"` 等价旧路径，文件名不变，旧 ckpt 仍可加载。
- **dry mask 始终需要真实 h**：依赖 `load_event_file` 永远返回 3 通道。CLAUDE.md 已规定数据格式恒为 `(T, N, 3)`，无破坏风险。
- **删除噪声代码**：项目内确认不使用；如未来需要，重新加回。
- **`channels="h"` 等单通道训练的物理合理性**：本设计只保证机制能跑通，物理结论由用户实验得出。
