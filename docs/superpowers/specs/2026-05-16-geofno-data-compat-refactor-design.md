# Geo-FNO 数据兼容性重构设计

**日期**: 2026-05-16
**作者**: brainstorming session

## 背景

`CLAUDE.md` 定义的数据格式（`storm_boundary`、`inner_boundary`、UUID 命名的 .pt 文件、`coordinates.mat` 中的 `coordinates`/`boundary` 等多种字段）与现有 `model/` 代码（单一 `boundary` 键、硬编码 `train_1_zscore.pt` 路径、`center_coordinates` 默认键）完全不兼容，且现有 Dataset 用全量预加载方式打开几百 GB 数据集时不可行。

本文档描述把现有 Geo-FNO 实现适配到新数据格式所需的全部改动。

## 目标

1. Dataset 能正确加载新格式 .pt 文件（包含 `storm_boundary` per-node 空间场、`inner_boundary` 稀疏边界场、`coordinates.mat` 边界元数据）
2. 模型摄入 per-node 边界场 + 静态边界类型 mask，物理意义清晰
3. 支持 `data/train`、`data/val`、`data/test` 三目录下数百个独立 storm 事件 .pt 文件，无需全量 RAM 预加载
4. 多卡 DDP（单机 4 卡）训练 RAM 占用不随 rank 数线性膨胀
5. 训练只保留 bundle 范式，删除 pushforward 与 recurrent
6. 测试链路（test_all.py）适配新数据格式与多文件评测

## 非目标

- 不修改 Geo-FNO 模型主体结构（IPHI + FNO 块保持原样，只通道数变）
- 不引入图神经网络或显式邻接（暂时不使用 `triangles`）
- 不做物理约束硬注入（边界节点的真实 GT 不在前向中强制覆盖）
- 不为多机训练做特殊优化（仅单机 4 卡）

## 数据格式（输入侧）

### .pt 文件结构

每个 .pt 字典包含：

| 键 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `graph` | `(T, 190764, 3)` | float32 | 状态 (u, v, h)，已 zscore 归一化 |
| `storm_boundary` | `(T, 190764, 3)` | float32 | (气压, x 风, y 风)，per-node 空间场，已归一化 |
| `inner_boundary` | `(T, 190764, 2)` | float32 | (边界水位, 边界流量)，仅在 94+36 个边界节点上有值，其余为 0，已归一化 |
| `run_id` | str | — | storm 事件 ID |
| `source_dir` | str | — | 原始数据来源路径 |

T 因事件而异（约 10–120 小时）。每个文件代表一个独立 storm 事件，**样本绝不跨文件拼接**。

### coordinates.mat 结构

| 键 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `coordinates` | `(190764, 3)` | float64 | 节点 3D 坐标，未归一化（IPHI 只用前 2 维） |
| `boundary` | `(190764, 1)` | int8 | 0=无边界, 1=水位边界, 2=流量边界 |
| `boundary_index` | `(190764, 1)` | int32 | 节点在对应 fort 文件中的索引 |
| `fort19_index` / `fort20_index` | `(190764, 1)` | int32 | 仅特定类型边界节点的 fort.19 / fort.20 索引 |
| `fort19_nodes` | `(94, 1)` | int32 | fort.19 时刻块顺序中的节点号 |
| `fort20_nodes` | `(36, 1)` | int32 | fort.20 时刻块顺序中的节点号 |
| `triangles` | `(345297, 3)` | int32 | 三角网格连接（本设计暂不使用） |

本设计仅消费 `coordinates[:, :2]` 与 `boundary` 两个字段；其余字段保留供未来使用（如显式邻接 GNN、边界条件 fort 文件直读）。

## 模型输入特征布局

每个节点的输入特征向量按以下顺序拼接，设 `S = bundle_size`：

```
位置          通道数      内容
[0]           3           当前状态 (u_t, v_t, h_t)
[3 ..]        3 * (S+1)   storm 窗口 (P, Wx, Wy) 在时刻 t, t+1, ..., t+S
[..]          2 * (S+1)   inner 窗口 (h_b, q_b) 在时刻 t, t+1, ..., t+S
[末尾]        3           边界类型 one-hot (none, water_level, flux)
```

**通道总数**：

```
C_in  = 3 + 5 * (S + 1) + 3  =  5*S + 11
C_out = 3 * S
```

| bundle_size S | C_in | C_out |
|---|---|---|
| 1 | 16 | 3 |
| 24 | 131 | 72 |
| 72 | 371 | 216 |

storm 与 inner 窗口都包含 t..t+S 共 S+1 个时刻，原因：这些是"已知"的外强迫（数值天气预报 + 潮汐/入流预报）。

边界类型 one-hot 是静态张量 `(N, 3)`，每个样本重复拼上去；模型借此区分"非边界节点上的 0"与"边界节点上恰好为 0"。

## 模型主体（无改动）

`model/model.py` 的 `GeoFNO2d`、`IPHI`、`SpectralConv2d` 全部保持原样。

- `fc0 = nn.Linear(in_channels, width)` 已是参数化的，自动适配新 `C_in`
- `fc2 = nn.Linear(128, out_channels)` 自动适配新 `C_out`
- 唯一前置约束 `out_channels % 3 == 0`（[model.py:236-238](../../../model/model.py#L236-L238)），新公式 `C_out = 3*S` 自动满足

## Dataset 设计：lazy 加载 + LRU + file-affine sampler

### 加载架构

```
┌─────────────────────────────────────────────────┐
│ MultiStormSurgeDataset (主进程, 每 rank 一个)   │
│  ├─ manifest: [{path, T}, ...] 来自 JSON        │
│  ├─ flat_index: [(file_idx, t_local), ...]      │
│  ├─ coords_norm (shared mem, ~4 MB)             │
│  └─ btype_oh   (shared mem, ~2 MB)              │
└─────────────────────────────────────────────────┘
                      │ fork (DataLoader workers)
                      ▼
┌─────────────────────────────────────────────────┐
│ DataLoader worker 进程                          │
│  ├─ LRU cache: {file_idx: (graph, storm, inner)}│
│  ├─ 容量 = lru_files_per_worker (默认 2)         │
│  ├─ 命中: 切时间片 + 拼通道                       │
│  └─ 未命中: torch.load → 入 LRU                 │
└─────────────────────────────────────────────────┘
```

总 RAM ≈ 4 rank × 4 worker × 2 file × ~300 MB ≈ **10 GB**（vs 全量 163 GB）。

### Manifest 生成

工具脚本 `scripts/build_manifest.py`：

- 输入: 单一目录路径（如 `data/train`）
- 输出: 同目录下 `manifest.json`
- 逻辑: glob `*.pt`、过滤 `._` 前缀的 macOS 元数据、逐文件 `torch.load` 取 `graph.shape[0]`、写 JSON

输出格式：

```json
{
  "num_nodes": 190764,
  "files": [
    {"path": "uuid1.pt", "T": 113},
    {"path": "uuid2.pt", "T": 67}
  ],
  "created_at": "2026-05-16T..."
}
```

主程序启动时若 `data/<split>/manifest.json` 缺失则报错并提示用户先运行 `build_manifest.py`。

### File-Affine Distributed Sampler

```python
class FileChunkedDistributedSampler(Sampler):
    """每个 epoch:
       1. shuffle 文件列表 (seed=epoch)
       2. rank_i 拿 files[i::world_size]
       3. rank 内: 把样本按 file_idx 分组, 组间 shuffle, 组内 shuffle
       4. 输出: 连续的同文件样本块 → 触发 LRU 高命中"""
```

效果：同一 worker 连续 ~ (T_i - S) 个样本来自同一文件，LRU 命中率 ≈ 99%。

### Dataset 返回签名

```python
def __getitem__(idx) -> tuple[Tensor, Tensor]:
    # features:     (N, C_in)
    # target_block: (S, N, 3)
```

返回 **2 元组**。坐标 `coords_2d_norm (N, 2)` 不在每个样本中重复传输（原代码每样本传一份是冗余的），改为：

- `MultiStormSurgeDataset` 暴露属性 `dataset.coords_2d_norm` 与 `dataset.btype_oh`，均为 share_memory_ 张量
- main.py / train.py 启动时把 `coords_2d_norm.to(device)` 取出，每个 batch 内 `expand(B, -1, -1)` 喂给模型
- 节省 `batch_size × N × 2 × 4 = ~24 MB / batch` 的 DataLoader collate 复制

`train.py` 和 `test_all.py` 中对原 4 元组的解包需要全部改为 2 元组。

### 坐标与边界类型加载（按用户要求挪进 Dataset）

```python
def load_static_coords(coords_path):
    mat = scipy.io.loadmat(coords_path)
    coords = mat["coordinates"][:, :2].astype(np.float32)
    cmin, cmax = coords.min(0), coords.max(0)
    coords_norm = (coords - cmin) / np.maximum(cmax - cmin, 1e-8)
    bt = mat["boundary"].astype(np.int64).flatten()  # (N,)
    btype_oh = np.eye(3, dtype=np.float32)[bt]       # (N, 3)
    coords_t = torch.from_numpy(coords_norm); coords_t.share_memory_()
    btype_t  = torch.from_numpy(btype_oh);    btype_t.share_memory_()
    return coords_t, btype_t
```

main.py 不再处理坐标归一化，只从 Dataset 取 `coords_2d_norm.to(device)`。

## 删除项

| 文件 | 删除内容 |
|---|---|
| `model/temporal_utils.py` | `validate_recurrent_params`, `recurrent_target_bounds`, `pushforward_steps` 相关 helper；`input_channels_for_bundle` / `output_channels_for_bundle` 更新公式 |
| `model/train.py` | `predict_recurrent_rollout`, `recurrent_rel_l2_loss`, `predict_final_block`（简化为一行 model forward） |
| `model/Dataset.py` | 旧文件整体替换为新 `model/dataset.py` |
| `model/main.py` | CONFIG 中 `pushforward_steps`、`recurrent_steps`、`train_paths`、`test_path` 字段；坐标归一化代码段 |
| `model/test_all.py` | `--pushforward_steps`、`--recurrent_steps` 命令行参数；归一化路径默认值；`dataset.boundary` 引用 |

## CONFIG（main.py）

```python
CONFIG = {
    "train_dir":   "data/train",
    "val_dir":     "data/val",
    "test_dir":    "data/test",
    "coords_path": "data/coordinates.mat",
    "norm_path":   "data/normalization.mat",

    "bundle_size":          72,
    "batch_size":           16,
    "num_workers":          4,
    "lru_files_per_worker": 2,

    "modes":           16,
    "width":           32,
    "s1":              64,
    "s2":              64,
    "num_fno_layers":  3,

    "num_epochs":      100,
    "lr":              1e-3,
    "weight_decay":    1e-4,
    "warmup_ratio":    0.05,
    "grad_clip":       1.0,
    "accum_steps":     1,
    "loss_type":       "rel_l2",

    "add_noise":       False,
    "uvh_noise_std":   [0.005, 0.005, 0.001],

    "seed":            42,
    "tb_dir":          "runs",
}
```

## 测试链路（test_all.py）

测试与训练的数据访问模式不同：自回归级联需要在同一文件内**连续访问** group_len 个时刻。lazy + LRU Dataset 适合训练的随机 idx 访问，但对测试不合适。因此：

- 测试用**独立的外层循环**：按 `data/test/` 中每个 .pt 文件，单次 `torch.load` 整文件进 RAM，做完一个文件再释放
- 每个 storm 事件独立做"多 bundle 级联"自回归评测：
  - bundle k=0: `state_t → pred_block (S, N, 3)`
  - bundle k≥1: 取上一 bundle 最后一步作为新 `state`，新 storm/inner 窗口从文件读 `t + k*S .. t + (k+1)*S`
  - 重复直到累计 `group_len` 步
- 约束 `group_len % bundle_size == 0`（保留原 [test_all.py:239-242](../../../model/test_all.py#L239-L242) 的断言逻辑）
- 边界窗口直接从文件读真实未来值（已知预报）
- 干网格 mask：`apply_dry_grid_error_mask` 逻辑保留
- 指标在 normalized + physical 两个空间分别聚合
- 跨文件聚合：所有 storm 事件的 per-step 指标按节点-时刻数加权合并
- 输出两份文件（normalized / physical），不再按 model 路径自动猜测

归一化路径硬编码到 `data/normalization.mat`（不再是 `data/normalize/normalization_origin.mat`）。

## 实施步骤

| # | 文件 | 操作 |
|---|---|---|
| 1 | `model/dataset.py` (新) | lazy + LRU + file-affine sampler + manifest 解析 + 坐标归一化 |
| 2 | `model/temporal_utils.py` | 删 push/recurrent，更新通道公式为 `C_in = 5S+11`, `C_out = 3S` |
| 3 | `model/train.py` | 删 rollout 路径；`build_feature_block` 由 dataset 内化后可整体移除；评估流程简化 |
| 4 | `model/main.py` | 新 CONFIG、扫描目录、新 dataset 接入、删坐标归一化代码 |
| 5 | `model/test_all.py` | 归一化路径、新 dataset 接入、多文件评测、删 push/recurrent 选项 |
| 6 | `scripts/build_manifest.py` (新) | 一次性 manifest 生成工具 |
| 7 | (服务器) 用户执行 | 单卡 bundle=1 单文件冒烟 → 全量 DDP bundle=72 |

本机不执行 Python 运行验证，所有 PR 以静态正确性 + 服务器跑通为准。

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| LRU 命中率不达预期（< 90%）导致每 batch I/O 阻塞 | file-affine sampler 强制连续样本同文件；监控 worker `__getitem__` 耗时 |
| 单个 storm 事件 T 太小（如 11 小时）以致 bundle=72 时 (T-S) ≤ 0 | manifest 构建阶段 warn；Dataset 初始化时丢弃 T ≤ S 的文件 |
| DDP 4 rank × 4 worker = 16 进程同时 torch.load 同一文件 | 实际不同 rank 文件不交叉（rank_i 持 `files[i::4]`），同 rank 内 workers 几乎错开访问 |
| coordinates.mat 中 boundary 字段值域异常 | `load_static_coords` 中 assert dtype 与值域 ∈ {0, 1, 2} |
| manifest 与实际文件不同步（用户增删 .pt 未重建） | main.py 启动校验 manifest 中文件均存在；不一致则报错提示重建 |

## 验收准则

- 单卡 bundle=1 在 1 个 .pt 文件上能跑通
- 全量 DDP 4 卡 bundle=72 启动实验不在本机进行，不需要考虑
