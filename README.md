# Geo-FNO 风暴潮预热预测模型

基于 Geometry-Aware Fourier Neural Operator 的风暴潮水位直接预测模型。输入 24 小时气象边界条件，直接预测第 25 小时的水位 h。

## 1. 项目结构

```
nansha/
├── model/                          # 核心代码
│   ├── main.py                     # 训练入口（单卡 / DDP 多卡）
│   ├── model.py                    # GeoFNO2d + SpectralConv2d + IPHI
│   ├── train.py                    # 训练循环 + EMA + 损失函数
│   ├── dataset.py                  # 数据加载 + DDP 采样器
│   ├── scheduler.py                # Cosine Warmup 学习率调度
│   ├── temporal_utils.py           # 固定常量定义
│   └── test_all.py                 # 单步直接预测评估
├── scripts/                        # 数据工具
│   ├── build_manifest.py           # 生成 manifest.json
│   └── check_data_integrity.py     # 校验 .pt 文件完整性
├── tests/                          # 单元测试（50 个）
├── data/                           # 数据目录
│   ├── coordinates.mat             # 节点坐标与边界属性
│   ├── normalization.mat           # z-score 归一化参数
│   ├── train/*.pt + manifest.json  # 训练集
│   ├── val/*.pt + manifest.json    # 验证集
│   └── test/*.pt + manifest.json   # 测试集
└── runs/                           # TensorBoard 日志
```

## 2. 完整流程

### Step 1：校验数据完整性

```bash
python scripts/check_data_integrity.py --data_root data --deep
```

检查所有 `.pt` 文件是否可正常加载，以及 key/shape 是否符合预期。

### Step 2：生成 manifest

```bash
python scripts/build_manifest.py data/train --min_T_warn 25
python scripts/build_manifest.py data/val   --min_T_warn 25
python scripts/build_manifest.py data/test  --min_T_warn 25
```

扫描每个 split 目录下的 `.pt` 文件，记录每个文件的 T（时间步数）和 N（节点数），生成 `manifest.json`。过滤掉 `T < 25` 的文件。

### Step 3：训练

```bash
# 单 GPU
python model/main.py

# 多 GPU (DDP，如 4 卡)
torchrun --nproc_per_node=4 model/main.py
```

训练过程：
- 每个 epoch 遍历训练集，梯度累积 + Cosine 学习率调度
- epoch 结束后在验证集上评估 5 个指标
- 保存最优模型为 `best_geofno.pt`
- TensorBoard 日志写入 `runs/`

### Step 4：评估

```bash
python model/test_all.py \
    --test_dir data/test \
    --coords data/coordinates.mat \
    --norm data/normalization.mat \
    --model best_geofno.pt \
    --output results.txt
```

对测试集每个文件的所有有效时间点做单步预测，输出归一化和物理空间的指标。

## 3. 数据格式

### 3.1 `.pt` 文件格式（每个风暴事件一个文件）

| Key | Shape | 说明 |
|-----|-------|------|
| `graph` | (T, N, 3) | 状态量 [u流速, v流速, 水位h]，T 时刻，N 节点 |
| `storm_boundary` | (T, N, 3) | 风暴边界 [气压P, x风速Wx, y风速Wy] |
| `inner_boundary` | (T, N, 2) | 内部边界 [边界水位, 边界流量]，非边界节点为 0 |

数据已进行 z-score 归一化。

### 3.2 `coordinates.mat` 格式

| Key | Shape | 说明 |
|-----|-------|------|
| `coordinates` | (N, 3) | 节点三维坐标（无归一化） |
| `boundary` | (N, 1) | 边界类型：0=无边界, 1=水位边界, 2=流量边界 |
| `boundary_index` | (N, 1) | 节点对应边界文件中的序号 |
| `fort19_index` | (N, 1) | 水位边界节点到 fort.19 序号的映射 |
| `fort20_index` | (N, 1) | 流量边界节点到 fort.20 序号的映射 |
| `fort19_nodes` | (94, 1) | fort.19 每个时刻 94 个值对应的节点号 |
| `fort20_nodes` | (36, 1) | fort.20 每个时刻 36 个值对应的节点号 |

### 3.3 `normalization.mat` 格式

两种兼容格式：

```python
# 格式 A（推荐）
{"graph_mean": (3,) float32, "graph_std": (3,) float32}

# 格式 B（兼容旧版）
{"u_mean": ..., "u_std": ..., "v_mean": ..., "v_std": ..., "h_mean": ..., "h_std": ...}
```

### 3.4 `manifest.json` 格式

```json
{
  "num_nodes": 190764,
  "files": [
    {"path": "event_001.pt", "T": 120},
    {"path": "event_002.pt", "T": 96}
  ],
  "created_at": "2026-05-25T00:00:00+00:00"
}
```

## 4. 特征构建

每个时间点 t 构建一个样本：

```
输入（120 通道）:
┌──────────────────────────────────┬──────────────────────────────────┐
│ storm[t : t+24]                  │ inner[t : t+24]                  │
│ 24 步 × 3 通道 = 72 通道          │ 24 步 × 2 通道 = 48 通道          │
│ [P₀,Wx₀,Wy₀, ..., P₂₃,Wx₂₃,Wy₂₃]│ [hB₀,qB₀, ..., hB₂₃,qB₂₃]      │
└──────────────────────────────────┴──────────────────────────────────┘

输出（1 通道）:
  target = graph[t+24, :, 2:3]   ← 第 25 小时的水位 h
```

每个文件的样本数 = `T - 24`（窗口长 24，需要 1 个目标步）。

## 5. 模型架构

```
输入 (B, N, 120)
    │
    ▼
┌─────────────────────────────────────────────┐
│  fc0: Linear(120 → width)                    │  升维
│  permute → (B, width, N)                    │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  conv0: SpectralConv2d + b0(grid bias)       │  不规则网格 → 规则网格
│  通过 IPHI 将不规则坐标映射到 [0,1]²          │  (IPHI: 可学习映射网络)
│  + GELU                                      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  Middle FNO Blocks × num_fno_layers          │  规则网格上的傅里叶层
│  ┌───────────────────────────────────────┐   │
│  │ SpectralConv2d + W(1×1 conv) + b(grid)│   │  跳跃连接: 谱域 + 空间域
│  │ + GELU                                │   │
│  └───────────────────────────────────────┘   │
│           ... 重复 num_fno_layers 次 ...      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  conv4: SpectralConv2d + b4(x_out bias)      │  规则网格 → 不规则网格
│  通过 IPHI 逆映射                             │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  permute → (B, N, width)                    │
│  fc1: Linear(width → fc1_hidden) + GELU      │  投影到单值输出
│  fc2: Linear(fc1_hidden → 1)                │
└──────────────────┬──────────────────────────┘
                   ▼
输出 (B, N, 1)  ← 直接预测 h(t+24)，不做残差
```

### 关键组件

| 组件 | 说明 |
|------|------|
| **IPHI** | 可学习的不规则网格→规则域映射。输入 (x,y,angle,radius) 4 个特征，通过 NeRF 风格的位置编码 + 4 层 MLP (tanh) 输出坐标偏移 |
| **SpectralConv2d** | 2D 傅里叶层。支持规则网格 FFT 和不规则网格（通过傅里叶基函数 einsum 计算）两种模式 |
| **残差连接** | 中层 FNO block 使用谱域卷积 + 1×1 空间卷积的跳跃连接 |

## 6. 超参数配置

配置定义在 `model/main.py` 的 `CONFIG` 字典中。

### 6.1 数据超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INPUT_WINDOW` | 24 | 输入时间窗口（小时），硬编码于 `temporal_utils.py` |
| `PREDICT_OFFSET` | 24 | 预测偏移量（小时） |
| `C_IN` (派生) | 120 | `24×3 + 24×2`，无需手动设置 |
| `C_OUT` (派生) | 1 | 输出通道数（h-only），无需手动设置 |
| `train_dir` | `data/train` | 训练集路径 |
| `val_dir` | `data/val` | 验证集路径 |
| `test_dir` | `data/test` | 测试集路径 |
| `coords_path` | `data/coordinates.mat` | 坐标文件路径 |
| `norm_path` | `data/normalization.mat` | 归一化参数路径 |
| `batch_size` | 16 | 全局 batch size（DDP 下自动均分到各卡） |
| `num_workers` | 4 | DataLoader 工作进程数 |
| `lru_files_per_worker` | 2 | 每个 worker 的 LRU 文件缓存数 |

### 6.2 模型超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `modes` | 24 | 傅里叶模态数（每个维度保留的低频分量数） |
| `width` | 48 | 模型宽度（FNO 内部通道数） |
| `s1` | 64 | 规则网格尺寸 (x 方向) |
| `s2` | 64 | 规则网格尺寸 (y 方向) |
| `num_fno_layers` | 4 | 规则网格上的 FNO 中间层数 |
| `fc1_hidden` | 256 | 输出投影层 fc1 的隐藏维度 |

### 6.3 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_epochs` | 200 | 训练总 epoch 数 |
| `lr` | 1e-3 | 初始学习率（AdamW） |
| `weight_decay` | 1e-4 | AdamW 权重衰减 |
| `warmup_ratio` | 0.05 | 学习率 warmup 比例（总步数的 5%） |
| `min_lr_ratio` | 0.01 | 学习率衰减下限（初始 lr 的 1%） |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `accum_steps` | 1 | 梯度累积步数（>1 时增大等效 batch size） |
| `loss_type` | `rel_l2` | 损失函数：`rel_l2`（相对 L2）或 `rmse` |
| `ema_decay` | 0.999 | EMA（指数移动平均）衰减率，用于模型平滑 |
| `seed` | 42 | 随机种子 |
| `tb_dir` | `runs` | TensorBoard 日志目录 |

### 6.4 学习率调度

采用 **Cosine Annealing + Linear Warmup**：

```
                  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
lr ▲             █              ██
   │           ▄█                █
   │         ▄█                  █▄
   │       ▄█                      █▄
   │     ▄█                          █
   │   ▄█                              █▄
   │ ▄█                                  █▄
   └─┴────┴────────────────────────────┴──► step
      warmup_steps            total_steps

- warmup_steps  = total_steps × warmup_ratio (默认 5%)
- min_lr        = lr × min_lr_ratio (默认 1e-5)
- 预热阶段：线性从 0 升至 lr
- 衰减阶段：cosine 从 lr 降至 min_lr
```

## 7. 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MSE | `mean((pred - target)²)` | 均方误差 |
| RMSE | `sqrt(MSE)` | 均方根误差 |
| MAE | `mean(|pred - target|)` | 平均绝对误差 |
| R² | `1 - SSE / SS_tot` | 决定系数 |
| Rel L2 | `||pred - target||₂ / ||target||₂` | 相对 L2 误差 |

评估在归一化空间和物理空间分别计算。物理空间通过 `normalization.mat` 反归一化得到。干网格节点（物理水位 < 0.005m）自动从评估中排除。

## 8. 推理示例

```python
import torch
from model import GeoFNO2d
from dataset import load_static_coords, build_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载坐标
coords = load_static_coords("data/coordinates.mat").to(device)

# 加载模型
model = GeoFNO2d(modes1=24, modes2=24, width=48,
                 in_channels=120, out_channels=1,
                 s1=64, s2=64, num_fno_layers=4, fc1_hidden=256)
model.load_state_dict(torch.load("best_geofno.pt", map_location=device))
model.to(device).eval()

# 加载数据（单样本）
data = torch.load("event.pt", map_location=device)
storm = data["storm_boundary"]   # (24, N, 3)
inner = data["inner_boundary"]   # (24, N, 2)

# 构建特征
features = build_features(storm, inner).unsqueeze(0)  # (1, N, 120)
x_in = coords.unsqueeze(0)                             # (1, N, 2)

# 推理
with torch.no_grad():
    pred_h = model(features, x_in)  # (1, N, 1) — 第 25 小时水位预测
```

## 9. 常用命令速查

```bash
# 数据准备
python scripts/check_data_integrity.py --deep
python scripts/build_manifest.py data/train --min_T_warn 25
python scripts/build_manifest.py data/val   --min_T_warn 25
python scripts/build_manifest.py data/test  --min_T_warn 25

# 训练
python model/main.py                              # 单 GPU
torchrun --nproc_per_node=4 model/main.py         # 4 GPU DDP

# 评估
python model/test_all.py --test_dir data/test

# 测试
python -m pytest tests/ -v

# TensorBoard
tensorboard --logdir runs
```
