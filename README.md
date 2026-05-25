# Geo-FNO 风暴潮预热预测模型

基于 Geometry-Aware Fourier Neural Operator 的风暴潮水位直接预测模型。
输入 24 小时气象边界条件，直接预测第 25 小时的水位 h。

## 1. 项目结构

```
nansha/
├── model/                          # 核心代码
│   ├── main.py                     # 训练入口（单卡 / DDP 多卡）
│   ├── model.py                    # GeoFNO2d + SpectralConv2d + IPHI
│   ├── train.py                    # 训练循环 + EMA + 损失函数
│   ├── dataset.py                  # 数据加载 + DDP 采样器
│   ├── scheduler.py                # Cosine Warmup 学习率调度
│   ├── temporal_utils.py           # 固定常量定义 (INPUT_WINDOW=24, C_IN=120, C_OUT=1)
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

---

## 2. 训练数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             训练数据流                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                               │
│  │ train/*.pt│    │  val/*.pt│    │ test/*.pt│    每个 .pt 文件:              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    T × N 节点                  │
│       │               │               │          graph / storm / inner       │
│       ▼               ▼               ▼                                      │
│  ┌─────────────────────────────────────────┐                                │
│  │        build_manifest.py                │  扫描 T, N, 生成 manifest.json  │
│  │  --min_T_warn 25                       │  过滤 T < 25 的文件             │
│  └────────────────┬────────────────────────┘                                │
│                   ▼                                                          │
│  ┌─────────────────────────────────────────┐                                │
│  │        MultiStormSurgeDataset            │  T < 25 的文件自动过滤          │
│  │                                         │  构建 flat_index               │
│  │  flat_index = [                        │  T=41 → 17 样本                 │
│  │    (file_0, t=0),                      │  T=36 → 12 样本                 │
│  │    (file_0, t=1),                      │  ...                            │
│  │    ...                                 │                                 │
│  │  ]                                     │                                 │
│  └────────────────┬────────────────────────┘                                │
│                   │                                                          │
│                   ▼ __getitem__                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                        单个样本构造                                │       │
│  │                                                                   │       │
│  │  storm(t..t+23) 展开 (N, 72) ───┐                                 │       │
│  │  inner(t..t+23) 展开 (N, 48) ───┤ cat ──► features (N, 120)      │       │
│  │                                  │                                 │       │
│  │  target = graph[t+24, :, 2:3] ──► (N, 1)  h(t+24)                │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  FileChunkedDistributedSampler (DDP)                                         │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │ 贪心文件分配 → 大文件优先 → 各 rank 尽量均分样本数               │           │
│  │ rank 内: 文件 shuffle → 文件内样本 shuffle → drop_last/pad     │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 数据格式

**`.pt` 文件**（每个风暴事件一个文件）：

| Key | Shape | 说明 |
| --- | --- | --- |
| `graph` | (T, N, 3) | [u流速, v流速, 水位h]，已 z-score 归一化 |
| `storm_boundary` | (T, N, 3) | [气压P, x风速Wx, y风速Wy] |
| `inner_boundary` | (T, N, 2) | [边界水位, 边界流量]，非边界节点为 0 |

**`coordinates.mat`**：节点坐标 N×3，`boundary` 边界类型 N×1（0/1/2），以及 fort.19/20 映射信息。仅 `coordinates` 的 xy 分量被模型使用——通过 IPHI 做不规则网格到规则域的空间映射。

**`normalization.mat`**：

```python
# 格式 A（推荐）
{"graph_mean": (3,) float32, "graph_std": (3,) float32}
# 格式 B（兼容旧版）
{"u_mean": ..., "u_std": ..., "v_mean": ..., "v_std": ..., "h_mean": ..., "h_std": ...}
```

**`manifest.json`**（由 `build_manifest.py` 生成）：

```json
{
  "num_nodes": 190764,
  "files": [{"path": "event_001.pt", "T": 120}, {"path": "event_002.pt", "T": 96}],
  "created_at": "2026-05-25T00:00:00+00:00"
}
```

---

## 3. 模型架构 GeoFNO2d

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             模型架构 GeoFNO2d                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  features (B, N, 120)              x_in (B, N, 2) xy 坐标                   │
│       │                                    │                                │
│       ▼                                    ▼                                │
│  ┌──────────┐                      ┌──────────────┐                         │
│  │ fc0      │ Linear(120, 48)      │ IPHI         │ xy → 变形坐标 ξ         │
│  └────┬─────┘                      │              │ 极坐标 + NeRF 位置编码   │
│       │                            │              │ MLP: 4→32→128→128→2     │
│       ▼                            └──────┬───────┘                         │
│  u = permute → (B, 48, N)                  │ ξ ∈ [0,1]²                    │
│       │                                    │                                │
│       ├────────────────────────────────────┤                                │
│       ▼                                    ▼                                │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  conv0: 不规则网格 → 64×64 规则网格                       │                │
│  │  fft2d(u, x_in, iphi): 傅里叶基 e^{-i·2π·K·ξ} 映射       │                │
│  │  + b0(grid) 网格偏置 → GELU                             │                │
│  └─────────────────────────────────────────────────────────┘                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  4 × Middle FNO Blocks (64×64 规则网格上)                  │                │
│  │                                                          │                │
│  │  ┌─────────────────┐    ┌──────────┐    ┌─────────────┐ │                │
│  │  │ SpectralConv2d  │ +  │ Conv2d   │ +  │ Conv2d(grid)│ │                │
│  │  │ (频域 FFT 卷积)   │    │ (1×1 空间)│    │ (网格偏置)    │ │                │
│  │  │ 保留 24×24 modes │    │          │    │             │ │                │
│  │  └─────────────────┘    └──────────┘    └─────────────┘ │                │
│  │                         → GELU                           │                │
│  └─────────────────────────────────────────────────────────┘                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  conv4: 64×64 规则网格 → 不规则网格                        │                │
│  │  ifft2d(u_ft, x_out, iphi): 逆傅里叶映射                   │                │
│  │  + b4(x_out) 位置偏置                                     │                │
│  └─────────────────────────────────────────────────────────┘                │
│       │                                                                     │
│       ▼                                                                     │
│  u = permute → (B, N, 48)                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────┐                                                                │
│  │ fc1      │ Linear(48, 256) → GELU                                        │
│  │ fc2      │ Linear(256, 1)                                                 │
│  └────┬─────┘                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  pred = fc2(fc1(u)) → (B, N, 1)  直接预测 h(t+24)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键组件

| 组件 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| **IPHI** | 不规则网格坐标 (x,y) | 变形坐标 ξ∈[0,1]² | 将网格节点映射到规则计算域。对每个节点计算极坐标 (angle, radius)，经 NeRF 风格 sin/cos 位置编码后通过 4 层 MLP (tanh) 输出偏移 |
| **SpectralConv2d** | 节点特征 / 网格特征 | 卷积后特征 | 2D 傅里叶层。规则网格上用 FFT；不规则网格上通过傅里叶基函数 `e^{±i·2π·K·ξ}` 做 einsum 映射 |
| **Middle Block** | 规则网格特征 (B,C,64,64) | 同 shape | 谱域卷积 + 1×1 空间卷积 + 网格偏置的跳跃连接，后接 GELU |

---

## 4. 训练循环

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             训练循环                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   for epoch in 1..200:                                                      │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │  train_loader (FileChunkedDistributedSampler)              │             │
│   │  每卡分配整文件, 贪心平衡负载                                 │             │
│   │  batch_size=16 → 4卡 × 4/卡                                │             │
│   └───────────────────────────────────────────────────────────┘             │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │  for each micro_batch:                                     │             │
│   │                                                            │             │
│   │    features (B, N, 120) ─► model(features, x_in)          │             │
│   │                                    │                       │             │
│   │                                    ▼                       │             │
│   │                            pred (B, N, 1)                  │             │
│   │                                                            │             │
│   │    loss = rel_l2(pred, target)                            │             │
│   │         = avg( ||pred-target||₂ / ||target||₂ )           │             │
│   │                                                            │             │
│   │    loss = loss / accum_steps                               │             │
│   │    loss.backward()                    ← DDP no_sync 处理   │             │
│   │                                                            │             │
│   │    if accum_steps ready:                                   │             │
│   │        clip_grad_norm_(1.0)                                │             │
│   │        optimizer.step()      ← AdamW, lr=1e-3              │             │
│   │        scheduler.step()      ← Warmup + Cosine Decay      │             │
│   │        ema.update(model)     ← decay=0.999                 │             │
│   │        optimizer.zero_grad()                                │             │
│   └───────────────────────────────────────────────────────────┘             │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │  evaluate_model (val_loader)                               │             │
│   │                                                            │             │
│   │  torch.no_grad():                                          │             │
│   │    features → ema.shadow(features, x_in) → pred            │             │
│   │    计算 MSE, RMSE, MAE, R², Rel-L2                          │             │
│   │    all_reduce 汇总多卡指标                                   │             │
│   └───────────────────────────────────────────────────────────┘             │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │  if val_rel_l2 < best_loss:                                │             │
│   │      torch.save(ema.shadow.state_dict(),                   │             │
│   │                 "best_geofno.pt")                          │             │
│   └───────────────────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 损失函数

| 名称 | `loss_type` 值 | 公式 |
| --- | --- | --- |
| 相对 L2（默认） | `"rel_l2"` | `mean(‖pred - target‖₂ / ‖target‖₂)` |
| RMSE | `"rmse"` | `sqrt(MSE(pred, target))` |

### 学习率调度

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
      warmup (5%)           cosine decay → 1% of lr

   total_steps  = num_epochs × optimizer_steps_per_epoch
   warmup_steps = total_steps × 0.05
   min_lr       = lr × 0.01
```

---

## 5. 超参数总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             超参数总览                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT_WINDOW = 24         输入时间窗口 (h)，硬编码于 temporal_utils.py      │
│   C_in  = 120               输入特征: 24×3 storm + 24×2 inner               │
│   C_out = 1                 输出: 单通道 h(t+24)                             │
│                                                                             │
│   modes        = 24         傅里叶模态数 (频域保留的低频分量)                  │
│   width        = 48         隐藏通道维度 (FNO 内部表示)                       │
│   s1, s2       = 64, 64     规则网格空间分辨率                                │
│   num_fno_layers = 4        FNO 中间层数 (规则网格上的谱卷积块数)              │
│   fc1_hidden   = 256        输出投影层隐藏维度                                │
│                                                                             │
│   batch_size   = 16         全局批次 (DDP 下均分到各卡)                       │
│   num_workers  = 4          DataLoader 进程数                               │
│                                                                             │
│   num_epochs   = 200        总训练轮次                                       │
│   lr           = 1e-3       峰值学习率 (AdamW)                               │
│   weight_decay = 1e-4       AdamW 权重衰减                                   │
│   warmup_ratio = 0.05       LR 线性预热比例 (总步数的 5%)                     │
│   min_lr_ratio = 0.01       LR 衰减下限 (峰值 lr 的 1%)                      │
│   grad_clip    = 1.0        梯度裁剪阈值                                      │
│   accum_steps  = 1          梯度累积步数 (>1 增大等效 batch)                  │
│   loss_type    = "rel_l2"   损失函数: rel_l2 或 rmse                         │
│   ema_decay    = 0.999      EMA 指数移动平均衰减率                            │
│                                                                             │
│   params       ≈ 31.9M      模型总参数量                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 评估

```bash
python model/test_all.py \
    --test_dir data/test \
    --coords data/coordinates.mat \
    --norm data/normalization.mat \
    --model best_geofno.pt \
    --output results.txt
```

对测试集每个文件的所有有效时间点做单步直接预测，评估指标：

| 指标 | 公式 | 说明 |
| --- | --- | --- |
| MSE | `mean((pred - target)²)` | 均方误差 |
| RMSE | `sqrt(MSE)` | 均方根误差 |
| MAE | `mean(\|pred - target\|)` | 平均绝对误差 |
| R² | `1 - SSE / SS_tot` | 决定系数 |
| Rel L2 | `‖pred - target‖₂ / ‖target‖₂` | 相对 L2 误差 |

分别在归一化空间和物理空间（通过 `normalization.mat` 反归一化）计算。干网格节点（物理水位 < 0.005m）自动排除。

---

## 7. 推理示例

```python
import torch
from model import GeoFNO2d
from dataset import load_static_coords, build_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coords = load_static_coords("data/coordinates.mat").to(device)

model = GeoFNO2d(modes1=24, modes2=24, width=48,
                 in_channels=120, out_channels=1,
                 s1=64, s2=64, num_fno_layers=4, fc1_hidden=256)
model.load_state_dict(torch.load("best_geofno.pt", map_location=device))
model.to(device).eval()

data = torch.load("event.pt", map_location=device)
storm = data["storm_boundary"]   # (24, N, 3)
inner = data["inner_boundary"]   # (24, N, 2)

features = build_features(storm, inner).unsqueeze(0)  # (1, N, 120)
x_in = coords.unsqueeze(0)                             # (1, N, 2)

with torch.no_grad():
    pred_h = model(features, x_in)  # (1, N, 1) — 第 25 小时水位预测
```

---

## 8. 常用命令

```bash
# 数据准备
python scripts/check_data_integrity.py --data_root data --deep
python scripts/build_manifest.py data/train --min_T_warn 25
python scripts/build_manifest.py data/val   --min_T_warn 25
python scripts/build_manifest.py data/test  --min_T_warn 25

# 训练
python model/main.py                              # 单 GPU
torchrun --nproc_per_node=4 model/main.py         # 4 GPU DDP

# 评估
python model/test_all.py --test_dir data/test

# 测试 & 监控
python -m pytest tests/ -v
tensorboard --logdir runs
```
