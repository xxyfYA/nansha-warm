# 设计：训练时对 h 状态通道注入随机噪音以缓解自回归漂移

- 日期：2026-05-24
- 范围：仅 `model/train.py` 与 `model/main.py`，零侵入推理路径
- 默认行为：完全不改变（`noise_sigma=0.0` 等价于现行训练）

## 1. 背景与动机

当前模型 `GeoFNO2d`（[model/model.py](../../../model/model.py)）以 residual 形式预测 h 通道未来 `bundle_size` 步：

```
pred_block = state_in + delta,   state_in = u[..., :1]
```

训练为 teacher-forced 单次预测：每个样本输入 ground-truth `h(t)`，输出 `h(t+1..t+S)`（[train.py:202-209](../../../model/train.py#L202)）。

推理为块自回归 rollout：从第二个 bundle 块开始，`input_state` 是上一个 bundle 的预测值（[test_all.py:321](../../../model/test_all.py#L321)）。因此训练分布与推理分布存在系统差异，rollout 误差随步数累积，长序推演损失显著恶化。

本方案在训练循环中向 `features[..., 0:1]`（h state 通道）注入 i.i.d. 高斯噪音，使模型学会从"带偏差的 h"出发仍能预测正确未来，从而对 rollout drift 鲁棒。这是 PDE 神经代理（Brandstetter et al. 2022, *Message Passing Neural PDE Solvers*）与气象代理（FourCastNet, GraphCast）中的标准技巧。

## 2. 设计决策（已与用户确认）

| 决策项 | 取值 | 理由 |
|---|---|---|
| 噪音作用通道 | 仅 `features[..., 0:1]`（h state） | 这是 rollout 唯一会被反馈的量，与 drift 源精确对应；storm/inner_boundary 在 rollout 时仍是已知外力，加噪音会改变问题语义 |
| 噪音幅度 | CONFIG 暴露 `noise_sigma`，默认 0.0 | 手动扫值（如 {0, 0.005, 0.01, 0.02, 0.05}）；不写死也不引入额外调度参数 |
| 注入位置 | `train.py` 训练循环内，`features.to(device)` 之后 | GPU 上抽样最快；与 dataset/model 解耦；评估路径自动豁免 |
| 目标是否同步加噪音 | 否，target 保持干净 | denoising 风格；与"修正 rollout drift"初衷一致 |
| 空间结构 | 节点独立 i.i.d. Gaussian | 实现最简；已被多项 PDE 代理工作经验证有效 |
| 应用频率 | 每个 training step 都加 | 仅一个超参；首次实验最小化变量 |
| 训练期 rollout 验证 | 不加 | 本 PR 范围保持最小；用户用 `test_all.py` 手工验证 |

## 3. 代码改动

### 3.1 `model/main.py`

在 `CONFIG` 字典（[main.py:49](../../../model/main.py#L49)）新增一项：

```python
"noise_sigma": 0.0,   # σ on the h state-in channel during training; 0 disables.
```

在 `main()` 调用 `train_model` 处透传该参数：

```python
train_model(
    ...
    noise_sigma=CONFIG["noise_sigma"],
)
```

无须新增 tensorboard 专项 scalar — 该值已经会通过 `config/all` markdown 表（[main.py:330-343](../../../model/main.py#L330)）记录到 TB，可在不同 run 之间溯源。

### 3.2 `model/train.py`

`train_model` 签名新增参数：

```python
def train_model(
    ...
    noise_sigma: float = 0.0,
):
```

在函数顶部新增校验（与既有 `accum_steps` 校验同风格）：

```python
if noise_sigma < 0.0:
    raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")
```

在 micro-batch 循环里，`features.to(device)` 之后、`model.forward` 之前注入：

```python
features = features.to(device, non_blocking=True)
target_block = target_block.to(device, non_blocking=True)
if noise_sigma > 0.0:
    features[..., 0:1].add_(torch.randn_like(features[..., 0:1]) * noise_sigma)
batch_size = features.shape[0]
```

要点：
- `features[..., 0:1].add_(...)`：保形 view，原地加，仅动 h state 通道；`features[..., 1:]` 不受影响。
- `torch.randn_like(...)`：在 `features` 所在 device 抽样，节点独立 i.i.d.，batch 维度自然广播到正确形状。
- `noise_sigma == 0.0` 路径完全跳过，零开销；默认零回归。
- 仅训练循环内生效；`evaluate_model()`（同文件第 76 行）保持现状，不引入任何路径分支。

## 4. 与现有特性的交互

- **EMA**（[train.py:36-54](../../../model/train.py#L36)）：噪音仅影响 forward 输入；live 权重因此略偏移，shadow 模型自然跟着指数平均。这正是我们想要的鲁棒解，无须特殊处理。
- **梯度累积**：每个 micro-batch 独立抽噪音，等价于在一个 effective batch 内见多种扰动，正面效应。
- **DDP**：每张卡独立抽噪音（不同 rank RNG 状态不同），不需要 broadcast；梯度 all-reduce 仍照常。
- **可复现性**：`set_seed()`（[main.py:84](../../../model/main.py#L84)）已设全局 seed，但 `cudnn.deterministic=False`，本身非 bit-exact 配置；新增噪音不让情况更差。
- **Checkpoint 兼容**：state_dict 结构不变；旧 checkpoint 可直接被 `test_all.py` 加载，新 checkpoint 也兼容旧推理代码。
- **DataLoader worker / shared memory**：噪音在主进程 GPU 上加，worker 不参与，避免 fork / RNG 复杂性。

## 5. 推理路径与有效性评估

- `model/test_all.py` 完全不动 — 这是衡量"噪音注入是否真的改善了 rollout"的客观尺。
- `evaluate_model()` 完全不动；它是 bundle 单次 teacher-forced 指标，加噪音后该指标可能略微变差，**这是正常现象，不代表退步**。
- 判断有效的方法：训练完成后，分别用 `noise_sigma=0` 与扫值跑 `test_all.py`，对比 `geofno_autoregressive_results_*_physical.txt` 中 step-wise RMSE 与 AUC (Rel L2 Area)。后者应当在长 rollout（如 ≥24 步）上有改善。
- Checkpoint 命名：当前 [temporal_utils.py:56](../../../model/temporal_utils.py#L56) `build_checkpoint_name` 只看 `bundle_size`。若并行扫多个 σ，需手动重命名输出文件（如 `best_geofno_b8_noise0p01.pt`）。**本次 PR 不改文件命名**，保持最小变动。

## 6. 测试计划

新增轻量单元测试（位于 [tests/](../../../tests/)），不依赖真实数据：

1. **`test_noise_zero_is_noop`**：构造固定 features / target，调用一步 train loop，`noise_sigma=0.0`；同 seed 下 loss 与无 PR 代码 bit-identical（或在容差内一致）。
2. **`test_noise_only_h_channel`**：clone 一份 features，应用注入逻辑后断言 `features[..., 1:]` 完全未变，`features[..., 0:1]` 改变；验证扰动统计量（mean ≈ 0，std ≈ σ，在合理样本量下）。
3. **`test_eval_path_clean`**：走一次 `evaluate_model()` 路径，确认 features 未被加噪音（结构上即成立，作为 belt-and-suspenders 保护）。
4. **`test_negative_sigma_rejected`**：传 `noise_sigma=-0.1` 应抛 `ValueError`。

非测试性校验：
- `python -c "from train import train_model"` import sanity。
- 一次小 batch 假数据的 forward + backward smoke test（与本仓库其它 spec 验收风格一致）。

不写"σ>0 真能改善 rollout"端到端测试 — 太慢、太脆、属实证而非单测，应当离线手工跑 `test_all.py` 对比。

## 7. 范围之外（明确不做）

- σ 的训练期调度（ramp-up / decay）：作为方案 B 后续可扩展，本 PR 不实现。
- 噪音的空间相关结构（RBF / 邻居平滑）：复杂度大幅上升，本 PR 不实现。
- 训练期自回归 rollout 验证指标：增加 PR 范围与训练时间，本 PR 不实现。
- 输出 checkpoint 文件名包含 σ：本 PR 不改命名约定。
- 对 storm_boundary / inner_boundary / btype 等其它通道加噪音：与 drift 源不对应，本 PR 不做。

## 8. 回滚

设 `CONFIG["noise_sigma"] = 0.0` 即等价于 PR 前行为。代码层面，删除 `train_model` 内 ~3 行注入逻辑、CONFIG 一项、`train_model` 一个参数即可彻底回滚；无 schema/checkpoint/数据迁移负担。
