# `test_all.py` 自回归 rollout 重设计

- 日期：2026-05-20
- 涉及文件：`model/test_all.py`
- 数据约束：64 个测试文件，`T ∈ [11, 121]`，13 个唯一 T 值

## 1. 背景与动机

当前 `model/test_all.py` 用「`group_len` 步定长 rollout × 多起点」的方式评估模型。判定文件可用的条件是 `T > group_len`：当 `group_len=72`（默认）时，`data/test/` 里 **64 个文件中有 44 个会被自动跳过**（T ∈ {11, 17, 25, 29, 37, 41, 43, 49, 53, 65}），相当于 69% 的测试集没有被评估。

但模型的 bundle 预测在数据层面只需要 `T > bundle_size`（一次 bundle 调用要 `bundle_size+1` 个 storm/inner 时间点）。短文件其实可以通过「贪心向前自回归 + 必要时尾部回退」的方式被完整 rollout 出来。本设计调整 `test_all.py` 的 rollout 与聚合逻辑，使得：

- 跳过条件下降到 `T ≤ bundle_size`
- 每个文件只跑一次 rollout（从 index 0 出发），自回归 bundle 调用次数为 `ceil(target_steps / bundle_size)`
- rollout 长度由新参数 `max_rollout` 统一封顶；超出此长度的文件按上限截断，不足此长度的文件用尾部 shift-back 补齐

## 2. 目标

- 让 `T > bundle_size` 的所有文件都能产出指标
- 每个文件每个步只产生一个预测，最少化 bundle 调用次数（`ceil(target_steps/B)`）
- 报告里展示 1..bucket_len 的 per-step 指标，并附「贡献文件数」列
- 与现有的归一化/物理两种 metric space、CHANNEL 子集、checkpoint 命名、`--allow_random_weights` 等行为保持兼容

## 3. 非目标

- 不改动模型架构、训练流程、loss 与归一化逻辑
- 不实现跨文件 batch 并行（每文件单独跑，沿用现有的「文件粒度循环」结构）
- 不引入 T 值缓存机制（pass-1 + pass-2 重新 load 文件，64 文件量级 I/O 可接受）

## 4. 参数变化

| 旧 | 新 | 说明 |
|---|---|---|
| `--group_len`（int，默认 72） | `--max_rollout`（int，默认 72） | 每个文件 rollout 的步数上限 |
| `--batch_size`（int，默认 1） | 删除 | 新逻辑下每个文件只 1 个起点，无 batch 维 |
| 校验 `group_len % bundle_size == 0` | 删除 | 算法在 `max_rollout % bundle_size != 0` 时通过 tail shift-back 处理 |

其它参数（`--test_dir / --coords / --norm / --model / --output / --num_files / --bundle_size / --channels / --allow_random_weights / --modes / --width / --s1 / --s2 / --num_fno_layers / --device`）保持不变。

## 5. 算法

### 5.1 跳过条件

```
若 T ≤ bundle_size 则跳过；否则评估
```

### 5.2 每文件目标步数

```
target_steps = min(max_rollout, T - 1)
```

含义：要为 rel_idx ∈ {1, 2, …, target_steps} 这些时间步都产出 1 个预测，并与 ground truth 比较。

### 5.3 主循环

```python
predictions = [None] * target_steps    # rel_idx i 的预测放在 predictions[i-1]
covered = 0

while covered < target_steps:
    remaining = target_steps - covered
    if remaining >= bundle_size:
        # —— 贪心向前 ——
        input_rel = covered                                   # 0 表示用 real[0]
        input_state = real[0] if covered == 0 else predictions[covered - 1]
        storm_window = storm_all[input_rel : input_rel + B + 1]
        inner_window = inner_all[input_rel : input_rel + B + 1]
        features = build_features_batch(input_state, storm_window, inner_window, btype_oh)
        bundle_out = model(features, x_in)                    # shape: [1, B, N, K]
        for i in range(bundle_size):
            predictions[covered + i] = bundle_out[:, i]
        covered += bundle_size
    else:
        # —— 尾部 shift-back ——
        shift_rel = target_steps - bundle_size                # 始终 >= 1（因为 covered>=B 时才进入）
        input_state = predictions[shift_rel - 1]              # 已被之前的 forward bundle 填好
        storm_window = storm_all[shift_rel : shift_rel + B + 1]
        inner_window = inner_all[shift_rel : shift_rel + B + 1]
        features = build_features_batch(input_state, storm_window, inner_window, btype_oh)
        bundle_out = model(features, x_in)
        for j in range(remaining):
            bundle_idx = bundle_size - remaining + j
            predictions[covered + j] = bundle_out[:, bundle_idx]
        covered = target_steps
```

bundle 调用次数始终 = `ceil(target_steps / bundle_size)`。

### 5.4 算法不变量

- 进入 `else` 分支时 `covered >= bundle_size`，所以 `shift_rel = target_steps - bundle_size ≥ 1`，`predictions[shift_rel - 1]` 一定已被前面的 forward bundle 写入。
- `else` 分支只会触发 1 次（用完即 `covered = target_steps`）。
- forward 分支与 tail 分支的 storm/inner window slice 都在 `[0, T-1]` 范围内：
  - forward 时 `input_rel + B = covered + B ≤ target_steps ≤ T - 1`
  - tail 时 `shift_rel + B = target_steps ≤ T - 1`

### 5.5 不同 T 下的行为对照

| T | bundle_size | max_rollout | target_steps | bundle 次数 | tail 是否触发 |
|---|---|---|---|---|---|
| 8 | 8 | 72 | — | — | 跳过 |
| 9 | 8 | 72 | 8 | 1 | 否 |
| 11 | 8 | 72 | 10 | 2 | 是（remaining=2） |
| 17 | 8 | 72 | 16 | 2 | 否 |
| 41 | 8 | 72 | 40 | 5 | 否 |
| 43 | 8 | 72 | 42 | 6 | 是（remaining=2） |
| 65 | 8 | 72 | 64 | 8 | 否 |
| 72 | 8 | 72 | 71 | 9 | 是（remaining=7） |
| 73 | 8 | 72 | 72 | 9 | 否 |
| 89 | 8 | 72 | 72 | 9 | 否 |
| 121 | 8 | 72 | 72 | 9 | 否 |
| 121 | 8 | 70 | 70 | 9 | 是（remaining=6） |

## 6. 数据流

### 6.1 Pass 1：预扫描

```
对每个 .pt：
    data = torch.load(path, map_location="cpu", weights_only=False)
    T = data["graph"].shape[0]
    del data
    若 T ≤ bundle_size：归入 skipped_files
    否则：记录 (path, T, target_steps = min(max_rollout, T - 1))
bucket_len = max(target_steps across evaluable files, default=0)
若 bucket_len == 0：raise（与现有「No evaluation groups」错误等价）
```

### 6.2 Pass 2：每文件推理

- 对每个 evaluable 文件：load → 跑 5.3 主循环 → 按 5.6 的方式更新桶 → del 释放
- 不再有「外层 group/batch 循环」，每个文件就是一次 rollout

### 6.3 桶分配

```
per_step_metrics_by_space = {
    metric_space: [init_bucket(device, num_channels) for _ in range(bucket_len)]
    for metric_space in METRIC_SPACES
}
```

不再动态增长（pass-1 已知大小）。

## 7. Metric 累加（保留现有逻辑，每步 batch=1）

对 `step ∈ [0, target_steps)`：

```python
rel_idx = step + 1
pred_norm     = predictions[step]                         # shape: [1, N, K]
target_full   = graph_all[rel_idx : rel_idx + 1].to(device)
target_norm_sub = target_full[..., list(state_channels)]

for metric_space in METRIC_SPACES:
    if metric_space == "physical":
        pred_metric   = denormalize(pred_norm, mean_sub, std_sub)
        target_metric = denormalize(target_norm_sub, mean_sub, std_sub)
    else:
        pred_metric, target_metric = pred_norm, target_norm_sub

    diff = pred_metric - target_metric
    diff = apply_dry_grid_error_mask(diff, target_full, mean_full, std_full)

    bucket = per_step_metrics_by_space[metric_space][step]
    bucket["sse"]       += torch.sum(diff ** 2, dim=(0, 1))
    bucket["sae"]       += torch.sum(torch.abs(diff), dim=(0, 1))
    bucket["sum_gt"]    += torch.sum(target_metric, dim=(0, 1))
    bucket["sum_sq_gt"] += torch.sum(target_metric ** 2, dim=(0, 1))
    l2_err = torch.norm(diff.permute(0, 2, 1), p=2, dim=2)
    l2_gt  = torch.norm(target_metric.permute(0, 2, 1), p=2, dim=2).clamp(min=1e-8)
    bucket["rel_l2_sum"] += (l2_err / l2_gt).sum(dim=0)
    bucket["count"]      += 1
```

Dry-mask、normalize/denormalize、相对 L2、SSE/SAE 累加规则全部保留。

## 8. 输出报告

`metric_output_path(...)` 命名规则不变（仍是 `<stem><ch_suffix>_<metric_space>.<suffix>`）。

每个 metric space 文件的 header：

```
Autoregressive Test Results
Max rollout: <max_rollout>
Bundle size: <bundle_size>
Channels: <selected_channel_names joined>
Checkpoint: <path | "random weights">
Metric Space: <physical | normalized>
Total files: <total> (evaluated: <eval>, skipped: <skip>)
====================================================================================================
Step  | Channel | MSE         | RMSE        | MAE         | R2          | Rel L2      | N
----------------------------------------------------------------------------------------------------
```

- 删除原 header 中的 `group_len=` / `bundle_size=`（合并到独立行）和 `Evaluated groups`
- 新增 `Max rollout` / `Bundle size` 字段
- 行结构里增加 `N` 列（= `bucket["count"]`，即贡献该 step 的文件数）

每个 step 的 3 行（u/v/wl，按 `selected_channel_names`）：每个 channel 行末附 `N`。`N` 值在同一个 step 内对每个 channel 相同。AUC 表保持原样，沿用梯形积分覆盖整个 `bucket_len`（深 step 的统计噪声大由 `N` 列体现）。

## 9. 受影响的函数清单

| 函数 | 修改类型 | 内容 |
|---|---|---|
| `parse_args` | 修改 | 删除 `--batch_size`，把 `--group_len` 改名为 `--max_rollout` |
| `main` | 修改 | 删除 `group_len % bundle_size` 校验；新增 pass-1 预扫描；桶按 `bucket_len` 预分配；不再传 `args.batch_size`；改向 `autoregressive_one_file` 传 `target_steps` |
| `autoregressive_one_file` | 重写 | 实现 5.3 主循环（贪心向前 + tail shift-back）；签名改：去 `group_len/batch_size/per_step_metrics_by_space/start_indices` 不再需要 num_groups 计算；返回 `target_steps` |
| `init_bucket / compute_stats` | 不变 | — |
| `compute_auc` | 不变 | 仍按梯形积分覆盖输入的全部 step |
| `write_results` | 修改 | header 字段更新；step 行加 `N` 列；签名调整 |
| `build_features_batch` | 不变 | batch=1 时退化使用 |
| `load_event_file / find_test_files / load_normalization_stats / strip_module_prefix / extract_state_dict / resolve_checkpoint_path / load_checkpoint` | 不变 | — |
| `metric_output_path` | 不变 | — |

## 10. 错误处理与边界

- `T ≤ bundle_size`：归 skipped，pass-2 不读取
- `evaluable == []`：raise 与现有「No evaluation groups」语义一致的 `RuntimeError`
- 文件 load 失败：保持现有 behavior（直接抛错；本设计不引入额外的 try/except）
- `max_rollout < bundle_size`：raise `ValueError`（target_steps 永远会进 tail 分支但 `shift_rel < 1` 无效）—— 在 args 解析后立即校验
- 与现有 `validate_temporal_params(bundle_size)` 校验顺序保持一致

## 11. 兼容性

- Checkpoint：完全兼容（模型架构不变）
- 输出文件名：保持现状（`<stem><ch_suffix>_<metric_space>.<suffix>`）
- CLI：`--group_len` 和 `--batch_size` 删除属于 breaking change；如果有外部脚本依赖，需要同步更新（仓库内目前未见有 shell 脚本固定调用 `test_all.py` 的位置——实现阶段会再 grep 一次确认）
- 报告文本格式：表头多了若干行、step 行多了 `N` 列；解析者需要更新

## 12. 测试策略

不引入新的 pytest 基建。手动 smoke：

1. **小合成 .pt**：写一个临时脚本，在 tmp 目录造若干 .pt（T 分别为 8、9、11、17、72、73），调用 `test_all.py --test_dir tmp --allow_random_weights --max_rollout 72 --bundle_size 8`，断言：
   - T=8 的文件 skip
   - 其他 5 个文件 evaluable
   - 输出 `*_physical.txt` 与 `*_normalized.txt` 都存在
   - `Step 1` 行 `N=5`；`Step 8` 行 `N=5`；`Step 9..10` 行 `N=4`（T=9 已停止贡献）；`Step 11..16` 行 `N=3`；`Step 17..71` 行 `N=2`（仅 T=72/73）；`Step 72` 行 `N=1`（仅 T=73 达到 target_steps=72）
2. **真实 test 集**：`python test_all.py --test_dir data/test --allow_random_weights --max_rollout 72 --num_files 8 --bundle_size 8`，确认 64 个文件 64 evaluated / 0 skipped，桶长度 72。
3. **回归：与旧逻辑在长文件子集上数值对比**：旧逻辑 group_len=72 时长文件（T=89/113/121）的 step 1..72 指标应能用新逻辑（max_rollout=72）在「同 batch_size=1、同种子」下复现。如有偏差应该来自「多起点 vs 单起点」聚合差异，逐文件 dump 单起点的预测应该 byte-equal。

## 13. 风险与权衡

- **每文件 N=1**：深 step（接近 max_rollout）的统计样本数由长文件数量决定。在当前 64 文件分布下：display step 72 处 `N = 20`（T=89 的 10 个 + T=113 的 9 个 + T=121 的 1 个）。比起旧逻辑（按 num_groups 累加，每个长文件贡献多个起点）样本数更少。这是有意的设计折中——「更易解释的单 rollout 指标 vs 更多起点带来的噪声平滑」，符合用户「用尽量小的自回归数量」原则。
- **跨文件 batch 损失的吞吐量**：在 64 文件量级、单 GPU、单卡显存能装下一个文件 forward 的情形下，差距可接受；如需进一步加速，后续可以单独立项做「同 T 分组 batch」优化。
- **Pass-1 I/O 翻倍**：每个 .pt 在 pass-1 和 pass-2 各 load 一次。64 文件 × 单文件 ~350 MB，全部读两次约 45 GB。在 SSD 上几分钟可完成，不视为瓶颈。如后续测试集规模扩大可加 T-cache。

## 14. 实施顺序（供后续 writing-plans 参考）

1. CLI 改动（`--max_rollout` 替换 `--group_len`，删除 `--batch_size`，新增校验）
2. 抽离 `prescan_files` 函数（pass-1）
3. 重写 `autoregressive_one_file` 为单文件单起点 + 贪心+shift 算法
4. `main` 串联 pass-1 / pass-2 / 桶分配 / 输出
5. 更新 `write_results` 的 header 与 step 行
6. 手动 smoke + 真实 test 集回归对比
