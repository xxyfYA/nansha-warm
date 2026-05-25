# test_all.py Rollout Redesign 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 改造 `model/test_all.py`：把跳过门槛从 `T > group_len` 降到 `T > bundle_size`，每文件单 rollout，用「贪心向前 + 必要时尾部 shift-back」算法，让 `data/test/` 里 64 个文件全部得到评估。

**Architecture:** 两遍式：pass-1 预扫描所有 `.pt` 拿 `T` 决定桶大小与跳过集合；pass-2 对每个 evaluable 文件单线程跑一次 rollout，target_steps = `min(max_rollout, T-1)`，bundle 调用次数始终 = `ceil(target_steps / bundle_size)`。指标累加结构、`init_bucket / compute_stats / compute_auc / metric_output_path / load_event_file / build_features_batch` 等不变；`parse_args / autoregressive_one_file / write_results / main` 修改；新增 `prescan_files` 辅助函数。

**Tech Stack:** Python, PyTorch, scipy.io, tqdm（已有依赖，无需新增）

**Reference spec:** `docs/superpowers/specs/2026-05-20-test_all-rollout-redesign.md`

---

## 文件结构

只动一个文件：`model/test_all.py`。无新文件，无测试基建变更。

| 函数 | 修改类型 | 行号位置 |
|---|---|---|
| `parse_args` | 修改参数 | 39–61 |
| `prescan_files` | 新增 | 插入到 `find_test_files` (241) 之后 |
| `autoregressive_one_file` | 重写函数体与签名 | 275–347 |
| `write_results` | 改 header + 新增 N 列 + 改签名 | 350–411 |
| `main` | 改流程：pass-1/桶分配/pass-2/result 拼装/调用新 write_results | 414–576 |

其它函数（`init_bucket / compute_stats / compute_auc / metric_output_path / load_event_file / load_normalization_stats / denormalize / apply_dry_grid_error_mask / strip_module_prefix / extract_state_dict / resolve_checkpoint_path / load_checkpoint / build_features_batch / find_test_files`）保持不变。

---

## Task 1: 新增 `prescan_files` 辅助函数

**Files:**
- Modify: `model/test_all.py`（在 `find_test_files` 之后插入）

- [ ] **Step 1: 在 `find_test_files` 后插入 `prescan_files`**

定位 `model/test_all.py` 中 `find_test_files` 函数（约 241–242 行）。在其后插入下面这段（保留前后各 1 个空行）：

```python
def prescan_files(file_paths, bundle_size, max_rollout):
    """Pass-1 scan: load each .pt to read graph T, classify evaluable vs skipped.

    Returns:
        evaluable: list of (path, T, target_steps) for files with T > bundle_size
        skipped: list of (path, T) for files with T <= bundle_size
        bucket_len: max target_steps across evaluable files (0 if none)
    """
    evaluable = []
    skipped = []
    for path in tqdm(file_paths, desc="Pre-scan T"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "graph" not in data:
            raise KeyError(f"{path}: missing 'graph' key; got {list(data.keys())}")
        num_time = int(data["graph"].shape[0])
        del data
        if num_time <= bundle_size:
            skipped.append((path, num_time))
            continue
        target_steps = min(max_rollout, num_time - 1)
        evaluable.append((path, num_time, target_steps))
    bucket_len = max((target_steps for _, _, target_steps in evaluable), default=0)
    return evaluable, skipped, bucket_len
```

- [ ] **Step 2: 验证函数被正确添加且 import 都齐全**

Run:
```bash
python -c "from model.test_all import prescan_files; print(prescan_files.__doc__.splitlines()[0])"
```
Expected: 打印 `Pass-1 scan: load each .pt to read graph T, classify evaluable vs skipped.`

- [ ] **Step 3: 在小子集上烟雾测试 prescan_files**

Run:
```bash
python -c "
from pathlib import Path
from model.test_all import prescan_files, find_test_files
files = find_test_files('data/test')[:8]
evaluable, skipped, bucket_len = prescan_files(files, bundle_size=8, max_rollout=72)
print(f'evaluable={len(evaluable)} skipped={len(skipped)} bucket_len={bucket_len}')
for p, T, ts in evaluable[:3]:
    print(f'  {p.name}: T={T} target_steps={ts}')
"
```
Expected: 输出 `evaluable=8 skipped=0 bucket_len=<某个值>`，并列出 3 个文件的 T / target_steps。

- [ ] **Step 4: 提交**

```bash
git add model/test_all.py
git commit -m "$(cat <<'EOF'
feat(test_all): add prescan_files helper for pass-1 T scan

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 改造 `parse_args`

**Files:**
- Modify: `model/test_all.py:39-61`

- [ ] **Step 1: 用下面的完整新版本替换 `parse_args` 函数**

整段 `def parse_args():` 到 `return parser.parse_args()` 替换为：

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Geo-FNO autoregressive test across a split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test split directory.")
    parser.add_argument("--coords", type=str, default="data/coordinates.mat", help="Path to coordinates.mat.")
    parser.add_argument("--norm", type=str, default="data/normalization.mat", help="Path to normalization.mat.")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--output", type=str, default="geofno_autoregressive_results.txt", help="Base output path.")
    parser.add_argument("--max_rollout", type=int, default=72, help="Max autoregressive rollout length per file (steps).")
    parser.add_argument("--num_files", type=int, default=None, help="Limit number of files for smoke tests.")
    parser.add_argument("--bundle_size", type=int, default=8, help="Bundle prediction size.")
    parser.add_argument("--channels", type=str, default="uvh", help="State channels to predict: any subset of u, v, h.")
    parser.add_argument("--allow_random_weights", action="store_true", help="Run without a checkpoint.")
    parser.add_argument("--modes", type=int, default=16, help="Fourier modes per axis.")
    parser.add_argument("--width", type=int, default=32, help="Model width.")
    parser.add_argument("--s1", type=int, default=64, help="Internal grid size along axis 1.")
    parser.add_argument("--s2", type=int, default=64, help="Internal grid size along axis 2.")
    parser.add_argument("--num_fno_layers", type=int, default=3, help="Number of FNO layers.")
    parser.add_argument("--device", type=str, default="auto", help="Device string or auto.")
    return parser.parse_args()
```

具体变化：
- 删除 `--group_len`、`--batch_size`
- 新增 `--max_rollout`（默认 72）
- 其余参数保持

> ⚠ 这一步会让旧版 `main()` 的 `args.group_len / args.batch_size` 引用瞬时失效；下一 Task 立刻补上。**本 Task 不单独提交**——`parse_args` 改完直接进入 Task 3 完成全部联动修改后一并提交。

---

## Task 3: 重写 `autoregressive_one_file`

**Files:**
- Modify: `model/test_all.py:275-347`

- [ ] **Step 1: 用下面的完整新版本替换 `autoregressive_one_file` 函数**

整段从 `def autoregressive_one_file(` 到该函数结尾的 `return {"evaluated_groups": int(num_groups), "skipped": False}` 替换为：

```python
def autoregressive_one_file(
    model,
    file_path,
    coords_2d_device,
    btype_oh_device,
    mean_sub,
    std_sub,
    mean_full,
    std_full,
    device,
    target_steps,
    bundle_size,
    per_step_metrics_by_space,
    state_channels,
):
    """Run a single autoregressive rollout (1 start) for `target_steps` and accumulate metrics.

    Uses greedy-forward bundle chaining; if the residual at the end is smaller than
    `bundle_size`, falls back to a shift-back tail bundle so the final output lands
    exactly at rel_idx == target_steps. Caller (prescan) guarantees T > bundle_size
    and target_steps >= bundle_size.
    """
    graph_all, storm_all, inner_all = load_event_file(file_path, coords_2d_device.size(0))
    num_time = graph_all.size(0)
    if target_steps > num_time - 1:
        raise ValueError(
            f"{file_path}: target_steps={target_steps} exceeds T-1={num_time - 1}"
        )

    x_in = coords_2d_device.unsqueeze(0)
    real_start = graph_all[0:1][..., list(state_channels)].to(device)
    predictions = [None] * target_steps

    covered = 0
    with torch.no_grad():
        while covered < target_steps:
            remaining = target_steps - covered
            if remaining >= bundle_size:
                input_rel = covered
                input_state = real_start if covered == 0 else predictions[covered - 1]
                use_full_block = True
            else:
                input_rel = target_steps - bundle_size
                input_state = predictions[input_rel - 1]
                use_full_block = False

            storm_window = storm_all[input_rel : input_rel + bundle_size + 1].unsqueeze(0).to(device)
            inner_window = inner_all[input_rel : input_rel + bundle_size + 1].unsqueeze(0).to(device)
            features = build_features_batch(input_state, storm_window, inner_window, btype_oh_device)
            bundle_out = model(features, x_in)

            if use_full_block:
                for i in range(bundle_size):
                    predictions[covered + i] = bundle_out[:, i]
                covered += bundle_size
            else:
                for j in range(remaining):
                    bundle_idx = bundle_size - remaining + j
                    predictions[covered + j] = bundle_out[:, bundle_idx]
                covered = target_steps

    with torch.no_grad():
        for step in range(target_steps):
            rel_idx = step + 1
            pred_norm = predictions[step]
            target_full_norm = graph_all[rel_idx : rel_idx + 1].to(device)
            target_norm_sub = target_full_norm[..., list(state_channels)]

            for metric_space in METRIC_SPACES:
                if metric_space == "physical":
                    pred_metric = denormalize(pred_norm, mean_sub, std_sub)
                    target_metric = denormalize(target_norm_sub, mean_sub, std_sub)
                else:
                    pred_metric = pred_norm
                    target_metric = target_norm_sub

                diff = pred_metric - target_metric
                diff = apply_dry_grid_error_mask(diff, target_full_norm, mean_full, std_full)
                bucket = per_step_metrics_by_space[metric_space][step]
                bucket["sse"] += torch.sum(diff ** 2, dim=(0, 1))
                bucket["sae"] += torch.sum(torch.abs(diff), dim=(0, 1))
                bucket["sum_gt"] += torch.sum(target_metric, dim=(0, 1))
                bucket["sum_sq_gt"] += torch.sum(target_metric ** 2, dim=(0, 1))

                l2_err = torch.norm(diff.permute(0, 2, 1), p=2, dim=2)
                l2_gt = torch.norm(target_metric.permute(0, 2, 1), p=2, dim=2).clamp(min=1e-8)
                bucket["rel_l2_sum"] += (l2_err / l2_gt).sum(dim=0)
                bucket["count"] += 1
```

具体变化：
- 签名：去除 `group_len`、`batch_size`；新增 `target_steps`
- 不再有外层 batch / `start_indices` 循环；每文件 1 start，固定从 index 0
- rollout 主循环改为「贪心向前 + 必要时 tail shift-back」
- 不再返回 `{"evaluated_groups", "skipped"}`——pass-1 已经决定一切，pass-2 只跑 evaluable 文件

> ⚠ 本 Task 不单独提交。

---

## Task 4: 改 `write_results`

**Files:**
- Modify: `model/test_all.py:350-411`

- [ ] **Step 1: 用下面的完整新版本替换 `write_results` 函数**

整段从 `def write_results(` 到该函数结尾替换为：

```python
def write_results(
    results_by_space,
    output_path,
    max_rollout,
    bundle_size,
    model_path,
    total_files,
    evaluated_files,
    skipped_files,
    selected_channel_names,
    channels_suffix,
):
    for metric_space in METRIC_SPACES:
        out = metric_output_path(output_path, metric_space, channels_suffix)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("Autoregressive Test Results\n")
            f.write(f"Max rollout: {max_rollout}\n")
            f.write(f"Bundle size: {bundle_size}\n")
            f.write(f"Channels: {''.join(selected_channel_names)}\n")
            f.write(f"Checkpoint: {model_path if model_path is not None else 'random weights'}\n")
            f.write(f"Metric Space: {metric_space}\n")
            f.write(
                f"Total files: {total_files} "
                f"(evaluated: {evaluated_files}, skipped: {skipped_files})\n"
            )
            f.write("=" * 110 + "\n")
            f.write(
                f"{'Step':<6} | {'Channel':<7} | {'MSE':<12} | {'RMSE':<12} | "
                f"{'MAE':<12} | {'R2':<12} | {'Rel L2':<12} | {'N':<6}\n"
            )
            f.write("-" * 110 + "\n")

            for result in results_by_space[metric_space]:
                step = result["step"]
                count = result["count"]
                for idx, channel in enumerate(selected_channel_names):
                    display_name = "wl" if channel == "h" else channel
                    step_label = str(step) if idx == 0 else ""
                    count_label = str(count) if idx == 0 else ""
                    metrics = result[channel]
                    f.write(
                        f"{step_label:<6} | {display_name:<7} | {metrics['mse']:<12.6f} | "
                        f"{metrics['rmse']:<12.6f} | {metrics['mae']:<12.6f} | "
                        f"{metrics['r2']:<12.6f} | {metrics['rel_l2']:<12.6f} | {count_label:<6}\n"
                    )
                f.write("-" * 110 + "\n")

            auc = compute_auc(results_by_space[metric_space], selected_channel_names)
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"AUC Summary Over {len(results_by_space[metric_space])} Steps\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"{'Channel':<11} | {'MSE Area':<12} | {'RMSE Area':<12} | "
                f"{'MAE Area':<12} | {'R2 Area':<12} | {'Rel L2 Area':<12}\n"
            )
            f.write("-" * 100 + "\n")
            for channel in selected_channel_names:
                display_name = "wl" if channel == "h" else channel
                metrics = auc[channel]
                f.write(
                    f"{display_name:<11} | {metrics['mse']:<12.6f} | {metrics['rmse']:<12.6f} | "
                    f"{metrics['mae']:<12.6f} | {metrics['r2']:<12.6f} | "
                    f"{metrics['rel_l2']:<12.6f}\n"
                )
            f.write("=" * 100 + "\n")
        print(f"[test] results -> {out}")
```

具体变化：
- 签名：`group_len` → `max_rollout`；`evaluated_groups` 删除；新增 `total_files`
- header 拆成多行：`Max rollout / Bundle size / Channels / Checkpoint / Metric Space / Total files (...)`
- per-step 表头加 `N` 列；分隔线从 100 字符扩到 110 字符
- 每个 step 内首个 channel 行的 N 列填值，其余 channel 行的 N 列留空（与 step 列一致的留空风格）

> ⚠ 本 Task 不单独提交。

---

## Task 5: 改 `main`

**Files:**
- Modify: `model/test_all.py:414-576`

- [ ] **Step 1: 用下面的完整新版本替换 `main` 函数**

整段从 `def main():` 到 `print("[test] done.")` 替换为：

```python
def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[test] device={device}")

    set_seed(3407)
    validate_temporal_params(args.bundle_size)
    if args.max_rollout < args.bundle_size:
        raise ValueError(
            f"max_rollout must be >= bundle_size: max_rollout={args.max_rollout}, "
            f"bundle_size={args.bundle_size}"
        )
    state_channels = parse_channels(args.channels)
    ch_suffix = channels_suffix(state_channels)
    num_channels = len(state_channels)
    selected_channel_names = tuple(CHANNEL_ORDER[channel] for channel in state_channels)

    coords_2d_cpu, btype_oh_cpu = load_static_coords(args.coords)
    coords_2d_device = coords_2d_cpu.to(device)
    btype_oh_device = btype_oh_cpu.to(device)
    num_nodes = coords_2d_cpu.size(0)

    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(
        args.norm,
        device=device,
        state_channels=state_channels,
    )

    in_channels = input_channels_for_bundle(args.bundle_size, num_channels)
    out_channels = output_channels_for_bundle(args.bundle_size, num_channels)
    model_args = {
        "bundle_size": args.bundle_size,
        "channels": "".join(selected_channel_names),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_channels": num_channels,
        "modes": args.modes,
        "width": args.width,
        "s1": args.s1,
        "s2": args.s2,
        "num_fno_layers": args.num_fno_layers,
    }
    model = GeoFNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        in_channels=in_channels,
        out_channels=out_channels,
        s1=args.s1,
        s2=args.s2,
        num_fno_layers=args.num_fno_layers,
        num_channels=num_channels,
    ).to(device)
    print(f"[test] model params={sum(p.numel() for p in model.parameters()):,}")

    default_checkpoint = build_checkpoint_name(args.bundle_size, ch_suffix)
    try:
        model_path = resolve_checkpoint_path(args.model, default_checkpoint)
    except FileNotFoundError:
        if not args.allow_random_weights:
            raise
        model_path = None

    if model_path is not None:
        print(f"[test] loading checkpoint {model_path}")
        load_checkpoint(model, model_path, device, model_args)
    else:
        print("[test] warning: using random weights (--allow_random_weights)")

    model.eval()

    test_files = find_test_files(args.test_dir)
    if args.num_files is not None:
        test_files = test_files[: args.num_files]
    if not test_files:
        raise FileNotFoundError(f"No .pt files in {args.test_dir}")

    evaluable, skipped, bucket_len = prescan_files(
        test_files, args.bundle_size, args.max_rollout
    )
    print(
        f"[test] prescan: total={len(test_files)} "
        f"evaluable={len(evaluable)} skipped={len(skipped)} bucket_len={bucket_len}"
    )
    if not evaluable:
        raise RuntimeError(
            f"No files eligible for evaluation. total={len(test_files)}, "
            f"skipped={len(skipped)}, bundle_size={args.bundle_size}"
        )

    per_step_metrics_by_space = {
        metric_space: [init_bucket(device, num_channels) for _ in range(bucket_len)]
        for metric_space in METRIC_SPACES
    }

    for path, _T, target_steps in tqdm(evaluable, desc="Test files"):
        autoregressive_one_file(
            model,
            path,
            coords_2d_device,
            btype_oh_device,
            mean_sub,
            std_sub,
            mean_full,
            std_full,
            device,
            target_steps,
            args.bundle_size,
            per_step_metrics_by_space,
            state_channels,
        )

    for metric_space in METRIC_SPACES:
        for step in range(bucket_len):
            bucket = per_step_metrics_by_space[metric_space][step]
            for key in ("sse", "sae", "sum_gt", "sum_sq_gt", "rel_l2_sum"):
                bucket[key] = bucket[key].detach().cpu().numpy()

    results_by_space = {metric_space: [] for metric_space in METRIC_SPACES}
    for metric_space in METRIC_SPACES:
        for step, bucket in enumerate(per_step_metrics_by_space[metric_space]):
            stats = compute_stats(bucket, num_nodes, num_channels)
            result = {"step": step + 1, "count": int(bucket["count"])}
            for idx, channel in enumerate(selected_channel_names):
                result[channel] = {
                    "mse": float(stats["mse_channels"][idx]),
                    "rmse": float(stats["rmse_channels"][idx]),
                    "mae": float(stats["mae_channels"][idx]),
                    "r2": float(stats["r2_channels"][idx]),
                    "rel_l2": float(stats["rel_l2_channels"][idx]),
                }
            results_by_space[metric_space].append(result)

            summary = f"[step {step + 1:02d}][{metric_space}][N={int(bucket['count'])}] "
            for channel in selected_channel_names:
                display_name = "wl" if channel == "h" else channel
                metrics = result[channel]
                summary += (
                    f"{display_name}: mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} "
                    f"mae={metrics['mae']:.6f} r2={metrics['r2']:.6f} "
                    f"rel_l2={metrics['rel_l2']:.6f} | "
                )
            print(summary.rstrip(" | "))

    write_results(
        results_by_space,
        args.output,
        args.max_rollout,
        args.bundle_size,
        model_path,
        total_files=len(test_files),
        evaluated_files=len(evaluable),
        skipped_files=len(skipped),
        selected_channel_names=selected_channel_names,
        channels_suffix=ch_suffix,
    )
    print("[test] done.")
```

具体变化：
- 删除 `if args.group_len % args.bundle_size != 0` 校验
- 新增 `if args.max_rollout < args.bundle_size` 校验
- 在 `find_test_files / --num_files` 截取后调用 `prescan_files`
- 桶按 `bucket_len` 预分配（不再按 `args.group_len`）
- 主循环改为遍历 `evaluable` 列表，每个文件 1 次 `autoregressive_one_file` 调用，签名匹配 Task 3
- 删除 `evaluated_files / evaluated_groups / skipped_files` 的逐文件累加逻辑——直接用 `prescan_files` 的返回值
- `result["count"]` 注入桶的 `count`，供 write_results 写 N 列
- 控制台逐 step 摘要里加 `[N=...]` tag
- 调用 `write_results` 用新签名（含 `total_files`，去 `evaluated_groups`）

- [ ] **Step 2: 静态检查文件能被 Python 解释器解析**

Run:
```bash
python -c "import ast; ast.parse(open('model/test_all.py').read()); print('OK')"
```
Expected: 输出 `OK`，无 SyntaxError。

- [ ] **Step 3: 静态检查 import 链与 `--help` 输出**

Run:
```bash
python model/test_all.py --help
```
Expected: 显示新参数表，包含 `--max_rollout` 默认 72，**不包含** `--group_len`、`--batch_size`。

- [ ] **Step 4: 烟雾测试——小子集 + 随机权重**

Run（如机器无 CUDA 把 `--device` 换成 `cpu`；data/test/、data/coordinates.mat、data/normalization.mat 都在仓库相对路径下）:
```bash
python model/test_all.py \
  --test_dir data/test \
  --allow_random_weights \
  --num_files 8 \
  --max_rollout 72 \
  --bundle_size 8 \
  --output /tmp/test_smoke.txt
```
Expected:
- 控制台先打印一行 `[test] prescan: total=8 evaluable=8 skipped=0 bucket_len=<n>`（`<n>` 在 10..72 之间，取决于这 8 个文件里最大 T-1 与 72 的较小值）
- 然后是 tqdm 进度条 + 每 step 一行 `[step XX][physical][N=...] ...` 与 `[normalized]` 对应输出
- 最终 `/tmp/test_smoke_physical.txt`、`/tmp/test_smoke_normalized.txt` 两个文件生成

- [ ] **Step 5: 检查输出文件 header 与 N 列**

Run:
```bash
head -10 /tmp/test_smoke_physical.txt
```
Expected: 显示
```
Autoregressive Test Results
Max rollout: 72
Bundle size: 8
Channels: uvh
Checkpoint: random weights
Metric Space: physical
Total files: 8 (evaluated: 8, skipped: 0)
==============================================================================================================
Step   | Channel | MSE          | RMSE         | MAE          | R2           | Rel L2       | N
--------------------------------------------------------------------------------------------------------------
```

Run:
```bash
grep -E "^[0-9]+ " /tmp/test_smoke_physical.txt | head -3
```
Expected: 显示前 3 行类似 `1      | u       | <num>        | <num>        | <num>        | <num>        | <num>        | 8     `，最后一列是当前 step 的贡献文件数。

- [ ] **Step 6: 全集烟雾——验证 64 文件全部 evaluable**

Run:
```bash
python model/test_all.py \
  --test_dir data/test \
  --allow_random_weights \
  --max_rollout 72 \
  --bundle_size 8 \
  --output /tmp/test_full.txt
```
Expected: 控制台第一条 `[test]` 输出包含 `prescan: total=64 evaluable=64 skipped=0 bucket_len=72`。

- [ ] **Step 7: 验证 N 列单调不增**

Run:
```bash
grep -E "^[0-9]+ " /tmp/test_full_physical.txt | awk -F'|' '{print $1, $NF}' | head -30
```
Expected: 前 10 行 N 都是 64（所有文件 target_steps >= 10），随 step 增大 N 单调不增；step 72 那一行 N 应该是 20（T=89/10 个 + T=113/9 个 + T=121/1 个）。

> 若 N 列对不上，回到 Task 3 / Task 5 找 bug；常见原因：`result["count"]` 没填进去、或 prescan 的 target_steps 算错。

- [ ] **Step 8: 提交（合并 Task 2–5 的所有修改）**

Run:
```bash
git status
git diff --stat
```
Expected: `model/test_all.py` 是唯一被修改的文件。

Run:
```bash
git add model/test_all.py
git commit -m "$(cat <<'EOF'
refactor(test_all): single-rollout per file with greedy+shift tail

Replace fixed-group_len multi-start rollout with one rollout per file:
target_steps = min(max_rollout, T-1). Greedy forward bundle chaining,
falling back to a shift-back tail when the residual is < bundle_size,
so the final prediction lands exactly at rel_idx == target_steps.
Skip threshold drops from T <= group_len to T <= bundle_size.

CLI:
- rename --group_len to --max_rollout
- drop --batch_size (single start per file)
- drop group_len % bundle_size divisibility check
- add max_rollout >= bundle_size validation

Output reports gain a per-step N column (files contributing to that step)
and a "Total files: X (evaluated: A, skipped: B)" header. Per-step metrics
are now aggregated across one prediction per file, with deep-step samples
naturally smaller as reflected by N.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Run:
```bash
git status
```
Expected: working tree clean.

---

## Self-Review 结果（计划写完后已 inline 修订）

- **Spec coverage 检查**：spec §4 参数变化 → Task 2；§5 算法 → Task 3；§6.1 pass-1 → Task 1；§6.2/6.3 pass-2 与桶分配 → Task 5；§7 metric 累加 → Task 3 后半段；§8 输出报告 → Task 4；§10 错误处理（`max_rollout < bundle_size`）→ Task 5；§12 烟雾测试 → Task 5 Steps 4–7。无缺失。
- **Placeholder 扫描**：全文已搜索 TBD/TODO/etc，无残留。每个 Step 都给了具体代码或具体命令。
- **类型/签名一致性**：`autoregressive_one_file` 新签名 `(model, path, coords, btype, mean_sub, std_sub, mean_full, std_full, device, target_steps, bundle_size, per_step_metrics_by_space, state_channels)` 与 Task 5 `main` 里的调用对齐；`write_results` 新签名 `(results_by_space, output_path, max_rollout, bundle_size, model_path, total_files, evaluated_files, skipped_files, selected_channel_names, channels_suffix)` 与 Task 5 调用对齐；`prescan_files` 返回 `(evaluable, skipped, bucket_len)` 与 main 解包对齐。
- **CHANNEL 子集兼容**：`real_start = graph_all[0:1][..., list(state_channels)]` 跟旧代码 `graph_all[batch_starts][..., list(state_channels)]` 一致；target 端 `target_full_norm = graph_all[rel_idx : rel_idx + 1]`（保留全 3 通道用于 dry mask）后再切 `state_channels`。
