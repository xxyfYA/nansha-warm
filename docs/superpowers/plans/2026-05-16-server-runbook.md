# Server Runbook -- Geo-FNO Refactor 验证

本文档用于在服务器上验证 Geo-FNO data-compat refactor。当前仓库本地不强制跑训练或全量测试；以下命令面向具备数据、GPU 和依赖环境的服务器。

## 前置环境

- Python 环境包含 PyTorch 2.x、CUDA 可用版本、`scipy`、`tqdm`、`tensorboard`、`pytest`。
- 数据目录存在：`data/train`、`data/val`、`data/test`。
- 静态文件存在：`data/coordinates.mat`、`data/normalization.mat`。
- 当前训练入口使用 `model/main.py` 中的 `CONFIG`；默认 b72 配置为 `bundle_size=72`。
- 当前测试入口 `model/test_all.py` 默认对齐 b72：`--bundle_size 72`、`--group_len 72`、`--batch_size 1`。

## 1. 构建 Manifest

在仓库根目录执行：

```bash
python scripts/build_manifest.py data/train --bundle_size_warn 72
python scripts/build_manifest.py data/val --bundle_size_warn 72
python scripts/build_manifest.py data/test --bundle_size_warn 72
```

期望结果：

- 三个 split 目录下分别生成或更新 `manifest.json`。
- 命令会对 `T <= 72` 的文件给出 warning；这类文件在 b72 训练或测试中不会产生有效样本，需要结合输出数量确认是否符合预期。

## 2. 单元测试

执行：

```bash
pytest tests/ -v
```

期望结果：

- `tests/test_temporal_utils.py` 当前 8 项全部 PASS。
- `tests/test_dataset.py` 当前 17 项全部 PASS。
- 总计当前应为 25 项 PASS。若测试数量变化，以当前 `tests/` 中实际 `test_*.py` 为准，但需要确认 temporal 和 dataset 两类覆盖仍在。

## 3. 单卡冒烟：`bundle_size=1`

目的：先验证数据读取、feature layout、模型 forward、loss、checkpoint、TensorBoard 写入的最小训练链路。

临时修改 `model/main.py` 的 `CONFIG`：

```python
"bundle_size": 1,
"batch_size": 4,
"num_epochs": 2,
```

建议只使用 `train` 中 1-2 个文件试跑，避免把冒烟变成全量训练。可选做法：

- 新建临时小目录，软链接或复制少量 `.pt` 文件后重新构建该目录的 `manifest.json`，并把 `CONFIG["train_dir"]` / `CONFIG["val_dir"]` 临时指向小目录。
- 或临时保存小 manifest，只包含 1-2 个训练文件和少量验证文件。

单卡执行：

```bash
python model/main.py
```

观察点：

- 训练能完成 2 个 epoch。
- loss 非 NaN/Inf。
- TensorBoard run 目录写入正常。
- 生成的 checkpoint 名称符合 `bundle_size=1` 语义：`best_geofno.pt`。

冒烟完成后，恢复 `CONFIG` 到目标 b72 训练设置。

## 4. 全量 4 卡 DDP：`bundle_size=72`

确认 `model/main.py` 的 `CONFIG` 使用目标设置，例如：

```python
"bundle_size": 72,
"batch_size": 16,
"num_epochs": 100,
```

全量训练：

```bash
torchrun --nproc_per_node=4 model/main.py
```

监控点：

- 每张 GPU 显存占用是否稳定，确认 lazy 加载和 file-affine sampler 没有导致 rank 间显存或内存异常增长。
- TensorBoard 中 train/val loss 是否正常写入并下降或至少保持数值稳定。
- DataLoader worker 耗时是否可接受；如果 worker 成为瓶颈，优先检查 `num_workers`、`lru_files_per_worker`、磁盘吞吐和 manifest 中过短文件比例。
- DDP 日志中没有 rank hang、NCCL error、manifest 缺失或 shape mismatch。

期望 checkpoint：

- b72 默认最优权重文件为 `best_geofno_b72.pt`。

## 5. 测试链路：4 文件 smoke

先用少量测试文件验证 autoregressive 测试链路：

```bash
python model/test_all.py --test_dir data/test --model best_geofno_b72.pt --bundle_size 72 --group_len 72 --num_files 4
```

说明：

- 当前 `model/test_all.py` 默认 `--batch_size 1`，上面命令可省略；如需显式固定，可追加 `--batch_size 1`。
- 当前默认 `--bundle_size 72` 和 `--group_len 72` 已对齐 b72，这里显式写出是为了 runbook 可读性和防止未来默认值变化。

期望结果：

- 测试可以完成，不出现 checkpoint shape mismatch、group length 非整除、数据 shape mismatch。
- 输出两份指标文件：
  - `geofno_autoregressive_results_physical.txt`
  - `geofno_autoregressive_results_normalized.txt`
- 两份文件分别包含 physical 和 normalized 指标空间下的 per-step/channel metrics 与 AUC summary。

## 6. 全量 Test

4 文件 smoke 通过后，去掉 `--num_files` 跑完整测试集：

```bash
python model/test_all.py --test_dir data/test --model best_geofno_b72.pt --bundle_size 72 --group_len 72
```

同样确认：

- 输出 physical / normalized 两份结果文件。
- `skipped_files` 数量符合数据中 `T <= group_len` 的文件比例预期。
- 最终指标文件可归档到对应训练 run 的结果目录或实验记录中。
