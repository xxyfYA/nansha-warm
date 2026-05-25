# Geo-FNO 数据兼容性重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `model/` 下现有 Geo-FNO 实现适配 `CLAUDE.md` 定义的新数据格式（per-node `storm_boundary`/`inner_boundary`、UUID 命名 .pt 多文件），同时引入 lazy 加载 + LRU 缓存 + file-affine sampler，使单机 4 卡 DDP 训练 RAM 占用不随 rank 数线性膨胀。

**Architecture:** Dataset 层重写——每个 storm 事件独立 .pt 文件，DataLoader worker 进程内维护 LRU 缓存；自定义 `FileChunkedDistributedSampler` 保证连续样本来自同文件以提高命中率；坐标归一化与 boundary one-hot 在 Dataset 内完成并 `share_memory_()` 给所有 rank/worker。模型主体 (`model.py`) 不变，仅通道公式更新为 `C_in = 5S + 11`、`C_out = 3S`，并删除 pushforward/recurrent 训练路径。

**Tech Stack:** PyTorch 2.x + DDP（torchrun），scipy.io（读 `.mat`），tqdm，TensorBoard，pytest。

---

## 前置说明

- 本机不执行 Python 运行验证。每个 task 的"run test"步骤是给服务器端的参考指令；本地仅检查代码静态正确性。
- 测试代码（`tests/`）和实现一并 commit，服务器上 `pytest tests/` 可批量跑。
- 全部依赖现有 git 仓库（已 init，远程 `origin → http://10.10.41.205:45077/TRAY/nansha.git`）。
- 严格忽略 `data/` 目录与 macOS `._*` 文件（.gitignore 已配置）。

## Spec 参考

实现遵循 [docs/superpowers/specs/2026-05-16-geofno-data-compat-refactor-design.md](../specs/2026-05-16-geofno-data-compat-refactor-design.md)。每个 Task 落到 spec 的一个或多个章节。

---

### Task 0: Baseline commit（冻结重构前代码）

**目的：** spec 已 commit，但 `model/`、`AGENTS.md`、`claude.md` 仍未 tracked。先把它们冻结到 git，后续重构的 diff 才有意义。

**Files:**
- Track: `AGENTS.md`, `claude.md`, `model/Dataset.py`, `model/main.py`, `model/model.py`, `model/temporal_utils.py`, `model/test_all.py`, `model/train.py`

- [ ] **Step 1: 加入现有代码到 git stage**

```bash
git add AGENTS.md claude.md model/Dataset.py model/main.py model/model.py model/temporal_utils.py model/test_all.py model/train.py
```

- [ ] **Step 2: 验证 stage 内容不含数据/元数据**

```bash
git status --short
```

期望输出仅显示 `A` 行（model/*.py、AGENTS.md、claude.md），不含 `data/`、`._*`、`__pycache__/`。

- [ ] **Step 3: Commit baseline**

```bash
git commit -m "$(cat <<'EOF'
chore: baseline commit of pre-refactor Geo-FNO codebase

Captures the existing model/ implementation and project markdown files
as the starting point for the data-compat refactor described in
docs/superpowers/specs/2026-05-16-geofno-data-compat-refactor-design.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1: 重写 `model/temporal_utils.py`

**目的：** 删除 pushforward / recurrent 相关 helper，仅保留 bundle 范式；按新公式更新通道函数。

实现 spec 章节：「删除项 - temporal_utils.py」「模型输入特征布局 - C_in/C_out 公式」。

**Files:**
- Modify: `model/temporal_utils.py` (整体重写)
- Create: `tests/__init__.py`
- Create: `tests/test_temporal_utils.py`

- [ ] **Step 1: 创建 `tests/__init__.py`（空文件，让 pytest 把 tests 作为 package）**

```python
# tests/__init__.py
```

- [ ] **Step 2: 写测试 `tests/test_temporal_utils.py`**

```python
# tests/test_temporal_utils.py
import sys
from pathlib import Path

import pytest

# 让 tests 能直接 import model/ 下的模块（项目无 setup.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from temporal_utils import (  # noqa: E402
    TemporalConfig,
    build_checkpoint_name,
    build_run_suffix,
    input_channels_for_bundle,
    num_temporal_samples,
    output_channels_for_bundle,
    validate_temporal_params,
)


def test_input_channels_formula():
    # C_in = 5*S + 11
    assert input_channels_for_bundle(1) == 16
    assert input_channels_for_bundle(24) == 131
    assert input_channels_for_bundle(72) == 371


def test_output_channels_formula():
    # C_out = 3*S
    assert output_channels_for_bundle(1) == 3
    assert output_channels_for_bundle(24) == 72
    assert output_channels_for_bundle(72) == 216


def test_num_temporal_samples_basic():
    assert num_temporal_samples(100, 72) == 28
    assert num_temporal_samples(73, 72) == 1


def test_num_temporal_samples_too_short_raises():
    with pytest.raises(ValueError):
        num_temporal_samples(72, 72)
    with pytest.raises(ValueError):
        num_temporal_samples(10, 72)


def test_validate_temporal_params_rejects_nonpositive():
    with pytest.raises(ValueError):
        validate_temporal_params(0)
    with pytest.raises(ValueError):
        validate_temporal_params(-1)


def test_temporal_config_exposes_channels():
    cfg = TemporalConfig(bundle_size=72)
    assert cfg.bundle_size == 72
    assert cfg.required_future_steps == 72
    assert cfg.input_channels == 371
    assert cfg.out_channels == 216


def test_build_checkpoint_name():
    assert build_checkpoint_name(1) == "best_geofno.pt"
    assert build_checkpoint_name(72) == "best_geofno_b72.pt"
    assert build_checkpoint_name(72, "_noise") == "best_geofno_b72_noise.pt"


def test_build_run_suffix():
    assert build_run_suffix(1) == ""
    assert build_run_suffix(72) == "_b72"
    assert build_run_suffix(72, "_noise") == "_b72_noise"
```

- [ ] **Step 3: 重写 `model/temporal_utils.py` 内容**

```python
# model/temporal_utils.py
"""Bundle-only temporal helpers for Geo-FNO storm-surge model.

Pushforward and recurrent training paths have been removed; only bundle
prediction (one forward → predict S future steps) is supported.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalConfig:
    bundle_size: int = 1

    def __post_init__(self):
        validate_temporal_params(self.bundle_size)

    @property
    def required_future_steps(self) -> int:
        return self.bundle_size

    @property
    def input_channels(self) -> int:
        return input_channels_for_bundle(self.bundle_size)

    @property
    def out_channels(self) -> int:
        return output_channels_for_bundle(self.bundle_size)


def validate_temporal_params(bundle_size: int) -> None:
    if bundle_size < 1:
        raise ValueError(f"bundle_size must be >= 1, got {bundle_size}")


def num_temporal_samples(num_time: int, bundle_size: int) -> int:
    """Number of valid sample start indices in a file of length num_time."""
    validate_temporal_params(bundle_size)
    num_samples = num_time - bundle_size
    if num_samples <= 0:
        raise ValueError(
            f"not enough time steps: num_time={num_time}, "
            f"bundle_size={bundle_size}, required_future_steps={bundle_size}"
        )
    return num_samples


def input_channels_for_bundle(bundle_size: int) -> int:
    """C_in = state(3) + storm_window(3*(S+1)) + inner_window(2*(S+1)) + btype_oh(3) = 5S + 11."""
    validate_temporal_params(bundle_size)
    return 5 * bundle_size + 11


def output_channels_for_bundle(bundle_size: int) -> int:
    """C_out = 3*S residual states."""
    validate_temporal_params(bundle_size)
    return 3 * bundle_size


def build_checkpoint_name(bundle_size: int, noise_suffix: str = "") -> str:
    validate_temporal_params(bundle_size)
    if bundle_size == 1:
        return f"best_geofno{noise_suffix}.pt"
    return f"best_geofno_b{bundle_size}{noise_suffix}.pt"


def build_run_suffix(bundle_size: int, noise_suffix: str = "") -> str:
    validate_temporal_params(bundle_size)
    if bundle_size == 1:
        return noise_suffix
    return f"_b{bundle_size}{noise_suffix}"
```

- [ ] **Step 4: 服务器跑测试**

```bash
pytest tests/test_temporal_utils.py -v
```

期望：8 个测试全部 PASS。本机跳过此步骤。

- [ ] **Step 5: Commit**

```bash
git add model/temporal_utils.py tests/__init__.py tests/test_temporal_utils.py
git commit -m "$(cat <<'EOF'
refactor(temporal): drop pushforward/recurrent, update channel formulas

- C_in = 5*S + 11 (state + storm_window + inner_window + boundary_type_oh)
- C_out = 3*S (residual bundle prediction)
- TemporalConfig simplified to bundle_size only
- Removed validate_recurrent_params, recurrent_target_bounds,
  pushforward_steps-aware helpers

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 新增 `scripts/build_manifest.py`

**目的：** 一次性扫描 `data/<split>/*.pt`，记录每个文件的 T 和 N，输出 `manifest.json`。后续 Dataset 启动时读 manifest 而不必再次打开每个文件。

实现 spec 章节：「Dataset 设计 - Manifest 生成」。

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/build_manifest.py`
- Test: 无单元测试（脚本式工具，依赖真实文件，留作服务器端联调）

- [ ] **Step 1: 创建 `scripts/__init__.py`（空文件）**

```python
# scripts/__init__.py
```

- [ ] **Step 2: 写 `scripts/build_manifest.py`**

```python
#!/usr/bin/env python
"""Generate manifest.json for a data/<split> directory.

Usage:
    python scripts/build_manifest.py data/train
    python scripts/build_manifest.py data/test --output data/test/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm


REQUIRED_KEYS = ("graph", "storm_boundary", "inner_boundary")


def find_pt_files(data_dir: Path) -> list[Path]:
    """Glob *.pt; drop macOS metadata files (._*)."""
    return sorted(p for p in data_dir.glob("*.pt") if not p.name.startswith("._"))


def read_file_metadata(path: Path) -> tuple[int, int]:
    """Open one .pt file, validate keys/shapes, return (T, N)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"{path}: missing key {key!r}; got {list(data.keys())}")
    graph = data["graph"]
    storm = data["storm_boundary"]
    inner = data["inner_boundary"]
    if graph.dim() != 3 or graph.size(-1) != 3:
        raise ValueError(f"{path}: graph shape must be (T,N,3), got {tuple(graph.shape)}")
    if storm.shape != graph.shape:
        raise ValueError(
            f"{path}: storm_boundary {tuple(storm.shape)} != graph {tuple(graph.shape)}"
        )
    if inner.dim() != 3 or inner.size(-1) != 2 or inner.size(1) != graph.size(1) \
            or inner.size(0) != graph.size(0):
        raise ValueError(
            f"{path}: inner_boundary {tuple(inner.shape)} incompatible with graph "
            f"{tuple(graph.shape)} (expected (T,N,2))"
        )
    return int(graph.size(0)), int(graph.size(1))


def build_manifest(data_dir: Path, bundle_size_warn: int | None = None) -> dict:
    files = find_pt_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .pt files (excluding ._*) in {data_dir}")

    entries: list[dict] = []
    num_nodes: int | None = None
    skipped: list[tuple[str, int]] = []

    for path in tqdm(files, desc=f"scanning {data_dir.name}"):
        T, N = read_file_metadata(path)
        if num_nodes is None:
            num_nodes = N
        elif N != num_nodes:
            raise ValueError(
                f"{path}: num_nodes={N} disagrees with first-file num_nodes={num_nodes}"
            )
        if bundle_size_warn is not None and T <= bundle_size_warn:
            skipped.append((path.name, T))
        entries.append({"path": path.name, "T": T})

    if skipped:
        print(
            f"[manifest] warning: {len(skipped)} files have T <= bundle_size_warn={bundle_size_warn} "
            f"and will yield no samples:",
            file=sys.stderr,
        )
        for name, T in skipped[:10]:
            print(f"  {name}: T={T}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more", file=sys.stderr)

    return {
        "num_nodes": num_nodes,
        "files": entries,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manifest.json for a data split.")
    parser.add_argument("data_dir", type=Path, help="Path to data/<split> directory.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output path (default: <data_dir>/manifest.json).",
    )
    parser.add_argument(
        "--bundle_size_warn", type=int, default=None,
        help="Warn for files with T <= this value (no samples will be produced).",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"error: {args.data_dir} is not a directory", file=sys.stderr)
        return 2

    manifest = build_manifest(args.data_dir, bundle_size_warn=args.bundle_size_warn)
    out = args.output or (args.data_dir / "manifest.json")
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {len(manifest['files'])} entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Commit**

```bash
git add scripts/__init__.py scripts/build_manifest.py
git commit -m "$(cat <<'EOF'
feat(scripts): add build_manifest.py for lazy-loading dataset

Scans data/<split>/*.pt, validates required keys (graph,
storm_boundary, inner_boundary) and shape contracts, and writes
manifest.json with {num_nodes, files: [{path, T}], created_at}.

Manifest enables main.py to avoid reading every .pt at startup; the
Dataset only needs (path, T) to enumerate flat sample indices.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 创建 `model/dataset.py`（lazy + LRU + sampler + 静态坐标）

**目的：** 实现 spec「Dataset 设计：lazy 加载 + LRU + file-affine sampler」全部内容。包含：

1. `load_static_coords` - 一次性加载坐标并归一化、boundary one-hot、`share_memory_`
2. `StormSurgeDataset` - 单文件 lazy 加载封装（含 LRU、特征拼接）
3. `MultiStormSurgeDataset` - 多文件聚合 + flat 索引
4. `FileChunkedDistributedSampler` - 文件亲和性 sampler

**Files:**
- Create: `model/dataset.py`
- Create: `tests/test_dataset.py`
- Delete: `model/Dataset.py`（旧文件，被新文件取代）

- [ ] **Step 1: 写测试 `tests/test_dataset.py`**

```python
# tests/test_dataset.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from dataset import (  # noqa: E402
    FileChunkedDistributedSampler,
    MultiStormSurgeDataset,
    StormSurgeDataset,
    load_static_coords,
)


N_NODES = 64


@pytest.fixture
def coords_mat(tmp_path):
    """Fake coordinates.mat with 64 nodes; first 8 are wl-bdy, next 4 are flux."""
    coords3 = np.random.rand(N_NODES, 3).astype(np.float64) * 100.0
    boundary = np.zeros((N_NODES, 1), dtype=np.int8)
    boundary[:8] = 1
    boundary[8:12] = 2
    p = tmp_path / "coords.mat"
    scipy.io.savemat(p, {"coordinates": coords3, "boundary": boundary})
    return p


def _make_pt(path, T):
    data = {
        "graph": torch.randn(T, N_NODES, 3),
        "storm_boundary": torch.randn(T, N_NODES, 3),
        "inner_boundary": torch.zeros(T, N_NODES, 2),
        "run_id": path.stem,
        "source_dir": "synthetic",
    }
    # only boundary nodes have non-zero inner_boundary
    data["inner_boundary"][:, :12] = torch.randn(T, 12, 2)
    torch.save(data, path)


@pytest.fixture
def split_dir(tmp_path):
    d = tmp_path / "split"
    d.mkdir()
    sizes = [80, 100, 50]
    for i, T in enumerate(sizes):
        _make_pt(d / f"e{i}.pt", T)
    manifest = {
        "num_nodes": N_NODES,
        "files": [{"path": f"e{i}.pt", "T": sizes[i]} for i in range(len(sizes))],
        "created_at": "2026-05-16T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_load_static_coords_normalizes(coords_mat):
    coords, btype = load_static_coords(coords_mat)
    assert coords.shape == (N_NODES, 2)
    assert coords.dtype == torch.float32
    assert torch.all(coords >= 0.0) and torch.all(coords <= 1.0)
    # min/max per dim should hit endpoints
    assert torch.isclose(coords.min(0).values, torch.zeros(2)).all()
    assert torch.isclose(coords.max(0).values, torch.ones(2)).all()
    assert btype.shape == (N_NODES, 3)
    assert torch.all(btype.sum(dim=1) == 1.0)
    # 0..7 are wl-bdy → column 1
    assert torch.all(btype[:8, 1] == 1.0)
    assert torch.all(btype[8:12, 2] == 1.0)
    assert torch.all(btype[12:, 0] == 1.0)


def test_load_static_coords_rejects_bad_boundary(tmp_path):
    p = tmp_path / "bad.mat"
    scipy.io.savemat(p, {
        "coordinates": np.zeros((4, 3)),
        "boundary": np.array([[0], [3], [0], [0]], dtype=np.int8),  # 3 is invalid
    })
    with pytest.raises(ValueError, match="boundary"):
        load_static_coords(p)


def test_single_dataset_shapes(split_dir, coords_mat):
    coords, btype = load_static_coords(coords_mat)
    ds = StormSurgeDataset(
        path=split_dir / "e0.pt",
        bundle_size=4,
        btype_oh=btype,
        lru_capacity=1,
    )
    assert len(ds) == 80 - 4  # T - S
    feat, target = ds[0]
    assert feat.shape == (N_NODES, 5 * 4 + 11)  # C_in
    assert target.shape == (4, N_NODES, 3)


def test_single_dataset_btype_concatenated(split_dir, coords_mat):
    coords, btype = load_static_coords(coords_mat)
    ds = StormSurgeDataset(
        path=split_dir / "e0.pt",
        bundle_size=2,
        btype_oh=btype,
        lru_capacity=1,
    )
    feat, _ = ds[0]
    # last 3 channels should equal btype_oh
    assert torch.allclose(feat[:, -3:], btype)


def test_multi_dataset_index_flattening(split_dir, coords_mat):
    coords, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(
        data_dir=split_dir,
        bundle_size=4,
        btype_oh=btype,
        lru_files_per_worker=1,
    )
    # samples = sum(T_i - S) = 76 + 96 + 46 = 218
    assert len(mds) == (80 - 4) + (100 - 4) + (50 - 4)
    feat, target = mds[0]
    assert feat.shape == (N_NODES, 31)
    assert target.shape == (4, N_NODES, 3)


def test_multi_dataset_drops_too_short_files(tmp_path, coords_mat):
    d = tmp_path / "split"
    d.mkdir()
    _make_pt(d / "small.pt", 5)
    _make_pt(d / "ok.pt", 50)
    manifest = {
        "num_nodes": N_NODES,
        "files": [{"path": "small.pt", "T": 5}, {"path": "ok.pt", "T": 50}],
        "created_at": "2026-05-16T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    coords, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(d, bundle_size=20, btype_oh=btype, lru_files_per_worker=1)
    # small.pt T=5 ≤ S=20 → dropped; only ok.pt 50-20=30 samples
    assert len(mds) == 30


def test_file_chunked_sampler_groups_by_file(split_dir, coords_mat):
    coords, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(split_dir, bundle_size=4, btype_oh=btype, lru_files_per_worker=1)
    sampler = FileChunkedDistributedSampler(mds, num_replicas=1, rank=0, shuffle=True, seed=0)
    indices = list(iter(sampler))
    assert len(indices) == len(mds)
    # check that within long runs, same file_idx wins
    file_seq = [mds.flat_index[i][0] for i in indices]
    # number of file transitions should equal number of unique files - 1 if grouping is perfect
    transitions = sum(1 for a, b in zip(file_seq, file_seq[1:]) if a != b)
    assert transitions <= len(set(file_seq)) - 1 + 2  # allow a bit of slack for shuffle


def test_file_chunked_sampler_disjoint_across_ranks(split_dir, coords_mat):
    coords, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(split_dir, bundle_size=4, btype_oh=btype, lru_files_per_worker=1)
    s0 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=0, shuffle=False, seed=0)
    s1 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=1, shuffle=False, seed=0)
    i0 = set(iter(s0))
    i1 = set(iter(s1))
    assert i0.isdisjoint(i1)
    # files per rank: rank_i gets files[i::2]; we have 3 files → rank0 gets 2 files, rank1 gets 1
    f0 = {mds.flat_index[i][0] for i in i0}
    f1 = {mds.flat_index[i][0] for i in i1}
    assert f0.isdisjoint(f1)


def test_lru_eviction(split_dir, coords_mat):
    coords, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(split_dir, bundle_size=4, btype_oh=btype, lru_files_per_worker=2)
    # touch samples from 3 different files
    for fi in range(3):
        first_idx_for_file = next(i for i, (f, _) in enumerate(mds.flat_index) if f == fi)
        _ = mds[first_idx_for_file]
    # cache should not exceed capacity
    assert len(mds._cache) <= 2
```

- [ ] **Step 2: 写 `model/dataset.py`**

```python
# model/dataset.py
"""Storm-surge lazy-loading dataset for Geo-FNO training.

Layout per node, in input feature vector (S = bundle_size):

    [u_t, v_t, h_t]                                   # 3   state
    [P, Wx, Wy] @ t, t+1, ..., t+S                    # 3*(S+1)  storm window
    [h_bdy, q_bdy] @ t, t+1, ..., t+S                 # 2*(S+1)  inner window
    [type_none, type_wl, type_flux]                   # 3   boundary type one-hot

Total C_in = 5*S + 11. See spec channel layout.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, Sampler


REQUIRED_PT_KEYS = ("graph", "storm_boundary", "inner_boundary")


# -------------------------------------------------------------------------
# Static coords / boundary-type loader
# -------------------------------------------------------------------------

def load_static_coords(coords_path):
    """Load coordinates.mat → (coords_norm (N,2) float32, btype_oh (N,3) float32).

    Coords are min/max-normalized into [0,1]^2 per axis.
    Both returned tensors are placed in shared memory.
    """
    coords_path = Path(coords_path)
    mat = scipy.io.loadmat(coords_path)
    if "coordinates" not in mat or "boundary" not in mat:
        raise KeyError(
            f"{coords_path}: missing 'coordinates' or 'boundary'; got "
            f"{[k for k in mat if not k.startswith('__')]}"
        )

    coords = mat["coordinates"][:, :2].astype(np.float32)
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = np.maximum(cmax - cmin, 1e-8)
    coords_norm = (coords - cmin) / span  # (N, 2) in [0,1]

    bt = mat["boundary"].astype(np.int64).flatten()  # (N,)
    if bt.min() < 0 or bt.max() > 2:
        raise ValueError(
            f"{coords_path}: boundary values must be in {{0,1,2}}, "
            f"got range [{int(bt.min())}, {int(bt.max())}]"
        )
    btype_oh = np.eye(3, dtype=np.float32)[bt]  # (N, 3)

    coords_t = torch.from_numpy(coords_norm).contiguous()
    btype_t = torch.from_numpy(btype_oh).contiguous()
    coords_t.share_memory_()
    btype_t.share_memory_()
    return coords_t, btype_t


# -------------------------------------------------------------------------
# Single-file lazy dataset (used standalone; also the storage backing
# MultiStormSurgeDataset's LRU)
# -------------------------------------------------------------------------

def _load_pt(path: Path) -> dict[str, torch.Tensor]:
    """torch.load one .pt and validate required keys/shapes."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in REQUIRED_PT_KEYS:
        if key not in data:
            raise KeyError(f"{path}: missing key {key!r}; got {list(data.keys())}")
    graph = data["graph"].float()
    storm = data["storm_boundary"].float()
    inner = data["inner_boundary"].float()
    if graph.dim() != 3 or graph.size(-1) != 3:
        raise ValueError(f"{path}: graph must be (T,N,3), got {tuple(graph.shape)}")
    if storm.shape != graph.shape:
        raise ValueError(
            f"{path}: storm_boundary {tuple(storm.shape)} != graph {tuple(graph.shape)}"
        )
    if (inner.dim() != 3 or inner.size(0) != graph.size(0)
            or inner.size(1) != graph.size(1) or inner.size(-1) != 2):
        raise ValueError(
            f"{path}: inner_boundary {tuple(inner.shape)} incompatible with graph "
            f"{tuple(graph.shape)} (expected (T,N,2))"
        )
    return {"graph": graph, "storm": storm, "inner": inner}


def _build_features(
    state_t: torch.Tensor,            # (N, 3)
    storm_window: torch.Tensor,       # (S+1, N, 3)
    inner_window: torch.Tensor,       # (S+1, N, 2)
    btype_oh: torch.Tensor,           # (N, 3)
) -> torch.Tensor:
    """Concat per-node feature vector. Returns (N, 5*S + 11)."""
    N = state_t.size(0)
    storm_flat = storm_window.permute(1, 0, 2).reshape(N, -1)   # (N, 3*(S+1))
    inner_flat = inner_window.permute(1, 0, 2).reshape(N, -1)   # (N, 2*(S+1))
    return torch.cat([state_t, storm_flat, inner_flat, btype_oh], dim=-1).contiguous()


class StormSurgeDataset(Dataset):
    """Lazy single-file dataset.

    Holds at most lru_capacity files in self._cache. For the multi-file
    case use MultiStormSurgeDataset instead.
    """

    def __init__(self, path, bundle_size, btype_oh, lru_capacity: int = 1):
        from temporal_utils import validate_temporal_params, num_temporal_samples

        validate_temporal_params(bundle_size)
        self.path = Path(path)
        self.bundle_size = int(bundle_size)
        self.btype_oh = btype_oh
        self.lru_capacity = int(lru_capacity)
        self._cache: "OrderedDict[Path, dict]" = OrderedDict()

        entry = self._get_entry()
        self.T = entry["graph"].size(0)
        self.N = entry["graph"].size(1)
        if self.btype_oh.size(0) != self.N:
            raise ValueError(
                f"{self.path}: btype_oh N={self.btype_oh.size(0)} != file N={self.N}"
            )
        self._num_samples = num_temporal_samples(self.T, self.bundle_size)

    def __len__(self) -> int:
        return self._num_samples

    def _get_entry(self) -> dict:
        if self.path in self._cache:
            self._cache.move_to_end(self.path)
            return self._cache[self.path]
        entry = _load_pt(self.path)
        self._cache[self.path] = entry
        while len(self._cache) > self.lru_capacity:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(idx)
        entry = self._get_entry()
        S = self.bundle_size
        graph = entry["graph"]
        storm = entry["storm"]
        inner = entry["inner"]
        state_t = graph[idx]                          # (N, 3)
        storm_w = storm[idx : idx + S + 1]            # (S+1, N, 3)
        inner_w = inner[idx : idx + S + 1]            # (S+1, N, 2)
        target = graph[idx + 1 : idx + S + 1].contiguous()  # (S, N, 3)
        features = _build_features(state_t, storm_w, inner_w, self.btype_oh)
        return features, target


# -------------------------------------------------------------------------
# Multi-file lazy dataset
# -------------------------------------------------------------------------

class MultiStormSurgeDataset(Dataset):
    """Lazy aggregation across multiple .pt files in one data split.

    Reads <data_dir>/manifest.json (built by scripts/build_manifest.py).
    Each worker process keeps its own LRU cache keyed by file_idx.
    """

    def __init__(self, data_dir, bundle_size, btype_oh, lru_files_per_worker: int = 2):
        from temporal_utils import validate_temporal_params

        validate_temporal_params(bundle_size)
        self.data_dir = Path(data_dir)
        self.bundle_size = int(bundle_size)
        self.btype_oh = btype_oh
        self.lru_files_per_worker = int(lru_files_per_worker)

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run: python scripts/build_manifest.py {self.data_dir}"
            )
        manifest = json.loads(manifest_path.read_text())
        if btype_oh.size(0) != manifest["num_nodes"]:
            raise ValueError(
                f"btype_oh N={btype_oh.size(0)} != manifest num_nodes={manifest['num_nodes']}"
            )
        self.num_nodes = int(manifest["num_nodes"])

        # Filter out files that don't exist or are too short
        self.files: list[Path] = []
        self.file_T: list[int] = []
        dropped: list[tuple[str, int]] = []
        for entry in manifest["files"]:
            p = self.data_dir / entry["path"]
            if not p.exists():
                raise FileNotFoundError(
                    f"manifest references missing file {p}; rebuild manifest "
                    f"with: python scripts/build_manifest.py {self.data_dir}"
                )
            T = int(entry["T"])
            if T <= self.bundle_size:
                dropped.append((entry["path"], T))
                continue
            self.files.append(p)
            self.file_T.append(T)
        if dropped:
            print(
                f"[dataset] {self.data_dir.name}: dropped {len(dropped)} files with "
                f"T <= bundle_size={self.bundle_size}"
            )
        if not self.files:
            raise RuntimeError(
                f"{self.data_dir}: no files survive bundle_size={self.bundle_size} filter"
            )

        # Flat index: list of (file_idx, t_local)
        self.flat_index: list[tuple[int, int]] = []
        for fi, T in enumerate(self.file_T):
            for t in range(T - self.bundle_size):
                self.flat_index.append((fi, t))

        # Per-process LRU cache
        self._cache: "OrderedDict[int, dict]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.flat_index)

    def _get_entry(self, fi: int) -> dict:
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        entry = _load_pt(self.files[fi])
        self._cache[fi] = entry
        while len(self._cache) > self.lru_files_per_worker:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.flat_index):
            raise IndexError(idx)
        fi, t = self.flat_index[idx]
        entry = self._get_entry(fi)
        S = self.bundle_size
        graph = entry["graph"]
        storm = entry["storm"]
        inner = entry["inner"]
        state_t = graph[t]
        storm_w = storm[t : t + S + 1]
        inner_w = inner[t : t + S + 1]
        target = graph[t + 1 : t + S + 1].contiguous()
        features = _build_features(state_t, storm_w, inner_w, self.btype_oh)
        return features, target


# -------------------------------------------------------------------------
# File-affine distributed sampler
# -------------------------------------------------------------------------

class FileChunkedDistributedSampler(Sampler[int]):
    """DistributedSampler variant that keeps each file's samples contiguous.

    File assignment is fixed at __init__: rank_i always owns files[i::num_replicas]
    (under a deterministic sort). Per-epoch shuffle only reorders file traversal
    order and samples-within-file. This keeps DDP step counts identical across
    ranks (avoiding all_reduce hangs) and is friendly to LRU.

    Per epoch:
      1. seed = self.seed + epoch
      2. shuffle this rank's assigned file list
      3. within each file: shuffle its sample list
      4. truncate or wrap-around so each rank yields exactly len(dataset) // num_replicas
         samples (when drop_last=True) — keeps DDP step counts equal across ranks.
    """

    def __init__(
        self,
        dataset: MultiStormSurgeDataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Group flat indices by file_idx
        self._by_file: dict[int, list[int]] = {}
        for flat_i, (fi, _) in enumerate(dataset.flat_index):
            self._by_file.setdefault(fi, []).append(flat_i)

        # Fixed per-rank file assignment (sorted, then strided by rank)
        all_files = sorted(self._by_file.keys())
        self._my_files: list[int] = all_files[self.rank::self.num_replicas]

        # Per-rank target sample count (kept equal across ranks for DDP correctness)
        world_total = len(dataset)
        if self.drop_last:
            self._my_target = world_total // self.num_replicas
        else:
            self._my_target = (world_total + self.num_replicas - 1) // self.num_replicas

    def __len__(self) -> int:
        return self._my_target

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        my_files = list(self._my_files)
        if self.shuffle and len(my_files) > 1:
            perm = torch.randperm(len(my_files), generator=g).tolist()
            my_files = [my_files[i] for i in perm]

        ordered: list[int] = []
        for fi in my_files:
            samples = list(self._by_file[fi])
            if self.shuffle and len(samples) > 1:
                perm = torch.randperm(len(samples), generator=g).tolist()
                samples = [samples[i] for i in perm]
            ordered.extend(samples)

        # Align length to _my_target via truncate or wrap-around pad
        if len(ordered) >= self._my_target:
            ordered = ordered[: self._my_target]
        else:
            if not ordered:
                raise RuntimeError(
                    f"rank {self.rank} has no samples; check file assignment and "
                    f"that each file has T > bundle_size"
                )
            extra = self._my_target - len(ordered)
            ordered.extend(ordered[i % len(ordered)] for i in range(extra))

        for s in ordered:
            yield s

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
```

- [ ] **Step 3: 删除旧 `model/Dataset.py`**

```bash
git rm model/Dataset.py
```

- [ ] **Step 4: 服务器跑测试**

```bash
pytest tests/test_dataset.py -v
```

期望：10 个测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add model/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(dataset): lazy multi-file dataset with LRU and file-affine sampler

Replaces model/Dataset.py with model/dataset.py:
- load_static_coords: normalize 2D coords to [0,1]^2 + boundary one-hot,
  share_memory_ both tensors
- StormSurgeDataset: single-file lazy dataset
- MultiStormSurgeDataset: aggregates a data split via manifest.json,
  per-worker LRU cache (default 2 files)
- FileChunkedDistributedSampler: rank_i takes files[i::world_size],
  consecutive indices stay on the same file to maximize LRU hits

Feature vector layout per node (C_in = 5*S + 11):
  [state(3), storm_window(3*(S+1)), inner_window(2*(S+1)), btype_oh(3)]

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 重写 `model/train.py`

**目的：** 删除 pushforward / recurrent rollout 路径；适配新 Dataset 返回的 2 元组 `(features, target_block)`；`build_feature_block` 由 Dataset 内化后从 train.py 移除；evaluate / train 流程同步精简。

**Files:**
- Modify: `model/train.py`（整体重写）

- [ ] **Step 1: 完整重写 `model/train.py`**

```python
# model/train.py
"""Training loop for Geo-FNO bundle-only mode."""
from __future__ import annotations

import torch
import torch.distributed as dist
from tqdm import tqdm


def is_distributed(dist_ctx: dict | None) -> bool:
    return bool(dist_ctx and dist_ctx.get("distributed", False))


def is_rank0(dist_ctx: dict | None) -> bool:
    return dist_ctx is None or dist_ctx.get("is_rank0", True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_sums(values, device, dist_ctx: dict | None):
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if is_distributed(dist_ctx):
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.cpu().tolist()


def barrier_if_distributed(dist_ctx: dict | None):
    if is_distributed(dist_ctx):
        dist.barrier()


def make_uvh_noise_std_tensor(uvh_noise_std, device):
    t = torch.as_tensor(uvh_noise_std, dtype=torch.float32, device=device)
    if t.numel() != 3:
        raise ValueError(f"uvh_noise_std must contain 3 values for u, v, h; got {uvh_noise_std}")
    if torch.any(t < 0):
        raise ValueError(f"uvh_noise_std must be non-negative; got {uvh_noise_std}")
    return t.reshape(1, 1, 3)


def add_uvh_training_noise(features: torch.Tensor, uvh_noise_std_tensor: torch.Tensor) -> torch.Tensor:
    noisy = features.clone()
    noise = torch.randn_like(features[..., :3]) * uvh_noise_std_tensor.to(dtype=features.dtype)
    noisy[..., :3] = features[..., :3] + noise
    return noisy


class RMSELoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


def rel_l2_loss(pred, target, eps: float = 1e-8):
    B = pred.shape[0]
    diff_flat = (pred - target).reshape(B, -1)
    target_flat = target.reshape(B, -1)
    diff_norm = torch.linalg.vector_norm(diff_flat, ord=2, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp(min=eps)
    return (diff_norm / target_norm).mean()


def _channel_rel_l2(pred: torch.Tensor, target: torch.Tensor, channel: int, eps: float = 1e-8):
    diff = (pred[..., channel] - target[..., channel]).reshape(pred.size(0), -1)
    base = target[..., channel].reshape(pred.size(0), -1)
    num = torch.linalg.vector_norm(diff, ord=2, dim=1)
    den = torch.linalg.vector_norm(base, ord=2, dim=1).clamp(min=eps)
    return (num / den).mean()


def evaluate_model(model, test_loader, device, coords_2d_device, dist_ctx: dict | None = None):
    """Bundle evaluation in normalized space; no autoregressive rollout."""
    model.eval()
    total_sse = total_sae = total_rel_l2 = 0.0
    total_rel_u = total_rel_v = total_rel_h = 0.0
    num_samples = 0
    total_elements = 0

    x_in_base = coords_2d_device.unsqueeze(0)
    with torch.no_grad():
        for features, target_block in test_loader:
            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            B = features.shape[0]
            x_in = x_in_base.expand(B, -1, -1)

            pred_block = model(features, x_in)
            diff = pred_block - target_block

            total_sse += (diff ** 2).sum().item()
            total_sae += diff.abs().sum().item()

            diff_flat = diff.reshape(B, -1)
            target_flat = target_block.reshape(B, -1)
            diff_norm = torch.linalg.vector_norm(diff_flat, ord=2, dim=1)
            target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp(min=1e-8)
            total_rel_l2 += (diff_norm / target_norm).sum().item()

            total_rel_u += _channel_rel_l2(pred_block, target_block, 0).item() * B
            total_rel_v += _channel_rel_l2(pred_block, target_block, 1).item() * B
            total_rel_h += _channel_rel_l2(pred_block, target_block, 2).item() * B

            num_samples += B
            total_elements += target_block.numel()

    totals = reduce_sums(
        [total_sse, total_sae, total_rel_l2, total_rel_u, total_rel_v, total_rel_h,
         num_samples, total_elements],
        device, dist_ctx,
    )
    sse, sae, rl2, ru, rv, rh, ns, ne = totals
    ns = max(1.0, ns)
    ne = max(1.0, ne)
    mse = sse / ne
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": sae / ne,
        "rel_l2": rl2 / ns,
        "rel_u": ru / ns,
        "rel_v": rv / ns,
        "rel_h": rh / ns,
    }


def train_model(model, train_loader, test_loader, num_epochs, device,
                optimizer, scheduler, coords_2d_device, writer, grad_clip=None,
                loss_type: str = "rel_l2",
                add_noise: bool = False, uvh_noise_std=(0.005, 0.005, 0.001),
                checkpoint_path: str = "best_geofno.pt",
                train_sampler=None, dist_ctx: dict | None = None,
                accum_steps: int = 1):
    if loss_type == "rmse":
        criterion = RMSELoss()
    elif loss_type == "rel_l2":
        criterion = None
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    global_step = 0
    best_loss = float("inf")
    noise_t = make_uvh_noise_std_tensor(uvh_noise_std, device) if add_noise else None

    for epoch in range(num_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train()
        local_loss_sum = 0.0
        local_n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                    leave=False, disable=not is_rank0(dist_ctx))

        x_in_base = coords_2d_device.unsqueeze(0)

        steps_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = steps_per_epoch // accum_steps
        usable_micro_batches = optimizer_steps_per_epoch * accum_steps

        optimizer.zero_grad(set_to_none=True)

        for micro_idx, (features, target_block) in enumerate(pbar):
            if micro_idx >= usable_micro_batches:
                break

            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            B = features.shape[0]

            if add_noise:
                features = add_uvh_training_noise(features, noise_t)

            x_in = x_in_base.expand(B, -1, -1)
            pred_block = model(features, x_in)

            if loss_type == "rmse":
                loss = criterion(pred_block, target_block)
            else:
                loss = rel_l2_loss(pred_block, target_block)
            loss = loss / accum_steps

            is_boundary = ((micro_idx + 1) % accum_steps == 0)
            if (not is_boundary) and is_distributed(dist_ctx):
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            loss_unscaled = loss.item() * accum_steps
            local_loss_sum += loss_unscaled * B
            local_n += B

            if is_boundary:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if is_rank0(dist_ctx) and writer is not None:
                    writer.add_scalar("train/loss_step", loss_unscaled, global_step)
                global_step += 1

            if is_rank0(dist_ctx):
                pbar.set_postfix({"loss": f"{loss_unscaled:.6f}"})

        global_loss_sum, global_n = reduce_sums([local_loss_sum, local_n], device, dist_ctx)
        avg_loss = global_loss_sum / max(1.0, global_n)
        if is_rank0(dist_ctx) and writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        test_metrics = evaluate_model(model, test_loader, device, coords_2d_device, dist_ctx=dist_ctx)
        current_lr = optimizer.param_groups[0]["lr"]

        if is_rank0(dist_ctx):
            if writer is not None:
                writer.add_scalar("val/loss_epoch", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/rel_l2", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/mse", test_metrics["mse"], epoch)
                writer.add_scalar("val/rmse", test_metrics["rmse"], epoch)
                writer.add_scalar("val/mae", test_metrics["mae"], epoch)
                writer.add_scalar("val/rel_u", test_metrics["rel_u"], epoch)
                writer.add_scalar("val/rel_v", test_metrics["rel_v"], epoch)
                writer.add_scalar("val/rel_h", test_metrics["rel_h"], epoch)
                writer.add_scalar("train/lr", current_lr, epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_loss:.6f} | "
                f"Test RMSE: {test_metrics['rmse']:.6f} | "
                f"Test Rel-L2: {test_metrics['rel_l2']:.6f} | "
                f"Test Rel-U: {test_metrics['rel_u']:.6f} | "
                f"Test Rel-V: {test_metrics['rel_v']:.6f} | "
                f"Test Rel-H: {test_metrics['rel_h']:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        current_test_loss = (test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"])
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                torch.save(unwrap_model(model).state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")

        barrier_if_distributed(dist_ctx)

    if is_rank0(dist_ctx):
        print("Training finished.")
```

- [ ] **Step 2: Commit**

```bash
git add model/train.py
git commit -m "$(cat <<'EOF'
refactor(train): drop pushforward/recurrent rollouts; bundle-only path

- Remove predict_recurrent_rollout, predict_final_block, build_feature_block,
  recurrent_rel_l2_loss (feature construction is now Dataset's job)
- DataLoader yields 2-tuples (features, target_block); unpacking simplified
- evaluate_model: single forward per batch, no rollout
- train_model: signature drops pushforward_steps/recurrent_steps params

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 重写 `model/main.py`

**目的：** 适配新 CONFIG schema、自动扫描数据目录、用新 Dataset/Sampler、删除手动坐标归一化、删除 pushforward/recurrent 字段。

**Files:**
- Modify: `model/main.py`（整体重写）

- [ ] **Step 1: 完整重写 `model/main.py`**

```python
# model/main.py
"""Training entrypoint for Geo-FNO storm-surge model (bundle-only, multi-file).

Single GPU:
    python model/main.py

Single-node multi-GPU DDP (N GPUs):
    torchrun --nproc_per_node=N model/main.py
"""
from __future__ import annotations

import math
import os
import platform
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    FileChunkedDistributedSampler,
    MultiStormSurgeDataset,
    load_static_coords,
)
from model import GeoFNO2d
from temporal_utils import (
    build_checkpoint_name,
    build_run_suffix,
    input_channels_for_bundle,
    output_channels_for_bundle,
    validate_temporal_params,
)
from train import train_model


# ===================== CONFIG =====================
CONFIG = {
    "train_dir":   "data/train",
    "val_dir":     "data/val",
    "test_dir":    "data/test",
    "coords_path": "data/coordinates.mat",
    "norm_path":   "data/normalization.mat",
    "tb_dir":      "runs",

    "seed": 42,

    "bundle_size":          72,
    "batch_size":           16,
    "num_workers":          4,
    "lru_files_per_worker": 2,

    "modes":           16,
    "width":           32,
    "s1":              64,
    "s2":              64,
    "num_fno_layers":  3,

    "num_epochs":   100,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.05,
    "grad_clip":    1.0,
    "accum_steps":  1,
    "loss_type":    "rel_l2",

    "add_noise":     False,
    "uvh_noise_std": [0.005, 0.005, 0.001],
}
# ==================================================


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, min_lr=0.0, last_epoch=-1):
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        ratio = min_lr / base_lr
        return ratio + (1.0 - ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def init_distributed() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA, but torch.cuda.is_available() is False")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_rank0": rank == 0,
    }


def cleanup_distributed(dist_ctx: dict):
    if dist_ctx.get("distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def get_device(dist_ctx: dict) -> torch.device:
    if dist_ctx["distributed"]:
        return torch.device("cuda", dist_ctx["local_rank"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_device_batch_size(global_batch_size: int, dist_ctx: dict) -> int:
    ws = dist_ctx["world_size"]
    if global_batch_size % ws != 0:
        raise ValueError(
            f"CONFIG['batch_size']={global_batch_size} must be divisible by world_size={ws}"
        )
    return global_batch_size // ws


def rank0_print(dist_ctx: dict, *args, **kwargs):
    if dist_ctx["is_rank0"]:
        print(*args, **kwargs)


def format_noise_value(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def build_noise_run_suffix(add_noise: bool, uvh_noise_std) -> str:
    if not add_noise:
        return ""
    return "_noise_uvh_" + "_".join(format_noise_value(v) for v in uvh_noise_std)


def main():
    dist_ctx = init_distributed()
    writer = None
    try:
        if CONFIG["accum_steps"] < 1:
            raise ValueError(f"accum_steps must be >= 1, got {CONFIG['accum_steps']}")
        validate_temporal_params(CONFIG["bundle_size"])
        set_seed(CONFIG["seed"])

        in_channels = input_channels_for_bundle(CONFIG["bundle_size"])
        out_channels = output_channels_for_bundle(CONFIG["bundle_size"])
        noise_suffix = build_noise_run_suffix(CONFIG["add_noise"], CONFIG["uvh_noise_std"])
        checkpoint_name = build_checkpoint_name(CONFIG["bundle_size"], noise_suffix)
        run_tag = "GeoFNO" + build_run_suffix(CONFIG["bundle_size"], noise_suffix) \
            + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        device = get_device(dist_ctx)
        rank0_print(dist_ctx,
                    f"[main] device={device}, distributed={dist_ctx['distributed']}, "
                    f"world_size={dist_ctx['world_size']}")
        rank0_print(dist_ctx,
                    f"[main] bundle_size={CONFIG['bundle_size']}, "
                    f"in_channels={in_channels}, out_channels={out_channels}")
        rank0_print(dist_ctx, f"[main] checkpoint name: {checkpoint_name}")

        # ---- Static coords / boundary type ----
        rank0_print(dist_ctx, f"[main] loading coords from {CONFIG['coords_path']}")
        coords_2d_cpu, btype_oh_cpu = load_static_coords(CONFIG["coords_path"])
        coords_2d_device = coords_2d_cpu.to(device)
        rank0_print(dist_ctx, f"[main] coords normalized, shape={tuple(coords_2d_cpu.shape)}, "
                              f"btype_oh shape={tuple(btype_oh_cpu.shape)}")

        # ---- Datasets ----
        rank0_print(dist_ctx, f"[main] loading train manifest from {CONFIG['train_dir']}")
        train_dataset = MultiStormSurgeDataset(
            data_dir=CONFIG["train_dir"],
            bundle_size=CONFIG["bundle_size"],
            btype_oh=btype_oh_cpu,
            lru_files_per_worker=CONFIG["lru_files_per_worker"],
        )
        rank0_print(dist_ctx, f"[main] loading val manifest from {CONFIG['val_dir']}")
        val_dataset = MultiStormSurgeDataset(
            data_dir=CONFIG["val_dir"],
            bundle_size=CONFIG["bundle_size"],
            btype_oh=btype_oh_cpu,
            lru_files_per_worker=CONFIG["lru_files_per_worker"],
        )
        rank0_print(dist_ctx, f"[main] train samples={len(train_dataset)}, val samples={len(val_dataset)}")
        rank0_print(dist_ctx, f"[main] nodes per sample={train_dataset.num_nodes}")

        batch_size = per_device_batch_size(CONFIG["batch_size"], dist_ctx)

        train_sampler = FileChunkedDistributedSampler(
            train_dataset,
            num_replicas=dist_ctx["world_size"],
            rank=dist_ctx["rank"],
            shuffle=True,
            seed=CONFIG["seed"],
            drop_last=True,
        )
        val_sampler = FileChunkedDistributedSampler(
            val_dataset,
            num_replicas=dist_ctx["world_size"],
            rank=dist_ctx["rank"],
            shuffle=False,
            seed=CONFIG["seed"],
            drop_last=False,
        )

        loader_kwargs = dict(num_workers=CONFIG["num_workers"], pin_memory=True)
        if CONFIG["num_workers"] > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            drop_last=True, **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            drop_last=False, **loader_kwargs,
        )

        # ---- Model ----
        model = GeoFNO2d(
            modes1=CONFIG["modes"],
            modes2=CONFIG["modes"],
            width=CONFIG["width"],
            in_channels=in_channels,
            out_channels=out_channels,
            s1=CONFIG["s1"],
            s2=CONFIG["s2"],
            num_fno_layers=CONFIG["num_fno_layers"],
        ).to(device)
        rank0_print(dist_ctx, f"[main] model params={sum(p.numel() for p in model.parameters()):,}")

        if dist_ctx["distributed"]:
            model = DDP(model, device_ids=[dist_ctx["local_rank"]], broadcast_buffers=False)

        # ---- Optimizer / scheduler ----
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        optimizer_steps_per_epoch = len(train_loader) // CONFIG["accum_steps"]
        if optimizer_steps_per_epoch < 1:
            raise ValueError(
                f"accum_steps={CONFIG['accum_steps']} too large for steps_per_epoch={len(train_loader)}"
            )
        total_steps = CONFIG["num_epochs"] * optimizer_steps_per_epoch
        warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr=CONFIG["lr"] * 0.01
        )
        rank0_print(dist_ctx, f"[main] total_steps={total_steps}, warmup_steps={warmup_steps}")

        # ---- TensorBoard ----
        if dist_ctx["is_rank0"]:
            tb_run_dir = os.path.join(CONFIG["tb_dir"], run_tag)
            os.makedirs(tb_run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_run_dir)
            rank0_print(dist_ctx, f"[main] tensorboard log dir={tb_run_dir}")

            md = "### Training Configuration\n| Parameter | Value |\n|---|---|\n"
            for k, v in CONFIG.items():
                md += f"| {k} | {v} |\n"
            md += f"| in_channels (derived) | {in_channels} |\n"
            md += f"| out_channels (derived) | {out_channels} |\n"
            md += "\n### System\n| Parameter | Value |\n|---|---|\n"
            md += f"| OS | {platform.system()} {platform.release()} |\n"
            md += f"| CPU Cores | {os.cpu_count()} |\n"
            md += f"| World Size | {dist_ctx['world_size']} |\n"
            try:
                md += f"| GPU | {torch.cuda.get_device_name(0)} |\n"
            except Exception:
                pass
            writer.add_text("config/all", md, 0)
        else:
            writer = None

        # ---- Train ----
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            num_epochs=CONFIG["num_epochs"],
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            coords_2d_device=coords_2d_device,
            writer=writer,
            grad_clip=CONFIG["grad_clip"],
            loss_type=CONFIG["loss_type"],
            add_noise=CONFIG["add_noise"],
            uvh_noise_std=CONFIG["uvh_noise_std"],
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
        )

        rank0_print(dist_ctx, "[main] done.")

    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add model/main.py
git commit -m "$(cat <<'EOF'
refactor(main): adapt to multi-file dataset, drop push/recurrent

- CONFIG drops pushforward_steps, recurrent_steps, train_paths, test_path
- Scans data/{train,val,test}/ via manifest.json
- Uses MultiStormSurgeDataset + FileChunkedDistributedSampler
- Drops manual coord normalization (now done inside load_static_coords)
- coords_2d_device passed to train_model as before; per-sample coord
  collation removed

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 重写 `model/test_all.py`

**目的：** 用文件外层循环（不走 lazy LRU Dataset）做自回归级联评测；归一化路径改 `data/normalization.mat`；删 push/recurrent 选项；新拼通道格式。

**Files:**
- Modify: `model/test_all.py`（整体重写）

- [ ] **Step 1: 完整重写 `model/test_all.py`**

```python
# model/test_all.py
"""Autoregressive multi-step test for Geo-FNO bundle model across a test split.

Each .pt file in --test_dir is loaded once, used to roll the model forward
group_len steps via consecutive bundle predictions, then released. Metrics
are aggregated globally (normalized + physical spaces).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from dataset import load_static_coords
from main import set_seed
from model import GeoFNO2d
from temporal_utils import (
    build_checkpoint_name,
    input_channels_for_bundle,
    output_channels_for_bundle,
    validate_temporal_params,
)


METRIC_SPACES = ("physical", "normalized")
CHANNEL_NAMES = ("u", "v", "h")
WATER_LEVEL_CHANNEL = 2
DRY_WATER_LEVEL_THRESHOLD = 0.005
REQUIRED_KEYS = ("graph", "storm_boundary", "inner_boundary")


def parse_args():
    p = argparse.ArgumentParser(description="Run Geo-FNO autoregressive test across a split.")
    p.add_argument("--test_dir", type=str, default="data/test", help="Path to test split directory.")
    p.add_argument("--coords", type=str, default="data/coordinates.mat")
    p.add_argument("--norm",   type=str, default="data/normalization.mat")
    p.add_argument("--model",  type=str, default=None, help="Checkpoint path.")
    p.add_argument("--output", type=str, default="geofno_autoregressive_results.txt")
    p.add_argument("--group_len",   type=int, default=24)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_files",   type=int, default=None, help="Limit number of files (smoke test).")
    p.add_argument("--bundle_size", type=int, default=1)
    p.add_argument("--allow_random_weights", action="store_true")
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--s1", type=int, default=64)
    p.add_argument("--s2", type=int, default=64)
    p.add_argument("--num_fno_layers", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def metric_output_path(base_path, metric_space):
    p = Path(base_path)
    return p.with_name(f"{p.stem}_{metric_space}{p.suffix or '.txt'}")


def load_normalization_stats(stats_path, device):
    print(f"[test] loading normalization stats from {stats_path}")
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(f"Normalization stats not found: {p}")
    f = scipy.io.loadmat(p)
    required = ["u_mean", "u_std", "v_mean", "v_std", "h_mean", "h_std"]
    missing = [k for k in required if k not in f]
    if missing:
        raise KeyError(f"{p} missing keys: {missing}")
    mean = torch.tensor(
        [float(f["u_mean"].item()), float(f["v_mean"].item()), float(f["h_mean"].item())],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3)
    std = torch.tensor(
        [float(f["u_std"].item()), float(f["v_std"].item()), float(f["h_std"].item())],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3)
    return mean, std


def denormalize(t, mean, std):
    return t * std + mean


def apply_dry_grid_error_mask(diff, target_norm, mean, std):
    target_wl = denormalize(
        target_norm[..., WATER_LEVEL_CHANNEL],
        mean[..., WATER_LEVEL_CHANNEL],
        std[..., WATER_LEVEL_CHANNEL],
    )
    dry_mask = target_wl < DRY_WATER_LEVEL_THRESHOLD
    return diff.masked_fill(dry_mask.unsqueeze(-1), 0.0)


def strip_module_prefix(state_dict):
    if not all(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            v = checkpoint.get(k)
            if isinstance(v, dict):
                return v
    return checkpoint


def resolve_checkpoint_path(explicit, default_name):
    if explicit is not None:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return explicit
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    candidates = [Path.cwd() / default_name, base_dir / default_name, script_dir / default_name]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError("Checkpoint not found. Checked: " + ", ".join(str(p) for p in candidates))
    return str(max(existing, key=lambda p: p.stat().st_mtime))


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = strip_module_prefix(extract_state_dict(ckpt))
    model.load_state_dict(sd)


def init_bucket(device):
    return {
        "sse": torch.zeros(3, device=device),
        "sae": torch.zeros(3, device=device),
        "sum_gt": torch.zeros(3, device=device),
        "sum_sq_gt": torch.zeros(3, device=device),
        "rel_l2_sum": torch.zeros(3, device=device),
        "count": 0,
    }


def compute_stats(bucket, num_nodes):
    count = bucket["count"]
    if count == 0:
        z = np.zeros(3, dtype=np.float64)
        return {"mse_channels": z, "rmse_channels": z, "mae_channels": z,
                "r2_channels": z, "rel_l2_channels": z}
    N = count * num_nodes
    sse = bucket["sse"]
    sae = bucket["sae"]
    mse_c = sse / N
    rmse_c = np.sqrt(mse_c)
    mae_c = sae / N
    ss_tot = bucket["sum_sq_gt"] - (bucket["sum_gt"] ** 2) / N
    ss_tot = np.maximum(ss_tot, 1e-8)
    r2_c = 1.0 - (sse / ss_tot)
    rl2_c = bucket["rel_l2_sum"] / count
    return {"mse_channels": mse_c, "rmse_channels": rmse_c, "mae_channels": mae_c,
            "r2_channels": r2_c, "rel_l2_channels": rl2_c}


def compute_auc(results, channel_names):
    auc = {ch: {} for ch in channel_names}
    steps = [r["step"] for r in results]
    if len(steps) < 2:
        for ch in channel_names:
            for m in ("mse", "rmse", "mae", "r2", "rel_l2"):
                auc[ch][m] = 0.0
        return auc
    for ch in channel_names:
        for metric in [k for k in results[0][ch].keys() if k != "step"]:
            y = [r[ch][metric] for r in results]
            auc[ch][metric] = float(np.trapz(y, steps))
    return auc


def build_features_batch(state_t, storm_w, inner_w, btype_oh):
    """state_t (B,N,3); storm_w (B,S+1,N,3); inner_w (B,S+1,N,2); btype_oh (N,3)."""
    B, _, N, _ = storm_w.shape
    storm_flat = storm_w.permute(0, 2, 1, 3).reshape(B, N, -1)
    inner_flat = inner_w.permute(0, 2, 1, 3).reshape(B, N, -1)
    btype_b = btype_oh.unsqueeze(0).expand(B, -1, -1)
    return torch.cat([state_t, storm_flat, inner_flat, btype_b], dim=-1).contiguous()


def find_test_files(test_dir: Path) -> list[Path]:
    return sorted(p for p in test_dir.glob("*.pt") if not p.name.startswith("._"))


def autoregressive_one_file(model, file_path, coords_2d_device, btype_oh_device,
                             mean, std, device, group_len, bundle_size, batch_size,
                             per_step_metrics_by_space):
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    for k in REQUIRED_KEYS:
        if k not in data:
            raise KeyError(f"{file_path}: missing key {k!r}")
    graph_all = data["graph"].float()
    storm_all = data["storm_boundary"].float()
    inner_all = data["inner_boundary"].float()
    T = graph_all.size(0)
    N = graph_all.size(1)
    if T <= group_len:
        return  # not enough timesteps for one group
    num_groups = T - group_len
    start_idx = torch.arange(num_groups, dtype=torch.long)
    x_in_base = coords_2d_device.unsqueeze(0)

    with torch.no_grad():
        for b0 in range(0, num_groups, batch_size):
            b1 = min(b0 + batch_size, num_groups)
            batch_starts = start_idx[b0:b1]
            B = batch_starts.shape[0]
            x_in = x_in_base.expand(B, -1, -1)

            current_state = graph_all[batch_starts].to(device)  # (B, N, 3)

            for block_offset in range(0, group_len, bundle_size):
                offsets = torch.arange(bundle_size + 1)
                idx_grid = batch_starts[:, None] + block_offset + offsets[None, :]
                storm_w = storm_all[idx_grid].to(device)        # (B, S+1, N, 3)
                inner_w = inner_all[idx_grid].to(device)        # (B, S+1, N, 2)
                features = build_features_batch(current_state, storm_w, inner_w, btype_oh_device)

                pred_block = model(features, x_in)              # (B, S, N, 3)

                for k in range(bundle_size):
                    step = block_offset + k
                    t_next = batch_starts + step + 1
                    pred_norm = pred_block[:, k]
                    target_norm = graph_all[t_next].to(device)

                    for ms in METRIC_SPACES:
                        if ms == "physical":
                            p_m = denormalize(pred_norm, mean, std)
                            t_m = denormalize(target_norm, mean, std)
                        else:
                            p_m = pred_norm
                            t_m = target_norm
                        diff = p_m - t_m
                        diff = apply_dry_grid_error_mask(diff, target_norm, mean, std)
                        bucket = per_step_metrics_by_space[ms][step]
                        bucket["sse"] += torch.sum(diff ** 2, dim=(0, 1))
                        bucket["sae"] += torch.sum(torch.abs(diff), dim=(0, 1))
                        bucket["sum_gt"] += torch.sum(t_m, dim=(0, 1))
                        bucket["sum_sq_gt"] += torch.sum(t_m ** 2, dim=(0, 1))
                        l2_err = torch.norm(diff.permute(0, 2, 1), p=2, dim=2)
                        l2_gt = torch.norm(t_m.permute(0, 2, 1), p=2, dim=2).clamp(min=1e-8)
                        bucket["rel_l2_sum"] += (l2_err / l2_gt).sum(dim=0)
                        bucket["count"] += B

                current_state = pred_block[:, -1]


def write_results(results_by_space, output_path, group_len, bundle_size, model_path):
    for ms in METRIC_SPACES:
        out = metric_output_path(output_path, ms)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(f"Autoregressive Test Results (group_len={group_len}, bundle_size={bundle_size})\n")
            f.write(f"Metric Space: {ms}\n")
            f.write(f"Checkpoint: {model_path if model_path is not None else 'random weights'}\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Step':<6} | {'Channel':<7} | {'MSE':<12} | {'RMSE':<12} | "
                    f"{'MAE':<12} | {'R2':<12} | {'Rel L2':<12}\n")
            f.write("-" * 100 + "\n")
            for r in results_by_space[ms]:
                step = r["step"]
                for j, ch in enumerate(CHANNEL_NAMES):
                    disp = "wl" if ch == "h" else ch
                    label = str(step) if j == 0 else ""
                    m = r[ch]
                    f.write(
                        f"{label:<6} | {disp:<7} | {m['mse']:<12.6f} | {m['rmse']:<12.6f} | "
                        f"{m['mae']:<12.6f} | {m['r2']:<12.6f} | {m['rel_l2']:<12.6f}\n"
                    )
                f.write("-" * 100 + "\n")
            auc = compute_auc(results_by_space[ms], CHANNEL_NAMES)
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"AUC Summary Over {group_len} Steps\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Channel':<11} | {'MSE Area':<12} | {'RMSE Area':<12} | "
                    f"{'MAE Area':<12} | {'R2 Area':<12} | {'Rel L2 Area':<12}\n")
            f.write("-" * 100 + "\n")
            for ch in CHANNEL_NAMES:
                m = auc[ch]
                disp = "wl" if ch == "h" else ch
                f.write(
                    f"{disp:<11} | {m['mse']:<12.6f} | {m['rmse']:<12.6f} | "
                    f"{m['mae']:<12.6f} | {m['r2']:<12.6f} | {m['rel_l2']:<12.6f}\n"
                )
            f.write("=" * 100 + "\n")
        print(f"[test] results -> {out}")


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[test] device={device}")

    set_seed(3407)
    validate_temporal_params(args.bundle_size)
    if args.group_len % args.bundle_size != 0:
        raise ValueError(
            f"group_len must be divisible by bundle_size: group_len={args.group_len}, "
            f"bundle_size={args.bundle_size}"
        )

    coords_2d_cpu, btype_oh_cpu = load_static_coords(args.coords)
    coords_2d_device = coords_2d_cpu.to(device)
    btype_oh_device = btype_oh_cpu.to(device)
    num_nodes = coords_2d_cpu.size(0)

    mean, std = load_normalization_stats(args.norm, device=device)

    in_channels = input_channels_for_bundle(args.bundle_size)
    out_channels = output_channels_for_bundle(args.bundle_size)
    model = GeoFNO2d(
        modes1=args.modes, modes2=args.modes, width=args.width,
        in_channels=in_channels, out_channels=out_channels,
        s1=args.s1, s2=args.s2, num_fno_layers=args.num_fno_layers,
    ).to(device)
    print(f"[test] model params={sum(p.numel() for p in model.parameters()):,}")

    default_ckpt = build_checkpoint_name(args.bundle_size)
    try:
        model_path = resolve_checkpoint_path(args.model, default_ckpt)
    except FileNotFoundError:
        if not args.allow_random_weights:
            raise
        model_path = None

    if model_path is not None:
        print(f"[test] loading checkpoint {model_path}")
        load_checkpoint(model, model_path, device)
    else:
        print("[test] warning: using random weights (--allow_random_weights)")

    model.eval()

    per_step_metrics_by_space = {
        space: [init_bucket(device) for _ in range(args.group_len)]
        for space in METRIC_SPACES
    }

    test_files = find_test_files(Path(args.test_dir))
    if args.num_files is not None:
        test_files = test_files[: args.num_files]
    if not test_files:
        raise FileNotFoundError(f"No .pt files in {args.test_dir}")

    for fp in tqdm(test_files, desc="Test files"):
        autoregressive_one_file(
            model, fp, coords_2d_device, btype_oh_device,
            mean, std, device, args.group_len, args.bundle_size, args.batch_size,
            per_step_metrics_by_space,
        )

    for ms in METRIC_SPACES:
        for step in range(args.group_len):
            b = per_step_metrics_by_space[ms][step]
            for k in ("sse", "sae", "sum_gt", "sum_sq_gt", "rel_l2_sum"):
                b[k] = b[k].detach().cpu().numpy()

    results_by_space = {space: [] for space in METRIC_SPACES}
    for ms in METRIC_SPACES:
        for step, bucket in enumerate(per_step_metrics_by_space[ms]):
            stats = compute_stats(bucket, num_nodes)
            entry = {"step": step + 1}
            for i, ch in enumerate(CHANNEL_NAMES):
                entry[ch] = {
                    "mse": float(stats["mse_channels"][i]),
                    "rmse": float(stats["rmse_channels"][i]),
                    "mae": float(stats["mae_channels"][i]),
                    "r2": float(stats["r2_channels"][i]),
                    "rel_l2": float(stats["rel_l2_channels"][i]),
                }
            results_by_space[ms].append(entry)
            summary = f"[step {step + 1:02d}][{ms}] "
            for ch in CHANNEL_NAMES:
                disp = "wl" if ch == "h" else ch
                m = entry[ch]
                summary += (f"{disp}: mse={m['mse']:.6f} rmse={m['rmse']:.6f} "
                            f"mae={m['mae']:.6f} r2={m['r2']:.6f} rel_l2={m['rel_l2']:.6f} | ")
            print(summary.rstrip(" | "))

    write_results(results_by_space, args.output, args.group_len, args.bundle_size, model_path)
    print("[test] done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add model/test_all.py
git commit -m "$(cat <<'EOF'
refactor(test): file-by-file autoregressive evaluation across data/test/

- Outer loop over each .pt file (load once, free after), avoids LRU
  thrash that lazy multi-file Dataset would cause for sequential access
- Adapts to new feature layout: build_features_batch concatenates
  [state, storm_window, inner_window, btype_oh]
- Removes --pushforward_steps / --recurrent_steps options
- Default --norm path → data/normalization.mat
- Preserves apply_dry_grid_error_mask, metric_space split (physical /
  normalized), AUC summary

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: 服务器联调清单（不在本机执行，作为交付文档）

**目的：** 给服务器端跑通的精确指令清单，便于用户在服务器上一次性执行。

**Files:**
- Create: `docs/superpowers/plans/2026-05-16-server-runbook.md`

- [ ] **Step 1: 写 runbook**

```markdown
# Server Runbook — Geo-FNO Refactor 验证

## 前置环境

服务器上确认 PyTorch 2.x、CUDA、scipy、tqdm、tensorboard 已装。

## 1. 构建 manifest

```bash
python scripts/build_manifest.py data/train --bundle_size_warn 72
python scripts/build_manifest.py data/val   --bundle_size_warn 72
python scripts/build_manifest.py data/test  --bundle_size_warn 72
```

## 2. 单元测试

```bash
pytest tests/ -v
```

期望：`test_temporal_utils.py` 8 项、`test_dataset.py` 10 项全部 PASS。

## 3. 单卡冒烟（bundle_size=1）

临时把 `CONFIG["bundle_size"] = 1`、`CONFIG["batch_size"] = 4`、
`CONFIG["num_epochs"] = 2`，仅用 train 的 1-2 个文件试跑：

```bash
CUDA_VISIBLE_DEVICES=0 python model/main.py
```

观察：第一 epoch loss 下降、第二 epoch 进入 val。无运行时报错。

## 4. 全量 4 卡 DDP（bundle_size=72）

恢复 CONFIG 至 `bundle_size=72, batch_size=16, num_epochs=100`，执行：

```bash
torchrun --nproc_per_node=4 model/main.py
```

监控：
- 单卡显存峰值
- `train/loss_step` TB 曲线
- worker `__getitem__` 耗时（可加临时打印）

## 5. 测试链路

训练得到 `best_geofno_b72.pt` 后：

```bash
python model/test_all.py \
    --test_dir data/test \
    --model best_geofno_b72.pt \
    --bundle_size 72 \
    --group_len 72 \
    --num_files 4   # 先冒烟 4 个文件
```

确认两份输出文件 `geofno_autoregressive_results_normalized.txt` 与
`geofno_autoregressive_results_physical.txt` 生成且数值合理。

## 6. 全量 test

去掉 `--num_files`，跑完 64 个测试文件。
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-05-16-server-runbook.md
git commit -m "$(cat <<'EOF'
docs(plans): add server runbook for refactor verification

Step-by-step commands the user runs on the 4×RTX-PRO-6000 server to
verify the refactor: build manifests, run pytest, smoke-test single-GPU
bundle=1, full 4-GPU DDP bundle=72, then autoregressive test evaluation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec 覆盖审查

| Spec 章节 | 由哪个 Task 实现 |
|---|---|
| 目标 1 (Dataset 新格式) | T3 (`MultiStormSurgeDataset`) |
| 目标 2 (per-node 边界 + 类型 mask) | T3 (`_build_features` + `load_static_coords`) |
| 目标 3 (多文件无预加载) | T3 + T2 (manifest) |
| 目标 4 (RAM 不 ×4 膨胀) | T3 (lazy + LRU + file-affine sampler) |
| 目标 5 (只保留 bundle) | T1 + T4 + T5 |
| 目标 6 (test_all 多文件) | T6 |
| 数据格式 .pt 验证 | T2 (`read_file_metadata`) + T3 (`_load_pt`) |
| coordinates.mat 解析 | T3 (`load_static_coords`) |
| 通道布局 5*S + 11 | T1 (`input_channels_for_bundle`) + T3 (`_build_features`) |
| 模型主体不变 | （无 Task；`model/model.py` 保持原样） |
| 删除 push/recurrent | T1 + T4 + T5 + T6 |
| CONFIG 新字段 | T5 |
| 测试链路 文件外层循环 + group_len % S 约束 | T6 |
| 风险 - manifest 不同步 | T3 (`MultiStormSurgeDataset.__init__` 校验文件存在) |
| 风险 - T ≤ S 丢弃 | T2 (warn) + T3 (drop) |
| 风险 - boundary 值域 | T3 (`load_static_coords` assert) |
| 验收 - 单卡 bundle=1 跑通 | T7 (runbook step 3) |

### Placeholder 扫描

无 "TBD"、"TODO"、"implement later"、"similar to"。每个 step 都有完整代码或具体命令。

### 类型与名称一致性

- `MultiStormSurgeDataset.flat_index` → T3 定义，T5 (sampler 构造) 与 test_dataset.py 都用此名
- `load_static_coords` 在 T3 定义，T5 main.py 与 T6 test_all.py 都按此 import
- `FileChunkedDistributedSampler.set_epoch` 在 T3 定义，T4 train.py 通过 `hasattr` 调用——兼容
- `_build_features` (T3 单样本版) 与 T6 `build_features_batch`（B 维批量版）一致 layout
- `build_checkpoint_name(bundle_size, noise_suffix)` 在 T1 定义，T5/T6 都按此签名调用
