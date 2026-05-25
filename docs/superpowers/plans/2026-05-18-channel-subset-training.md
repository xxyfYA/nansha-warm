# 通道子集训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 Geo-FNO 加一个 `channels` 字符串参数，端到端控制哪些 graph 状态通道（u/v/h 任意子集）参与训练与自回归评测，同时清理掉本项目不使用的 UVH 输入噪声代码。

**Architecture:** 在 `temporal_utils.py` 加 `parse_channels` / `channels_suffix` 工具，把 `num_channels` 作为新参数贯穿 dataset / model / train / test 整条链；外部驱动（storm、inner、btype）始终全量输入，只对 `state_t` 和 `target` 做切片；checkpoint / run tag / test 输出文件按通道后缀命名；dry-grid mask 在 test_all 中始终用真实 h 构建（与是否预测 h 解耦）。

**Tech Stack:** PyTorch 2.x + DDP（torchrun），scipy.io，tqdm，TensorBoard，pytest。

---

## 前置说明

- 本机不执行 GPU 训练验证。每个 task 的"run test"步骤可直接在本地跑 CPU pytest；服务器侧只在 Task 8 跑训练 smoke。
- 实现遵循 [docs/superpowers/specs/2026-05-18-channel-subset-training-design.md](../specs/2026-05-18-channel-subset-training-design.md)。
- 默认 `channels="uvh"` 路径必须与改动前数值完全等价（关键回归点）。
- 严格按 TDD：每个 task 先写 / 改测试，再改实现，确保红→绿，再 commit。

---

### Task 0: 调整 `.gitignore` 让 `tests/` 不再被 ignore

**目的：** 上一次重构中 `tests/` 在 `.gitignore` 末尾被列为 ignored；现有 3 个测试文件是早期 `git add -f` 留下的，但后续新增测试必须再 `-f` 才能加进来。本 task 把这一行清掉，避免 plan 后续每个 task 都要 `-f`。

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 读 `.gitignore` 末尾确认 `tests/` 行**

Run: `tail -3 .gitignore`
Expected: 最后一行是 `tests/`（前面有空行）。

- [ ] **Step 2: 删除 `tests/` 行**

把 `.gitignore` 末尾的 `tests/` 那一行删掉（连同它前面的空行也删，保持文件干净）。

- [ ] **Step 3: 验证**

Run: `git check-ignore -v tests/test_temporal_utils.py`
Expected: exit code 1，无输出（说明文件不再被 ignore）。

Run: `git status --short`
Expected: 只显示 `.gitignore` 一行 `M` 改动。

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "$(cat <<'EOF'
chore(gitignore): stop ignoring tests/ directory

The tests/ entry was inherited from an earlier scaffold but tests are
checked in. Removing it so new test files don't require git add -f.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1: `model/temporal_utils.py` 重构（channel 工具 + 签名变更）

**目的：** 引入 `parse_channels` / `channels_suffix`；把 `num_channels` 作为新参数加入 `input_channels_for_bundle` / `output_channels_for_bundle` / `TemporalConfig`；`build_checkpoint_name` / `build_run_suffix` 把 `noise_suffix` 参数换成 `channels_suffix`。

实现 spec 章节：「§ 通道规范」「§ 涉及模块改动 - 1. `model/temporal_utils.py`」。

**Files:**
- Modify: `model/temporal_utils.py`
- Modify: `tests/test_temporal_utils.py`

- [ ] **Step 1: 改写 `tests/test_temporal_utils.py` 加入新测试，并把 `_noise` 的旧测试 case 换成 `_chh`**

完整替换文件内容为：

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from temporal_utils import (  # noqa: E402
    CHANNEL_ORDER,
    CHANNEL_TO_INDEX,
    TemporalConfig,
    build_checkpoint_name,
    build_run_suffix,
    channels_suffix,
    input_channels_for_bundle,
    num_temporal_samples,
    output_channels_for_bundle,
    parse_channels,
    validate_temporal_params,
)


def test_input_channels_formula_default_3():
    # C_in = K + 5*S + 8; default K=3 -> 5S+11
    assert input_channels_for_bundle(1) == 16
    assert input_channels_for_bundle(24) == 131
    assert input_channels_for_bundle(72) == 371


def test_input_channels_formula_varies_with_num_channels():
    assert input_channels_for_bundle(1, num_channels=1) == 14
    assert input_channels_for_bundle(1, num_channels=2) == 15
    assert input_channels_for_bundle(8, num_channels=1) == 49
    assert input_channels_for_bundle(8, num_channels=2) == 50
    assert input_channels_for_bundle(8, num_channels=3) == 51


def test_output_channels_formula_default_3():
    assert output_channels_for_bundle(1) == 3
    assert output_channels_for_bundle(24) == 72
    assert output_channels_for_bundle(72) == 216


def test_output_channels_formula_varies_with_num_channels():
    assert output_channels_for_bundle(8, num_channels=1) == 8
    assert output_channels_for_bundle(8, num_channels=2) == 16
    assert output_channels_for_bundle(8, num_channels=3) == 24


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


def test_channel_order_constants():
    assert CHANNEL_ORDER == ("u", "v", "h")
    assert CHANNEL_TO_INDEX == {"u": 0, "v": 1, "h": 2}


def test_parse_channels_full():
    assert parse_channels("uvh") == (0, 1, 2)
    assert parse_channels("UVH") == (0, 1, 2)


def test_parse_channels_subset_normalizes_order():
    assert parse_channels("h") == (2,)
    assert parse_channels("vh") == (1, 2)
    assert parse_channels("hv") == (1, 2)
    assert parse_channels("uh") == (0, 2)
    assert parse_channels("uv") == (0, 1)


def test_parse_channels_dedups():
    assert parse_channels("hh") == (2,)
    assert parse_channels("uuvh") == (0, 1, 2)


def test_parse_channels_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        parse_channels("")


def test_parse_channels_rejects_unknown_char():
    with pytest.raises(ValueError, match="unknown"):
        parse_channels("uvx")
    with pytest.raises(ValueError, match="unknown"):
        parse_channels("p")


def test_channels_suffix_uvh_is_empty():
    assert channels_suffix((0, 1, 2)) == ""


def test_channels_suffix_subsets():
    assert channels_suffix((2,)) == "_chh"
    assert channels_suffix((0,)) == "_chu"
    assert channels_suffix((1,)) == "_chv"
    assert channels_suffix((0, 1)) == "_chuv"
    assert channels_suffix((1, 2)) == "_chvh"
    assert channels_suffix((0, 2)) == "_chuh"


def test_temporal_config_default_num_channels():
    cfg = TemporalConfig(bundle_size=72)
    assert cfg.bundle_size == 72
    assert cfg.required_future_steps == 72
    assert cfg.input_channels == 371
    assert cfg.out_channels == 216
    assert cfg.num_channels == 3


def test_temporal_config_with_num_channels():
    cfg = TemporalConfig(bundle_size=8, num_channels=1)
    assert cfg.bundle_size == 8
    assert cfg.input_channels == 49
    assert cfg.out_channels == 8
    assert cfg.num_channels == 1


def test_build_checkpoint_name_default_uvh():
    assert build_checkpoint_name(1) == "best_geofno.pt"
    assert build_checkpoint_name(72) == "best_geofno_b72.pt"


def test_build_checkpoint_name_with_channels_suffix():
    assert build_checkpoint_name(1, "_chh") == "best_geofno_chh.pt"
    assert build_checkpoint_name(72, "_chuv") == "best_geofno_b72_chuv.pt"


def test_build_run_suffix_default_uvh():
    assert build_run_suffix(1) == ""
    assert build_run_suffix(72) == "_b72"


def test_build_run_suffix_with_channels_suffix():
    assert build_run_suffix(1, "_chh") == "_chh"
    assert build_run_suffix(72, "_chuv") == "_b72_chuv"
```

- [ ] **Step 2: 运行测试确认它们失败（实现还没改）**

Run: `pytest tests/test_temporal_utils.py -v`
Expected: ImportError on `CHANNEL_ORDER` / `parse_channels` / `channels_suffix` — 红。

- [ ] **Step 3: 重写 `model/temporal_utils.py`**

完整文件内容：

```python
"""Bundle-only temporal helpers for Geo-FNO storm-surge model.

Provides:
- channel-subset utilities (parse_channels, channels_suffix)
- per-bundle input/output channel counts parameterized by num_channels
- checkpoint / run-tag naming with optional channel suffix
"""
from dataclasses import dataclass


CHANNEL_ORDER = ("u", "v", "h")
CHANNEL_TO_INDEX = {"u": 0, "v": 1, "h": 2}


@dataclass(frozen=True)
class TemporalConfig:
    bundle_size: int = 1
    num_channels: int = 3

    def __post_init__(self):
        validate_temporal_params(self.bundle_size)
        if self.num_channels < 1 or self.num_channels > 3:
            raise ValueError(f"num_channels must be in [1, 3], got {self.num_channels}")

    @property
    def required_future_steps(self) -> int:
        return self.bundle_size

    @property
    def input_channels(self) -> int:
        return input_channels_for_bundle(self.bundle_size, self.num_channels)

    @property
    def out_channels(self) -> int:
        return output_channels_for_bundle(self.bundle_size, self.num_channels)


def validate_temporal_params(bundle_size: int) -> None:
    if bundle_size < 1:
        raise ValueError(f"bundle_size must be >= 1, got {bundle_size}")


def num_temporal_samples(num_time: int, bundle_size: int) -> int:
    validate_temporal_params(bundle_size)
    num_samples = num_time - bundle_size
    if num_samples <= 0:
        raise ValueError(
            f"not enough time steps: num_time={num_time}, "
            f"bundle_size={bundle_size}, required_future_steps={bundle_size}"
        )
    return num_samples


def parse_channels(spec: str) -> tuple[int, ...]:
    """Parse a channel-spec string into an ordered tuple of indices.

    "h" -> (2,); "uvh" -> (0,1,2); "vh" / "hv" -> (1,2). Case-insensitive,
    deduplicates, and returns indices sorted ascending so downstream code can
    rely on (u<v<h) order.
    """
    s = spec.lower().strip()
    if not s:
        raise ValueError("channels spec is empty")
    seen: list[int] = []
    for ch in s:
        if ch not in CHANNEL_TO_INDEX:
            raise ValueError(f"unknown channel {ch!r}; allowed: u, v, h")
        idx = CHANNEL_TO_INDEX[ch]
        if idx not in seen:
            seen.append(idx)
    return tuple(sorted(seen))


def channels_suffix(indices: tuple[int, ...]) -> str:
    """ "" for (0,1,2) (full uvh, backward-compatible); else "_ch" + names."""
    if indices == (0, 1, 2):
        return ""
    return "_ch" + "".join(CHANNEL_ORDER[i] for i in indices)


def input_channels_for_bundle(bundle_size: int, num_channels: int = 3) -> int:
    """C_in = K + 3*(S+1) + 2*(S+1) + 3 = K + 5*S + 8."""
    validate_temporal_params(bundle_size)
    if num_channels < 1 or num_channels > 3:
        raise ValueError(f"num_channels must be in [1, 3], got {num_channels}")
    return num_channels + 5 * bundle_size + 8


def output_channels_for_bundle(bundle_size: int, num_channels: int = 3) -> int:
    """C_out = K * S residual states."""
    validate_temporal_params(bundle_size)
    if num_channels < 1 or num_channels > 3:
        raise ValueError(f"num_channels must be in [1, 3], got {num_channels}")
    return num_channels * bundle_size


def build_checkpoint_name(bundle_size: int, channels_suffix: str = "") -> str:
    validate_temporal_params(bundle_size)
    if bundle_size == 1:
        return f"best_geofno{channels_suffix}.pt"
    return f"best_geofno_b{bundle_size}{channels_suffix}.pt"


def build_run_suffix(bundle_size: int, channels_suffix: str = "") -> str:
    validate_temporal_params(bundle_size)
    if bundle_size == 1:
        return channels_suffix
    return f"_b{bundle_size}{channels_suffix}"
```

- [ ] **Step 4: 运行测试确认绿**

Run: `pytest tests/test_temporal_utils.py -v`
Expected: 全部 PASS（约 19 项）。

- [ ] **Step 5: Commit**

```bash
git add model/temporal_utils.py tests/test_temporal_utils.py
git commit -m "$(cat <<'EOF'
feat(temporal_utils): add channel-subset helpers and parameterize counts

- parse_channels/channels_suffix for u/v/h subset specs
- input/output_channels_for_bundle accept num_channels (default 3)
- TemporalConfig exposes num_channels
- build_checkpoint_name/build_run_suffix swap noise_suffix for channels_suffix

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `model/dataset.py` 加 `state_channels` 字段

**目的：** `_build_features` 只把选定的 state 通道送进特征；两个 Dataset 类的 `__init__` 加 `state_channels: tuple[int, ...] = (0, 1, 2)` 字段；`__getitem__` 同步裁 target。

实现 spec 章节：「§ 涉及模块改动 - 2. `model/dataset.py`」。

**Files:**
- Modify: `model/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: 在 `tests/test_dataset.py` 末尾追加 3 个新测试**

在文件最末尾追加：

```python


def test_single_dataset_with_state_channels_h_only(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    ds = StormSurgeDataset(
        path=split_dir / "e0.pt",
        bundle_size=4,
        btype_oh=btype,
        lru_capacity=1,
        state_channels=(2,),
    )
    feat, target = ds[0]
    # C_in = K + 5*S + 8 = 1 + 20 + 8 = 29
    assert feat.shape == (N_NODES, 29)
    assert target.shape == (4, N_NODES, 1)


def test_single_dataset_with_state_channels_uv(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    ds = StormSurgeDataset(
        path=split_dir / "e0.pt",
        bundle_size=4,
        btype_oh=btype,
        lru_capacity=1,
        state_channels=(0, 1),
    )
    feat, target = ds[0]
    # C_in = 2 + 20 + 8 = 30
    assert feat.shape == (N_NODES, 30)
    assert target.shape == (4, N_NODES, 2)


def test_multi_dataset_with_state_channels_h_only(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(
        data_dir=split_dir,
        bundle_size=4,
        btype_oh=btype,
        lru_files_per_worker=1,
        state_channels=(2,),
    )
    feat, target = mds[0]
    assert feat.shape == (N_NODES, 29)
    assert target.shape == (4, N_NODES, 1)


def test_dataset_state_channels_target_matches_graph_slice(tmp_path):
    """target[..., i] must equal graph[t+1:t+S+1, :, state_channels[i]]."""
    d = tmp_path / "layout"
    d.mkdir()
    path = d / "event.pt"
    _make_deterministic_pt(path, T=6, num_nodes=3)
    btype = torch.eye(3, dtype=torch.float32)
    ds = StormSurgeDataset(
        path=path,
        bundle_size=2,
        btype_oh=btype,
        lru_capacity=1,
        state_channels=(2,),
    )
    _, target = ds[1]
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = data["graph"][2:4, :, 2:3]
    assert torch.equal(target, expected)


def test_dataset_state_channels_features_state_prefix(tmp_path):
    """features[:, :K] must equal graph[t, :, state_channels]."""
    d = tmp_path / "layout"
    d.mkdir()
    path = d / "event.pt"
    _make_deterministic_pt(path, T=6, num_nodes=3)
    btype = torch.eye(3, dtype=torch.float32)
    ds = StormSurgeDataset(
        path=path,
        bundle_size=2,
        btype_oh=btype,
        lru_capacity=1,
        state_channels=(1, 2),
    )
    features, _ = ds[1]
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected_state = data["graph"][1, :, 1:3]
    assert torch.equal(features[:, :2], expected_state)
```

- [ ] **Step 2: 跑测试确认它们失败**

Run: `pytest tests/test_dataset.py -v -k "state_channels"`
Expected: TypeError on `state_channels` keyword — 红。

- [ ] **Step 3: 修改 `model/dataset.py` 的 `_build_features`**

把 `_build_features` 整个替换为：

```python
def _build_features(
    state_t: torch.Tensor,
    storm_window: torch.Tensor,
    inner_window: torch.Tensor,
    btype_oh: torch.Tensor,
    state_channels: tuple[int, ...] = (0, 1, 2),
) -> torch.Tensor:
    """Build one per-node feature matrix from temporal windows.

    Only the columns in state_channels are kept from state_t; storm and inner
    windows are always passed through in full.
    """
    state_sub = state_t[..., list(state_channels)]
    num_nodes = state_sub.size(0)
    storm_flat = storm_window.permute(1, 0, 2).reshape(num_nodes, -1)
    inner_flat = inner_window.permute(1, 0, 2).reshape(num_nodes, -1)
    return torch.cat([state_sub, storm_flat, inner_flat, btype_oh], dim=-1).contiguous()
```

- [ ] **Step 4: 修改 `StormSurgeDataset.__init__` 与 `__getitem__`**

`__init__` 签名改为：

```python
def __init__(
    self,
    path,
    bundle_size,
    btype_oh,
    lru_capacity: int = 1,
    state_channels: tuple[int, ...] = (0, 1, 2),
):
```

在 `__init__` body 现有校验之后、`self._cache = ...` 之前加入：

```python
self.state_channels = tuple(state_channels)
if not self.state_channels or any(c < 0 or c > 2 for c in self.state_channels):
    raise ValueError(f"state_channels must be a non-empty subset of (0,1,2); got {state_channels}")
```

`__getitem__` 末尾两行改成：

```python
target = graph[idx + 1 : idx + bundle_size + 1][..., list(self.state_channels)].contiguous()
features = _build_features(state_t, storm_window, inner_window, self.btype_oh, self.state_channels)
return features, target
```

- [ ] **Step 5: 修改 `MultiStormSurgeDataset.__init__` 与 `__getitem__`**

`__init__` 签名改为：

```python
def __init__(
    self,
    data_dir,
    bundle_size,
    btype_oh,
    lru_files_per_worker: int = 2,
    state_channels: tuple[int, ...] = (0, 1, 2),
):
```

在 `__init__` 现有校验之后加入相同的 `state_channels` 校验：

```python
self.state_channels = tuple(state_channels)
if not self.state_channels or any(c < 0 or c > 2 for c in self.state_channels):
    raise ValueError(f"state_channels must be a non-empty subset of (0,1,2); got {state_channels}")
```

`__getitem__` 末尾改成：

```python
target = graph[t + 1 : t + bundle_size + 1][..., list(self.state_channels)].contiguous()
features = _build_features(state_t, storm_window, inner_window, self.btype_oh, self.state_channels)
return features, target
```

- [ ] **Step 6: 跑全部 dataset 测试**

Run: `pytest tests/test_dataset.py -v`
Expected: 全部 PASS（原 17 项 + 新 5 项）。

- [ ] **Step 7: Commit**

```bash
git add model/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(dataset): support state_channels subset in feature/target slicing

Both StormSurgeDataset and MultiStormSurgeDataset accept a state_channels
tuple (default (0,1,2)). _build_features keeps only the selected state
columns; targets are sliced the same way. Storm/inner/btype windows are
always passed through in full so external forcing remains available.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `model/model.py` 的 `GeoFNO2d` 加 `num_channels`

**目的：** 模型只看到抽象的 K 个状态通道；`state_in = u[..., :num_channels]`；`out_channels % num_channels == 0`；输出 `(B, S, N, num_channels)`。

实现 spec 章节：「§ 涉及模块改动 - 3. `model/model.py`」。

**Files:**
- Modify: `model/model.py`
- Create: `tests/test_geofno_num_channels.py`

- [ ] **Step 1: 新建 `tests/test_geofno_num_channels.py`**

```python
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import GeoFNO2d  # noqa: E402


def _build(num_channels, bundle_size=2):
    return GeoFNO2d(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=num_channels + 5 * bundle_size + 8,
        out_channels=num_channels * bundle_size,
        s1=4,
        s2=4,
        num_fno_layers=1,
        num_channels=num_channels,
    )


def test_geofno_default_num_channels_is_3():
    model = GeoFNO2d(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=18,
        out_channels=6,
        s1=4,
        s2=4,
        num_fno_layers=1,
    )
    assert model.num_channels == 3
    assert model.bundle_size == 2


def test_geofno_forward_shape_uvh():
    model = _build(num_channels=3, bundle_size=2)
    B, N = 1, 5
    u = torch.randn(B, N, 3 + 5 * 2 + 8)
    x = torch.rand(B, N, 2)
    out = model(u, x)
    assert out.shape == (B, 2, N, 3)


def test_geofno_forward_shape_h_only():
    model = _build(num_channels=1, bundle_size=2)
    B, N = 1, 5
    u = torch.randn(B, N, 1 + 5 * 2 + 8)
    x = torch.rand(B, N, 2)
    out = model(u, x)
    assert out.shape == (B, 2, N, 1)


def test_geofno_forward_shape_uv():
    model = _build(num_channels=2, bundle_size=3)
    B, N = 2, 4
    u = torch.randn(B, N, 2 + 5 * 3 + 8)
    x = torch.rand(B, N, 2)
    out = model(u, x)
    assert out.shape == (B, 3, N, 2)


def test_geofno_rejects_out_channels_not_divisible_by_num_channels():
    with pytest.raises(ValueError, match="divisible"):
        GeoFNO2d(
            modes1=2,
            modes2=2,
            width=4,
            in_channels=14,
            out_channels=5,  # not divisible by 2
            s1=4,
            s2=4,
            num_fno_layers=1,
            num_channels=2,
        )


def test_geofno_residual_uses_first_K_columns():
    """The residual base must come from features[..., :num_channels], so passing
    zero delta means output equals state_in."""
    model = _build(num_channels=1, bundle_size=2)
    # Zero out fc2 so delta is bias-only; easier: just check that state_in is
    # the first column.
    B, N = 1, 3
    u = torch.zeros(B, N, 1 + 5 * 2 + 8)
    u[..., 0] = 1.234  # the single state channel
    x = torch.rand(B, N, 2)
    out = model(u, x)
    # delta has no reason to be exactly zero, but state_in must be the broadcasted 1.234.
    # Test instead that out - delta == state_in for each bundle step.
    # Simpler: just check that disabling fc2 yields out == state_in.
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
    out = model(u, x)
    assert torch.allclose(out, torch.full((B, 2, N, 1), 1.234), atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_geofno_num_channels.py -v`
Expected: TypeError on `num_channels` keyword — 红。

- [ ] **Step 3: 修改 `GeoFNO2d.__init__`**

把 `__init__` 签名改为：

```python
def __init__(
    self,
    modes1,
    modes2,
    width,
    in_channels,
    out_channels,
    s1=40,
    s2=40,
    num_fno_layers: int = 3,
    num_channels: int = 3,
):
```

在 body 开头（`if num_fno_layers < 1` 之前）加：

```python
if num_channels < 1 or num_channels > 3:
    raise ValueError(f"num_channels must be in [1, 3], got {num_channels}")
self.num_channels = num_channels
```

把原本：

```python
if out_channels % 3 != 0:
    raise ValueError(f"GeoFNO2d requires out_channels divisible by 3, got {out_channels}")
self.bundle_size = out_channels // 3
```

替换为：

```python
if out_channels % num_channels != 0:
    raise ValueError(
        f"GeoFNO2d requires out_channels divisible by num_channels={num_channels}, "
        f"got out_channels={out_channels}"
    )
self.bundle_size = out_channels // num_channels
```

- [ ] **Step 4: 修改 `GeoFNO2d.forward`**

把开头的两条 3 通道硬校验：

```python
if u.size(-1) < 3:
    raise ValueError(f"GeoFNO2d residual output requires at least 3 input channels, got {u.size(-1)}")
if self.out_channels % 3 != 0:
    raise ValueError(f"GeoFNO2d requires out_channels divisible by 3, got {self.out_channels}")

state_in = u[..., :3]
```

替换为：

```python
if u.size(-1) < self.num_channels:
    raise ValueError(
        f"GeoFNO2d expects at least num_channels={self.num_channels} input columns, "
        f"got {u.size(-1)}"
    )

state_in = u[..., :self.num_channels]
```

并把最后的 reshape：

```python
delta = delta_flat.view(batch_size, num_nodes, self.bundle_size, 3)
```

替换为：

```python
delta = delta_flat.view(batch_size, num_nodes, self.bundle_size, self.num_channels)
```

更新 docstring 中"first 3 channels must be the current normalized state (u, v, h)"为"first `num_channels` columns must be the current normalized state in the order of the selected channels (subset of u, v, h)"。

返回行的 `(batch, bundle_size, N, 3)` 注释改成 `(batch, bundle_size, N, num_channels)`。

- [ ] **Step 5: 跑测试确认绿**

Run: `pytest tests/test_geofno_num_channels.py tests/test_spectralconv.py -v`
Expected: 全 PASS。

- [ ] **Step 6: Commit**

```bash
git add model/model.py tests/test_geofno_num_channels.py
git commit -m "$(cat <<'EOF'
feat(model): parameterize GeoFNO2d on num_channels

Model now treats state as an abstract K-channel block. Residual base is
features[..., :num_channels], output is (B, bundle_size, N, num_channels).
out_channels must be divisible by num_channels. Default num_channels=3
keeps existing checkpoints loadable.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `model/train.py` 删除 noise + 加 `state_channels`

**目的：** 彻底删 `make_uvh_noise_std_tensor`、`add_uvh_training_noise`、`train_model` 中 `add_noise`/`uvh_noise_std`/`noise_t` 分支；`evaluate_model` / `train_model` 接受 `state_channels`，按选定通道动态报告 per-channel rel_l2 与 tensorboard scalar。

实现 spec 章节：「§ 涉及模块改动 - 4. `model/train.py`」。

**Files:**
- Modify: `model/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: 新建 `tests/test_train.py`**

```python
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import GeoFNO2d  # noqa: E402
from train import evaluate_model  # noqa: E402


def _build(num_channels=3, bundle_size=2):
    return GeoFNO2d(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=num_channels + 5 * bundle_size + 8,
        out_channels=num_channels * bundle_size,
        s1=4,
        s2=4,
        num_fno_layers=1,
        num_channels=num_channels,
    )


class _FixedLoader:
    """Returns a fixed list of (features, target_block) batches once."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _make_batch(num_channels, bundle_size, B=1, N=6):
    features = torch.randn(B, N, num_channels + 5 * bundle_size + 8)
    target = torch.randn(B, bundle_size, N, num_channels)
    return features, target


def test_evaluate_model_uvh_reports_three_per_channel():
    torch.manual_seed(0)
    model = _build(num_channels=3, bundle_size=2)
    coords = torch.rand(6, 2)
    loader = _FixedLoader([_make_batch(3, 2)])
    metrics = evaluate_model(model, loader, torch.device("cpu"), coords, state_channels=(0, 1, 2))
    assert set(metrics.keys()) >= {"mse", "rmse", "mae", "rel_l2", "rel_u", "rel_v", "rel_h"}


def test_evaluate_model_h_only_reports_only_rel_h():
    torch.manual_seed(0)
    model = _build(num_channels=1, bundle_size=2)
    coords = torch.rand(6, 2)
    loader = _FixedLoader([_make_batch(1, 2)])
    metrics = evaluate_model(model, loader, torch.device("cpu"), coords, state_channels=(2,))
    assert "rel_h" in metrics
    assert "rel_u" not in metrics
    assert "rel_v" not in metrics


def test_evaluate_model_uv_reports_only_rel_u_and_rel_v():
    torch.manual_seed(0)
    model = _build(num_channels=2, bundle_size=2)
    coords = torch.rand(6, 2)
    loader = _FixedLoader([_make_batch(2, 2)])
    metrics = evaluate_model(model, loader, torch.device("cpu"), coords, state_channels=(0, 1))
    assert "rel_u" in metrics
    assert "rel_v" in metrics
    assert "rel_h" not in metrics


def test_train_module_no_longer_exposes_noise_helpers():
    import train as train_mod
    assert not hasattr(train_mod, "make_uvh_noise_std_tensor")
    assert not hasattr(train_mod, "add_uvh_training_noise")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_train.py -v`
Expected: TypeError on `state_channels` keyword or AssertionError on noise helpers — 红。

- [ ] **Step 3: 改写 `model/train.py`**

完整替换文件内容为：

```python
"""Training loop for Geo-FNO bundle-only mode."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm import tqdm

from temporal_utils import CHANNEL_ORDER


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


class RMSELoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


def rel_l2_loss(pred, target, eps: float = 1e-8):
    batch_size = pred.shape[0]
    diff_flat = (pred - target).reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    diff_norm = torch.linalg.vector_norm(diff_flat, ord=2, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp(min=eps)
    return (diff_norm / target_norm).mean()


def _channel_rel_l2(pred: torch.Tensor, target: torch.Tensor, channel_idx: int, eps: float = 1e-8):
    """channel_idx is the index into the K-channel prediction tensor."""
    diff = (pred[..., channel_idx] - target[..., channel_idx]).reshape(pred.size(0), -1)
    base = target[..., channel_idx].reshape(pred.size(0), -1)
    num = torch.linalg.vector_norm(diff, ord=2, dim=1)
    den = torch.linalg.vector_norm(base, ord=2, dim=1).clamp(min=eps)
    return (num / den).mean()


def evaluate_model(
    model,
    test_loader,
    device,
    coords_2d_device,
    state_channels: tuple[int, ...],
    dist_ctx: dict | None = None,
):
    """Bundle evaluation in normalized space; no autoregressive rollout."""
    num_channels = len(state_channels)
    model.eval()
    total_sse = 0.0
    total_sae = 0.0
    total_rel_l2 = 0.0
    per_channel_rel = [0.0] * num_channels
    num_samples = 0
    total_elements = 0

    x_in_base = coords_2d_device.to(device, non_blocking=True).unsqueeze(0)
    with torch.no_grad():
        for features, target_block in test_loader:
            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            batch_size = features.shape[0]
            x_in = x_in_base.expand(batch_size, -1, -1)

            pred_block = model(features, x_in)
            diff = pred_block - target_block

            total_sse += (diff ** 2).sum().item()
            total_sae += diff.abs().sum().item()

            diff_flat = diff.reshape(batch_size, -1)
            target_flat = target_block.reshape(batch_size, -1)
            diff_norm = torch.linalg.vector_norm(diff_flat, ord=2, dim=1)
            target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp(min=1e-8)
            total_rel_l2 += (diff_norm / target_norm).sum().item()

            for k in range(num_channels):
                per_channel_rel[k] += _channel_rel_l2(pred_block, target_block, k).item() * batch_size

            num_samples += batch_size
            total_elements += target_block.numel()

    totals_input = [total_sse, total_sae, total_rel_l2, *per_channel_rel, num_samples, total_elements]
    totals = reduce_sums(totals_input, device, dist_ctx)
    sse = totals[0]
    sae = totals[1]
    rel_l2 = totals[2]
    rel_per_channel = totals[3 : 3 + num_channels]
    sample_count = max(1.0, totals[3 + num_channels])
    element_count = max(1.0, totals[4 + num_channels])
    mse = sse / element_count
    out = {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": sae / element_count,
        "rel_l2": rel_l2 / sample_count,
    }
    for k, ch_global_idx in enumerate(state_channels):
        ch_name = CHANNEL_ORDER[ch_global_idx]
        out[f"rel_{ch_name}"] = rel_per_channel[k] / sample_count
    return out


def _ddp_sync_context(model, should_sync: bool, dist_ctx: dict | None):
    if should_sync or not is_distributed(dist_ctx):
        return nullcontext()
    if not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    device,
    optimizer,
    scheduler,
    coords_2d_device,
    writer,
    state_channels: tuple[int, ...],
    grad_clip=None,
    loss_type: str = "rel_l2",
    checkpoint_path: str = "best_geofno.pt",
    train_sampler=None,
    dist_ctx: dict | None = None,
    accum_steps: int = 1,
):
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    if loss_type == "rmse":
        criterion = RMSELoss()
    elif loss_type == "rel_l2":
        criterion = None
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    global_step = 0
    best_loss = float("inf")
    x_in_base = coords_2d_device.to(device, non_blocking=True).unsqueeze(0)

    for epoch in range(num_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train()
        local_loss_sum = 0.0
        local_n = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
            disable=not is_rank0(dist_ctx),
        )

        steps_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = steps_per_epoch // accum_steps
        usable_micro_batches = optimizer_steps_per_epoch * accum_steps

        optimizer.zero_grad(set_to_none=True)

        for micro_idx, (features, target_block) in enumerate(pbar):
            if micro_idx >= usable_micro_batches:
                break

            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            batch_size = features.shape[0]

            should_sync = (micro_idx + 1) % accum_steps == 0
            with _ddp_sync_context(model, should_sync, dist_ctx):
                x_in = x_in_base.expand(batch_size, -1, -1)
                pred_block = model(features, x_in)
                if loss_type == "rmse":
                    loss = criterion(pred_block, target_block)
                else:
                    loss = rel_l2_loss(pred_block, target_block)
                loss = loss / accum_steps
                loss.backward()

            loss_unscaled = loss.item() * accum_steps
            local_loss_sum += loss_unscaled * batch_size
            local_n += batch_size

            if should_sync:
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

        test_metrics = evaluate_model(
            model, test_loader, device, coords_2d_device, state_channels, dist_ctx=dist_ctx
        )
        current_lr = optimizer.param_groups[0]["lr"]

        if is_rank0(dist_ctx):
            if writer is not None:
                writer.add_scalar("val/loss_epoch", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/rel_l2", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/mse", test_metrics["mse"], epoch)
                writer.add_scalar("val/rmse", test_metrics["rmse"], epoch)
                writer.add_scalar("val/mae", test_metrics["mae"], epoch)
                for ch_global_idx in state_channels:
                    ch_name = CHANNEL_ORDER[ch_global_idx]
                    writer.add_scalar(f"val/rel_{ch_name}", test_metrics[f"rel_{ch_name}"], epoch)
                writer.add_scalar("train/lr", current_lr, epoch)
            per_ch_str = " | ".join(
                f"Test Rel-{CHANNEL_ORDER[c].upper()}: {test_metrics[f'rel_{CHANNEL_ORDER[c]}']:.6f}"
                for c in state_channels
            )
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_loss:.6f} | "
                f"Test RMSE: {test_metrics['rmse']:.6f} | "
                f"Test Rel-L2: {test_metrics['rel_l2']:.6f} | "
                f"{per_ch_str} | "
                f"LR: {current_lr:.2e}"
            )

        current_test_loss = test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                torch.save(unwrap_model(model).state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")

        barrier_if_distributed(dist_ctx)

    if is_rank0(dist_ctx):
        print("Training finished.")
```

- [ ] **Step 4: 跑测试确认绿**

Run: `pytest tests/test_train.py -v`
Expected: 全部 PASS（4 项）。

- [ ] **Step 5: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "$(cat <<'EOF'
refactor(train): remove noise augmentation, add state_channels plumbing

Drops make_uvh_noise_std_tensor, add_uvh_training_noise, and the
add_noise/uvh_noise_std branch from train_model — this project does
not use input noise. evaluate_model and train_model now take a
state_channels tuple and report per-channel rel_l2 / tensorboard
scalars only for selected channels.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `model/main.py` 整合（CONFIG + 调用链）

**目的：** CONFIG 增加 `"channels": "uvh"`、删除 noise 条目；启动时解析 channels、构造 num_channels / channels_suffix；checkpoint 名 / run tag / 数据集 / 模型 / train_model 全部串起来；rank0 print 摘要更新。

实现 spec 章节：「§ 涉及模块改动 - 5. `model/main.py`」。

**Files:**
- Modify: `model/main.py`

- [ ] **Step 1: 修改 CONFIG 块**

把：

```python
    "add_noise": False,
    "uvh_noise_std": [0.005, 0.005, 0.001],
}
```

替换为：

```python
    "channels": "uvh",
}
```

- [ ] **Step 2: 更新 imports**

把：

```python
from temporal_utils import (
    build_checkpoint_name,
    build_run_suffix,
    input_channels_for_bundle,
    output_channels_for_bundle,
    validate_temporal_params,
)
```

替换为：

```python
from temporal_utils import (
    build_checkpoint_name,
    build_run_suffix,
    channels_suffix,
    input_channels_for_bundle,
    output_channels_for_bundle,
    parse_channels,
    validate_temporal_params,
)
```

- [ ] **Step 3: 删除 noise helper 函数**

把 `format_noise_value` 和 `build_noise_run_suffix` 两个函数（连同它们之间的空行）整段删掉。

- [ ] **Step 4: 重写 `main()` 中 CONFIG-driven 的派生量**

把：

```python
        validate_temporal_params(CONFIG["bundle_size"])
        preflight_manifest_files(CONFIG)
        set_seed(CONFIG["seed"])

        in_channels = input_channels_for_bundle(CONFIG["bundle_size"])
        out_channels = output_channels_for_bundle(CONFIG["bundle_size"])
        noise_suffix = build_noise_run_suffix(CONFIG["add_noise"], CONFIG["uvh_noise_std"])
        checkpoint_name = build_checkpoint_name(CONFIG["bundle_size"], noise_suffix)
        run_tag = (
            "GeoFNO"
            + build_run_suffix(CONFIG["bundle_size"], noise_suffix)
            + "_"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
```

替换为：

```python
        validate_temporal_params(CONFIG["bundle_size"])
        preflight_manifest_files(CONFIG)
        set_seed(CONFIG["seed"])

        state_channels = parse_channels(CONFIG["channels"])
        num_channels = len(state_channels)
        ch_suffix = channels_suffix(state_channels)

        in_channels = input_channels_for_bundle(CONFIG["bundle_size"], num_channels)
        out_channels = output_channels_for_bundle(CONFIG["bundle_size"], num_channels)
        checkpoint_name = build_checkpoint_name(CONFIG["bundle_size"], ch_suffix)
        run_tag = (
            "GeoFNO"
            + build_run_suffix(CONFIG["bundle_size"], ch_suffix)
            + "_"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
```

- [ ] **Step 5: 改 rank0 启动摘要打印**

把：

```python
        rank0_print(
            dist_ctx,
            f"[main] bundle_size={CONFIG['bundle_size']}, "
            f"in_channels={in_channels}, out_channels={out_channels}",
        )
        rank0_print(
            dist_ctx,
            f"[main] noise: add_noise={CONFIG['add_noise']}, "
            f"uvh_noise_std={CONFIG['uvh_noise_std']}",
        )
        rank0_print(dist_ctx, f"[main] checkpoint name: {checkpoint_name}")
```

替换为：

```python
        rank0_print(
            dist_ctx,
            f"[main] bundle_size={CONFIG['bundle_size']}, "
            f"in_channels={in_channels}, out_channels={out_channels}",
        )
        rank0_print(
            dist_ctx,
            f"[main] channels={CONFIG['channels']!r} -> "
            f"state_channels={state_channels}, num_channels={num_channels}",
        )
        rank0_print(dist_ctx, f"[main] checkpoint name: {checkpoint_name}")
```

- [ ] **Step 6: 把 `state_channels` 传给数据集构造**

把：

```python
        train_dataset = MultiStormSurgeDataset(
            data_dir=CONFIG["train_dir"],
            bundle_size=CONFIG["bundle_size"],
            btype_oh=btype_oh_cpu,
            lru_files_per_worker=CONFIG["lru_files_per_worker"],
        )
```

替换为：

```python
        train_dataset = MultiStormSurgeDataset(
            data_dir=CONFIG["train_dir"],
            bundle_size=CONFIG["bundle_size"],
            btype_oh=btype_oh_cpu,
            lru_files_per_worker=CONFIG["lru_files_per_worker"],
            state_channels=state_channels,
        )
```

把 `val_dataset = MultiStormSurgeDataset(...)` 块也加入 `state_channels=state_channels,`。

- [ ] **Step 7: 把 `num_channels` 传给模型构造**

把 `model = GeoFNO2d(...)` 块在 `num_fno_layers=CONFIG["num_fno_layers"],` 后面加一行：

```python
            num_channels=num_channels,
```

- [ ] **Step 8: 把 `state_channels` 传给 train_model，删除 noise 参数**

把：

```python
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
```

替换为：

```python
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
            state_channels=state_channels,
            grad_clip=CONFIG["grad_clip"],
            loss_type=CONFIG["loss_type"],
            checkpoint_path=checkpoint_name,
            train_sampler=train_sampler,
            dist_ctx=dist_ctx,
            accum_steps=CONFIG["accum_steps"],
        )
```

- [ ] **Step 9: 本地静态检查**

Run: `python -c "import sys; sys.path.insert(0, 'model'); import main"`
Expected: 无 ImportError、无 SyntaxError，进程退出码 0。

- [ ] **Step 10: Commit**

```bash
git add model/main.py
git commit -m "$(cat <<'EOF'
feat(main): wire channels CONFIG end-to-end and drop noise plumbing

CONFIG['channels'] (default 'uvh') is parsed into state_channels and
num_channels which flow into dataset, model, checkpoint name, run tag,
and train_model. CONFIG no longer carries add_noise / uvh_noise_std.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `model/test_all.py` 核心数据流（接口 + 自回归 + dry mask）

**目的：** 加 `--channels` 参数；`load_normalization_stats` 同时返回裁剪与完整 mean/std；`autoregressive_one_file` 沿 K 通道前进，dry mask 始终用真实 h；`apply_dry_grid_error_mask` 接收完整 mean/std。

实现 spec 章节：「§ 涉及模块改动 - 6. `model/test_all.py`」（接口/数据流子集，不动 init_bucket/compute_stats/输出格式 — 这些放 Task 7）。

**Files:**
- Modify: `model/test_all.py`

- [ ] **Step 1: 更新 imports，引入 channel 工具**

把：

```python
from temporal_utils import (
    build_checkpoint_name,
    input_channels_for_bundle,
    output_channels_for_bundle,
    validate_temporal_params,
)
```

替换为：

```python
from temporal_utils import (
    CHANNEL_ORDER,
    build_checkpoint_name,
    channels_suffix,
    input_channels_for_bundle,
    output_channels_for_bundle,
    parse_channels,
    validate_temporal_params,
)
```

- [ ] **Step 2: `parse_args` 加 `--channels`**

在 `parser.add_argument("--bundle_size", ...)` 之后插入一行：

```python
    parser.add_argument("--channels", type=str, default="uvh", help="State channels to predict: any non-empty subset of uvh.")
```

- [ ] **Step 3: `load_normalization_stats` 返回 sub + full**

把函数签名和返回处改为：

```python
def load_normalization_stats(stats_path, device, state_channels: tuple[int, ...] = (0, 1, 2)):
    print(f"[test] loading normalization stats from {stats_path}")
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {path}")

    stats = scipy.io.loadmat(path)
    if all(key in stats for key in GRAPH_STATS_KEYS):
        mean_values = np.asarray(stats["graph_mean"], dtype=np.float32).reshape(-1)
        std_values = np.asarray(stats["graph_std"], dtype=np.float32).reshape(-1)
        if mean_values.size < 3 or std_values.size < 3:
            raise ValueError(
                f"{path} graph_mean/graph_std must contain at least 3 channels; "
                f"got graph_mean={mean_values.size}, graph_std={std_values.size}"
            )
        mean_values = mean_values[:3]
        std_values = std_values[:3]
    elif all(key in stats for key in LEGACY_STATS_KEYS):
        mean_values = np.asarray(
            [stats["u_mean"].item(), stats["v_mean"].item(), stats["h_mean"].item()],
            dtype=np.float32,
        )
        std_values = np.asarray(
            [stats["u_std"].item(), stats["v_std"].item(), stats["h_std"].item()],
            dtype=np.float32,
        )
    else:
        raise KeyError(
            f"{path} missing normalization stats. Acceptable key sets: "
            f"{GRAPH_STATS_KEYS} or {LEGACY_STATS_KEYS}. "
            f"Available keys: {sorted(key for key in stats if not key.startswith('__'))}"
        )

    mean_full = torch.tensor(mean_values, device=device, dtype=torch.float32).view(1, 1, 3)
    std_full = torch.tensor(std_values, device=device, dtype=torch.float32).view(1, 1, 3)
    sub_idx = list(state_channels)
    mean_sub = mean_full[..., sub_idx]
    std_sub = std_full[..., sub_idx]
    return mean_sub, std_sub, mean_full, std_full
```

- [ ] **Step 4: 重写 `apply_dry_grid_error_mask`**

替换为：

```python
def apply_dry_grid_error_mask(diff, target_full_norm, mean_full, std_full):
    """Zero out diff at dry-grid nodes (physical water level < threshold).

    The mask is always built from the *true* h column in target_full_norm,
    independent of which channels the model predicts. diff has last-dim K
    (selected channels); the mask broadcasts over K naturally.
    """
    target_wl = denormalize(
        target_full_norm[..., WATER_LEVEL_CHANNEL],
        mean_full[..., WATER_LEVEL_CHANNEL],
        std_full[..., WATER_LEVEL_CHANNEL],
    )
    dry_mask = target_wl < DRY_WATER_LEVEL_THRESHOLD
    return diff.masked_fill(dry_mask.unsqueeze(-1), 0.0)
```

- [ ] **Step 5: 修改 `autoregressive_one_file` 签名和循环**

把签名扩展为：

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
    group_len,
    bundle_size,
    batch_size,
    state_channels: tuple[int, ...],
    per_step_metrics_by_space,
):
```

函数体内：

把 `current_state = graph_all[batch_starts].to(device)` 改为：

```python
current_state = graph_all[batch_starts][..., list(state_channels)].to(device)
```

把每个 bundle_step 内的 metric 计算块（从 `target_indices = batch_starts + step + 1` 到 `bucket["count"] += current_batch_size`）整段替换为：

```python
                    target_indices = batch_starts + step + 1
                    pred_norm = pred_block[:, bundle_step]
                    target_full_norm = graph_all[target_indices].to(device)
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
                        bucket["count"] += current_batch_size
```

`current_state = pred_block[:, -1]` 这行不动（pred_block 已是 K 通道）。

- [ ] **Step 6: 留 `init_bucket / compute_stats / write_results` 暂不动**

它们仍然假设 dim=3，会在 Task 7 处理。本 task 末尾代码尚不能运行 — 这是预期的（Task 7 完成后才能 import-run）。

- [ ] **Step 7: 静态语法检查**

Run: `python -c "import ast; ast.parse(open('model/test_all.py').read())"`
Expected: 无输出（语法解析通过）。

- [ ] **Step 8: Commit**

```bash
git add model/test_all.py
git commit -m "$(cat <<'EOF'
feat(test_all): add --channels and route subset through autoregressive loop

load_normalization_stats now returns (mean_sub, std_sub, mean_full,
std_full); autoregressive_one_file walks forward in the K-channel
subset; dry-grid mask is always built from the real h column via
mean_full/std_full regardless of which channels the model predicts.

NOTE: init_bucket / compute_stats / write_results still assume dim=3
and will be updated in the next commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `model/test_all.py` 指标聚合 + 输出全 K 通道化

**目的：** `init_bucket(device, num_channels)`；`compute_stats` / `compute_auc` / `write_results` / 终端 summary / `metric_output_path` 全部按 `state_channels` 维度循环；`main()` 把 channels 全链路串起来。

实现 spec 章节：「§ 涉及模块改动 - 6. `model/test_all.py`」（剩余部分）。

**Files:**
- Modify: `model/test_all.py`
- Create: `tests/test_test_all_helpers.py`

- [ ] **Step 1: 改 `init_bucket` 接受 `num_channels`**

把：

```python
def init_bucket(device):
    return {
        "sse": torch.zeros(3, device=device),
        "sae": torch.zeros(3, device=device),
        "sum_gt": torch.zeros(3, device=device),
        "sum_sq_gt": torch.zeros(3, device=device),
        "rel_l2_sum": torch.zeros(3, device=device),
        "count": 0,
    }
```

替换为：

```python
def init_bucket(device, num_channels: int):
    return {
        "sse": torch.zeros(num_channels, device=device),
        "sae": torch.zeros(num_channels, device=device),
        "sum_gt": torch.zeros(num_channels, device=device),
        "sum_sq_gt": torch.zeros(num_channels, device=device),
        "rel_l2_sum": torch.zeros(num_channels, device=device),
        "count": 0,
    }
```

- [ ] **Step 2: 改 `compute_stats` 用动态 K**

把：

```python
def compute_stats(bucket, num_nodes):
    count = bucket["count"]
    if count == 0:
        zeros = np.zeros(3, dtype=np.float64)
        return {
            "mse_channels": zeros,
            "rmse_channels": zeros,
            "mae_channels": zeros,
            "r2_channels": zeros,
            "rel_l2_channels": zeros,
        }
    ...
```

替换为：

```python
def compute_stats(bucket, num_nodes, num_channels: int):
    count = bucket["count"]
    if count == 0:
        zeros = np.zeros(num_channels, dtype=np.float64)
        return {
            "mse_channels": zeros,
            "rmse_channels": zeros,
            "mae_channels": zeros,
            "r2_channels": zeros,
            "rel_l2_channels": zeros,
        }

    num_values = count * num_nodes
    sse = bucket["sse"]
    sae = bucket["sae"]
    mse_channels = sse / num_values
    rmse_channels = np.sqrt(mse_channels)
    mae_channels = sae / num_values
    ss_tot = bucket["sum_sq_gt"] - (bucket["sum_gt"] ** 2) / num_values
    ss_tot = np.maximum(ss_tot, 1e-8)
    r2_channels = 1.0 - (sse / ss_tot)
    rel_l2_channels = bucket["rel_l2_sum"] / count
    return {
        "mse_channels": mse_channels,
        "rmse_channels": rmse_channels,
        "mae_channels": mae_channels,
        "r2_channels": r2_channels,
        "rel_l2_channels": rel_l2_channels,
    }
```

- [ ] **Step 3: 改 `compute_auc` 按 `selected_channel_names` 走**

替换函数体为：

```python
def compute_auc(results, selected_channel_names):
    auc = {channel: {} for channel in selected_channel_names}
    steps = [entry["step"] for entry in results]
    if len(steps) < 2:
        for channel in selected_channel_names:
            for metric in ("mse", "rmse", "mae", "r2", "rel_l2"):
                auc[channel][metric] = 0.0
        return auc

    for channel in selected_channel_names:
        metrics = [key for key in results[0][channel].keys() if key != "step"]
        for metric in metrics:
            values = [entry[channel][metric] for entry in results]
            auc[channel][metric] = float(np.trapz(values, steps))
    return auc
```

- [ ] **Step 4: 改 `metric_output_path` 接受 `channels_suffix`**

替换为：

```python
def metric_output_path(base_path, metric_space, channels_suffix: str = ""):
    path = Path(base_path)
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}{channels_suffix}_{metric_space}{suffix}")
```

- [ ] **Step 5: 改 `write_results` 接收 `selected_channel_names` 与 `channels_suffix`**

把函数签名改为：

```python
def write_results(
    results_by_space,
    output_path,
    group_len,
    bundle_size,
    model_path,
    evaluated_files,
    evaluated_groups,
    skipped_files,
    selected_channel_names,
    channels_suffix: str = "",
):
```

把函数体里所有 `for metric_space in METRIC_SPACES:` 那段循环改为（替换整段 `for metric_space in METRIC_SPACES: ... print(f"[test] results -> {out}")`）：

```python
    for metric_space in METRIC_SPACES:
        out = metric_output_path(output_path, metric_space, channels_suffix)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(f"Autoregressive Test Results (group_len={group_len}, bundle_size={bundle_size})\n")
            f.write(f"Metric Space: {metric_space}\n")
            f.write(f"Channels: {''.join(selected_channel_names)}\n")
            f.write(f"Checkpoint: {model_path if model_path is not None else 'random weights'}\n")
            f.write(f"Evaluated files: {evaluated_files}\n")
            f.write(f"Evaluated groups: {evaluated_groups}\n")
            f.write(f"Skipped files: {skipped_files}\n")
            f.write("=" * 100 + "\n")
            f.write(
                f"{'Step':<6} | {'Channel':<7} | {'MSE':<12} | {'RMSE':<12} | "
                f"{'MAE':<12} | {'R2':<12} | {'Rel L2':<12}\n"
            )
            f.write("-" * 100 + "\n")

            for result in results_by_space[metric_space]:
                step = result["step"]
                for idx, channel in enumerate(selected_channel_names):
                    display_name = "wl" if channel == "h" else channel
                    step_label = str(step) if idx == 0 else ""
                    metrics = result[channel]
                    f.write(
                        f"{step_label:<6} | {display_name:<7} | {metrics['mse']:<12.6f} | "
                        f"{metrics['rmse']:<12.6f} | {metrics['mae']:<12.6f} | "
                        f"{metrics['r2']:<12.6f} | {metrics['rel_l2']:<12.6f}\n"
                    )
                f.write("-" * 100 + "\n")

            auc = compute_auc(results_by_space[metric_space], selected_channel_names)
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"AUC Summary Over {group_len} Steps\n")
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

- [ ] **Step 6: 重写 `main()` 把 channels 串起来**

把 `main()` 整个函数体替换为：

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
    if args.group_len % args.bundle_size != 0:
        raise ValueError(
            f"group_len must be divisible by bundle_size: group_len={args.group_len}, "
            f"bundle_size={args.bundle_size}"
        )

    state_channels = parse_channels(args.channels)
    num_channels = len(state_channels)
    ch_suffix = channels_suffix(state_channels)
    selected_channel_names = tuple(CHANNEL_ORDER[c] for c in state_channels)
    print(f"[test] channels={args.channels!r} -> state_channels={state_channels}, num_channels={num_channels}")

    coords_2d_cpu, btype_oh_cpu = load_static_coords(args.coords)
    coords_2d_device = coords_2d_cpu.to(device)
    btype_oh_device = btype_oh_cpu.to(device)
    num_nodes = coords_2d_cpu.size(0)

    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(
        args.norm, device=device, state_channels=state_channels
    )

    in_channels = input_channels_for_bundle(args.bundle_size, num_channels)
    out_channels = output_channels_for_bundle(args.bundle_size, num_channels)
    model_args = {
        "bundle_size": args.bundle_size,
        "channels": args.channels,
        "in_channels": in_channels,
        "out_channels": out_channels,
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
    per_step_metrics_by_space = {
        metric_space: [init_bucket(device, num_channels) for _ in range(args.group_len)]
        for metric_space in METRIC_SPACES
    }

    test_files = find_test_files(args.test_dir)
    if args.num_files is not None:
        test_files = test_files[: args.num_files]
    if not test_files:
        raise FileNotFoundError(f"No .pt files in {args.test_dir}")

    evaluated_files = 0
    evaluated_groups = 0
    skipped_files = 0
    for file_path in tqdm(test_files, desc="Test files"):
        file_result = autoregressive_one_file(
            model,
            file_path,
            coords_2d_device,
            btype_oh_device,
            mean_sub,
            std_sub,
            mean_full,
            std_full,
            device,
            args.group_len,
            args.bundle_size,
            args.batch_size,
            state_channels,
            per_step_metrics_by_space,
        )
        if file_result["skipped"]:
            skipped_files += 1
        else:
            evaluated_files += 1
            evaluated_groups += file_result["evaluated_groups"]

    if evaluated_groups == 0:
        raise RuntimeError(
            "No evaluation groups were produced. "
            f"evaluated_files={evaluated_files}, evaluated_groups={evaluated_groups}, "
            f"skipped_files={skipped_files}, group_len={args.group_len}, "
            f"bundle_size={args.bundle_size}"
        )

    for metric_space in METRIC_SPACES:
        for step in range(args.group_len):
            bucket = per_step_metrics_by_space[metric_space][step]
            for key in ("sse", "sae", "sum_gt", "sum_sq_gt", "rel_l2_sum"):
                bucket[key] = bucket[key].detach().cpu().numpy()

    results_by_space = {metric_space: [] for metric_space in METRIC_SPACES}
    for metric_space in METRIC_SPACES:
        for step, bucket in enumerate(per_step_metrics_by_space[metric_space]):
            stats = compute_stats(bucket, num_nodes, num_channels)
            result = {"step": step + 1}
            for idx, channel in enumerate(selected_channel_names):
                result[channel] = {
                    "mse": float(stats["mse_channels"][idx]),
                    "rmse": float(stats["rmse_channels"][idx]),
                    "mae": float(stats["mae_channels"][idx]),
                    "r2": float(stats["r2_channels"][idx]),
                    "rel_l2": float(stats["rel_l2_channels"][idx]),
                }
            results_by_space[metric_space].append(result)

            summary = f"[step {step + 1:02d}][{metric_space}] "
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
        args.group_len,
        args.bundle_size,
        model_path,
        evaluated_files,
        evaluated_groups,
        skipped_files,
        selected_channel_names,
        ch_suffix,
    )
    print("[test] done.")
```

也把全局的 `CHANNEL_NAMES = ("u", "v", "h")` 那一行删掉（现在用 `CHANNEL_ORDER` from temporal_utils，且通过 `selected_channel_names` 局部变量传递）。

- [ ] **Step 7: 新建 `tests/test_test_all_helpers.py`**

```python
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from test_all import (  # noqa: E402
    apply_dry_grid_error_mask,
    compute_stats,
    init_bucket,
    load_normalization_stats,
    metric_output_path,
)


def _write_norm_mat(path, mean=(1.0, 2.0, 3.0), std=(0.1, 0.2, 0.3)):
    scipy.io.savemat(path, {"graph_mean": np.array(mean), "graph_std": np.array(std)})


def test_load_normalization_stats_default_full(tmp_path):
    p = tmp_path / "norm.mat"
    _write_norm_mat(p)
    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(p, torch.device("cpu"))
    assert mean_full.shape == (1, 1, 3)
    assert std_full.shape == (1, 1, 3)
    assert mean_sub.shape == (1, 1, 3)
    assert std_sub.shape == (1, 1, 3)
    assert torch.allclose(mean_full, torch.tensor([[[1.0, 2.0, 3.0]]]))


def test_load_normalization_stats_subset_h(tmp_path):
    p = tmp_path / "norm.mat"
    _write_norm_mat(p)
    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(
        p, torch.device("cpu"), state_channels=(2,)
    )
    assert mean_sub.shape == (1, 1, 1)
    assert torch.allclose(mean_sub, torch.tensor([[[3.0]]]))
    assert torch.allclose(std_sub, torch.tensor([[[0.3]]]))
    assert mean_full.shape == (1, 1, 3)


def test_init_bucket_sizes_match_num_channels():
    b1 = init_bucket(torch.device("cpu"), 1)
    assert b1["sse"].shape == (1,)
    b3 = init_bucket(torch.device("cpu"), 3)
    assert b3["sse"].shape == (3,)


def test_compute_stats_zero_count_returns_zeros_of_right_size():
    bucket = init_bucket(torch.device("cpu"), 2)
    for key in ("sse", "sae", "sum_gt", "sum_sq_gt", "rel_l2_sum"):
        bucket[key] = bucket[key].numpy()
    stats = compute_stats(bucket, num_nodes=10, num_channels=2)
    assert stats["mse_channels"].shape == (2,)
    assert (stats["mse_channels"] == 0).all()


def test_metric_output_path_includes_channels_suffix(tmp_path):
    base = tmp_path / "results.txt"
    p = metric_output_path(base, "physical", "_chh")
    assert p.name == "results_chh_physical.txt"
    p_full = metric_output_path(base, "normalized", "")
    assert p_full.name == "results_normalized.txt"


def test_apply_dry_grid_error_mask_uses_full_h_regardless_of_diff_dim():
    """Even when diff is (B, N, 1) for h-only model, the mask is built from the
    real h column in target_full_norm."""
    B, N = 1, 4
    # target_full_norm: h column has 0.0 at node 0 (dry), 1.0 elsewhere (wet)
    target_full = torch.zeros(B, N, 3)
    target_full[..., 2] = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    mean_full = torch.tensor([[[0.0, 0.0, 0.0]]])
    std_full = torch.tensor([[[1.0, 1.0, 1.0]]])
    diff_h_only = torch.ones(B, N, 1)  # K=1
    masked = apply_dry_grid_error_mask(diff_h_only, target_full, mean_full, std_full)
    assert masked[0, 0, 0].item() == 0.0
    assert masked[0, 1, 0].item() == 1.0
```

- [ ] **Step 8: 跑全部测试**

Run: `pytest tests/ -v`
Expected: 全 PASS。

- [ ] **Step 9: Commit**

```bash
git add model/test_all.py tests/test_test_all_helpers.py
git commit -m "$(cat <<'EOF'
feat(test_all): K-channel aggregation and channels-aware outputs

init_bucket/compute_stats/compute_auc/write_results all walk over the
selected channel subset. metric_output_path embeds channels_suffix so
results from different runs no longer collide. main() parses --channels
and threads it through every helper.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 端到端集成验证

**目的：** 跑全套 pytest；并给出服务器侧验证 `channels="uvh"` 数值等价、`channels="h"` 能跑通的两条命令。

实现 spec 章节：「§ 测试与验证」。

**Files:**
- (no code changes; verification only)

- [ ] **Step 1: 本地跑全部 pytest**

Run: `pytest tests/ -v`
Expected: 全部 PASS（约 36 项左右）。

- [ ] **Step 2: 静态 import 检查**

Run:
```bash
python -c "import sys; sys.path.insert(0, 'model'); import main, test_all, train, model as model_mod, dataset, temporal_utils; print('imports ok')"
```
Expected: 打印 `imports ok`，无 ImportError。

- [ ] **Step 3: 写服务器侧验证脚本到 plan 末尾备忘（不入仓）**

在本地终端输出（不写文件）以下命令，供用户在服务器上执行：

回归对照（必须数值等价）：
```bash
# 旧 checkpoint 路径名未变；CONFIG["channels"]="uvh" 是默认值
python model/main.py            # smoke
torchrun --nproc_per_node=4 model/main.py
python model/test_all.py --channels uvh --num_files 1 --allow_random_weights
```

子集 smoke：
```bash
# 改 CONFIG["channels"]="h" 后训练，checkpoint 会落到 best_geofno_b8_chh.pt
python model/main.py
python model/test_all.py --channels h --num_files 1 --allow_random_weights
```

- [ ] **Step 4: 给用户发回验证清单**

把以下 3 条粘到回复，等用户在服务器跑完反馈：

1. `pytest tests/ -v` 全绿
2. `channels="uvh"` smoke 训练 1 epoch 后 val Rel-L2 与改动前数值等价（在同 seed 下）
3. `channels="h"` smoke 训练能跑通；checkpoint 文件名带 `_chh`；`test_all.py --channels h` 自回归输出文件名带 `_chh`，dry mask 仍生效（用真实 h）

- [ ] **Step 5: 最终 commit（仅 plan 内 markdown 改动，如有）**

如果本 task 没改任何文件就跳过 commit；如果只是 plan 文档的勾选状态改动，可一并 commit：

```bash
git status --short
# 若仅 plan 文件有勾选变化:
git add docs/superpowers/plans/2026-05-18-channel-subset-training.md
git commit -m "$(cat <<'EOF'
chore(plan): mark channel-subset training plan complete

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## 验收清单（spec 覆盖回查）

| Spec 章节 | 实现 Task |
|---|---|
| § 通道规范（解析、命名、num_channels 范围） | Task 1 |
| § 1. temporal_utils.py | Task 1 |
| § 2. dataset.py | Task 2 |
| § 3. model.py | Task 3 |
| § 4. train.py（删 noise + state_channels 化） | Task 4 |
| § 5. main.py（CONFIG + 调用链） | Task 5 |
| § 6. test_all.py（接口 + 自回归 + dry mask） | Task 6 |
| § 6. test_all.py（指标聚合 + 输出格式） | Task 7 |
| § 输入/输出形状汇总（表） | Task 1（公式）+ Task 2/3（运行时） |
| § 命名约定（表） | Task 1（命名函数）+ Task 5（main 调用）+ Task 7（test 输出文件名） |
| § 测试与验证（3 种 smoke） | Task 8 |
| § 风险与缓解 | 文档性，无独立 task |
