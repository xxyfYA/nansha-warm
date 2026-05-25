# 移除多通道支持，硬编码 h-only 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把所有 `state_channels` / `num_channels` / `parse_channels` / `channels_suffix` 多通道抽象从代码库中删除，h-only 硬编码到 dataset / model / train / test_all。

**Architecture:** 自底向上重写 [model/temporal_utils.py](../../../model/temporal_utils.py)、[model/dataset.py](../../../model/dataset.py)、[model/model.py](../../../model/model.py)、[model/train.py](../../../model/train.py)、[model/main.py](../../../model/main.py)、[model/test_all.py](../../../model/test_all.py)；同步删除 / 更新 `tests/` 中的相关测试；每步保持 import 与 pytest 绿色。

**Tech Stack:** PyTorch、pytest、numpy、scipy、TensorBoard。

**关联 Spec:** [docs/superpowers/specs/2026-05-24-remove-multi-channel-support-design.md](../specs/2026-05-24-remove-multi-channel-support-design.md)

---

### Task 1: 重写 `temporal_utils.py` + 同步测试

**Files:**
- Modify: [model/temporal_utils.py](../../../model/temporal_utils.py)
- Modify: [tests/test_temporal_utils.py](../../../tests/test_temporal_utils.py)

- [ ] **Step 1: 重写 `model/temporal_utils.py` 为下列完整内容**

```python
"""Bundle-only temporal helpers for Geo-FNO storm-surge model (h-only)."""
from dataclasses import dataclass


CHANNEL_NAME = "h"


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
    validate_temporal_params(bundle_size)
    num_samples = num_time - bundle_size
    if num_samples <= 0:
        raise ValueError(
            f"not enough time steps: num_time={num_time}, "
            f"bundle_size={bundle_size}, required_future_steps={bundle_size}"
        )
    return num_samples


def input_channels_for_bundle(bundle_size: int) -> int:
    """C_in = 1 + 5*S + 8 = 5*S + 9 (h-only state + storm + inner + btype)."""
    validate_temporal_params(bundle_size)
    return 5 * bundle_size + 9


def output_channels_for_bundle(bundle_size: int) -> int:
    """C_out = S (single-channel residual)."""
    validate_temporal_params(bundle_size)
    return bundle_size


def build_checkpoint_name(bundle_size: int) -> str:
    validate_temporal_params(bundle_size)
    if bundle_size == 1:
        return "best_geofno.pt"
    return f"best_geofno_b{bundle_size}.pt"


def build_run_suffix(bundle_size: int) -> str:
    validate_temporal_params(bundle_size)
    if bundle_size == 1:
        return ""
    return f"_b{bundle_size}"
```

- [ ] **Step 2: 重写 `tests/test_temporal_utils.py` 为下列完整内容**

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from temporal_utils import (  # noqa: E402
    CHANNEL_NAME,
    TemporalConfig,
    build_checkpoint_name,
    build_run_suffix,
    input_channels_for_bundle,
    num_temporal_samples,
    output_channels_for_bundle,
    validate_temporal_params,
)


def test_input_channels_formula():
    # C_in = 5*S + 9
    assert input_channels_for_bundle(1) == 14
    assert input_channels_for_bundle(8) == 49
    assert input_channels_for_bundle(24) == 129
    assert input_channels_for_bundle(72) == 369


def test_output_channels_formula():
    assert output_channels_for_bundle(1) == 1
    assert output_channels_for_bundle(8) == 8
    assert output_channels_for_bundle(24) == 24
    assert output_channels_for_bundle(72) == 72


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


def test_channel_name_constant():
    assert CHANNEL_NAME == "h"


def test_temporal_config_basic():
    cfg = TemporalConfig(bundle_size=8)
    assert cfg.bundle_size == 8
    assert cfg.required_future_steps == 8
    assert cfg.input_channels == 49
    assert cfg.out_channels == 8


def test_temporal_config_default():
    cfg = TemporalConfig()
    assert cfg.bundle_size == 1
    assert cfg.input_channels == 14
    assert cfg.out_channels == 1


def test_temporal_config_rejects_invalid_bundle():
    with pytest.raises(ValueError):
        TemporalConfig(bundle_size=0)


def test_build_checkpoint_name():
    assert build_checkpoint_name(1) == "best_geofno.pt"
    assert build_checkpoint_name(8) == "best_geofno_b8.pt"
    assert build_checkpoint_name(72) == "best_geofno_b72.pt"


def test_build_run_suffix():
    assert build_run_suffix(1) == ""
    assert build_run_suffix(8) == "_b8"
    assert build_run_suffix(72) == "_b72"
```

- [ ] **Step 3: 跑 temporal_utils 测试**

```bash
pytest tests/test_temporal_utils.py -v
```

Expected: 所有测试 PASS（无 import 失败）。

- [ ] **Step 4: 提交**

```bash
git add model/temporal_utils.py tests/test_temporal_utils.py
git commit -m "refactor: drop multi-channel abstractions from temporal_utils"
```

---

### Task 2: 重写 `dataset.py` + 同步测试

**Files:**
- Modify: [model/dataset.py](../../../model/dataset.py)
- Modify: [tests/test_dataset.py](../../../tests/test_dataset.py)

- [ ] **Step 1: 修改 `model/dataset.py`** — 应用以下定向编辑

**1a. 文件头 docstring 替换为：**
```python
"""Storm-surge lazy-loading dataset for Geo-FNO training (h-only).

Layout per node, in input feature vector (S = bundle_size):

    h state (1 channel)                # current water-level value
    [P, Wx, Wy] @ t, t+1, ..., t+S     # 3*(S+1) storm window
    [h_bdy, q_bdy] @ t, t+1, ..., t+S  # 2*(S+1) inner window
    [type_none, type_wl, type_flux]    # 3 boundary type one-hot

Total C_in = 1 + 5*(S+1) + 3 = 5*S + 9.
"""
```

**1b. 删除整个 `_validate_state_channels` 函数（dataset.py 中第 130-142 行附近的版本）。**

**1c. 用以下替换 `_build_features`：**
```python
def _build_features(
    state_t: torch.Tensor,
    storm_window: torch.Tensor,
    inner_window: torch.Tensor,
    btype_oh: torch.Tensor,
) -> torch.Tensor:
    """Build one per-node feature matrix from temporal windows (h-only)."""
    num_nodes = state_t.size(0)
    state_sub = state_t[..., 2:3]
    storm_flat = storm_window.permute(1, 0, 2).reshape(num_nodes, -1)
    inner_flat = inner_window.permute(1, 0, 2).reshape(num_nodes, -1)
    return torch.cat([state_sub, storm_flat, inner_flat, btype_oh], dim=-1).contiguous()
```

**1d. `StormSurgeDataset.__init__` 移除 `state_channels` 参数与相关赋值/校验：**
```python
class StormSurgeDataset(Dataset):
    """Single-file storm-surge dataset with an LRU-backed event cache."""

    def __init__(
        self,
        path,
        bundle_size,
        btype_oh,
        lru_capacity: int = 1,
    ):
        from temporal_utils import num_temporal_samples, validate_temporal_params

        validate_temporal_params(bundle_size)
        if lru_capacity < 1:
            raise ValueError(f"lru_capacity must be >= 1, got {lru_capacity}")

        self.path = Path(path)
        self.bundle_size = int(bundle_size)
        self.btype_oh = btype_oh.float()
        self.lru_capacity = int(lru_capacity)
        self._cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()

        entry = self._get_entry()
        self.T = entry["graph"].size(0)
        self.N = entry["graph"].size(1)
        if self.btype_oh.size(0) != self.N:
            raise ValueError(f"{self.path}: btype_oh N={self.btype_oh.size(0)} != file N={self.N}")
        self._num_samples = num_temporal_samples(self.T, self.bundle_size)
```

**1e. `StormSurgeDataset.__getitem__` 中 target 切片硬编码：**
```python
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(idx)

        entry = self._get_entry()
        bundle_size = self.bundle_size
        graph = entry["graph"]
        state_t = graph[idx]
        storm_window = entry["storm"][idx : idx + bundle_size + 1]
        inner_window = entry["inner"][idx : idx + bundle_size + 1]
        target = graph[idx + 1 : idx + bundle_size + 1, :, 2:3].contiguous()
        features = _build_features(state_t, storm_window, inner_window, self.btype_oh)
        return features, target
```

**1f. `MultiStormSurgeDataset.__init__` 移除 `state_channels`：**
```python
class MultiStormSurgeDataset(Dataset):
    """Lazy aggregation over all usable files in one split directory."""

    def __init__(
        self,
        data_dir,
        bundle_size,
        btype_oh,
        lru_files_per_worker: int = 2,
    ):
        from temporal_utils import validate_temporal_params

        validate_temporal_params(bundle_size)
        if lru_files_per_worker < 1:
            raise ValueError(f"lru_files_per_worker must be >= 1, got {lru_files_per_worker}")

        self.data_dir = Path(data_dir)
        self.bundle_size = int(bundle_size)
        self.btype_oh = btype_oh.float()
        self.lru_files_per_worker = int(lru_files_per_worker)

        manifest_path = self.data_dir / "manifest.json"
        # ... rest of body unchanged from current code ...
```
（**保留** `__init__` 中从 `manifest_path = ...` 开始到结束的全部逻辑：manifest 读取、文件过滤、`flat_index` 构建、`_cache` 初始化。仅删除 `state_channels` 参数行与对应的 `self.state_channels = _validate_state_channels(...)` 行。）

**1g. `MultiStormSurgeDataset.__getitem__` 中 target 切片同样硬编码：**
```python
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.flat_index):
            raise IndexError(idx)

        file_idx, t = self.flat_index[idx]
        entry = self._get_entry(file_idx)
        bundle_size = self.bundle_size
        graph = entry["graph"]
        state_t = graph[t]
        storm_window = entry["storm"][t : t + bundle_size + 1]
        inner_window = entry["inner"][t : t + bundle_size + 1]
        target = graph[t + 1 : t + bundle_size + 1, :, 2:3].contiguous()
        features = _build_features(state_t, storm_window, inner_window, self.btype_oh)
        return features, target
```

- [ ] **Step 2: 修改 `tests/test_dataset.py`** — 应用以下编辑

**2a. 修改 `test_single_dataset_shapes`：**
```python
def test_single_dataset_shapes(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    ds = StormSurgeDataset(
        path=split_dir / "e0.pt",
        bundle_size=4,
        btype_oh=btype,
        lru_capacity=1,
    )
    assert len(ds) == 80 - 4
    feat, target = ds[0]
    # C_in = 5*S + 9 = 29
    assert feat.shape == (N_NODES, 29)
    assert target.shape == (4, N_NODES, 1)
```

**2b. 修改 `test_feature_layout_is_state_storm_inner_btype_order`：**
```python
def test_feature_layout_is_state_storm_inner_btype_order(tmp_path):
    d = tmp_path / "layout"
    d.mkdir()
    path = d / "event.pt"
    _make_deterministic_pt(path, T=5, num_nodes=3)
    btype = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    ds = StormSurgeDataset(path=path, bundle_size=2, btype_oh=btype, lru_capacity=1)

    features, target = ds[1]
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected_node0 = torch.cat(
        [
            data["graph"][1, 0, 2:3],
            data["storm_boundary"][1:4, 0].reshape(-1),
            data["inner_boundary"][1:4, 0].reshape(-1),
            btype[0],
        ]
    )

    assert torch.equal(features[0], expected_node0)
    assert torch.equal(target, data["graph"][2:4, :, 2:3])
```

**2c. 修改 `test_multi_dataset_index_flattening`：**
```python
def test_multi_dataset_index_flattening(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(
        data_dir=split_dir,
        bundle_size=4,
        btype_oh=btype,
        lru_files_per_worker=1,
    )
    assert len(mds) == (80 - 4) + (100 - 4) + (50 - 4)
    feat, target = mds[0]
    # C_in = 5*S + 9 = 29
    assert feat.shape == (N_NODES, 29)
    assert target.shape == (4, N_NODES, 1)
```

**2d. 删除以下整个测试函数（与 `state_channels` 参数化相关）：**
- `test_single_dataset_with_state_channels_h_only`
- `test_single_dataset_with_state_channels_uv`
- `test_multi_dataset_with_state_channels_h_only`
- `test_dataset_state_channels_target_matches_graph_slice`
- `test_dataset_state_channels_features_state_prefix`
- `test_dataset_rejects_invalid_state_channels`

（其他测试如 `test_single_dataset_btype_concatenated`、sampler 相关测试、`test_lru_eviction`、`test_load_static_coords_*` 等**保持不变**。）

- [ ] **Step 3: 跑 dataset 测试**

```bash
pytest tests/test_dataset.py -v
```

Expected: 所有测试 PASS。

- [ ] **Step 4: 提交**

```bash
git add model/dataset.py tests/test_dataset.py
git commit -m "refactor: hardcode h-only state channel in dataset"
```

---

### Task 3: 重写 `model.py` + 删除 `test_geofno_num_channels.py`

**Files:**
- Modify: [model/model.py](../../../model/model.py)
- Delete: [tests/test_geofno_num_channels.py](../../../tests/test_geofno_num_channels.py)
- Create: `tests/test_geofno_forward.py`

- [ ] **Step 1: 修改 `model/model.py` 中 `GeoFNO2d`** — 替换为：

```python
class GeoFNO2d(nn.Module):
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
        fc1_hidden: int = 256,
    ):
        super(GeoFNO2d, self).__init__()
        """
        Geo-FNO for 2D irregular mesh hydrodynamic prediction (h-only).

        Input: (batch, N_nodes, in_channels) -- node features on irregular mesh
        Output: (batch, bundle_size, N_nodes, 1) -- predicted h on same irregular mesh
        """
        if num_fno_layers < 1:
            raise ValueError(f"num_fno_layers must be >= 1, got {num_fno_layers}")
        self.num_fno_layers = num_fno_layers
        self.num_channels = 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.s1 = s1
        self.s2 = s2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc1_hidden = fc1_hidden
        self.bundle_size = out_channels

        # Lifting layer
        self.fc0 = nn.Linear(in_channels, self.width)

        # Boundary spectral convs (irregular <-> regular)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)

        # Middle FNO blocks on the regular grid — count = num_fno_layers
        self.middle_convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(num_fno_layers)
        ])
        self.middle_ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1) for _ in range(num_fno_layers)
        ])
        self.middle_bs = nn.ModuleList([
            nn.Conv2d(2, self.width, 1) for _ in range(num_fno_layers)
        ])

        # Boundary biases (fixed)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        # Projection layers
        self.fc1 = nn.Linear(self.width, self.fc1_hidden)
        self.fc2 = nn.Linear(self.fc1_hidden, out_channels)

        # IPHI: learnable irregular-to-regular mapping
        self.iphi = IPHI(width=32)
        grid_x = torch.linspace(0, 1, self.s1, dtype=torch.float32)
        grid_y = torch.linspace(0, 1, self.s2, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1).unsqueeze(0)
        self.register_buffer("grid", grid)

    def forward(self, u, x_in, x_out=None):
        """
        Args:
            u: (batch, N_nodes, in_channels) -- node features. The first channel
               must be the current normalized h state.
            x_in: (batch, N_nodes, 2) -- 2D coordinates of input nodes
            x_out: (batch, N_nodes, 2) -- 2D coordinates of output nodes (default: same as x_in)
        Returns:
            (batch, bundle_size, N_nodes, 1) -- predicted future normalized h states.
        """
        if x_out is None:
            x_out = x_in
        if u.size(-1) < 1:
            raise ValueError(
                f"GeoFNO2d requires at least 1 input channel, got {u.size(-1)}"
            )

        state_in = u[..., :1]

        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)

        # Lift to high-dimensional channel space
        u = self.fc0(u)
        u = u.permute(0, 2, 1)  # (batch, width, N)

        # Layer 0: irregular mesh -> regular grid via IPHI
        uc1 = self.conv0(u, x_in=x_in, iphi=self.iphi)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        # Middle FNO blocks (num_fno_layers of them)
        for conv, w, b in zip(self.middle_convs, self.middle_ws, self.middle_bs):
            uc = F.gelu(conv(uc) + w(uc) + b(grid))

        # Layer 4: regular grid -> irregular mesh via IPHI
        u = self.conv4(uc, x_out=x_out, iphi=self.iphi)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        # Project back to output space
        u = u.permute(0, 2, 1)  # (batch, N, width)
        u = self.fc1(u)
        u = F.gelu(u)
        delta_flat = self.fc2(u)
        batch_size, num_nodes, _ = delta_flat.shape
        delta = delta_flat.view(batch_size, num_nodes, self.bundle_size, 1)
        delta = delta.permute(0, 2, 1, 3).contiguous()
        pred_block = state_in.unsqueeze(1) + delta
        return pred_block  # (batch, bundle_size, N, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        return self.grid.expand(batchsize, -1, -1, -1)
```

（`SpectralConv2d` 与 `IPHI` 类**不动**。）

- [ ] **Step 2: 删除 `tests/test_geofno_num_channels.py`**

```bash
git rm tests/test_geofno_num_channels.py
```

- [ ] **Step 3: 新建 `tests/test_geofno_forward.py` 覆盖 h-only forward shape**

```python
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import GeoFNO2d  # noqa: E402


def _build(bundle_size=2):
    return GeoFNO2d(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=5 * bundle_size + 9,
        out_channels=bundle_size,
        s1=4,
        s2=4,
        num_fno_layers=1,
    )


def test_geofno_num_channels_is_1():
    model = _build()
    assert model.num_channels == 1
    assert model.bundle_size == 2


def test_geofno_forward_shape_h_only():
    model = _build(bundle_size=2)
    B, N = 1, 5
    u = torch.randn(B, N, 5 * 2 + 9)
    x = torch.rand(B, N, 2)
    out = model(u, x)
    assert out.shape == (B, 2, N, 1)


def test_geofno_forward_shape_bundle_3():
    model = _build(bundle_size=3)
    B, N = 2, 4
    u = torch.randn(B, N, 5 * 3 + 9)
    x = torch.rand(B, N, 2)
    out = model(u, x)
    assert out.shape == (B, 3, N, 1)


def test_geofno_residual_uses_first_column_as_state():
    """The residual base must come from features[..., :1], so zeroing fc2
    yields output equal to the broadcasted state_in."""
    model = _build(bundle_size=2)
    B, N = 1, 3
    u = torch.zeros(B, N, 5 * 2 + 9)
    u[..., 0] = 1.234
    x = torch.rand(B, N, 2)
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
    out = model(u, x)
    assert torch.allclose(out, torch.full((B, 2, N, 1), 1.234), atol=1e-5)
```

- [ ] **Step 4: 跑 model + dataset 测试，确保未受影响**

```bash
pytest tests/test_geofno_forward.py tests/test_dataset.py tests/test_temporal_utils.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add model/model.py tests/test_geofno_forward.py
git commit -m "refactor: hardcode num_channels=1 in GeoFNO2d"
```

---

### Task 4: 重写 `train.py`

**Files:**
- Modify: [model/train.py](../../../model/train.py)

- [ ] **Step 1: 修改 `model/train.py`** — 应用以下定向编辑

**1a. 删除 import 中 `CHANNEL_ORDER`：**
```python
# from temporal_utils import CHANNEL_ORDER  ← 删除整行
```

**1b. 删除以下三个函数：** `_validate_state_channels`、`_channel_rel_l2`、`mean_channel_rel_l2_loss`

**1c. 在原 `_validate_state_channels` 位置新增 `rel_l2_loss`：**
```python
def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Single-channel relative L2 loss, averaged over batch."""
    diff = (pred - target).reshape(pred.size(0), -1)
    base = target.reshape(pred.size(0), -1)
    num = torch.linalg.vector_norm(diff, ord=2, dim=1)
    den = torch.linalg.vector_norm(base, ord=2, dim=1).clamp(min=eps)
    return (num / den).mean()
```

**1d. 把 `evaluate_model` 整个替换为：**
```python
def evaluate_model(model, test_loader, device, coords_2d_device, dist_ctx: dict | None = None):
    """Bundle evaluation in normalized space; no autoregressive rollout (h-only)."""
    model.eval()
    total_sse = 0.0
    total_sae = 0.0
    total_rel_l2 = 0.0
    num_samples = 0
    total_elements = 0

    x_in_base = coords_2d_device.to(device, non_blocking=True).unsqueeze(0)
    with torch.no_grad():
        for features, target_block in test_loader:
            features = features.to(device, non_blocking=True)
            target_block = target_block.to(device, non_blocking=True)
            batch_size = features.shape[0]
            if target_block.shape[-1] != 1:
                raise ValueError(
                    f"target_block last dim {target_block.shape[-1]} != 1 (h-only)"
                )
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

            num_samples += batch_size
            total_elements += target_block.numel()

    totals = reduce_sums(
        [total_sse, total_sae, total_rel_l2, num_samples, total_elements],
        device,
        dist_ctx,
    )
    sse, sae, rel_l2, sample_count, element_count = totals
    sample_count = max(1.0, sample_count)
    element_count = max(1.0, element_count)
    mse = sse / element_count
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": sae / element_count,
        "rel_l2": rel_l2 / sample_count,
    }
```

**1e. 把 `train_model` 签名 + 内部循环替换为：**
```python
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
    grad_clip=None,
    loss_type: str = "rel_l2",
    ema_decay: float | None = None,
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

    ema = None
    if ema_decay is not None:
        ema = ExponentialMovingAverage(model, decay=ema_decay)

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
                if ema is not None:
                    ema.update(model)

                if is_rank0(dist_ctx) and writer is not None:
                    writer.add_scalar("train/loss_step", loss_unscaled, global_step)
                    writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            if is_rank0(dist_ctx):
                pbar.set_postfix({"loss": f"{loss_unscaled:.6f}"})

        global_loss_sum, global_n = reduce_sums([local_loss_sum, local_n], device, dist_ctx)
        avg_loss = global_loss_sum / max(1.0, global_n)
        if is_rank0(dist_ctx) and writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        eval_model = ema.shadow if ema is not None else model
        test_metrics = evaluate_model(
            eval_model,
            test_loader,
            device,
            coords_2d_device,
            dist_ctx=dist_ctx,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        if is_rank0(dist_ctx):
            if writer is not None:
                writer.add_scalar("val/loss_epoch", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/rel_l2", test_metrics["rel_l2"], epoch)
                writer.add_scalar("val/mse", test_metrics["mse"], epoch)
                writer.add_scalar("val/rmse", test_metrics["rmse"], epoch)
                writer.add_scalar("val/mae", test_metrics["mae"], epoch)
                writer.add_scalar("train/lr", current_lr, epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_loss:.6f} | "
                f"Test RMSE: {test_metrics['rmse']:.6f} | "
                f"Test Rel-L2: {test_metrics['rel_l2']:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        current_test_loss = test_metrics["rmse"] if loss_type == "rmse" else test_metrics["rel_l2"]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            if is_rank0(dist_ctx):
                save_target = ema.shadow if ema is not None else unwrap_model(model)
                torch.save(save_target.state_dict(), checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path} (metric={best_loss:.6f})")

        barrier_if_distributed(dist_ctx)

    if is_rank0(dist_ctx):
        print("Training finished.")
```

- [ ] **Step 2: 跑前面已重写模块的全部测试**

```bash
pytest tests/test_temporal_utils.py tests/test_dataset.py tests/test_geofno_forward.py -v
```

Expected: 全部 PASS（无 import 失败）。

- [ ] **Step 3: 静态确认 train.py 可 import**

```bash
python -c "import sys; sys.path.insert(0, 'model'); import train; print('train import ok')"
```

Expected: 打印 `train import ok`，无异常。

- [ ] **Step 4: 提交**

```bash
git add model/train.py
git commit -m "refactor: simplify train/eval loop to single-channel h"
```

---

### Task 5: 重写 `test_all.py` + 同步 `test_test_all_helpers.py`

**Files:**
- Modify: [model/test_all.py](../../../model/test_all.py)
- Modify: [tests/test_test_all_helpers.py](../../../tests/test_test_all_helpers.py)

- [ ] **Step 1: 修改 `model/test_all.py`** — 应用以下编辑

**1a. 替换文件顶部常量与 import：**
```python
"""Autoregressive multi-step test for Geo-FNO bundle model across a test split (h-only)."""
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
    CHANNEL_NAME,
    build_checkpoint_name,
    input_channels_for_bundle,
    output_channels_for_bundle,
    validate_temporal_params,
)


METRIC_SPACES = ("physical", "normalized")
WATER_LEVEL_CHANNEL = 2
DRY_WATER_LEVEL_THRESHOLD = 0.005
REQUIRED_KEYS = ("graph", "storm_boundary", "inner_boundary")
GRAPH_STATS_KEYS = ("graph_mean", "graph_std")
LEGACY_STATS_KEYS = ("u_mean", "u_std", "v_mean", "v_std", "h_mean", "h_std")
```

**1b. 替换 `parse_args`（去掉 `--channels`）：**
```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Geo-FNO autoregressive test across a split (h-only).",
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
    parser.add_argument("--allow_random_weights", action="store_true", help="Run without a checkpoint.")
    parser.add_argument("--modes", type=int, default=16, help="Fourier modes per axis.")
    parser.add_argument("--width", type=int, default=32, help="Model width.")
    parser.add_argument("--s1", type=int, default=64, help="Internal grid size along axis 1.")
    parser.add_argument("--s2", type=int, default=64, help="Internal grid size along axis 2.")
    parser.add_argument("--num_fno_layers", type=int, default=3, help="Number of FNO layers.")
    parser.add_argument("--fc1_hidden", type=int, default=256, help="Hidden dim of the post-FNO FC1 layer.")
    parser.add_argument("--device", type=str, default="auto", help="Device string or auto.")
    return parser.parse_args()
```

**1c. 替换 `metric_output_path`：**
```python
def metric_output_path(base_path, metric_space):
    path = Path(base_path)
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}_{metric_space}{suffix}")
```

**1d. 替换 `load_normalization_stats`：**
```python
def load_normalization_stats(stats_path, device):
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
    mean_sub = mean_full[..., 2:3]
    std_sub = std_full[..., 2:3]
    return mean_sub, std_sub, mean_full, std_full
```

**1e. 替换 `init_bucket`、`compute_stats`、`compute_auc`：**
```python
def init_bucket(device):
    return {
        "sse": torch.zeros(1, device=device),
        "sae": torch.zeros(1, device=device),
        "sum_gt": torch.zeros(1, device=device),
        "sum_sq_gt": torch.zeros(1, device=device),
        "rel_l2_sum": torch.zeros(1, device=device),
        "count": 0,
    }


def compute_stats(bucket, num_nodes):
    count = bucket["count"]
    if count == 0:
        zeros = np.zeros(1, dtype=np.float64)
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


def compute_auc(results):
    auc = {"h": {}}
    steps = [entry["step"] for entry in results]
    if len(steps) < 2:
        for metric in ("mse", "rmse", "mae", "r2", "rel_l2"):
            auc["h"][metric] = 0.0
        return auc

    metrics = [key for key in results[0]["h"].keys() if key != "step"]
    for metric in metrics:
        values = [entry["h"][metric] for entry in results]
        auc["h"][metric] = float(np.trapz(values, steps))
    return auc
```

**1f. 替换 `autoregressive_one_file`：**
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
):
    """Run a single autoregressive rollout (h-only)."""
    graph_all, storm_all, inner_all = load_event_file(file_path, coords_2d_device.size(0))
    num_time = graph_all.size(0)
    if target_steps > num_time - 1:
        raise ValueError(
            f"{file_path}: target_steps={target_steps} exceeds T-1={num_time - 1}"
        )

    x_in = coords_2d_device.unsqueeze(0)
    real_start = graph_all[0:1, :, 2:3].to(device)
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
            target_norm_sub = target_full_norm[..., 2:3]

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

**1g. 替换 `write_results`：**
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
):
    for metric_space in METRIC_SPACES:
        out = metric_output_path(output_path, metric_space)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("Autoregressive Test Results\n")
            f.write(f"Max rollout: {max_rollout}\n")
            f.write(f"Bundle size: {bundle_size}\n")
            f.write(f"Channels: {CHANNEL_NAME}\n")
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
                metrics = result["h"]
                f.write(
                    f"{step:<6} | {'wl':<7} | {metrics['mse']:<12.6f} | "
                    f"{metrics['rmse']:<12.6f} | {metrics['mae']:<12.6f} | "
                    f"{metrics['r2']:<12.6f} | {metrics['rel_l2']:<12.6f} | {count:<6}\n"
                )
                f.write("-" * 110 + "\n")

            auc = compute_auc(results_by_space[metric_space])
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"AUC Summary Over {len(results_by_space[metric_space])} Steps\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"{'Channel':<11} | {'MSE Area':<12} | {'RMSE Area':<12} | "
                f"{'MAE Area':<12} | {'R2 Area':<12} | {'Rel L2 Area':<12}\n"
            )
            f.write("-" * 100 + "\n")
            metrics = auc["h"]
            f.write(
                f"{'wl':<11} | {metrics['mse']:<12.6f} | {metrics['rmse']:<12.6f} | "
                f"{metrics['mae']:<12.6f} | {metrics['r2']:<12.6f} | "
                f"{metrics['rel_l2']:<12.6f}\n"
            )
            f.write("=" * 100 + "\n")
        print(f"[test] results -> {out}")
```

**1h. 替换 `main()`：**
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

    coords_2d_cpu, btype_oh_cpu = load_static_coords(args.coords)
    coords_2d_device = coords_2d_cpu.to(device)
    btype_oh_device = btype_oh_cpu.to(device)
    num_nodes = coords_2d_cpu.size(0)

    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(args.norm, device=device)

    in_channels = input_channels_for_bundle(args.bundle_size)
    out_channels = output_channels_for_bundle(args.bundle_size)
    model_args = {
        "bundle_size": args.bundle_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "modes": args.modes,
        "width": args.width,
        "s1": args.s1,
        "s2": args.s2,
        "num_fno_layers": args.num_fno_layers,
        "fc1_hidden": args.fc1_hidden,
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
        fc1_hidden=args.fc1_hidden,
    ).to(device)
    print(f"[test] model params={sum(p.numel() for p in model.parameters()):,}")

    default_checkpoint = build_checkpoint_name(args.bundle_size)
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
        metric_space: [init_bucket(device) for _ in range(bucket_len)]
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
        )

    for metric_space in METRIC_SPACES:
        for step in range(bucket_len):
            bucket = per_step_metrics_by_space[metric_space][step]
            for key in ("sse", "sae", "sum_gt", "sum_sq_gt", "rel_l2_sum"):
                bucket[key] = bucket[key].detach().cpu().numpy()

    results_by_space = {metric_space: [] for metric_space in METRIC_SPACES}
    for metric_space in METRIC_SPACES:
        for step, bucket in enumerate(per_step_metrics_by_space[metric_space]):
            stats = compute_stats(bucket, num_nodes)
            result = {"step": step + 1, "count": int(bucket["count"])}
            result["h"] = {
                "mse": float(stats["mse_channels"][0]),
                "rmse": float(stats["rmse_channels"][0]),
                "mae": float(stats["mae_channels"][0]),
                "r2": float(stats["r2_channels"][0]),
                "rel_l2": float(stats["rel_l2_channels"][0]),
            }
            results_by_space[metric_space].append(result)

            m = result["h"]
            print(
                f"[step {step + 1:02d}][{metric_space}][N={int(bucket['count'])}] "
                f"wl: mse={m['mse']:.6f} rmse={m['rmse']:.6f} "
                f"mae={m['mae']:.6f} r2={m['r2']:.6f} rel_l2={m['rel_l2']:.6f}"
            )

    write_results(
        results_by_space,
        args.output,
        args.max_rollout,
        args.bundle_size,
        model_path,
        total_files=len(test_files),
        evaluated_files=len(evaluable),
        skipped_files=len(skipped),
    )
    print("[test] done.")
```

（其它顶层 helper 如 `denormalize`、`apply_dry_grid_error_mask`、`strip_module_prefix`、`extract_state_dict`、`resolve_checkpoint_path`、`load_checkpoint`、`find_test_files`、`prescan_files`、`load_event_file`、`build_features_batch` **保持不变**。）

- [ ] **Step 2: 重写 `tests/test_test_all_helpers.py` 为下列完整内容**

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


def test_load_normalization_stats_h_only(tmp_path):
    stats_path = tmp_path / "normalization.mat"
    scipy.io.savemat(
        stats_path,
        {
            "graph_mean": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "graph_std": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        },
    )

    mean_sub, std_sub, mean_full, std_full = load_normalization_stats(
        stats_path,
        device=torch.device("cpu"),
    )

    assert tuple(mean_sub.shape) == (1, 1, 1)
    assert tuple(std_sub.shape) == (1, 1, 1)
    assert tuple(mean_full.shape) == (1, 1, 3)
    assert tuple(std_full.shape) == (1, 1, 3)
    torch.testing.assert_close(mean_sub, torch.tensor([[[3.0]]]))
    torch.testing.assert_close(std_sub, torch.tensor([[[6.0]]]))
    torch.testing.assert_close(mean_full, torch.tensor([[[1.0, 2.0, 3.0]]]))
    torch.testing.assert_close(std_full, torch.tensor([[[4.0, 5.0, 6.0]]]))


def test_init_bucket_single_channel():
    bucket = init_bucket(torch.device("cpu"))
    for key in ("sse", "sae", "sum_gt", "sum_sq_gt", "rel_l2_sum"):
        assert tuple(bucket[key].shape) == (1,)
    assert bucket["count"] == 0


def test_compute_stats_zero_count_returns_single_channel_shape():
    bucket = {
        "sse": np.zeros(1, dtype=np.float64),
        "sae": np.zeros(1, dtype=np.float64),
        "sum_gt": np.zeros(1, dtype=np.float64),
        "sum_sq_gt": np.zeros(1, dtype=np.float64),
        "rel_l2_sum": np.zeros(1, dtype=np.float64),
        "count": 0,
    }

    stats = compute_stats(bucket, num_nodes=10)

    for key in ("mse_channels", "rmse_channels", "mae_channels", "r2_channels", "rel_l2_channels"):
        assert stats[key].shape == (1,)
        assert stats[key][0] == 0.0


def test_metric_output_path_no_suffix():
    assert metric_output_path("results/out.txt", "physical") == Path("results/out_physical.txt")
    assert metric_output_path("results/out.txt", "normalized") == Path("results/out_normalized.txt")
    assert metric_output_path("results/out", "physical") == Path("results/out_physical.txt")


def test_apply_dry_grid_error_mask_uses_full_h_even_when_diff_dim_k1():
    diff = torch.tensor([[[9.0], [7.0]]])
    target_full_norm = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]]])
    mean_full = torch.tensor([[[0.0, 0.0, 1.0]]])
    std_full = torch.tensor([[[1.0, 1.0, 1.0]]])

    masked = apply_dry_grid_error_mask(diff, target_full_norm, mean_full, std_full)

    torch.testing.assert_close(masked, torch.tensor([[[0.0], [7.0]]]))
```

- [ ] **Step 3: 跑 test_all 相关测试**

```bash
pytest tests/test_test_all_helpers.py -v
```

Expected: 全部 PASS。

- [ ] **Step 4: 静态确认 test_all.py 可 import**

```bash
python -c "import sys; sys.path.insert(0, 'model'); import test_all; print('test_all import ok')"
```

Expected: 打印 `test_all import ok`。

- [ ] **Step 5: 提交**

```bash
git add model/test_all.py tests/test_test_all_helpers.py
git commit -m "refactor: drop --channels from test_all and simplify metric buckets"
```

---

### Task 6: 重写 `main.py`

**Files:**
- Modify: [model/main.py](../../../model/main.py)

- [ ] **Step 1: 修改 `model/main.py`** — 应用以下编辑

**1a. CONFIG 中删除 `channels` 键：**
```python
CONFIG = {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "coords_path": "data/coordinates.mat",
    "norm_path": "data/normalization.mat",
    "tb_dir": "runs",

    "seed": 42,

    "bundle_size": 8,
    "batch_size": 16,
    "num_workers": 4,
    "lru_files_per_worker": 2,

    "modes": 24,
    "width": 48,
    "s1": 64,
    "s2": 64,
    "num_fno_layers": 4,
    "fc1_hidden": 256,

    "num_epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.01,
    "grad_clip": 1.0,
    "accum_steps": 1,
    "loss_type": "rel_l2",
    "ema_decay": 0.999,
}
```

**1b. import 中去除 `channels_suffix`、`parse_channels`：**
```python
from temporal_utils import (
    build_checkpoint_name,
    build_run_suffix,
    input_channels_for_bundle,
    output_channels_for_bundle,
    validate_temporal_params,
)
```

**1c. `main()` 函数中删除通道相关初始化、构造、日志、调用** — 把整个 `main()` 替换为：

```python
def main():
    dist_ctx = init_distributed()
    writer = None
    try:
        if CONFIG["accum_steps"] < 1:
            raise ValueError(f"accum_steps must be >= 1, got {CONFIG['accum_steps']}")
        validate_temporal_params(CONFIG["bundle_size"])
        preflight_manifest_files(CONFIG)
        set_seed(CONFIG["seed"])

        in_channels = input_channels_for_bundle(CONFIG["bundle_size"])
        out_channels = output_channels_for_bundle(CONFIG["bundle_size"])
        checkpoint_name = build_checkpoint_name(CONFIG["bundle_size"])
        run_tag = (
            "GeoFNO"
            + build_run_suffix(CONFIG["bundle_size"])
            + "_"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        device = get_device(dist_ctx)
        rank0_print(
            dist_ctx,
            f"[main] device={device}, distributed={dist_ctx['distributed']}, "
            f"world_size={dist_ctx['world_size']}",
        )
        rank0_print(
            dist_ctx,
            f"[main] bundle_size={CONFIG['bundle_size']}, "
            f"in_channels={in_channels}, out_channels={out_channels}",
        )
        rank0_print(dist_ctx, f"[main] checkpoint name: {checkpoint_name}")

        rank0_print(dist_ctx, f"[main] loading coords from {CONFIG['coords_path']}")
        coords_2d_cpu, btype_oh_cpu = load_static_coords(CONFIG["coords_path"])
        coords_2d_device = coords_2d_cpu.to(device)
        rank0_print(
            dist_ctx,
            f"[main] coords shape={tuple(coords_2d_cpu.shape)}, "
            f"btype_oh shape={tuple(btype_oh_cpu.shape)}",
        )

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
        rank0_print(
            dist_ctx,
            f"[main] train samples={len(train_dataset)}, val samples={len(val_dataset)}",
        )
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
            pad_to_equal_length=False,
        )

        loader_kwargs = {"num_workers": CONFIG["num_workers"], "pin_memory": True}
        if CONFIG["num_workers"] > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            drop_last=False,
            **loader_kwargs,
        )

        model = GeoFNO2d(
            modes1=CONFIG["modes"],
            modes2=CONFIG["modes"],
            width=CONFIG["width"],
            in_channels=in_channels,
            out_channels=out_channels,
            s1=CONFIG["s1"],
            s2=CONFIG["s2"],
            num_fno_layers=CONFIG["num_fno_layers"],
            fc1_hidden=CONFIG["fc1_hidden"],
        ).to(device)
        rank0_print(dist_ctx, f"[main] model params={sum(p.numel() for p in model.parameters()):,}")

        if dist_ctx["distributed"]:
            model = DDP(
                model,
                device_ids=[dist_ctx["local_rank"]],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        optimizer_steps_per_epoch = len(train_loader) // CONFIG["accum_steps"]
        if optimizer_steps_per_epoch < 1:
            raise ValueError(
                f"accum_steps={CONFIG['accum_steps']} too large for "
                f"steps_per_epoch={len(train_loader)}"
            )
        scheduler = build_scheduler(
            optimizer,
            num_epochs=CONFIG["num_epochs"],
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            warmup_ratio=CONFIG["warmup_ratio"],
            min_lr_ratio=CONFIG["min_lr_ratio"],
        )
        total_steps = CONFIG["num_epochs"] * optimizer_steps_per_epoch
        warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
        rank0_print(
            dist_ctx,
            f"[main] Cosine: total_steps={total_steps}, warmup_steps={warmup_steps}, "
            f"min_lr={CONFIG['lr'] * CONFIG['min_lr_ratio']:.2e}",
        )

        if dist_ctx["is_rank0"]:
            tb_run_dir = os.path.join(CONFIG["tb_dir"], run_tag)
            os.makedirs(tb_run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_run_dir)
            rank0_print(dist_ctx, f"[main] tensorboard log dir={tb_run_dir}")

            config_md = "### Training Configuration\n| Parameter | Value |\n|---|---|\n"
            for key, value in CONFIG.items():
                config_md += f"| {key} | {value} |\n"
            config_md += f"| in_channels (derived) | {in_channels} |\n"
            config_md += f"| out_channels (derived) | {out_channels} |\n"
            config_md += "\n### System\n| Parameter | Value |\n|---|---|\n"
            config_md += f"| OS | {platform.system()} {platform.release()} |\n"
            config_md += f"| CPU Cores | {os.cpu_count()} |\n"
            config_md += f"| World Size | {dist_ctx['world_size']} |\n"
            try:
                config_md += f"| GPU | {torch.cuda.get_device_name(0)} |\n"
            except Exception:
                pass
            writer.add_text("config/all", config_md, 0)

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
            ema_decay=CONFIG.get("ema_decay"),
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
```

- [ ] **Step 2: 静态确认 main.py 可 import**

```bash
python -c "import sys; sys.path.insert(0, 'model'); import main; print('main import ok')"
```

Expected: 打印 `main import ok`。

- [ ] **Step 3: 跑全部 pytest，确认整体绿**

```bash
pytest tests/ -v
```

Expected: 全部 PASS。

- [ ] **Step 4: 提交**

```bash
git add model/main.py
git commit -m "refactor: drop channels config from main entrypoint"
```

---

### Task 7: 本机最终验证（pytest + import sanity + 文档扫描）

**Files:** 无源码改动；本任务只在本机做能做的验证。**实际训练 / DDP / test_all 冒烟由你在服务器上手动跑（见下方「服务器端 checklist」）。**

- [ ] **Step 1: 跑全量 pytest**

```bash
pytest tests/ -v
```

Expected: 全部 PASS，无 skip / error。

- [ ] **Step 2: 静态 import 检查 — 确认所有模块无 import 错误**

```bash
python -c "import sys; sys.path.insert(0, 'model'); import temporal_utils, dataset, model, train, test_all, main; print('all imports ok')"
```

Expected: 打印 `all imports ok`。

- [ ] **Step 3: 扫文档 markdown 中残留的 `--channels` / `state_channels` 引用**

```bash
grep -rn "state_channels\|--channels\|num_channels\|parse_channels\|channels_suffix\|CHANNEL_ORDER" AGENTS.md claude.md docs/ 2>/dev/null
```

对每个命中：
- 若是历史 spec / plan 文档（[docs/superpowers/specs/](../specs/) / [docs/superpowers/plans/](.)）：**不动**，作为历史记录。
- 若是 [AGENTS.md](../../../AGENTS.md) / [claude.md](../../../claude.md) / 其它运行手册：编辑掉相关段落或加注 "已移除 (2026-05-24)"。

- [ ] **Step 4: 如有文档改动，提交**

```bash
git add -u
git commit -m "docs: scrub multi-channel references from runtime docs"
```

（如 Step 3 grep 无命中或全部命中都在历史 spec/plan 中，跳过本提交。）

---

## 服务器端 checklist（用户在服务器上执行，不属于 Task）

本地 pytest + import sanity 跑过后，把代码同步到服务器，手动验证以下三件事即可。每件事只要"能跑几个 step / 几个 epoch 不报错"就过。

### A. Checkpoint 迁移

```bash
mv best_geofno_b8_chh.pt best_geofno_b8.pt
```

如 checkpoint 文件不存在或已被覆盖，跳过 — 重训就好。

### B. 训练冒烟（单卡 / 多卡）

临时把 `model/main.py` 中 `CONFIG["num_epochs"]` 改成 1（或 2），跑：

```bash
python model/main.py                       # 单卡
torchrun --nproc_per_node=N model/main.py  # 多卡 DDP
```

Expected：
- 控制台打印 `Epoch 1/1 | Train Loss: ... | Test RMSE: ... | Test Rel-L2: ... | LR: ...`，**无** `Test Rel-H` 字段
- 训练结束生成 `best_geofno_b8.pt`
- `runs/GeoFNO_b8_<ts>/` 目录创建，tensorboard 标量含 `val/rel_l2`、不含 `val/rel_h`

跑通后把 `num_epochs` 改回 200。

### C. test_all 冒烟

```bash
python model/test_all.py --num_files 1 --max_rollout 8
```

Expected：
- 加载 `best_geofno_b8.pt`（无 `_chh` 后缀）
- 输出 `geofno_autoregressive_results_physical.txt` 与 `_normalized.txt`（无 `_chh` 后缀）
- 表格内只含一行 `wl`

---

## 自审清单

- ✅ Spec 覆盖：每个 spec section 都有对应 Task（Task 1 = temporal_utils；Task 2 = dataset；Task 3 = model；Task 4 = train；Task 5 = test_all；Task 6 = main；Task 7 = 本机验证 + 文档扫描；checkpoint 迁移与训练/test_all 冒烟划归服务器端 checklist）。
- ✅ 占位符：所有代码块均为可执行完整代码，无 TBD / TODO。
- ✅ 类型一致：`rel_l2_loss` 在 Task 4 定义并在 Task 4 内调用；`build_features_batch` 在 Task 5 保持原签名；`init_bucket(device)`、`compute_stats(bucket, num_nodes)`、`compute_auc(results)` 在 Task 5 内统一签名。
- ✅ 公式一致：`input_channels_for_bundle(S) = 5*S+9` 在 Task 1 定义，Task 2 / 5 / 6 全部使用。
- ✅ Checkpoint：服务器端 checklist 显式重命名；本机不依赖 checkpoint 做任何验证。
