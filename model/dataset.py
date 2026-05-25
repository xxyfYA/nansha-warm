"""Storm-surge lazy-loading dataset for Geo-FNO warm-up prediction model.

Each .pt file is one pre-processed sample:
    storm_boundary  — (24, N, 3)   [P, Wx, Wy]
    inner_boundary  — (24, N, 2)   [h_bdy, q_bdy]
    target          — (N, 3)       [u, v, h] at t+24

The dataset takes target[:, 2:3] as the single-channel h supervision signal.
"""
from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, Sampler

REQUIRED_PT_KEYS = ("storm_boundary", "inner_boundary", "target")


def load_static_coords(coords_path):
    """Load node 2D coordinates for IPHI irregular-to-regular mapping.

    Returns:
        coords_t: (N, 2) float32, min/max-normalized to [0, 1] per axis.

    Shared memory is attempted for DataLoader worker reuse.
    """
    coords_path = Path(coords_path)
    mat = scipy.io.loadmat(coords_path)
    if "coordinates" not in mat:
        keys = [key for key in mat.keys() if not key.startswith("__")]
        raise KeyError(f"{coords_path}: missing 'coordinates'; got {keys}")

    coords = np.asarray(mat["coordinates"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{coords_path}: coordinates must be (N,>=2), got {coords.shape}")
    coords = coords[:, :2].astype(np.float32, copy=False)
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = np.maximum(cmax - cmin, np.float32(1e-8))
    coords_norm = (coords - cmin) / span

    coords_t = torch.from_numpy(np.ascontiguousarray(coords_norm)).float()
    try:
        coords_t.share_memory_()
    except (RuntimeError, OSError) as exc:
        warnings.warn(
            f"{coords_path}: shared memory unavailable; returning ordinary CPU tensors ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        coords_t = coords_t.clone()
    return coords_t


def load_pt(path) -> dict[str, torch.Tensor]:
    """Load and validate one pre-processed .pt file."""
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in REQUIRED_PT_KEYS:
        if key not in data:
            raise KeyError(f"{path}: missing key {key!r}; got {list(data.keys())}")

    storm = data["storm_boundary"].float()
    inner = data["inner_boundary"].float()
    target = data["target"].float()

    if storm.dim() != 3 or storm.size(-1) != 3:
        raise ValueError(f"{path}: storm_boundary must be (24,N,3), got {tuple(storm.shape)}")
    if inner.dim() != 3 or inner.size(0) != storm.size(0) or inner.size(1) != storm.size(1) or inner.size(-1) != 2:
        raise ValueError(
            f"{path}: inner_boundary {tuple(inner.shape)} incompatible with storm_boundary "
            f"{tuple(storm.shape)} (expected (24,N,2))"
        )
    if target.dim() != 2 or target.size(-1) != 3 or target.size(0) != storm.size(1):
        raise ValueError(
            f"{path}: target {tuple(target.shape)} must be (N,3) with N={storm.size(1)}"
        )

    return {"storm": storm, "inner": inner, "target": target}


def build_features(
    storm_window: torch.Tensor,
    inner_window: torch.Tensor,
) -> torch.Tensor:
    """Build per-node feature matrix from 24-hour forcing windows.

    Args:
        storm_window: (T, N, 3) — storm boundary [P, Wx, Wy] where T=24
        inner_window: (T, N, 2) — inner boundary [h_bdy, q_bdy] where T=24

    Returns:
        (N, 120) feature tensor: storm_flat (N, 72) + inner_flat (N, 48)
    """
    num_nodes = storm_window.size(1)
    storm_flat = storm_window.permute(1, 0, 2).reshape(num_nodes, -1)
    inner_flat = inner_window.permute(1, 0, 2).reshape(num_nodes, -1)
    return torch.cat([storm_flat, inner_flat], dim=-1).contiguous()


class MultiStormSurgeDataset(Dataset):
    """Dataset over pre-processed .pt files — each file is one sample."""

    def __init__(self, data_dir, lru_files_per_worker: int = 2):
        if lru_files_per_worker < 1:
            raise ValueError(f"lru_files_per_worker must be >= 1, got {lru_files_per_worker}")

        self.data_dir = Path(data_dir)
        self.lru_files_per_worker = int(lru_files_per_worker)

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run: python scripts/build_manifest.py "
                f"{self.data_dir}"
            )
        manifest = json.loads(manifest_path.read_text())
        if "num_nodes" not in manifest or "files" not in manifest:
            raise KeyError(f"{manifest_path}: manifest must contain 'num_nodes' and 'files'")
        if not manifest["files"]:
            raise RuntimeError(f"{manifest_path}: manifest files list is empty")

        self.num_nodes = int(manifest["num_nodes"])
        self.files: list[Path] = []
        for entry in manifest["files"]:
            rel_path = entry if isinstance(entry, str) else entry["path"]
            path = Path(rel_path)
            if not path.is_absolute():
                path = self.data_dir / path
            if not path.exists():
                raise FileNotFoundError(
                    f"manifest references missing file {path}; rebuild manifest "
                    f"with: python scripts/build_manifest.py {self.data_dir}"
                )
            self.files.append(path)

        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.files)

    def _get_entry(self, file_idx: int) -> dict[str, torch.Tensor]:
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]

        entry = load_pt(self.files[file_idx])
        if entry["storm"].size(1) != self.num_nodes:
            raise ValueError(
                f"{self.files[file_idx]}: file N={entry['storm'].size(1)} "
                f"!= manifest num_nodes={self.num_nodes}"
            )

        self._cache[file_idx] = entry
        while len(self._cache) > self.lru_files_per_worker:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.files):
            raise IndexError(idx)

        entry = self._get_entry(idx)
        features = build_features(entry["storm"], entry["inner"])
        target_h = entry["target"][:, 2:3].contiguous()  # (N, 1) — h only
        return features, target_h


class FileChunkedDistributedSampler(Sampler[int]):
    """Distributed sampler that greedily assigns files to balance per-rank sample counts."""

    def __init__(
        self,
        dataset: MultiStormSurgeDataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        pad_to_equal_length: bool | None = None,
    ):
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if pad_to_equal_length is None:
            pad_to_equal_length = not drop_last
        if drop_last and pad_to_equal_length:
            raise ValueError("pad_to_equal_length=True is incompatible with drop_last=True")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.pad_to_equal_length = bool(pad_to_equal_length)
        self.epoch = 0

        self._files_by_rank, self._rank_totals = self._build_fixed_assignment()
        self._assignment_error: str | None = None
        needs_equal_nonempty = self.drop_last or self.pad_to_equal_length
        if needs_equal_nonempty and any(total == 0 for total in self._rank_totals):
            self._assignment_error = (
                "FileChunkedDistributedSampler could not assign samples to every rank: "
                f"rank_totals={self._rank_totals}, num_replicas={self.num_replicas}, "
                f"num_files={len(dataset)}. Reduce num_replicas or add more files."
            )
        if self.drop_last:
            self._target_count = min(self._rank_totals)
        elif self.pad_to_equal_length:
            self._target_count = max(self._rank_totals)
        else:
            self._target_count = self._rank_totals[self.rank]

    def _build_fixed_assignment(self) -> tuple[list[list[int]], list[int]]:
        files_by_rank: list[list[int]] = [[] for _ in range(self.num_replicas)]
        totals = [0 for _ in range(self.num_replicas)]
        for file_idx in range(len(self.dataset)):
            target_rank = min(range(self.num_replicas), key=lambda rank: (totals[rank], rank))
            files_by_rank[target_rank].append(file_idx)
            totals[target_rank] += 1
        return files_by_rank, totals

    def __iter__(self) -> Iterator[int]:
        if self._assignment_error is not None:
            raise RuntimeError(self._assignment_error)

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        files = list(self._files_by_rank[self.rank])
        if self.shuffle and len(files) > 1:
            order = torch.randperm(len(files), generator=generator).tolist()
            files = [files[i] for i in order]

        indices = list(files)
        target_count = self._target_count
        if len(indices) >= target_count:
            indices = indices[:target_count]
        else:
            if not indices and target_count:
                raise RuntimeError(
                    f"rank {self.rank} has no samples; reduce num_replicas or rebuild manifest"
                )
            extra = target_count - len(indices)
            indices.extend(indices[i % len(indices)] for i in range(extra))

        yield from indices

    def __len__(self) -> int:
        if self._assignment_error is not None:
            raise RuntimeError(self._assignment_error)
        return self._target_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
