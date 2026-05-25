import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from dataset import (  # noqa: E402
    FileChunkedDistributedSampler,
    MultiStormSurgeDataset,
    build_features,
    load_pt,
    load_static_coords,
)

N_NODES = 64


@pytest.fixture
def coords_mat(tmp_path):
    """Fake coordinates.mat with 64 nodes."""
    coords3 = np.random.rand(N_NODES, 3).astype(np.float64) * 100.0
    p = tmp_path / "coords.mat"
    scipy.io.savemat(p, {"coordinates": coords3})
    return p


def _make_pt(path, num_nodes=N_NODES):
    """Create a new-format .pt file: one pre-processed sample."""
    data = {
        "storm_boundary": torch.randn(24, num_nodes, 3),
        "inner_boundary": torch.zeros(24, num_nodes, 2),
        "target": torch.randn(num_nodes, 3),
        "run_id": path.stem,
        "source_dir": "synthetic",
    }
    bdy_count = min(12, num_nodes)
    data["inner_boundary"][:, :bdy_count] = torch.randn(24, bdy_count, 2)
    torch.save(data, path)


def _make_deterministic_pt(path, num_nodes=3):
    storm_boundary = torch.arange(24 * num_nodes * 3, dtype=torch.float32).reshape(24, num_nodes, 3)
    inner_boundary = 1000 + torch.arange(24 * num_nodes * 2, dtype=torch.float32).reshape(24, num_nodes, 2)
    target = 2000 + torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3)
    torch.save(
        {
            "storm_boundary": storm_boundary,
            "inner_boundary": inner_boundary,
            "target": target,
        },
        path,
    )


@pytest.fixture
def split_dir(tmp_path):
    d = tmp_path / "split"
    d.mkdir()
    for i in range(3):
        _make_pt(d / f"e{i}.pt")
    manifest = {
        "num_nodes": N_NODES,
        "files": [f"e{i}.pt" for i in range(3)],
        "created_at": "2026-05-25T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


@pytest.fixture
def uneven_split_dir(tmp_path):
    d = tmp_path / "uneven"
    d.mkdir()
    for i in range(3):
        _make_pt(d / f"e{i}.pt")
    manifest = {
        "num_nodes": N_NODES,
        "files": [f"e{i}.pt" for i in range(3)],
        "created_at": "2026-05-25T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


# --- load_static_coords ---

def test_load_static_coords_normalizes(coords_mat):
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        coords = load_static_coords(coords_mat)
    assert coords.shape == (N_NODES, 2)
    assert coords.dtype == torch.float32
    shared_memory_warnings = [
        warning for warning in warnings_record
        if issubclass(warning.category, RuntimeWarning)
        and "shared memory" in str(warning.message)
    ]
    if shared_memory_warnings:
        assert not coords.is_shared()
    else:
        assert coords.is_shared()
    assert torch.all(coords >= 0.0) and torch.all(coords <= 1.0)
    assert torch.isclose(coords.min(0).values, torch.zeros(2)).all()
    assert torch.isclose(coords.max(0).values, torch.ones(2)).all()


def test_load_static_coords_warns_and_returns_cpu_tensors_when_share_memory_fails(
    coords_mat, monkeypatch
):
    original_share_memory = torch.Tensor.share_memory_

    def fail_once(self):
        raise RuntimeError("shm unavailable")

    monkeypatch.setattr(torch.Tensor, "share_memory_", fail_once)
    with pytest.warns(RuntimeWarning, match="shared memory"):
        coords = load_static_coords(coords_mat)

    assert coords.shape == (N_NODES, 2)
    assert not coords.is_shared()
    monkeypatch.setattr(torch.Tensor, "share_memory_", original_share_memory)


def test_load_static_coords_rejects_bad_coords(tmp_path):
    p = tmp_path / "bad.mat"
    scipy.io.savemat(p, {"wrong_key": np.zeros((4, 3))})
    with pytest.raises(KeyError, match="coordinates"):
        load_static_coords(p)


# --- load_pt ---

def test_load_pt_validates_new_format(tmp_path):
    path = tmp_path / "sample.pt"
    _make_pt(path, num_nodes=10)
    entry = load_pt(path)
    assert entry["storm"].shape == (24, 10, 3)
    assert entry["inner"].shape == (24, 10, 2)
    assert entry["target"].shape == (10, 3)


def test_load_pt_rejects_missing_key(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"storm_boundary": torch.randn(24, 5, 3)}, path)
    with pytest.raises(KeyError, match="missing key"):
        load_pt(path)


# --- build_features ---

def test_feature_layout_is_storm_inner_order(tmp_path):
    d = tmp_path / "layout"
    d.mkdir()
    path = d / "event.pt"
    _make_deterministic_pt(path, num_nodes=2)

    entry = load_pt(path)
    features = build_features(entry["storm"], entry["inner"])
    assert features.shape == (2, 120)

    expected = torch.cat([
        entry["storm"].permute(1, 0, 2).reshape(2, -1),   # (2, 72)
        entry["inner"].permute(1, 0, 2).reshape(2, -1),   # (2, 48)
    ], dim=-1)
    assert torch.equal(features, expected)


# --- MultiStormSurgeDataset ---

def test_multi_dataset_len_equals_num_files(split_dir):
    mds = MultiStormSurgeDataset(data_dir=split_dir, lru_files_per_worker=1)
    assert len(mds) == 3


def test_multi_dataset_getitem_shapes(split_dir):
    mds = MultiStormSurgeDataset(data_dir=split_dir, lru_files_per_worker=1)
    feat, target = mds[0]
    assert feat.shape == (N_NODES, 120)
    assert target.shape == (N_NODES, 1)


def test_multi_dataset_getitem_target_is_h_channel(split_dir):
    mds = MultiStormSurgeDataset(data_dir=split_dir, lru_files_per_worker=1)
    _, target = mds[0]
    # target should be column 2 (h) of the target tensor
    raw = torch.load(mds.files[0], map_location="cpu", weights_only=False)
    expected_h = raw["target"][:, 2:3]
    assert torch.equal(target, expected_h)


def test_multi_dataset_raises_on_missing_manifest(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest"):
        MultiStormSurgeDataset(d)


def test_multi_dataset_raises_on_empty_manifest(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"num_nodes": 10, "files": []}))
    with pytest.raises(RuntimeError, match="empty"):
        MultiStormSurgeDataset(d)


# --- FileChunkedDistributedSampler ---

def test_file_chunked_sampler_len_matches_dataset(split_dir):
    mds = MultiStormSurgeDataset(split_dir, lru_files_per_worker=1)
    sampler = FileChunkedDistributedSampler(mds, num_replicas=1, rank=0, shuffle=False, seed=0)
    assert len(sampler) == len(mds)


def test_file_chunked_sampler_disjoint_across_ranks(split_dir):
    mds = MultiStormSurgeDataset(split_dir, lru_files_per_worker=1)
    s0 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=0, shuffle=False, seed=0)
    s1 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=1, shuffle=False, seed=0)
    i0 = set(iter(s0))
    i1 = set(iter(s1))
    assert i0.isdisjoint(i1)


def test_file_chunked_sampler_len_is_stable_across_epochs(uneven_split_dir):
    mds = MultiStormSurgeDataset(uneven_split_dir)
    sampler = FileChunkedDistributedSampler(
        mds,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=3,
        drop_last=True,
    )

    lengths = []
    for epoch in range(8):
        sampler.set_epoch(epoch)
        lengths.append(len(sampler))

    assert len(set(lengths)) == 1


def test_balanced_sampler_drop_last_uses_min_rank_total_without_duplicates(uneven_split_dir):
    mds = MultiStormSurgeDataset(uneven_split_dir)
    samplers = [
        FileChunkedDistributedSampler(
            mds,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            seed=0,
            drop_last=True,
        )
        for rank in range(2)
    ]

    rank_indices = [list(iter(sampler)) for sampler in samplers]
    # With 3 files and 2 ranks, min rank total = 1, so each rank gets 1
    assert all(len(indices) == 1 for indices in rank_indices)
    assert all(len(indices) == len(set(indices)) for indices in rank_indices)
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])


def test_balanced_sampler_raises_when_a_rank_has_no_samples(tmp_path):
    d = tmp_path / "single"
    d.mkdir()
    _make_pt(d / "only.pt")
    manifest = {
        "num_nodes": N_NODES,
        "files": ["only.pt"],
        "created_at": "2026-05-25T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    mds = MultiStormSurgeDataset(d)
    sampler = FileChunkedDistributedSampler(
        mds,
        num_replicas=2,
        rank=1,
        shuffle=False,
        seed=0,
        drop_last=True,
    )

    with pytest.raises(RuntimeError, match="could not assign samples to every rank"):
        list(iter(sampler))
    with pytest.raises(RuntimeError, match="could not assign samples to every rank"):
        len(sampler)


def test_file_chunked_sampler_set_epoch_changes_shuffle(split_dir):
    mds = MultiStormSurgeDataset(split_dir, lru_files_per_worker=1)
    sampler = FileChunkedDistributedSampler(mds, num_replicas=1, rank=0, shuffle=True, seed=0)
    epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))
    assert len(epoch0) == len(epoch1) == len(mds)
    assert epoch0 != epoch1


def test_lru_eviction(split_dir):
    mds = MultiStormSurgeDataset(split_dir, lru_files_per_worker=2)
    for i in range(3):
        _ = mds[i]
    assert len(mds._cache) <= 2


def test_dataset_index_error(split_dir):
    mds = MultiStormSurgeDataset(split_dir, lru_files_per_worker=1)
    with pytest.raises(IndexError):
        _ = mds[-1]
    with pytest.raises(IndexError):
        _ = mds[len(mds)]


def test_sampler_pad_to_equal_length(uneven_split_dir):
    mds = MultiStormSurgeDataset(uneven_split_dir)
    samplers = [
        FileChunkedDistributedSampler(
            mds,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            seed=0,
            drop_last=False,
            pad_to_equal_length=True,
        )
        for rank in range(2)
    ]
    # With 3 files and 2 ranks: rank 0 gets 2, rank 1 gets 1
    # pad_to_equal_length=True → both get max=2
    rank_indices = [list(iter(sampler)) for sampler in samplers]
    assert all(len(indices) == 2 for indices in rank_indices)
    # The padded rank should have a duplicate
    assert len(set(rank_indices[1])) == 1


def test_sampler_rejects_pad_with_drop_last():
    import tempfile
    d = Path(tempfile.mkdtemp())
    _make_pt(d / "e0.pt")
    manifest = {
        "num_nodes": N_NODES,
        "files": ["e0.pt"],
        "created_at": "2026-05-25T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    mds = MultiStormSurgeDataset(d)
    with pytest.raises(ValueError, match="pad_to_equal_length.*incompatible"):
        FileChunkedDistributedSampler(
            mds, num_replicas=1, rank=0, drop_last=True, pad_to_equal_length=True,
        )
