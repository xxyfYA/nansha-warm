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
    StormSurgeDataset,
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


def _make_pt(path, T):
    data = {
        "graph": torch.randn(T, N_NODES, 3),
        "storm_boundary": torch.randn(T, N_NODES, 3),
        "inner_boundary": torch.zeros(T, N_NODES, 2),
        "run_id": path.stem,
        "source_dir": "synthetic",
    }
    data["inner_boundary"][:, :12] = torch.randn(T, 12, 2)
    torch.save(data, path)


def _make_deterministic_pt(path, T, num_nodes=3):
    graph = torch.arange(T * num_nodes * 3, dtype=torch.float32).reshape(T, num_nodes, 3)
    storm_boundary = 1000 + torch.arange(T * num_nodes * 3, dtype=torch.float32).reshape(
        T, num_nodes, 3
    )
    inner_boundary = 2000 + torch.arange(T * num_nodes * 2, dtype=torch.float32).reshape(
        T, num_nodes, 2
    )
    torch.save(
        {
            "graph": graph,
            "storm_boundary": storm_boundary,
            "inner_boundary": inner_boundary,
        },
        path,
    )


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


@pytest.fixture
def uneven_split_dir(tmp_path):
    d = tmp_path / "uneven"
    d.mkdir()
    sizes = [104, 54, 14]
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


def test_single_dataset_shapes(split_dir):
    ds = StormSurgeDataset(
        path=split_dir / "e0.pt",
        lru_capacity=1,
    )
    assert len(ds) == 80 - 24
    feat, target = ds[0]
    assert feat.shape == (N_NODES, 120)
    assert target.shape == (N_NODES, 1)


def test_single_dataset_too_short_rejects(tmp_path):
    d = tmp_path / "tiny"
    d.mkdir()
    _make_pt(d / "short.pt", T=20)
    with pytest.raises(ValueError, match="T="):
        StormSurgeDataset(path=d / "short.pt")


def test_feature_layout_is_storm_inner_order(tmp_path):
    d = tmp_path / "layout"
    d.mkdir()
    path = d / "event.pt"
    _make_deterministic_pt(path, T=30, num_nodes=2)
    ds = StormSurgeDataset(path=path, lru_capacity=1)

    features, target = ds[0]
    assert features.shape == (2, 120)
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = torch.cat([
        data["storm_boundary"][0:24, 0].reshape(-1),  # 72
        data["inner_boundary"][0:24, 0].reshape(-1),  # 48
    ])
    assert torch.equal(features[0], expected)
    assert target.shape == (2, 1)


def test_multi_dataset_index_flattening(split_dir):
    mds = MultiStormSurgeDataset(
        data_dir=split_dir,
        lru_files_per_worker=1,
    )
    assert len(mds) == (80 - 24) + (100 - 24) + (50 - 24)
    feat, target = mds[0]
    assert feat.shape == (N_NODES, 120)
    assert target.shape == (N_NODES, 1)


def test_multi_dataset_drops_too_short_files(tmp_path):
    d = tmp_path / "split"
    d.mkdir()
    _make_pt(d / "tiny.pt", 10)
    _make_pt(d / "ok.pt", 50)
    manifest = {
        "num_nodes": N_NODES,
        "files": [{"path": "tiny.pt", "T": 10}, {"path": "ok.pt", "T": 50}],
        "created_at": "2026-05-16T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    mds = MultiStormSurgeDataset(d, lru_files_per_worker=1)
    assert len(mds) == 50 - 24


def test_file_chunked_sampler_groups_by_file(split_dir):
    mds = MultiStormSurgeDataset(split_dir, lru_files_per_worker=1)
    sampler = FileChunkedDistributedSampler(mds, num_replicas=1, rank=0, shuffle=True, seed=0)
    indices = list(iter(sampler))
    assert len(indices) == len(mds)
    file_seq = [mds.flat_index[i][0] for i in indices]
    transitions = sum(1 for a, b in zip(file_seq, file_seq[1:]) if a != b)
    assert transitions <= len(set(file_seq)) - 1 + 2


def test_file_chunked_sampler_disjoint_across_ranks(split_dir):
    mds = MultiStormSurgeDataset(split_dir, lru_files_per_worker=1)
    s0 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=0, shuffle=False, seed=0)
    s1 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=1, shuffle=False, seed=0)
    i0 = set(iter(s0))
    i1 = set(iter(s1))
    assert i0.isdisjoint(i1)
    f0 = {mds.flat_index[i][0] for i in i0}
    f1 = {mds.flat_index[i][0] for i in i1}
    assert f0.isdisjoint(f1)


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
    assert all(len(indices) == len(set(indices)) for indices in rank_indices)
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])


def test_balanced_sampler_raises_when_a_rank_has_no_samples(tmp_path):
    d = tmp_path / "single"
    d.mkdir()
    _make_pt(d / "only.pt", 40)
    manifest = {
        "num_nodes": N_NODES,
        "files": [{"path": "only.pt", "T": 40}],
        "created_at": "2026-05-16T00:00:00+00:00",
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
    for fi in range(3):
        first_idx_for_file = next(i for i, (f, _) in enumerate(mds.flat_index) if f == fi)
        _ = mds[first_idx_for_file]
    assert len(mds._cache) <= 2
