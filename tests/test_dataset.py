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
        coords, btype = load_static_coords(coords_mat)
    assert coords.shape == (N_NODES, 2)
    assert coords.dtype == torch.float32
    shared_memory_warnings = [
        warning for warning in warnings_record
        if issubclass(warning.category, RuntimeWarning)
        and "shared memory" in str(warning.message)
    ]
    if shared_memory_warnings:
        assert not coords.is_shared()
        assert not btype.is_shared()
    else:
        assert coords.is_shared()
        assert btype.is_shared()
    assert torch.all(coords >= 0.0) and torch.all(coords <= 1.0)
    assert torch.isclose(coords.min(0).values, torch.zeros(2)).all()
    assert torch.isclose(coords.max(0).values, torch.ones(2)).all()
    assert btype.shape == (N_NODES, 3)
    assert torch.all(btype.sum(dim=1) == 1.0)
    assert torch.all(btype[:8, 1] == 1.0)
    assert torch.all(btype[8:12, 2] == 1.0)
    assert torch.all(btype[12:, 0] == 1.0)


def test_load_static_coords_warns_and_returns_cpu_tensors_when_share_memory_fails(
    coords_mat, monkeypatch
):
    original_share_memory = torch.Tensor.share_memory_

    def fail_once(self):
        raise RuntimeError("shm unavailable")

    monkeypatch.setattr(torch.Tensor, "share_memory_", fail_once)
    with pytest.warns(RuntimeWarning, match="shared memory"):
        coords, btype = load_static_coords(coords_mat)

    assert coords.shape == (N_NODES, 2)
    assert btype.shape == (N_NODES, 3)
    assert not coords.is_shared()
    assert not btype.is_shared()
    monkeypatch.setattr(torch.Tensor, "share_memory_", original_share_memory)


def test_load_static_coords_rejects_bad_boundary(tmp_path):
    p = tmp_path / "bad.mat"
    scipy.io.savemat(
        p,
        {
            "coordinates": np.zeros((4, 3)),
            "boundary": np.array([[0], [3], [0], [0]], dtype=np.int8),
        },
    )
    with pytest.raises(ValueError, match="boundary"):
        load_static_coords(p)


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
    assert feat.shape == (N_NODES, 29)
    assert target.shape == (4, N_NODES, 1)


def test_single_dataset_btype_concatenated(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    ds = StormSurgeDataset(
        path=split_dir / "e0.pt",
        bundle_size=2,
        btype_oh=btype,
        lru_capacity=1,
    )
    feat, _ = ds[0]
    assert torch.allclose(feat[:, -3:], btype)


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
    assert feat.shape == (N_NODES, 29)
    assert target.shape == (4, N_NODES, 1)


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
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(d, bundle_size=20, btype_oh=btype, lru_files_per_worker=1)
    assert len(mds) == 30


def test_file_chunked_sampler_groups_by_file(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(split_dir, bundle_size=4, btype_oh=btype, lru_files_per_worker=1)
    sampler = FileChunkedDistributedSampler(mds, num_replicas=1, rank=0, shuffle=True, seed=0)
    indices = list(iter(sampler))
    assert len(indices) == len(mds)
    file_seq = [mds.flat_index[i][0] for i in indices]
    transitions = sum(1 for a, b in zip(file_seq, file_seq[1:]) if a != b)
    assert transitions <= len(set(file_seq)) - 1 + 2


def test_file_chunked_sampler_disjoint_across_ranks(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(split_dir, bundle_size=4, btype_oh=btype, lru_files_per_worker=1)
    s0 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=0, shuffle=False, seed=0)
    s1 = FileChunkedDistributedSampler(mds, num_replicas=2, rank=1, shuffle=False, seed=0)
    i0 = set(iter(s0))
    i1 = set(iter(s1))
    assert i0.isdisjoint(i1)
    f0 = {mds.flat_index[i][0] for i in i0}
    f1 = {mds.flat_index[i][0] for i in i1}
    assert f0.isdisjoint(f1)


def test_file_chunked_sampler_len_is_stable_across_epochs(uneven_split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(uneven_split_dir, bundle_size=4, btype_oh=btype)
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

    assert lengths == [60] * 8


def test_balanced_sampler_drop_last_uses_min_rank_total_without_duplicates(uneven_split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(uneven_split_dir, bundle_size=4, btype_oh=btype)
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
    assert [len(indices) for indices in rank_indices] == [60, 60]
    assert [len(sampler) for sampler in samplers] == [60, 60]
    assert all(len(indices) == len(set(indices)) for indices in rank_indices)
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])

    all_used = set().union(*(set(indices) for indices in rank_indices))
    rank_file_counts = [
        {
            file_idx: sum(1 for index in indices if mds.flat_index[index][0] == file_idx)
            for file_idx in sorted({mds.flat_index[index][0] for index in indices})
        }
        for indices in rank_indices
    ]
    assert rank_file_counts == [{0: 60}, {1: 50, 2: 10}]
    assert all_used == set(range(60)) | set(range(100, 160))


def test_balanced_sampler_drop_last_false_pads_short_rank(uneven_split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(uneven_split_dir, bundle_size=4, btype_oh=btype)
    samplers = [
        FileChunkedDistributedSampler(
            mds,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            seed=0,
            drop_last=False,
        )
        for rank in range(2)
    ]

    rank_indices = [list(iter(sampler)) for sampler in samplers]
    assert [len(indices) for indices in rank_indices] == [100, 100]
    assert [len(sampler) for sampler in samplers] == [100, 100]
    assert len(set(rank_indices[0])) == 100
    assert len(set(rank_indices[1])) == 60
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])


def test_balanced_sampler_no_padding_covers_all_samples_without_duplicates(
    uneven_split_dir, coords_mat
):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(uneven_split_dir, bundle_size=4, btype_oh=btype)
    samplers = [
        FileChunkedDistributedSampler(
            mds,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            seed=0,
            drop_last=False,
            pad_to_equal_length=False,
        )
        for rank in range(2)
    ]

    rank_indices = [list(iter(sampler)) for sampler in samplers]
    assert [len(indices) for indices in rank_indices] == [100, 60]
    assert [len(sampler) for sampler in samplers] == [100, 60]
    assert all(len(indices) == len(set(indices)) for indices in rank_indices)
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert set().union(*(set(indices) for indices in rank_indices)) == set(range(len(mds)))


def test_balanced_sampler_raises_when_a_rank_has_no_samples(tmp_path, coords_mat):
    d = tmp_path / "single"
    d.mkdir()
    _make_pt(d / "only.pt", 20)
    manifest = {
        "num_nodes": N_NODES,
        "files": [{"path": "only.pt", "T": 20}],
        "created_at": "2026-05-16T00:00:00+00:00",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(d, bundle_size=4, btype_oh=btype)
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


def test_file_chunked_sampler_set_epoch_changes_shuffle(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(split_dir, bundle_size=4, btype_oh=btype, lru_files_per_worker=1)
    sampler = FileChunkedDistributedSampler(mds, num_replicas=1, rank=0, shuffle=True, seed=0)
    epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))
    assert len(epoch0) == len(epoch1) == len(mds)
    assert epoch0 != epoch1


def test_lru_eviction(split_dir, coords_mat):
    _, btype = load_static_coords(coords_mat)
    mds = MultiStormSurgeDataset(split_dir, bundle_size=4, btype_oh=btype, lru_files_per_worker=2)
    for fi in range(3):
        first_idx_for_file = next(i for i, (f, _) in enumerate(mds.flat_index) if f == fi)
        _ = mds[first_idx_for_file]
    assert len(mds._cache) <= 2

