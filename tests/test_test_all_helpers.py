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
