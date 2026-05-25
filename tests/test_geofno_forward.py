import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import GeoFNO2d  # noqa: E402


def _build():
    return GeoFNO2d(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=120,
        out_channels=1,
        s1=4,
        s2=4,
        num_fno_layers=1,
    )


def test_geofno_forward_shape():
    model = _build()
    batch_size, num_nodes = 1, 5
    u = torch.randn(batch_size, num_nodes, 120)
    x = torch.rand(batch_size, num_nodes, 2)
    out = model(u, x)
    assert out.shape == (batch_size, num_nodes, 1)


def test_geofno_forward_shape_batch_2():
    model = _build()
    batch_size, num_nodes = 2, 4
    u = torch.randn(batch_size, num_nodes, 120)
    x = torch.rand(batch_size, num_nodes, 2)
    out = model(u, x)
    assert out.shape == (batch_size, num_nodes, 1)


def test_geofno_forward_is_direct_prediction_not_residual():
    """Zeroing fc2 should give zero output (no residual base)."""
    model = _build()
    batch_size, num_nodes = 1, 3
    u = torch.randn(batch_size, num_nodes, 120)
    x = torch.rand(batch_size, num_nodes, 2)
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
    out = model(u, x)
    assert torch.allclose(out, torch.zeros(batch_size, num_nodes, 1), atol=1e-5)
