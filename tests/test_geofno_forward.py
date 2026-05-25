import sys
from pathlib import Path

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
    batch_size, num_nodes = 1, 5
    u = torch.randn(batch_size, num_nodes, 5 * 2 + 9)
    x = torch.rand(batch_size, num_nodes, 2)
    out = model(u, x)
    assert out.shape == (batch_size, 2, num_nodes, 1)


def test_geofno_forward_shape_bundle_3():
    model = _build(bundle_size=3)
    batch_size, num_nodes = 2, 4
    u = torch.randn(batch_size, num_nodes, 5 * 3 + 9)
    x = torch.rand(batch_size, num_nodes, 2)
    out = model(u, x)
    assert out.shape == (batch_size, 3, num_nodes, 1)


def test_geofno_residual_uses_first_column_as_state():
    """Zeroing fc2 should leave the residual base equal to features[..., :1]."""
    model = _build(bundle_size=2)
    batch_size, num_nodes = 1, 3
    u = torch.zeros(batch_size, num_nodes, 5 * 2 + 9)
    u[..., 0] = 1.234
    x = torch.rand(batch_size, num_nodes, 2)
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
    out = model(u, x)
    assert torch.allclose(out, torch.full((batch_size, 2, num_nodes, 1), 1.234), atol=1e-5)
