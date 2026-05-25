import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import GeoFNO2d  # noqa: E402


def _build(fc1_hidden=256, in_channels=120, out_channels=1):
    return GeoFNO2d(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=in_channels,
        out_channels=out_channels,
        s1=4,
        s2=4,
        num_fno_layers=1,
        fc1_hidden=fc1_hidden,
    )


def test_fc1_hidden_default_is_256():
    model = GeoFNO2d(
        modes1=2,
        modes2=2,
        width=4,
        in_channels=120,
        out_channels=1,
        s1=4,
        s2=4,
        num_fno_layers=1,
    )
    assert model.fc1.out_features == 256
    assert model.fc2.in_features == 256


def test_fc1_hidden_can_be_set_to_128():
    model = _build(fc1_hidden=128)
    assert model.fc1.out_features == 128
    assert model.fc2.in_features == 128


def test_forward_shape_unchanged_when_fc1_hidden_changes():
    model_default = _build(fc1_hidden=256)
    model_custom = _build(fc1_hidden=128)

    batch_size, num_nodes = 1, 5
    u = torch.randn(batch_size, num_nodes, 120)
    x = torch.rand(batch_size, num_nodes, 2)

    out_default = model_default(u, x)
    out_custom = model_custom(u, x)

    assert out_default.shape == (batch_size, num_nodes, 1)
    assert out_custom.shape == (batch_size, num_nodes, 1)


def test_parameter_count_increases_with_fc1_hidden():
    model_default = _build(fc1_hidden=128)
    model_custom = _build(fc1_hidden=256)

    params_default = sum(p.numel() for p in model_default.parameters())
    params_custom = sum(p.numel() for p in model_custom.parameters())

    assert params_custom > params_default
