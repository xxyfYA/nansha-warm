import sys
from pathlib import Path

import torch

# Allow tests to import modules from model/ directly; the project has no setup.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from model import SpectralConv2d  # noqa: E402


def test_batch_shared_coordinates_build_basis_once_per_node(monkeypatch):
    conv = SpectralConv2d(in_channels=2, out_channels=3, modes1=2, modes2=2)
    batch_size = 4
    num_nodes = 5
    coords = torch.rand(num_nodes, 2).unsqueeze(0).expand(batch_size, -1, -1)

    outer_input_sizes = []
    original_outer = torch.outer

    def recording_outer(left, right):
        outer_input_sizes.append(left.numel())
        return original_outer(left, right)

    monkeypatch.setattr(torch, "outer", recording_outer)

    u = torch.randn(batch_size, 2, num_nodes)
    conv.fft2d(u, coords)
    assert outer_input_sizes == [num_nodes, num_nodes]

    outer_input_sizes.clear()
    u_ft = torch.randn(batch_size, 3, 2 * conv.modes1, conv.modes2, dtype=torch.cfloat)
    conv.ifft2d(u_ft, coords)
    assert outer_input_sizes == [num_nodes, num_nodes]


def test_batch_shared_coordinate_outputs_match_batched_fallback():
    conv = SpectralConv2d(in_channels=2, out_channels=3, modes1=2, modes2=2)
    batch_size = 3
    num_nodes = 7
    coords_base = torch.rand(num_nodes, 2)
    coords_shared = coords_base.unsqueeze(0).expand(batch_size, -1, -1)
    coords_repeated = coords_base.unsqueeze(0).repeat(batch_size, 1, 1)

    u = torch.randn(batch_size, 2, num_nodes)
    shared_fft = conv.fft2d(u, coords_shared)
    fallback_fft = conv.fft2d(u, coords_repeated)
    assert torch.allclose(shared_fft, fallback_fft, atol=1e-5, rtol=1e-5)

    u_ft = torch.randn(batch_size, 3, 2 * conv.modes1, conv.modes2, dtype=torch.cfloat)
    shared_ifft = conv.ifft2d(u_ft, coords_shared)
    fallback_ifft = conv.ifft2d(u_ft, coords_repeated)
    assert torch.allclose(shared_ifft, fallback_ifft, atol=1e-5, rtol=1e-5)
