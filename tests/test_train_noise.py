"""Unit tests for h-channel input noise injection in train.py."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from train import _apply_h_state_noise  # noqa: E402


def _make_features(batch=2, num_nodes=4, channels=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, num_nodes, channels, generator=g)


def test_apply_h_state_noise_zero_sigma_is_noop():
    features = _make_features()
    original = features.clone()
    _apply_h_state_noise(features, noise_sigma=0.0)
    assert torch.equal(features, original)


def test_apply_h_state_noise_negative_sigma_is_noop():
    features = _make_features()
    original = features.clone()
    _apply_h_state_noise(features, noise_sigma=-0.1)
    assert torch.equal(features, original)


def test_apply_h_state_noise_only_touches_h_channel():
    features = _make_features()
    other_channels_before = features[..., 1:].clone()
    rng_state = torch.random.get_rng_state()
    try:
        _apply_h_state_noise(features, noise_sigma=0.05)
    finally:
        torch.random.set_rng_state(rng_state)
    assert torch.equal(features[..., 1:], other_channels_before)


def test_apply_h_state_noise_magnitude_matches_sigma():
    features = torch.zeros(64, 512, 9)
    sigma = 0.1
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(42)
        _apply_h_state_noise(features, noise_sigma=sigma)
    finally:
        torch.random.set_rng_state(rng_state)
    h_after = features[..., 0:1]
    assert h_after.mean().abs().item() < 0.01
    assert abs(h_after.std().item() - sigma) < 0.01


from train import train_model  # noqa: E402


def test_train_model_rejects_negative_noise_sigma():
    with pytest.raises(ValueError, match="noise_sigma"):
        train_model(
            model=None,
            train_loader=None,
            test_loader=None,
            num_epochs=0,
            device=None,
            optimizer=None,
            scheduler=None,
            coords_2d_device=None,
            writer=None,
            noise_sigma=-0.1,
        )
