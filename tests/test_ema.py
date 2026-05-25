import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from train import ExponentialMovingAverage  # noqa: E402


class _TinyModel(torch.nn.Module):
    def __init__(self, weight: float = 0.0, buffer: float = 0.0, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([weight], dtype=dtype))
        self.register_buffer("running", torch.tensor([buffer], dtype=dtype))


class _DDPStyleWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_ema_init_shadow_matches_live_and_is_deep_copy():
    model = _TinyModel(weight=3.0, buffer=7.0)

    ema = ExponentialMovingAverage(model, decay=0.9)

    assert ema.shadow is not model
    assert ema.shadow.weight.item() == pytest.approx(model.weight.item())
    assert ema.shadow.running.item() == pytest.approx(model.running.item())
    assert ema.shadow.weight.data_ptr() != model.weight.data_ptr()
    assert ema.shadow.running.data_ptr() != model.running.data_ptr()


def test_ema_decay_half_two_updates_match_expected_values():
    model = _TinyModel(weight=0.0)
    ema = ExponentialMovingAverage(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update()
    assert ema.shadow.weight.item() == pytest.approx(1.0)

    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update()
    assert ema.shadow.weight.item() == pytest.approx(2.5)


def test_ema_decay_high_multiple_updates_close_to_theory():
    model = _TinyModel(weight=0.0)
    ema = ExponentialMovingAverage(model, decay=0.999)

    with torch.no_grad():
        model.weight.fill_(1.0)

    steps = 3000
    for _ in range(steps):
        ema.update()

    expected = 1.0 - (0.999 ** steps)
    assert ema.shadow.weight.item() == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_ema_shadow_parameters_do_not_require_grad():
    model = _TinyModel(weight=1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)

    assert all(not p.requires_grad for p in ema.shadow.parameters())
    assert all(p.grad is None for p in ema.shadow.parameters())


def test_ema_update_copies_buffers_from_live_model():
    model = _TinyModel(weight=0.0, buffer=1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)

    with torch.no_grad():
        model.running.fill_(5.0)
    ema.update()

    assert ema.shadow.running.item() == pytest.approx(5.0)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 2.0])
def test_ema_invalid_decay_raises_value_error(decay):
    with pytest.raises(ValueError, match="EMA decay"):
        ExponentialMovingAverage(_TinyModel(), decay=decay)


def test_ema_update_works_with_ddp_style_module_wrapper():
    wrapped = _DDPStyleWrapper(_TinyModel(weight=0.0))
    ema = ExponentialMovingAverage(wrapped, decay=0.5)

    with torch.no_grad():
        wrapped.module.weight.fill_(2.0)
    ema.update()

    assert ema.shadow.weight.item() == pytest.approx(1.0)
