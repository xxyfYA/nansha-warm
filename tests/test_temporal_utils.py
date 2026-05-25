import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from temporal_utils import (  # noqa: E402
    CHANNEL_NAME,
    TemporalConfig,
    build_checkpoint_name,
    build_run_suffix,
    input_channels_for_bundle,
    num_temporal_samples,
    output_channels_for_bundle,
    validate_temporal_params,
)


def test_input_channels_formula():
    # C_in = 5*S + 9
    assert input_channels_for_bundle(1) == 14
    assert input_channels_for_bundle(8) == 49
    assert input_channels_for_bundle(24) == 129
    assert input_channels_for_bundle(72) == 369


def test_output_channels_formula():
    assert output_channels_for_bundle(1) == 1
    assert output_channels_for_bundle(8) == 8
    assert output_channels_for_bundle(24) == 24
    assert output_channels_for_bundle(72) == 72


def test_num_temporal_samples_basic():
    assert num_temporal_samples(100, 72) == 28
    assert num_temporal_samples(73, 72) == 1


def test_num_temporal_samples_too_short_raises():
    with pytest.raises(ValueError):
        num_temporal_samples(72, 72)
    with pytest.raises(ValueError):
        num_temporal_samples(10, 72)


def test_validate_temporal_params_rejects_nonpositive():
    with pytest.raises(ValueError):
        validate_temporal_params(0)
    with pytest.raises(ValueError):
        validate_temporal_params(-1)


def test_channel_name_constant():
    assert CHANNEL_NAME == "h"


def test_temporal_config_basic():
    cfg = TemporalConfig(bundle_size=8)
    assert cfg.bundle_size == 8
    assert cfg.required_future_steps == 8
    assert cfg.input_channels == 49
    assert cfg.out_channels == 8


def test_temporal_config_default():
    cfg = TemporalConfig()
    assert cfg.bundle_size == 1
    assert cfg.input_channels == 14
    assert cfg.out_channels == 1


def test_temporal_config_rejects_invalid_bundle():
    with pytest.raises(ValueError):
        TemporalConfig(bundle_size=0)


def test_build_checkpoint_name():
    assert build_checkpoint_name(1) == "best_geofno.pt"
    assert build_checkpoint_name(8) == "best_geofno_b8.pt"
    assert build_checkpoint_name(72) == "best_geofno_b72.pt"


def test_build_run_suffix():
    assert build_run_suffix(1) == ""
    assert build_run_suffix(8) == "_b8"
    assert build_run_suffix(72) == "_b72"
