import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from temporal_utils import (  # noqa: E402
    C_IN,
    C_OUT,
    CHANNEL_NAME,
    INPUT_WINDOW,
    PREDICT_OFFSET,
    TOTAL_STEPS_NEEDED,
    build_checkpoint_name,
)


def test_constants():
    assert INPUT_WINDOW == 24
    assert PREDICT_OFFSET == 24
    assert TOTAL_STEPS_NEEDED == 25
    assert C_IN == 120
    assert C_OUT == 1


def test_channel_name_constant():
    assert CHANNEL_NAME == "h"


def test_build_checkpoint_name():
    assert build_checkpoint_name() == "best_geofno.pt"
