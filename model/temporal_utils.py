"""Fixed-window constants for Geo-FNO warm-up prediction model.

Input:  24-hour forcing window (storm + inner boundary)
Output: water level h at hour t+24 (single-step direct prediction)
"""

INPUT_WINDOW: int = 24
PREDICT_OFFSET: int = 24
TOTAL_STEPS_NEEDED: int = INPUT_WINDOW + 1  # 25
C_IN: int = INPUT_WINDOW * 3 + INPUT_WINDOW * 2  # 120
C_OUT: int = 1
CHANNEL_NAME: str = "h"


def build_checkpoint_name() -> str:
    return "best_geofno.pt"
