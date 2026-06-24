import numpy as np

from ct_eus_vessel.thresholds import estimate_vessel_hu_window


def test_estimate_vessel_hu_window_uses_bright_soft_tissue_tail() -> None:
    body_hu = np.array([-100, 30, 35, 42, 90, 120, 160, 210, 240, 1000], dtype=np.float32)
    mask = np.ones(body_hu.shape, dtype=bool)
    organ_exclusion = body_hu >= 900

    window = estimate_vessel_hu_window(body_hu, mask, organ_exclusion)

    assert 90 <= window.low <= 130
    assert 200 <= window.high <= 260
    assert window.low < window.high
