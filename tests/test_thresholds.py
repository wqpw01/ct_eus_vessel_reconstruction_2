import numpy as np

from ct_eus_vessel.thresholds import estimate_phase_hu_window, estimate_vessel_hu_window


def test_estimate_vessel_hu_window_uses_bright_soft_tissue_tail() -> None:
    body_hu = np.array([-100, 30, 35, 42, 90, 120, 160, 210, 240, 1000], dtype=np.float32)
    mask = np.ones(body_hu.shape, dtype=bool)
    organ_exclusion = body_hu >= 900

    window = estimate_vessel_hu_window(body_hu, mask, organ_exclusion)

    assert 90 <= window.low <= 130
    assert 200 <= window.high <= 260
    assert window.low < window.high


def test_estimate_phase_hu_window_uses_arterial_seed_and_literature_floor() -> None:
    hu = np.array([40, 70, 120, 180, 240, 410, 460, 520, 560], dtype=np.float32)
    body = np.ones_like(hu, dtype=bool)
    exclusion = np.zeros_like(hu, dtype=bool)
    seed = hu >= 410

    window = estimate_phase_hu_window("arterial", hu, body, exclusion, seed)

    assert window.low >= 150
    assert window.high >= 500
    assert window.high <= 650


def test_estimate_phase_hu_window_uses_portal_seed_and_excludes_liver_tail() -> None:
    hu = np.array([60, 90, 110, 125, 135, 145, 160, 178, 190, 205, 220], dtype=np.float32)
    body = np.ones_like(hu, dtype=bool)
    exclusion = np.zeros_like(hu, dtype=bool)
    seed = hu >= 145

    window = estimate_phase_hu_window("portal", hu, body, exclusion, seed)

    assert 110 <= window.low <= 150
    assert 180 <= window.high <= 280
