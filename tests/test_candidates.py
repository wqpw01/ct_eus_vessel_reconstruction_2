import numpy as np

from ct_eus_vessel.candidates import extract_vessel_candidates
from ct_eus_vessel.thresholds import HUWindow


def test_extract_vessel_candidates_uses_hu_window_and_exclusion_mask() -> None:
    volume = np.full((5, 8, 8), 35, dtype=np.float32)
    body = np.ones_like(volume, dtype=bool)
    exclusion = np.zeros_like(volume, dtype=bool)
    volume[:, 3, 3] = 180
    volume[1:4, 6, 6] = 1000
    exclusion[1:4, 6, 6] = True

    result = extract_vessel_candidates(
        volume,
        body_mask=body,
        exclusion_mask=exclusion,
        min_component_voxels=3,
    )

    assert result.mask[:, 3, 3].all()
    assert not result.mask[2, 6, 6]
    assert result.bbox == ((0, 5), (3, 4), (3, 4))


def test_extract_vessel_candidates_can_gate_by_vesselness() -> None:
    volume = np.full((5, 8, 8), 35, dtype=np.float32)
    body = np.ones_like(volume, dtype=bool)
    volume[:, 2, 2] = 170
    volume[:, 5, 1:6] = 170
    vesselness = np.zeros_like(volume, dtype=np.float32)
    vesselness[:, 2, 2] = 0.9
    vesselness[:, 5, 1:6] = 0.1

    result = extract_vessel_candidates(
        volume,
        body_mask=body,
        vesselness=vesselness,
        vesselness_min=0.5,
        min_component_voxels=3,
    )

    assert result.mask[:, 2, 2].all()
    assert not result.mask[:, 5, 1:6].any()


def test_extract_vessel_candidates_honors_explicit_hu_window() -> None:
    volume = np.full((3, 5, 5), 40, dtype=np.float32)
    body = np.ones_like(volume, dtype=bool)
    volume[:, 2, 2] = 180

    result = extract_vessel_candidates(
        volume,
        body_mask=body,
        min_component_voxels=1,
        hu_window=HUWindow(low=170, high=190),
    )

    assert result.mask[:, 2, 2].all()
    assert int(result.mask.sum()) == 3
