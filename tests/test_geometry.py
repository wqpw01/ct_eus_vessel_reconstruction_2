import numpy as np

from ct_eus_vessel.geometry import bounding_box_zyx, keep_components_by_size


def test_bounding_box_zyx_returns_half_open_bounds() -> None:
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[1:4, 2:5, 3:6] = True

    assert bounding_box_zyx(mask) == ((1, 4), (2, 5), (3, 6))


def test_keep_components_by_size_filters_tiny_noise() -> None:
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0, 0, 0] = True
    mask[2:4, 2:4, 2:4] = True

    cleaned = keep_components_by_size(mask, min_voxels=4)

    assert cleaned[0, 0, 0] is np.False_
    assert cleaned[2:4, 2:4, 2:4].all()
