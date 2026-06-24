from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def bounding_box_zyx(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple((int(lo), int(hi)) for lo, hi in zip(mins, maxs, strict=True))  # type: ignore[return-value]


def keep_components_by_size(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if min_voxels <= 1:
        return binary.copy()

    labels, count = ndi.label(binary)
    if count == 0:
        return np.zeros_like(binary, dtype=bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_voxels)
    keep[0] = False
    return keep[labels]
