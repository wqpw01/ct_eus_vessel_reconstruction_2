from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import bounding_box_zyx, keep_components_by_size
from .thresholds import HUWindow, estimate_vessel_hu_window


@dataclass(frozen=True)
class CandidateResult:
    mask: np.ndarray
    hu_window: HUWindow
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None


def extract_vessel_candidates(
    hu_volume: np.ndarray,
    body_mask: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    vesselness: np.ndarray | None = None,
    vesselness_min: float | None = None,
    min_component_voxels: int = 8,
    hu_floor: float = 80.0,
    hu_ceiling: float = 300.0,
    hu_window: HUWindow | None = None,
) -> CandidateResult:
    body = np.asarray(body_mask, dtype=bool)
    exclusion = np.zeros_like(body, dtype=bool) if exclusion_mask is None else np.asarray(exclusion_mask, dtype=bool)
    hu = np.asarray(hu_volume, dtype=np.float32)
    if hu.shape != body.shape or exclusion.shape != body.shape:
        raise ValueError("hu_volume, body_mask and exclusion_mask must have the same shape")

    window = hu_window or estimate_vessel_hu_window(
        hu,
        body_mask=body,
        exclusion_mask=exclusion,
        floor=hu_floor,
        ceiling=hu_ceiling,
    )
    mask = body & ~exclusion & (hu >= window.low) & (hu <= window.high)

    if vesselness is not None and vesselness_min is not None:
        response = np.asarray(vesselness, dtype=np.float32)
        if response.shape != body.shape:
            raise ValueError("vesselness must have the same shape as hu_volume")
        mask &= response >= float(vesselness_min)

    cleaned = keep_components_by_size(mask, min_voxels=min_component_voxels)
    return CandidateResult(mask=cleaned, hu_window=window, bbox=bounding_box_zyx(cleaned))
