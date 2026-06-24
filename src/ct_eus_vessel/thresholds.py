from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HUWindow:
    low: float
    high: float


def estimate_vessel_hu_window(
    hu_volume: np.ndarray,
    body_mask: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    floor: float = 80.0,
    ceiling: float = 300.0,
) -> HUWindow:
    usable = np.asarray(body_mask, dtype=bool).copy()
    if exclusion_mask is not None:
        usable &= ~np.asarray(exclusion_mask, dtype=bool)

    values = np.asarray(hu_volume, dtype=np.float32)[usable]
    values = values[np.isfinite(values)]
    values = values[(values >= -100) & (values <= ceiling)]
    if values.size == 0:
        return HUWindow(low=floor, high=ceiling)

    low = float(np.percentile(values, 60))
    high = float(np.percentile(values, 98))
    low = max(floor, min(low, ceiling - 20))
    high = min(ceiling, max(high, low + 120))
    return HUWindow(low=low, high=high)
