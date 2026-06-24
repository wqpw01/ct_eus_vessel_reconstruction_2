from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HUWindow:
    low: float
    high: float


PHASE_HU_LIMITS: dict[str, tuple[float, float]] = {
    "arterial": (150.0, 650.0),
    "portal": (115.0, 280.0),
    "venous": (105.0, 260.0),
}


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


def estimate_phase_hu_window(
    phase: str,
    hu_volume: np.ndarray,
    body_mask: np.ndarray,
    exclusion_mask: np.ndarray | None,
    seed_mask: np.ndarray,
) -> HUWindow:
    floor, ceiling = PHASE_HU_LIMITS.get(phase, (80.0, 320.0))
    seed_values = _finite_values(hu_volume, np.asarray(body_mask, dtype=bool) & np.asarray(seed_mask, dtype=bool))
    seed_values = seed_values[(seed_values >= floor - 80) & (seed_values <= ceiling)]
    if seed_values.size < 4:
        return estimate_vessel_hu_window(
            hu_volume,
            body_mask=body_mask,
            exclusion_mask=exclusion_mask,
            floor=floor,
            ceiling=ceiling,
        )

    if phase == "arterial":
        low = float(np.percentile(seed_values, 10)) - 20.0
        high = float(np.percentile(seed_values, 95)) + 80.0
        min_width = 180.0
    elif phase == "portal":
        low = float(np.percentile(seed_values, 25)) - 20.0
        high = float(np.percentile(seed_values, 95)) + 40.0
        min_width = 90.0
    elif phase == "venous":
        low = float(np.percentile(seed_values, 25)) - 15.0
        high = float(np.percentile(seed_values, 95)) + 40.0
        min_width = 80.0
    else:
        low = float(np.percentile(seed_values, 20)) - 20.0
        high = float(np.percentile(seed_values, 95)) + 40.0
        min_width = 100.0

    low = max(floor, min(low, ceiling - min_width))
    high = min(ceiling, max(high, low + min_width))
    return HUWindow(low=float(low), high=float(high))


def _finite_values(hu_volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(hu_volume, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    return values[np.isfinite(values)]
