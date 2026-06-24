from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_PHASE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "arterial": ("arterial", "artery", "动脉", " a ", "_a"),
    "portal": ("portal", "门静脉", "pv"),
    "venous": ("venous", "vein", "delayed", "静脉"),
}


def classify_phase(
    series_description: str | None,
    protocol_name: str | None = None,
    keywords: Mapping[str, Iterable[str]] | None = None,
) -> str:
    text = f" {series_description or ''} {protocol_name or ''} ".casefold()
    phase_keywords = keywords or DEFAULT_PHASE_KEYWORDS
    for phase in ("arterial", "portal", "venous"):
        for keyword in phase_keywords.get(phase, ()):
            lowered = keyword.casefold()
            if lowered in {" a ", "_a"}:
                continue
            if lowered in text:
                return phase
    for keyword in phase_keywords.get("arterial", ()):
        if keyword.casefold() in {" a ", "_a"} and keyword.casefold() in text:
            return "arterial"
    return "other"


def choose_primary_series(series: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for item in series:
        phase = str(item.get("phase", "other"))
        if phase == "other":
            continue
        current = selected.get(phase)
        if current is None or _num_instances(item) > _num_instances(current):
            selected[phase] = item
    return selected


def _num_instances(item: Mapping[str, Any]) -> int:
    try:
        return int(item.get("num_instances", 0))
    except (TypeError, ValueError):
        return 0
