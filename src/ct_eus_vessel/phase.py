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


def infer_dynamic_phases(series: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(item) for item in series]
    candidates = [row for row in rows if _is_dynamic_contrast_series(row)]
    if len(candidates) < 3:
        for row in rows:
            row.setdefault("metadata_phase", row.get("phase", "other"))
            row.setdefault("dynamic_phase", "")
        return rows

    ordered = sorted(candidates, key=_series_order_key)
    middle = ordered[1:-1]
    portal = _choose_portal_candidate(middle)

    dynamic_by_uid: dict[str, str] = {str(ordered[0].get("series_uid", "")): "arterial"}
    dynamic_by_uid[str(ordered[-1].get("series_uid", ""))] = "venous"
    if portal is not None:
        dynamic_by_uid[str(portal.get("series_uid", ""))] = "portal"
    for row in middle:
        dynamic_by_uid.setdefault(str(row.get("series_uid", "")), "portal")

    for row in rows:
        metadata_phase = str(row.get("phase", "other"))
        dynamic_phase = dynamic_by_uid.get(str(row.get("series_uid", "")), "")
        row["metadata_phase"] = metadata_phase
        row["dynamic_phase"] = dynamic_phase
        if dynamic_phase:
            row["phase"] = dynamic_phase
    return rows


def _num_instances(item: Mapping[str, Any]) -> int:
    try:
        return int(item.get("num_instances", 0))
    except (TypeError, ValueError):
        return 0


def _is_dynamic_contrast_series(item: Mapping[str, Any]) -> bool:
    description = str(item.get("series_description", "") or "").casefold()
    protocol = str(item.get("protocol_name", "") or "").casefold()
    text = f"{description} {protocol}"
    blocked = ("mip", "scout", "dose", "lung", "med", "report")
    if any(token in text for token in blocked):
        return False
    if "1.0" not in description or "_a" not in description:
        return False
    return _num_instances(item) >= 100


def _series_order_key(item: Mapping[str, Any]) -> tuple[float, str]:
    raw = item.get("series_number", "")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        number = float("inf")
    return number, str(item.get("series_uid", ""))


def _choose_portal_candidate(items: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not items:
        return None
    return max(items, key=lambda item: (_num_instances(item), -_series_order_key(item)[0]))
