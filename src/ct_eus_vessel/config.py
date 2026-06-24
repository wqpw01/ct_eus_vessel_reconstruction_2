from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    case_id: str
    raw_data: Path
    reference_labels: Path
    output_root: Path
    work_root: Path
    arterial_keywords: tuple[str, ...]
    portal_keywords: tuple[str, ...]
    venous_keywords: tuple[str, ...]
    required_phases: tuple[str, ...]
    reference_labels_eval_only: bool = True


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    project = data.get("project", {})
    paths = data.get("paths", {})
    phase = data.get("phase", {})

    raw_data = _required_path(paths, "raw_data")
    reference_labels = _required_path(paths, "reference_labels")
    output_root = _required_path(paths, "output_root")
    work_root = _required_path(paths, "work_root")

    if reference_labels.resolve(strict=False) == output_root.resolve(strict=False):
        raise ValueError("reference_labels must not be the same path as output_root")
    if reference_labels.resolve(strict=False) == work_root.resolve(strict=False):
        raise ValueError("reference_labels must not be the same path as work_root")

    return ProjectConfig(
        case_id=str(project.get("case_id", "")).strip() or "unknown-case",
        raw_data=raw_data,
        reference_labels=reference_labels,
        output_root=output_root,
        work_root=work_root,
        arterial_keywords=_as_tuple(phase.get("arterial_keywords", ("arterial", "artery"))),
        portal_keywords=_as_tuple(phase.get("portal_keywords", ("portal", "pv"))),
        venous_keywords=_as_tuple(phase.get("venous_keywords", ("venous", "vein"))),
        required_phases=_as_tuple(phase.get("required_phases", ("arterial", "portal"))),
    )


def _required_path(mapping: dict[str, Any], key: str) -> Path:
    value = mapping.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required path: {key}")
    return Path(value).expanduser()


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
