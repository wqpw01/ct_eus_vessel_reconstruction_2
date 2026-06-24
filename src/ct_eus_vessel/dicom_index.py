from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

from .phase import classify_phase, infer_dynamic_phases


def index_dicom_series(root: str | Path) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = defaultdict(int)
    root_path = Path(root)

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not uid:
            continue
        counts[uid] += 1
        if uid not in groups:
            description = str(getattr(ds, "SeriesDescription", "") or "")
            protocol = str(getattr(ds, "ProtocolName", "") or "")
            groups[uid] = {
                "series_uid": uid,
                "series_description": description,
                "protocol_name": protocol,
                "study_date": str(getattr(ds, "StudyDate", "") or ""),
                "series_number": str(getattr(ds, "SeriesNumber", "") or ""),
                "phase": classify_phase(description, protocol),
                "first_file": str(path),
            }

    rows = []
    for uid, item in groups.items():
        row = dict(item)
        row["num_instances"] = counts[uid]
        rows.append(row)
    inferred = infer_dynamic_phases(rows)
    return sorted(inferred, key=lambda row: (-int(row["num_instances"]), str(row["series_number"])))


def write_series_index_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series_uid",
        "phase",
        "metadata_phase",
        "dynamic_phase",
        "num_instances",
        "series_description",
        "protocol_name",
        "study_date",
        "series_number",
        "first_file",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path
