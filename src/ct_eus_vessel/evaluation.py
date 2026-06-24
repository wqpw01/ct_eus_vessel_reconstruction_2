from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


DEFAULT_VESSEL_LABEL_IDS = {8, 9, 10, 15, 25}


@dataclass(frozen=True)
class LabelInfo:
    index: int
    name: str


def parse_itksnap_label_file(path: str | Path) -> dict[int, LabelInfo]:
    labels: dict[int, LabelInfo] = {}
    pattern = re.compile(r"^\s*(\d+)\s+.*?\"(.+)\"\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        index = int(match.group(1))
        labels[index] = LabelInfo(index=index, name=match.group(2))
    return labels


def compute_binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    return {
        "prediction_voxels": pred_voxels,
        "target_voxels": gt_voxels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": _safe_divide(2 * tp, pred_voxels + gt_voxels),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
    }


def compute_hausdorff_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    reference_image: sitk.Image,
) -> dict[str, float | None]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if not pred.any() or not gt.any():
        return {"hausdorff_mm": None, "average_hausdorff_mm": None}

    pred_img = _mask_to_reference_image(pred, reference_image)
    gt_img = _mask_to_reference_image(gt, reference_image)
    hausdorff = sitk.HausdorffDistanceImageFilter()
    hausdorff.Execute(pred_img, gt_img)
    return {
        "hausdorff_mm": float(hausdorff.GetHausdorffDistance()),
        "average_hausdorff_mm": float(hausdorff.GetAverageHausdorffDistance()),
    }


def summarize_label_overlaps(
    prediction: np.ndarray,
    label_image: np.ndarray,
    label_info: dict[int, LabelInfo],
    vessel_label_ids: set[int] | None = None,
    reference_image: sitk.Image | None = None,
) -> dict[str, Any]:
    vessel_ids = vessel_label_ids or DEFAULT_VESSEL_LABEL_IDS
    pred = np.asarray(prediction, dtype=bool)
    labels = np.asarray(label_image)
    vessel_target = np.isin(labels, list(vessel_ids))
    metrics = compute_binary_metrics(pred, vessel_target)
    if reference_image is not None:
        metrics.update(compute_hausdorff_metrics(pred, vessel_target, reference_image))
    non_vessel_overlap: dict[str, dict[str, Any]] = {}
    pred_voxels = max(int(pred.sum()), 1)
    for index, info in sorted(label_info.items()):
        if index == 0 or index in vessel_ids:
            continue
        label_mask = labels == index
        overlap = int((pred & label_mask).sum())
        label_voxels = int(label_mask.sum())
        if overlap == 0 and label_voxels == 0:
            continue
        non_vessel_overlap[info.name] = {
            "label_id": index,
            "label_voxels": label_voxels,
            "overlap_voxels": overlap,
            "overlap_fraction_of_prediction": overlap / pred_voxels,
            "overlap_fraction_of_label": _safe_divide(overlap, label_voxels),
        }
    labeled = labels != 0
    unlabeled_prediction = int((pred & ~labeled).sum())
    return {
        "vessel_label_ids": sorted(vessel_ids),
        "vessel_metrics": metrics,
        "non_vessel_overlap": non_vessel_overlap,
        "unlabeled_prediction_voxels": unlabeled_prediction,
        "unlabeled_prediction_fraction": unlabeled_prediction / pred_voxels,
    }


def evaluate_prediction(
    prediction_path: str | Path,
    label_image_path: str | Path,
    label_txt_path: str | Path,
    output_path: str | Path,
    vessel_label_ids: set[int] | None = None,
) -> dict[str, Any]:
    prediction_img = sitk.ReadImage(str(prediction_path))
    label_img = sitk.ReadImage(str(label_image_path))
    _assert_same_space(prediction_img, label_img)
    prediction = sitk.GetArrayFromImage(prediction_img) > 0
    labels = sitk.GetArrayFromImage(label_img)
    label_info = parse_itksnap_label_file(label_txt_path)
    summary = summarize_label_overlaps(
        prediction,
        labels,
        label_info,
        vessel_label_ids=vessel_label_ids,
        reference_image=prediction_img,
    )
    payload = {
        "prediction_path": str(prediction_path),
        "label_image_path": str(label_image_path),
        "label_txt_path": str(label_txt_path),
        "ground_truth_role": "evaluation_only",
        **summary,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _assert_same_space(a: sitk.Image, b: sitk.Image) -> None:
    if a.GetSize() != b.GetSize():
        raise ValueError(f"Image size mismatch: {a.GetSize()} != {b.GetSize()}")
    if not np.allclose(a.GetSpacing(), b.GetSpacing()):
        raise ValueError(f"Image spacing mismatch: {a.GetSpacing()} != {b.GetSpacing()}")
    if not np.allclose(a.GetOrigin(), b.GetOrigin()):
        raise ValueError(f"Image origin mismatch: {a.GetOrigin()} != {b.GetOrigin()}")
    if not np.allclose(a.GetDirection(), b.GetDirection()):
        raise ValueError("Image direction mismatch")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mask_to_reference_image(mask: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    image.CopyInformation(reference_image)
    return image
