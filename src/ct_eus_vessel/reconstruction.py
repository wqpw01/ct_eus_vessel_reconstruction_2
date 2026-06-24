from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import frangi

from .candidates import extract_vessel_candidates
from .config import ProjectConfig
from .dicom_index import index_dicom_series, write_series_index_csv
from .geometry import bounding_box_zyx
from .iteration import IterationRecord, write_iteration_log
from .thresholds import estimate_phase_hu_window


PHASE_LABELS = {"arterial": 1, "portal": 2, "venous": 3}
PHASE_OUTPUT_NAMES = {
    "arterial": "artery_candidate",
    "portal": "portal_vein_candidate",
    "venous": "systemic_vein_candidate",
}
TOTAL_SEGMENTATOR_EXCLUSION_ROIS = [
    "liver",
    "spleen",
    "kidney_left",
    "kidney_right",
    "gallbladder",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "esophagus",
    "small_bowel",
    "duodenum",
    "colon",
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "heart",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "sacrum",
    "sternum",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "humerus_left",
    "humerus_right",
]
TOTAL_SEGMENTATOR_VESSEL_ROIS = [
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
]
TOTAL_SEGMENTATOR_ROIS = TOTAL_SEGMENTATOR_EXCLUSION_ROIS + TOTAL_SEGMENTATOR_VESSEL_ROIS
EXCLUSION_DILATION_MM = 2.5
BONE_DILATION_MM = 1.5
VESSELNESS_PERCENTILES = {"arterial": 75.0, "portal": 80.0, "venous": 80.0}


@dataclass(frozen=True)
class FusedMasks:
    binary: np.ndarray
    multilabel: np.ndarray


@dataclass(frozen=True)
class ReconstructionResult:
    run_dir: Path
    selected_series: dict[str, dict[str, Any]]
    outputs: dict[str, str]
    metrics: dict[str, Any]
    totalseg_status: str


def assert_reference_labels_not_in_inputs(reference_labels: Path, input_paths: list[Path]) -> None:
    reference = reference_labels.resolve(strict=False)
    for path in input_paths:
        resolved = path.resolve(strict=False)
        if resolved == reference or reference in resolved.parents:
            raise ValueError(f"Ground Truth path must not be used for reconstruction input: {resolved}")


def select_reconstruction_series(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for phase in PHASE_LABELS:
        candidates = [row for row in rows if row.get("phase") == phase and _num_instances(row) >= 100]
        if not candidates:
            continue
        selected[phase] = max(candidates, key=lambda row: (_num_instances(row), -_series_number(row)))
    return selected


def fuse_phase_masks(masks: dict[str, np.ndarray]) -> FusedMasks:
    shape = next(iter(masks.values())).shape
    binary = np.zeros(shape, dtype=bool)
    multilabel = np.zeros(shape, dtype=np.uint8)
    for phase, label in PHASE_LABELS.items():
        if phase not in masks:
            continue
        mask = np.asarray(masks[phase], dtype=bool)
        binary |= mask
        multilabel[(multilabel == 0) & mask] = label
    return FusedMasks(binary=binary, multilabel=multilabel)


def run_reconstruction(
    config: ProjectConfig,
    version: str = "v0.3-reconstruction",
    run_totalseg: bool = True,
    run_frangi: bool = True,
) -> ReconstructionResult:
    assert_reference_labels_not_in_inputs(config.reference_labels, [config.raw_data, config.work_root, config.output_root])
    run_dir = config.output_root / f"{config.case_id}-{version}"
    raw_dir = run_dir / "raw_nifti"
    mask_dir = run_dir / "masks"
    nifti_dir = run_dir / "compat_nifti"
    mesh_dir = run_dir / "mesh"
    qc_dir = run_dir / "qc"
    index_dir = run_dir / "index"
    totalseg_dir = run_dir / "totalseg"
    for path in (raw_dir, mask_dir, nifti_dir, mesh_dir, qc_dir, index_dir, totalseg_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows = index_dicom_series(config.raw_data)
    write_series_index_csv(rows, index_dir / "series_index.csv")
    selected = select_reconstruction_series(rows)
    missing = [phase for phase in PHASE_LABELS if phase not in selected]
    if missing:
        raise ValueError(f"Missing required reconstruction phases after dynamic inference: {missing}")

    selected_path = qc_dir / "selected_series.json"
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    phase_images = {phase: read_dicom_series(row) for phase, row in selected.items()}
    for phase, image in phase_images.items():
        sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(raw_dir / f"{phase}.nii.gz"))

    reference_phase = "portal"
    reference_image = phase_images[reference_phase]
    reference_ct_nrrd = mask_dir / "reference_ct_portal.nrrd"
    sitk.WriteImage(sitk.Cast(reference_image, sitk.sitkFloat32), str(reference_ct_nrrd))

    totalseg_status = "skipped"
    if run_totalseg:
        totalseg_status = run_totalsegmentator(raw_dir / f"{reference_phase}.nii.gz", totalseg_dir)

    ts_masks = load_totalseg_masks(totalseg_dir)
    phase_masks_reference: dict[str, np.ndarray] = {}
    phase_metrics: dict[str, Any] = {}
    for phase, image in phase_images.items():
        exclusion = build_exclusion_mask(image, reference_image, ts_masks)
        seed = build_vessel_seed_mask(phase, image, reference_image, ts_masks)
        vesselness = compute_volume_frangi(sitk.GetArrayFromImage(image), sigmas=(1.0, 2.0, 3.5)) if run_frangi else None
        segmented, metrics = segment_phase_vessels(phase, image, exclusion, seed, vesselness)
        mask_image = array_to_image(segmented.astype(np.uint8), image)
        mask_reference = resample_mask(mask_image, reference_image)
        mask_reference_arr = sitk.GetArrayFromImage(mask_reference).astype(bool)
        phase_masks_reference[phase] = mask_reference_arr
        output_name = PHASE_OUTPUT_NAMES[phase]
        write_mask_pair(mask_reference_arr, reference_image, mask_dir / f"{output_name}.nrrd", nifti_dir / f"{output_name}.nii.gz")
        phase_metrics[phase] = metrics
        phase_metrics[phase]["selected_series"] = selected[phase]
        phase_metrics[phase]["reference_space_voxels"] = int(mask_reference_arr.sum())

    fused = fuse_phase_masks(phase_masks_reference)
    write_mask_pair(fused.binary, reference_image, mask_dir / "vessel_fused_binary.nrrd", nifti_dir / "vessel_fused_binary.nii.gz")
    write_mask_pair(fused.multilabel, reference_image, mask_dir / "vessel_fused_multilabel.nrrd", nifti_dir / "vessel_fused_multilabel.nii.gz")
    bbox = write_bbox_json(fused.binary, reference_image, qc_dir / "bbox.json")
    write_mesh(fused.binary, reference_image, mesh_dir / "vessel_fused_binary.stl")
    for phase, mask in phase_masks_reference.items():
        write_mesh(mask, reference_image, mesh_dir / f"{PHASE_OUTPUT_NAMES[phase]}.stl")

    outputs = {
        "reference_ct": str(reference_ct_nrrd),
        "fused_binary": str(mask_dir / "vessel_fused_binary.nrrd"),
        "fused_multilabel": str(mask_dir / "vessel_fused_multilabel.nrrd"),
        "bbox": str(qc_dir / "bbox.json"),
        "mesh": str(mesh_dir / "vessel_fused_binary.stl"),
        "selected_series": str(selected_path),
    }
    metrics = {
        "version": version,
        "phase_metrics": phase_metrics,
        "fused_voxels": int(fused.binary.sum()),
        "bbox": bbox,
        "totalseg_status": totalseg_status,
        "ground_truth_used": False,
        "ground_truth_note": "reference_labels path is reserved for evaluation only and was not read by reconstruction.",
    }
    (qc_dir / "reconstruction_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_v03_iteration_log(config, version, run_dir, selected, metrics, totalseg_status)
    return ReconstructionResult(run_dir=run_dir, selected_series=selected, outputs=outputs, metrics=metrics, totalseg_status=totalseg_status)


def read_dicom_series(row: dict[str, Any]) -> sitk.Image:
    series_dir = Path(row["first_file"]).parent
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir), str(row["series_uid"]))
    if not files:
        raise FileNotFoundError(f"No DICOM files found for series {row['series_uid']} in {series_dir}")
    reader.SetFileNames(files)
    image = reader.Execute()
    return sitk.Cast(image, sitk.sitkFloat32)


def run_totalsegmentator(input_image: Path, output_dir: Path) -> str:
    try:
        from totalsegmentator.python_api import totalsegmentator

        totalsegmentator(
            input=input_image,
            output=output_dir,
            task="total",
            roi_subset=TOTAL_SEGMENTATOR_ROIS,
            fast=True,
            body_seg=True,
            device="gpu",
            quiet=False,
            remove_small_blobs=True,
        )
        return "completed"
    except Exception as exc:  # pragma: no cover - exercised during real runs
        (output_dir / "totalseg_error.txt").write_text(str(exc), encoding="utf-8")
        return f"failed: {exc}"


def load_totalseg_masks(totalseg_dir: Path) -> dict[str, sitk.Image]:
    masks: dict[str, sitk.Image] = {}
    for name in TOTAL_SEGMENTATOR_ROIS:
        path = totalseg_dir / f"{name}.nii.gz"
        if path.exists():
            masks[name] = sitk.ReadImage(str(path), sitk.sitkUInt8)
    return masks


def build_exclusion_mask(image: sitk.Image, reference_image: sitk.Image, ts_masks: dict[str, sitk.Image]) -> np.ndarray:
    hu = sitk.GetArrayFromImage(image)
    spacing_xyz = image.GetSpacing()
    exclusion = dilate_mask_by_mm(hu > 700, spacing_xyz, BONE_DILATION_MM)
    for name in TOTAL_SEGMENTATOR_EXCLUSION_ROIS:
        if name not in ts_masks:
            continue
        roi = sitk.GetArrayFromImage(resample_mask(ts_masks[name], image, reference_image)).astype(bool)
        exclusion |= dilate_mask_by_mm(roi, spacing_xyz, EXCLUSION_DILATION_MM)
    return exclusion


def build_vessel_seed_mask(
    phase: str,
    image: sitk.Image,
    reference_image: sitk.Image,
    ts_masks: dict[str, sitk.Image],
) -> np.ndarray:
    seed_names = {
        "arterial": ["aorta"],
        "portal": ["portal_vein_and_splenic_vein"],
        "venous": ["inferior_vena_cava"],
    }[phase]
    seed = np.zeros(sitk.GetArrayFromImage(image).shape, dtype=bool)
    for name in seed_names:
        if name in ts_masks:
            seed |= sitk.GetArrayFromImage(resample_mask(ts_masks[name], image, reference_image)).astype(bool)
    return seed


def segment_phase_vessels(
    phase: str,
    image: sitk.Image,
    exclusion_mask: np.ndarray,
    seed_mask: np.ndarray,
    vesselness: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    hu = sitk.GetArrayFromImage(image)
    body = create_body_mask(hu)
    exclusion = apply_exclusion_seed_override(exclusion_mask, seed_mask)
    phase_window = estimate_phase_hu_window(
        phase,
        hu,
        body_mask=body,
        exclusion_mask=exclusion,
        seed_mask=seed_mask,
    )
    candidate = extract_vessel_candidates(
        hu,
        body_mask=body,
        exclusion_mask=exclusion,
        min_component_voxels=120,
        hu_window=phase_window,
    )
    vesselness_min = 0.0
    if vesselness is not None and candidate.mask.any():
        values = vesselness[candidate.mask]
        positive = values[values > 0]
        percentile = VESSELNESS_PERCENTILES.get(phase, 80.0)
        vesselness_min = float(np.percentile(positive, percentile)) if positive.size else 0.0
    mask = grow_seed_connected_mask(
        candidate.mask,
        seed_mask,
        vesselness,
        vesselness_min=vesselness_min,
        recovery_iterations=1,
    ) & body
    labels, count = ndi.label(mask)
    if count > 0:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= 80
        keep[0] = False
        mask = keep[labels]
    metrics = {
        "native_voxels": int(mask.sum()),
        "hu_window": {"low": candidate.hu_window.low, "high": candidate.hu_window.high},
        "vesselness_min": vesselness_min,
        "candidate_voxels": int(candidate.mask.sum()),
        "seed_voxels": int(seed_mask.sum()),
        "body_voxels": int(body.sum()),
        "exclusion_voxels": int(exclusion.sum()),
        "exclusion_dilation_mm": EXCLUSION_DILATION_MM,
        "bone_dilation_mm": BONE_DILATION_MM,
        "native_bbox_zyx": bounding_box_zyx(mask),
    }
    return mask.astype(bool), metrics


def dilate_mask_by_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if radius_mm <= 0:
        return binary.copy()
    spacing_zyx = spacing_xyz[::-1]
    radius_voxels = [max(1, int(np.ceil(radius_mm / spacing))) for spacing in spacing_zyx]
    grids = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radius_voxels)]
    distance_sq = sum((axis * spacing) ** 2 for axis, spacing in zip(grids, spacing_zyx, strict=True))
    structure = distance_sq <= radius_mm**2
    return ndi.binary_dilation(binary, structure=structure)


def apply_exclusion_seed_override(exclusion_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    return np.asarray(exclusion_mask, dtype=bool) & ~np.asarray(seed_mask, dtype=bool)


def grow_seed_connected_mask(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    vesselness: np.ndarray | None,
    vesselness_min: float,
    recovery_iterations: int = 1,
) -> np.ndarray:
    candidate = np.asarray(candidate_mask, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool)
    if candidate.shape != seed.shape:
        raise ValueError("candidate_mask and seed_mask must have the same shape")
    if vesselness is None:
        core = candidate.copy()
    else:
        response = np.asarray(vesselness, dtype=np.float32)
        if response.shape != candidate.shape:
            raise ValueError("vesselness must have the same shape as candidate_mask")
        core = candidate & (response >= float(vesselness_min))
    if not seed.any():
        return core
    allowed = core | seed
    grown = ndi.binary_propagation(seed, mask=allowed)
    if recovery_iterations > 0:
        grown |= ndi.binary_dilation(
            grown,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(recovery_iterations),
        ) & candidate
    return np.asarray(grown, dtype=bool)


def create_body_mask(hu: np.ndarray) -> np.ndarray:
    body = np.asarray(hu) > -500
    body = ndi.binary_closing(body, structure=np.ones((3, 5, 5), dtype=bool))
    labels, count = ndi.label(body)
    if count == 0:
        return np.ones_like(body, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def compute_slice_frangi(hu: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    clipped = np.clip(hu, -100, 320).astype(np.float32)
    normalized = (clipped + 100.0) / 420.0
    response = np.zeros_like(normalized, dtype=np.float32)
    for z in range(normalized.shape[0]):
        response[z] = frangi(normalized[z], sigmas=sigmas, black_ridges=False)
    return np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)


def compute_volume_frangi(
    hu: np.ndarray,
    sigmas: tuple[float, ...],
    max_slab_slices: int = 32,
    overlap_slices: int = 6,
) -> np.ndarray:
    clipped = np.clip(np.asarray(hu, dtype=np.float32), -100, 320)
    normalized = (clipped + 100.0) / 420.0
    try:
        if normalized.ndim == 3 and normalized.shape[0] > max_slab_slices > 0:
            response = _slabbed_volume_frangi(normalized, sigmas, max_slab_slices, overlap_slices)
        else:
            response = frangi(normalized, sigmas=sigmas, black_ridges=False)
    except Exception:
        return compute_slice_frangi(hu, sigmas=sigmas)
    return np.nan_to_num(np.asarray(response, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _slabbed_volume_frangi(
    normalized: np.ndarray,
    sigmas: tuple[float, ...],
    max_slab_slices: int,
    overlap_slices: int,
) -> np.ndarray:
    depth = normalized.shape[0]
    response = np.zeros_like(normalized, dtype=np.float32)
    core_start = 0
    overlap = max(0, int(overlap_slices))
    slab_size = max(1, int(max_slab_slices))
    while core_start < depth:
        core_end = min(depth, core_start + slab_size)
        slab_start = max(0, core_start - overlap)
        slab_end = min(depth, core_end + overlap)
        slab = normalized[slab_start:slab_end]
        slab_response = frangi(slab, sigmas=sigmas, black_ridges=False)
        trim_start = core_start - slab_start
        trim_end = trim_start + (core_end - core_start)
        response[core_start:core_end] = np.asarray(slab_response, dtype=np.float32)[trim_start:trim_end]
        core_start = core_end
    return response


def array_to_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def resample_mask(mask: sitk.Image, reference: sitk.Image, source_reference: sitk.Image | None = None) -> sitk.Image:
    del source_reference
    return sitk.Resample(
        sitk.Cast(mask, sitk.sitkUInt8),
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def write_mask_pair(mask: np.ndarray, reference: sitk.Image, nrrd_path: Path, nifti_path: Path) -> None:
    image = array_to_image(mask.astype(np.uint8), reference)
    sitk.WriteImage(image, str(nrrd_path))
    sitk.WriteImage(image, str(nifti_path))


def write_bbox_json(mask: np.ndarray, reference: sitk.Image, path: Path) -> dict[str, Any]:
    bbox = bounding_box_zyx(mask)
    payload: dict[str, Any] = {"bbox_zyx": bbox}
    if bbox is not None:
        (z0, z1), (y0, y1), (x0, x1) = bbox
        low = reference.TransformIndexToPhysicalPoint((x0, y0, z0))
        high = reference.TransformIndexToPhysicalPoint((x1 - 1, y1 - 1, z1 - 1))
        payload["bbox_physical_xyz"] = {"min": low, "max": high}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_mesh(mask: np.ndarray, reference: sitk.Image, path: Path) -> None:
    if int(mask.sum()) == 0:
        return
    try:
        import trimesh

        spacing_zyx = reference.GetSpacing()[::-1]
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing_zyx)
        verts_xyz = verts[:, [2, 1, 0]]
        mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
        mesh.export(path)
    except Exception as exc:  # pragma: no cover - exercised during real runs
        path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")


def write_v03_iteration_log(
    config: ProjectConfig,
    version: str,
    run_dir: Path,
    selected: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    totalseg_status: str,
) -> None:
    record = IterationRecord(
        version=version,
        case_id=config.case_id,
        output_dir=run_dir,
        pipeline_summary=[
            "真实 CT-0021 三期 DICOM 读取与动态期相选择",
            "TotalSegmentator 预训练模型前置器官/骨骼/血管锚点提取，不使用人工标注",
            "Seed 自适应分期 HU 窗、slabbed 3D Frangi 管状响应、seed-connected 生长与多期融合",
            "输出 Slicer NRRD、兼容 NIfTI、STL mesh、bbox 与 QC JSON",
        ],
        core_parameters={
            "ground_truth_used": False,
            "reference_labels_role": "evaluation_only",
            "selected_series": {
                phase: {
                    "series_uid": row.get("series_uid"),
                    "series_number": row.get("series_number"),
                    "num_instances": row.get("num_instances"),
                    "phase": row.get("phase"),
                    "metadata_phase": row.get("metadata_phase"),
                    "dynamic_phase": row.get("dynamic_phase"),
                }
                for phase, row in selected.items()
            },
            "totalseg_status": totalseg_status,
            "fused_voxels": metrics["fused_voxels"],
        },
        generated_outputs=[
            "masks/reference_ct_portal.nrrd",
            "masks/artery_candidate.nrrd",
            "masks/portal_vein_candidate.nrrd",
            "masks/systemic_vein_candidate.nrrd",
            "masks/vessel_fused_binary.nrrd",
            "masks/vessel_fused_multilabel.nrrd",
            "compat_nifti/*.nii.gz",
            "mesh/*.stl",
            "qc/bbox.json",
            "qc/reconstruction_summary.json",
        ],
        known_limits=[
            "当前版本仍是自动候选重建，人工标注仅用于 Dice/Hausdorff 等评价，不参与重建",
            "Slabbed 3D Frangi 在本地 CPU 上耗时较高，后续应做 ROI 裁剪或 GPU/服务器加速",
            "TotalSegmentator 若输出缺失会自动降级为 HU 骨骼排除和体表 mask",
        ],
    )
    write_iteration_log(Path("docs/iterations"), record)


def _num_instances(row: dict[str, Any]) -> int:
    try:
        return int(row.get("num_instances", 0))
    except (TypeError, ValueError):
        return 0


def _series_number(row: dict[str, Any]) -> float:
    try:
        return float(row.get("series_number", 0))
    except (TypeError, ValueError):
        return 0.0
