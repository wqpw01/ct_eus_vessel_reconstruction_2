from pathlib import Path

import numpy as np
import pytest

from ct_eus_vessel.reconstruction import (
    TOTAL_SEGMENTATOR_EXCLUSION_ROIS,
    TOTAL_SEGMENTATOR_VESSEL_ROIS,
    apply_exclusion_seed_override,
    assert_reference_labels_not_in_inputs,
    compute_volume_frangi,
    dilate_mask_by_mm,
    fuse_phase_masks,
    grow_seed_connected_mask,
    select_reconstruction_series,
)


def test_select_reconstruction_series_uses_dynamic_phases_and_excludes_mip() -> None:
    rows = [
        {"series_uid": "701", "phase": "portal", "series_number": "701", "num_instances": 257},
        {"series_uid": "901", "phase": "venous", "series_number": "901", "num_instances": 257},
        {"series_uid": "501", "phase": "arterial", "series_number": "501", "num_instances": 227},
        {"series_uid": "10003", "phase": "arterial", "series_number": "10003", "num_instances": 13},
    ]

    selected = select_reconstruction_series(rows)

    assert selected["arterial"]["series_uid"] == "501"
    assert selected["portal"]["series_uid"] == "701"
    assert selected["venous"]["series_uid"] == "901"


def test_reference_labels_are_rejected_as_reconstruction_inputs() -> None:
    reference = Path("/labels/CT-0021-union-nrrd")

    with pytest.raises(ValueError, match="Ground Truth"):
        assert_reference_labels_not_in_inputs(
            reference,
            [Path("/raw/CT-0021"), Path("/labels/CT-0021-union-nrrd/reference_ct.nrrd")],
        )


def test_fuse_phase_masks_keeps_separate_multilabel_outputs() -> None:
    artery = np.zeros((3, 4, 5), dtype=bool)
    portal = np.zeros_like(artery)
    venous = np.zeros_like(artery)
    artery[1, 1, 1] = True
    portal[1, 1, 2] = True
    venous[1, 1, 3] = True
    venous[1, 1, 1] = True

    fused = fuse_phase_masks({"arterial": artery, "portal": portal, "venous": venous})

    assert fused.binary.sum() == 3
    assert fused.multilabel[1, 1, 1] == 1
    assert fused.multilabel[1, 1, 2] == 2
    assert fused.multilabel[1, 1, 3] == 3


def test_totalseg_exclusion_rois_cover_major_false_positive_organs_and_bones() -> None:
    required = {
        "stomach",
        "pancreas",
        "duodenum",
        "small_bowel",
        "colon",
        "heart",
        "lung_upper_lobe_left",
        "lung_lower_lobe_right",
        "rib_left_1",
        "rib_right_12",
        "sternum",
    }

    assert required.issubset(set(TOTAL_SEGMENTATOR_EXCLUSION_ROIS))
    assert set(TOTAL_SEGMENTATOR_VESSEL_ROIS).isdisjoint(set(TOTAL_SEGMENTATOR_EXCLUSION_ROIS))


def test_dilate_mask_by_mm_expands_in_voxel_space() -> None:
    mask = np.zeros((5, 7, 7), dtype=bool)
    mask[2, 3, 3] = True

    expanded = dilate_mask_by_mm(mask, spacing_xyz=(1.0, 1.0, 1.0), radius_mm=1.5)

    assert int(expanded.sum()) > int(mask.sum())
    assert expanded[2, 3, 4]


def test_exclusion_seed_override_keeps_vessel_seed_available() -> None:
    exclusion = np.zeros((3, 5, 5), dtype=bool)
    exclusion[1, 2, 2] = True
    seed = np.zeros_like(exclusion)
    seed[1, 2, 2] = True

    final = apply_exclusion_seed_override(exclusion, seed)

    assert not final[1, 2, 2]


def test_grow_seed_connected_mask_drops_disconnected_bright_island() -> None:
    candidate = np.zeros((1, 1, 8), dtype=bool)
    candidate[0, 0, 1:4] = True
    candidate[0, 0, 6:8] = True
    seed = np.zeros_like(candidate)
    seed[0, 0, 1] = True
    vesselness = np.zeros_like(candidate, dtype=np.float32)
    vesselness[candidate] = 1.0

    grown = grow_seed_connected_mask(candidate, seed, vesselness, vesselness_min=0.5, recovery_iterations=0)

    assert grown[0, 0, 1]
    assert grown[0, 0, 3]
    assert not grown[0, 0, 6]


def test_compute_volume_frangi_returns_3d_response() -> None:
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[:, 3, 3] = 250.0

    response = compute_volume_frangi(volume, sigmas=(1.0, 2.0))

    assert response.shape == volume.shape
    assert float(response.max()) > 0.0
    assert response[:, 3, 3].mean() >= response[:, 0, 0].mean()


def test_compute_volume_frangi_can_process_overlapping_slabs() -> None:
    volume = np.zeros((9, 7, 7), dtype=np.float32)
    volume[:, 3, 3] = 250.0

    response = compute_volume_frangi(volume, sigmas=(1.0,), max_slab_slices=4, overlap_slices=1)

    assert response.shape == volume.shape
    assert float(response.max()) > 0.0
