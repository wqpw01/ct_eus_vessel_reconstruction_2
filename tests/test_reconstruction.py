from pathlib import Path

import numpy as np
import pytest

from ct_eus_vessel.reconstruction import (
    assert_reference_labels_not_in_inputs,
    fuse_phase_masks,
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
