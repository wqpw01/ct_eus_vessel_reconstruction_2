from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.evaluation import (
    LabelInfo,
    compute_binary_metrics,
    compute_hausdorff_metrics,
    parse_itksnap_label_file,
    summarize_label_overlaps,
)


def test_parse_itksnap_label_file_reads_names(tmp_path: Path) -> None:
    path = tmp_path / "label.txt"
    path.write_text(
        '\n'.join(
            [
                '    0     0    0    0        0  0  0    "Clear Label"',
                '    8     0    0  205        1  1  1    "Ao"',
                '   10   210  180  140        1  1  1    "PV"',
            ]
        ),
        encoding="utf-8",
    )

    labels = parse_itksnap_label_file(path)

    assert labels[8] == LabelInfo(index=8, name="Ao")
    assert labels[10].name == "PV"


def test_compute_binary_metrics_reports_precision_recall_dice() -> None:
    pred = np.array([1, 1, 1, 0, 0], dtype=bool)
    target = np.array([1, 0, 1, 1, 0], dtype=bool)

    metrics = compute_binary_metrics(pred, target)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["dice"] == 4 / 6
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3


def test_summarize_label_overlaps_separates_vessels_and_organs() -> None:
    pred = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    labels = np.array([8, 6, 10, 0, 6, 0], dtype=np.uint16)
    label_info = {
        6: LabelInfo(index=6, name="liver"),
        8: LabelInfo(index=8, name="Ao"),
        10: LabelInfo(index=10, name="PV"),
    }

    summary = summarize_label_overlaps(pred, labels, label_info, vessel_label_ids={8, 10})

    assert summary["vessel_metrics"]["tp"] == 2
    assert summary["non_vessel_overlap"]["liver"]["overlap_voxels"] == 1
    assert summary["unlabeled_prediction_voxels"] == 1


def test_compute_hausdorff_metrics_uses_image_spacing_in_mm() -> None:
    pred = np.zeros((1, 1, 3), dtype=bool)
    target = np.zeros_like(pred)
    pred[0, 0, 0] = True
    target[0, 0, 2] = True
    reference = sitk.GetImageFromArray(np.zeros_like(pred, dtype=np.uint8))
    reference.SetSpacing((2.0, 1.0, 1.0))

    metrics = compute_hausdorff_metrics(pred, target, reference)

    assert metrics["hausdorff_mm"] == 4.0
    assert metrics["average_hausdorff_mm"] == 4.0
