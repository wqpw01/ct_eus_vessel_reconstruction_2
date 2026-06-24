from pathlib import Path

from ct_eus_vessel.iteration import IterationRecord, write_iteration_log


def test_write_iteration_log_records_pipeline_and_gt_warning(tmp_path: Path) -> None:
    record = IterationRecord(
        version="v0.1",
        case_id="CT-0021",
        output_dir=Path("/outputs/CT-0021-v0.1"),
        pipeline_summary=["DICOM index", "phase classification"],
        core_parameters={"required_phases": ["arterial", "portal"]},
        generated_outputs=["series_index.csv"],
        known_limits=["No full reconstruction yet"],
    )

    path = write_iteration_log(tmp_path, record)
    text = path.read_text(encoding="utf-8")

    assert path.name == "CT-0021-v0.1.md"
    assert "Ground Truth 仅用于最终评价" in text
    assert "DICOM index" in text
    assert "required_phases" in text
