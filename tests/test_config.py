from pathlib import Path

import pytest

from ct_eus_vessel.config import ProjectConfig, load_config


def test_load_config_keeps_reference_labels_eval_only(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  case_id: CT-0021",
                "paths:",
                "  raw_data: /data/raw",
                "  reference_labels: /data/labels",
                "  output_root: /data/out",
                "  work_root: /data/work",
                "phase:",
                "  arterial_keywords: ['arterial']",
                "  portal_keywords: ['portal']",
                "  venous_keywords: ['venous']",
                "  required_phases: ['arterial', 'portal']",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, ProjectConfig)
    assert config.case_id == "CT-0021"
    assert config.reference_labels == Path("/data/labels")
    assert config.reference_labels_eval_only is True
    assert config.required_phases == ("arterial", "portal")


def test_load_config_rejects_reference_labels_as_output_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  case_id: CT-0021",
                "paths:",
                "  raw_data: /data/raw",
                "  reference_labels: /data/same",
                "  output_root: /data/same",
                "  work_root: /data/work",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference_labels"):
        load_config(config_path)
