from pathlib import Path

from ct_eus_vessel import cli
from ct_eus_vessel.cli import build_arg_parser


def test_cli_has_index_and_plan_commands() -> None:
    parser = build_arg_parser()

    index_args = parser.parse_args(["index", "--config", "configs/default.yml"])
    plan_args = parser.parse_args(["write-log", "--config", "configs/default.yml", "--version", "v0.1"])
    reconstruct_args = parser.parse_args(["reconstruct", "--config", "configs/default.yml", "--skip-totalseg"])
    eval_args = parser.parse_args(
        [
            "evaluate",
            "--prediction",
            "pred.nrrd",
            "--label-image",
            "pseudo_label-.nii",
            "--label-map",
            "label.txt",
            "--output",
            "eval.json",
        ]
    )

    assert index_args.command == "index"
    assert index_args.config == Path("configs/default.yml")
    assert plan_args.command == "write-log"
    assert plan_args.version == "v0.1"
    assert reconstruct_args.command == "reconstruct"
    assert reconstruct_args.skip_totalseg is True
    assert eval_args.command == "evaluate"
    assert eval_args.output == Path("eval.json")


def test_evaluate_command_does_not_require_project_config(monkeypatch, capsys) -> None:
    calls = {}

    def fake_evaluate_prediction(**kwargs):
        calls.update(kwargs)
        return {
            "vessel_metrics": {
                "dice": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "hausdorff_mm": 4.0,
                "average_hausdorff_mm": 2.0,
            }
        }

    monkeypatch.setattr(cli, "evaluate_prediction", fake_evaluate_prediction)

    exit_code = cli.main(
        [
            "evaluate",
            "--prediction",
            "pred.nrrd",
            "--label-image",
            "pseudo_label-.nii",
            "--label-map",
            "label.txt",
            "--output",
            "eval.json",
        ]
    )

    assert exit_code == 0
    assert calls["prediction_path"] == Path("pred.nrrd")
    assert "Hausdorff: 4.0000 mm" in capsys.readouterr().out
