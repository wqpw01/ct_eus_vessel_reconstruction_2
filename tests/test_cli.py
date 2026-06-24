from pathlib import Path

from ct_eus_vessel.cli import build_arg_parser


def test_cli_has_index_and_plan_commands() -> None:
    parser = build_arg_parser()

    index_args = parser.parse_args(["index", "--config", "configs/default.yml"])
    plan_args = parser.parse_args(["write-log", "--config", "configs/default.yml", "--version", "v0.1"])

    assert index_args.command == "index"
    assert index_args.config == Path("configs/default.yml")
    assert plan_args.command == "write-log"
    assert plan_args.version == "v0.1"
