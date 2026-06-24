from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .dicom_index import index_dicom_series, write_series_index_csv
from .iteration import IterationRecord, write_iteration_log
from .phase import choose_primary_series


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ct-eus-vessel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index DICOM series and infer phases.")
    index_parser.add_argument("--config", type=Path, required=True)

    log_parser = subparsers.add_parser("write-log", help="Write a pipeline iteration log.")
    log_parser.add_argument("--config", type=Path, required=True)
    log_parser.add_argument("--version", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "index":
        rows = index_dicom_series(config.raw_data)
        output_dir = config.output_root / config.case_id / "index"
        csv_path = write_series_index_csv(rows, output_dir / "series_index.csv")
        selected = choose_primary_series(rows)
        print(f"Wrote {csv_path}")
        for phase, item in selected.items():
            print(f"{phase}: {item['series_description']} ({item['num_instances']} instances)")
        return 0

    if args.command == "write-log":
        output_dir = config.output_root / f"{config.case_id}-{args.version}"
        record = IterationRecord(
            version=args.version,
            case_id=config.case_id,
            output_dir=output_dir,
            pipeline_summary=[
                "DICOM 多序列索引、metadata_phase 记录与 dynamic_phase 时序推断",
                "人工参考标签保持评价专用，不参与分割与参数估计",
                "TotalSegmentator、配准、多期融合和三维重建接口已纳入 v1 架构",
            ],
            core_parameters={
                "raw_data": str(config.raw_data),
                "output_root": str(config.output_root),
                "required_phases": list(config.required_phases),
                "reference_labels_eval_only": config.reference_labels_eval_only,
            },
            generated_outputs=[
                "series_index.csv",
                "iteration_log.md",
            ],
            known_limits=[
                "当前版本尚未执行完整血管分割和三维重建",
                "门静脉最佳序列目前仍需结合图像 HU 增强评分进一步确认",
            ],
        )
        path = write_iteration_log(Path("docs/iterations"), record)
        print(f"Wrote {path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
