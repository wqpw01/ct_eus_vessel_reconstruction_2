from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class IterationRecord:
    version: str
    case_id: str
    output_dir: Path
    pipeline_summary: list[str]
    core_parameters: dict[str, Any]
    generated_outputs: list[str]
    known_limits: list[str]


def write_iteration_log(root: str | Path, record: IterationRecord) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / f"{record.case_id}-{record.version}.md"
    path.write_text(_render_log(record), encoding="utf-8")
    return path


def _render_log(record: IterationRecord) -> str:
    params = yaml.safe_dump(record.core_parameters, allow_unicode=True, sort_keys=False).strip()
    summary = "\n".join(f"- {item}" for item in record.pipeline_summary)
    outputs = "\n".join(f"- {item}" for item in record.generated_outputs)
    limits = "\n".join(f"- {item}" for item in record.known_limits)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# {record.case_id} {record.version} 迭代记录

生成时间: {timestamp}

## 数据使用红线

Ground Truth 仅用于最终评价，不参与分割、阈值估计、候选生成、训练、后处理或参数搜索。

## Pipeline 摘要

{summary}

## 核心参数

```yaml
{params}
```

## 输出文件

输出目录: `{record.output_dir}`

{outputs}

## 已知局限

{limits}
"""
