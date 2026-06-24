# CT-EUS 血管重建 Pipeline 技术路线 v0.2-phase-inference

## Executive Summary

v0.2 修正了 CT-0021 的关键期相判定逻辑：本病例的多个 `1.0 x 1.0_A` 序列是注射造影剂后连续扫描得到的时间动态序列，不能只依赖 DICOM `ProtocolName`。根据临床业务规则，首个 `1.0 x 1.0_A` 为动脉显影最佳，最后一个 `1.0 x 1.0_A` 为静脉显影，门静脉最佳显影位于中间某个序列。

本版索引结果已经能够识别出 venous 序列，同时保留 DICOM 原始语义分类，方便团队追溯为什么某些 `ProtocolName=Portal Phase` 的序列会被动态推断为静脉期。

## Context

v0.1 的索引仅使用 `SeriesDescription` 与 `ProtocolName` 做语义分类，因此 CT-0021 中后续多个 `1.0 x 1.0_A` 序列都被标为 portal。用户提供了更准确的扫描协议解释：这些序列存在明确先后顺序，最后一个连续增强序列应视为静脉期候选。

## Methods

### 1. 双层期相字段

索引表现在包含三个期相相关字段：

- `metadata_phase`：仅由 DICOM 文字字段推断。
- `dynamic_phase`：由连续增强序列时序推断。
- `phase`：最终供下游选择使用的期相；若存在 `dynamic_phase`，优先采用动态推断。

### 2. 连续增强序列识别

动态期相推断只作用于满足以下条件的序列：

- `series_description` 包含 `1.0` 与 `_A`。
- 排除 `MIP`、`Scout`、`Dose`、`lung`、`med`、`report` 等非原始增强薄层序列。
- `num_instances >= 100`，避免把 MIP、报告或定位片纳入动态序列。

候选序列按 `SeriesNumber` 排序。首个候选标为 `arterial`，最后一个候选标为 `venous`，中间序列标为 `portal`。中间多个门静脉候选中，当前优先选择层数更多且时序更靠前的序列作为主 portal 候选；下一步将补充图像 HU 增强评分。

### 3. 测试保护

新增测试 `test_infer_dynamic_phases_relabels_continuous_1mm_contrast_series`，固定 CT-0021 的 501/601/701/801/901 序列行为，防止后续重构时再次丢失静脉期。

## Results

重新索引真实 CT-0021 后，关键序列如下：

| phase | metadata_phase | dynamic_phase | num_instances | series_number | protocol_name |
| --- | --- | --- | ---: | --- | --- |
| arterial | arterial | arterial | 227 | 501 | Arterial Phase |
| portal | portal | portal | 227 | 601 | Portal Phase |
| portal | portal | portal | 257 | 701 | Portal Phase |
| portal | portal | portal | 257 | 801 | Portal Phase |
| venous | portal | venous | 257 | 901 | Portal Phase |

输出文件：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/CT-EUS-vessel-pipeline-runs/CT-0021/index/series_index.csv`

## Discussion

这次修正说明期相识别必须结合扫描协议和时间动态，而不能把 DICOM 文本字段当作绝对真相。对于后续重建，动脉、门静脉、静脉三类输入已经可以从 CT-0021 的连续增强序列中选出。

仍需注意：门静脉“最佳显影”的中间序列目前还没有通过图像强度自动评分确认。下一步应在不使用 GT 的前提下，对肝门区/门静脉主干候选区域计算增强 HU、管状响应和连通性评分，再决定 601/701/801 中哪一个作为主门静脉期。

## Reproducibility Appendix

```bash
cd /home/zyt/ct_eus_vessel_pipeline
pytest -q
PYTHONPATH=src python -m ct_eus_vessel.cli index --config configs/default.yml
sed -n '1,12p' /mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/CT-EUS-vessel-pipeline-runs/CT-0021/index/series_index.csv
```
