# CT-EUS 血管重建 Pipeline 技术路线 v0.1

## Executive Summary

v0.1 建立了从零开始的独立工程基线，项目目录为 `/home/zyt/ct_eus_vessel_pipeline`。本版重点不是直接追求最终三维重建效果，而是先固定工程边界、数据红线、DICOM 多序列索引、期相识别、候选血管基础工具、自动测试和迭代文档机制。

当前真实数据索引显示：CT-0021 包含 1 个主要动脉期序列、多个门静脉期序列，暂未发现明确命名的静脉期序列。因此 v0.1 按“动脉期 + 门静脉期主线、静脉期可选缺失”处理，禁止凭主观推断生成静脉期输入。

## Context

目标任务是在多期增强 CT 中提取胸腹腔关键血管，用于 EUS 定位下游任务。人工标注路径只允许用于最终评价，不参与分割、候选生成、阈值估计、模型训练、后处理或参数搜索。

本版遵循以下工程约束：

- 项目不放在 `/home/zyt/projects/location2EUS`。
- 项目不放在 `/home/zyt/ct_eus_vessel_reconstruction`。
- 原始数据只读引用，代码与迭代文档独立管理。
- 输出写入配置指定的 `CT-EUS-vessel-pipeline-runs`，避免覆盖旧结果。

## Methods

### 1. DICOM 多序列索引

`ct_eus_vessel.dicom_index.index_dicom_series` 只读取 DICOM header，不加载像素数据。按 `SeriesInstanceUID` 聚合，并记录：

- `series_uid`
- `phase`
- `num_instances`
- `series_description`
- `protocol_name`
- `study_date`
- `series_number`
- `first_file`

### 2. 期相识别

`ct_eus_vessel.phase.classify_phase` 采用显式语义关键词优先策略。由于本例中 arterial 和 portal 都可能使用 `1.0 x 1.0_A` 序列名，分类器优先读取 `ProtocolName`/`SeriesDescription` 中的 `Arterial Phase`、`Portal Phase`、`Venous Phase` 等完整语义，再把 `_A` 作为动脉期弱提示。

### 3. 动态 HU 候选窗

`ct_eus_vessel.thresholds.estimate_vessel_hu_window` 在 body mask 内、排除器官/骨骼 mask 后估计增强血管候选窗。默认约束：

```yaml
vessel_hu_floor: 80
vessel_hu_ceiling: 300
bone_hu_floor: 700
```

上界至少覆盖 `low + 120 HU`，用于避免在小样本或血管占比很低时把 150-250 HU 的真实增强血管压出候选窗。

### 4. 候选血管提取

`ct_eus_vessel.candidates.extract_vessel_candidates` 当前输入为 HU 体数据、body mask、排除 mask 和可选 vesselness 响应，输出：

- 候选血管二值 mask
- 实际使用的 HU window
- z/y/x 半开区间 bounding box

该函数为后续 Frangi/Hessian、多期融合和 TotalSegmentator 前置剔除提供稳定接口。

### 5. 迭代记录

每个版本必须写入 `docs/iterations/<case>-<version>.md`，内容包括：

- Pipeline 摘要
- Ground Truth 使用红线
- 核心参数
- 输出文件
- 已知局限

## Results

真实 CT-0021 索引输出已写入：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/CT-EUS-vessel-pipeline-runs/CT-0021/index/series_index.csv`

观察到的关键序列：

| phase | num_instances | series_description | protocol_name | series_number |
| --- | ---: | --- | --- | --- |
| arterial | 227 | 1.0 x 1.0_A | Arterial Phase | 501 |
| portal | 257 | 1.0 x 1.0_A | Portal Phase | 701 |
| portal | 257 | 1.0 x 1.0_A | Portal Phase | 801 |
| portal | 257 | 1.0 x 1.0_A | Portal Phase | 901 |
| portal | 227 | 1.0 x 1.0_A | Portal Phase | 601 |

## Discussion

v0.1 已经解决工程起点问题，但还不是完整血管重建版本。下一版最关键的是在不使用 GT 的前提下，完成以下事项：

- 选择 portal 多序列中的主序列或多序列融合策略。
- 将动脉期和门静脉期配准到统一物理空间。
- 运行 TotalSegmentator 生成肝脏、脾脏、肾脏和骨骼排除 mask。
- 在排除 mask 后执行多尺度 Frangi/Hessian vesselness。
- 做多期候选融合、连通性保护、表面片状伪影清理和 ROI bbox 导出。

## Reproducibility Appendix

基础验证命令：

```bash
cd /home/zyt/ct_eus_vessel_pipeline
pytest -q
PYTHONPATH=src python -m ct_eus_vessel.cli index --config configs/default.yml
PYTHONPATH=src python -m ct_eus_vessel.cli write-log --config configs/default.yml --version v0.1
```
