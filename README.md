# CT-EUS Vessel Pipeline

独立的 CT-EUS 定位项目血管分割与三维重建工程目录。

本项目从零搭建，不放入 `/home/zyt/projects/location2EUS`，也不放入 `/home/zyt/ct_eus_vessel_reconstruction`。原始 DICOM、人工参考标签和输出结果继续按配置引用外部路径；代码、配置、测试与迭代文档保存在本目录。

## 当前范围

- 读取项目配置，并强制约束人工标注只能用于最终评价。
- 建立 DICOM 多序列索引与期相识别入口。
- 对连续 `1.0 x 1.0_A` 增强序列执行动态时序期相推断：首序列为动脉期，末序列为静脉期，中间序列为门静脉期候选。
- 提供动态 HU 血管候选窗、连通域过滤和 ROI bounding box 的基础工具。
- 生成每轮重建必须同步更新的迭代日志。
- 预留 TotalSegmentator 前置器官/骨骼剔除、配准、多期融合、重采样、三维网格与指标评价阶段。

## 环境

医学影像、CUDA/PyTorch、SimpleITK、TotalSegmentator 属于重科学计算依赖，默认使用 mamba/conda：

```bash
mamba env create -f environment.yml
mamba activate ct-eus-vessel
pip install -e ".[dev]"
```

## 基本命令

```bash
ct-eus-vessel index --config configs/default.yml
ct-eus-vessel reconstruct --config configs/default.yml --version v0.3b-conservative
ct-eus-vessel write-log --config configs/default.yml --version v0.1
pytest
```

## 数据红线

`reference_labels` 只允许在评价阶段读取。任何分割、阈值估计、模型训练、候选生成、后处理与参数搜索阶段都不得读取人工参考标签。
