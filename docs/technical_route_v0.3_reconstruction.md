# CT-EUS 血管重建 Pipeline 技术路线 v0.3/v0.3b

## Executive Summary

本轮首次对真实 CT-0021 跑完整自动血管候选重建。人工标注没有参与重建，只保留为后续评价输入。v0.3 是高召回候选版，v0.3b 是在观察到片状假阳性后立即收紧的保守版。

推荐团队优先查看 v0.3b 输出：

`/mnt/c/Users/zhangyutang/Desktop/CT-EUS血管重建结果/CT-EUS-vessel-pipeline-runs/CT-0021-v0.3b-conservative`

## Context

CT-0021 的动态期相已经在 v0.2 中修正：501 为 arterial，701 为 portal，901 为 venous。v0.3 以这三期为输入，先在各期原始空间识别候选，再把候选 mask 重采样到 portal 参考空间融合。

## Methods

### 数据红线

本轮重建只读取：

- 原始 CT DICOM
- TotalSegmentator 预训练模型输出
- 本轮运行产生的中间文件

本轮重建不读取：

- `reference_labels`
- 人工 NRRD 标签
- 任何人工标注派生参数

### 前置剔除与锚点

TotalSegmentator 在 portal 期 CT 上运行，输出器官、骨骼与血管锚点：

- 器官/骨骼剔除：liver, spleen, kidney_left, kidney_right, vertebrae_T12/L1-L5, sacrum
- 血管锚点：aorta, inferior_vena_cava, portal_vein_and_splenic_vein

骨骼另有 HU > 700 的硬排除兜底。

### 血管候选

每个期相分别执行：

- body mask：HU > -500 并保留最大连通域
- 动态 HU 窗：默认约 80-200 HU，按当前体数据估计
- slice-wise Frangi：`sigmas=(1.0, 2.0, 3.5)`，亮管状结构
- 连通域过滤：去除小碎片
- 保留 TotalSegmentator 血管锚点，避免主干断链

v0.3 使用较宽松的 Frangi/HU 联合门控；v0.3b 去掉全局填洞，并将 Frangi 门控提高到候选内第 70 百分位。

## Results

| version | fused voxels | arterial ref voxels | portal ref voxels | venous ref voxels | TotalSegmentator |
| --- | ---: | ---: | ---: | ---: | --- |
| v0.3-reconstruction | 5,118,493 | 1,102,062 | 3,586,452 | 3,851,921 | completed |
| v0.3b-conservative | 2,556,167 | 451,754 | 1,333,336 | 1,569,704 | completed |

v0.3b 输出包含：

- `masks/reference_ct_portal.nrrd`
- `masks/artery_candidate.nrrd`
- `masks/portal_vein_candidate.nrrd`
- `masks/systemic_vein_candidate.nrrd`
- `masks/vessel_fused_binary.nrrd`
- `masks/vessel_fused_multilabel.nrrd`
- `compat_nifti/*.nii.gz`
- `mesh/*.stl`
- `qc/bbox.json`
- `qc/reconstruction_summary.json`
- `qc/fused_slice_overlay.png`
- `qc/fused_mip_overlay.png`

## Discussion

v0.3b 已经实现端到端真实数据重建，但从 QC 图看仍有腹腔软组织、肝脾周边和部分体壁/骨旁片状假阳性。当前版本可作为第一版三维候选基线，不应作为最终临床级结果。

下一轮建议：

- 将 Frangi 从 slice-wise 升级为 spacing-aware 3D Hessian。
- 使用血管锚点做中心线区域生长，替代大范围 HU 阈值。
- 针对肝脏表面建立 surface distance suppression，降低肝脏表面片状伪影。
- 单独做 table/body 外部连通域剔除 QC。
- 在完成重建后，单独运行评价脚本读取人工标签计算 Dice/Hausdorff，评价阶段与重建阶段分离。

## Reproducibility Appendix

```bash
cd /home/zyt/ct_eus_vessel_pipeline
PYTHONPATH=src python -m ct_eus_vessel.cli reconstruct --config configs/default.yml --version v0.3-reconstruction
PYTHONPATH=src python -m ct_eus_vessel.cli reconstruct --config configs/default.yml --version v0.3b-conservative
pytest -q
```
