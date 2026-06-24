# CT-0021 v0.5 强约束血管过滤路线

## Executive Summary

v0.5 是在“强约束优先”思路下完成的真实重建版本。它没有使用人工标注参与重建，只在完成后做评价。

这版把问题从“尽量多找血管”改成“先证明它像目标期相的血管，再允许它保留”。结果是 Precision 从 v0.4 的 0.2686 提升到 0.7483，False Positive 从 884,212 降到 98,852，未标注区域占比从 54.02% 降到 15.28%。

## Context

上一版最严重的问题是大量肝外非血管组织仍被重建进来。根因不是单一阈值，而是：

- HU 窗过宽，增强器官边缘和血管混在一起
- 只做了 OR 型多期融合，没有硬性连通性约束
- 2D Frangi 容易接受片状或网状高亮结构
- 骨骼/胸腔/胃肠道排除没有物理膨胀，边界漏入明显

## Methods

v0.5 的主链路：

1. TotalSegmentator 预训练 ROI 前置剔除
2. 器官/骨骼 mask 物理膨胀
3. phase-specific HU window，依据 seed 分布自适应
4. slabbed 3D Frangi，避免整体内存爆炸
5. seed-connected growth，只保留和 aorta / PV-SV / IVC 锚点连通的候选
6. 多期融合

### Phase windows

- arterial: `224.0 - 628.0`
- portal: `124.0 - 242.0`
- venous: `110.0 - 210.0`

### Morphology

- bone dilation: `1.5 mm`
- exclusion dilation: `2.5 mm`
- seed override: seed 位置不被 exclusion 切掉

## Results

| Version | Dice | Precision | Recall | Hausdorff mm | Avg Hausdorff mm | Prediction voxels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.4-expanded-exclusion | 0.3703 | 0.2686 | 0.5959 | 103.7215 | 11.6154 | 1,208,875 |
| v0.5-strong-vessel-filter | 0.6270 | 0.7483 | 0.5395 | 76.8809 | 5.2920 | 392,806 |

### Residual overlaps

- liver: 10,361 voxels
- duodenum: 1,644 voxels
- pancreas: 535 voxels
- Label 16: 181 voxels
- right adr: 155 voxels
- Label 19: 93 voxels

## Discussion

v0.5 把“骨骼/器官/大块软组织混入”的问题明显压下去了，说明强约束策略是对的。

但 Recall 下降到 0.5395，说明仍然存在过筛偏紧的情况，尤其是细小支路和局部肝内血管。下一步不应该回到宽阈值，而应该做：

- 只在肝门区和血管近邻做局部回填
- 对小分支做单独的细尺度候选策略
- 把 slabbed 3D Frangi 迁移到 GPU 或先做 ROI 裁剪

## Appendix

Slicer 重点查看：

- `masks/vessel_fused_binary.nrrd`
- `masks/vessel_fused_multilabel.nrrd`
- `mesh/vessel_fused_binary.stl`
- `mesh/artery_candidate.stl`
- `mesh/portal_vein_candidate.stl`
- `mesh/systemic_vein_candidate.stl`
- `qc/bbox.json`
- `qc/evaluation_against_pseudo_label.json`
- `qc/reconstruction_summary.json`
- `qc/fused_slice_overlay.png`
- `qc/fused_mip_overlay.png`
