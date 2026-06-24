# CT-0021 v0.4 评价与改进方向

## Executive Summary

v0.4-expanded-exclusion 是当前三版中效果最好的自动重建结果。它没有读取或使用人工标注参与重建，只在重建完成后用 `label.txt` 和 `pseudo_label-.nii` 做评价。

相比 v0.3b，v0.4 将预测体素从 2,556,167 降到 1,208,875，Dice 从 0.2193 提升到 0.3703，Precision 从 0.1330 提升到 0.2686。胃、胰腺、十二指肠和未命名非血管 Label 19 的误重建显著下降。

主要残余问题仍然明确：未标注区域预测体素仍有 652,992 个，占预测结果 54.02%；最大 Hausdorff distance 仍为 103.7215 mm，说明仍存在远离目标血管的孤立假阳性或非血管片段。

## Context

本轮问题来自 3D Slicer 目视检查：两个早期版本均存在大量脊柱/肋骨、肺部/心脏、其他脏器和无关组织被重建进来的情况。

评价使用的人工相关文件为：

- `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\血管重建病例\label.txt`
- `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\血管重建病例\pseudo_label-.nii`
- `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\血管重建病例\ct.nii`

使用边界：

- 人工标注只用于评价和确定下一步改进方向
- 人工标注不进入重建输入、阈值估计、候选生成、训练、后处理或参数搜索
- 当前重建链路仍只使用真实 CT/DICOM、期相信息、HU、Frangi vesselness 和 TotalSegmentator 预训练输出

## Methods

血管类别按 `label.txt` 中的以下标签合并为评价目标：

- 8: Ao
- 9: IVC
- 10: PV
- 15: liver vein
- 25: celiac artery

v0.4 的核心改动是扩展 TotalSegmentator 前置排除结构，将下列高风险假阳性来源纳入排除：

- 肝、脾、肾、胆囊
- 胃、胰腺、食管、肠管、十二指肠、结肠、肾上腺
- 肺叶、心脏
- 胸腰椎、骶骨、肋骨、胸骨、肩胛骨、锁骨、肱骨

血管锚点仍保留为独立 ROI，不和排除 ROI 混用：

- arterial: aorta
- portal: portal_vein_and_splenic_vein
- venous: inferior_vena_cava

## Results

| Version | Prediction voxels | Dice | Precision | Recall | Hausdorff mm | Avg Hausdorff mm | Unlabeled voxels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.3-reconstruction | 5,118,493 | 0.1411 | 0.0781 | 0.7333 | 106.2424 | 15.7002 | 2,987,193 |
| v0.3b-conservative | 2,556,167 | 0.2193 | 0.1330 | 0.6240 | 103.7215 | 14.5824 | 1,286,783 |
| v0.4-expanded-exclusion | 1,208,875 | 0.3703 | 0.2686 | 0.5959 | 103.7215 | 11.6154 | 652,992 |

v0.4 相比 v0.3b 的主要非血管重叠变化：

| Label | v0.3b overlap | v0.4 overlap | Reduction |
| --- | ---: | ---: | ---: |
| Label 19 | 368,161 | 41,198 | 88.8% |
| stomach | 132,661 | 4,390 | 96.7% |
| pancreas | 70,913 | 3,012 | 95.8% |
| duodenum | 82,952 | 13,336 | 83.9% |
| Label 18 | 39,836 | 6,322 | 84.1% |
| spleen | 10,074 | 4,189 | 58.4% |

v0.4 中仍排名靠前的非血管重叠：

| Label | Label id | Overlap voxels |
| --- | ---: | ---: |
| Label 19 | 19 | 41,198 |
| liver | 6 | 18,873 |
| left kidney | 3 | 18,706 |
| right kidney | 2 | 13,444 |
| duodenum | 14 | 13,336 |
| Label 18 | 18 | 6,322 |
| stomach | 7 | 4,390 |
| spleen | 1 | 4,189 |
| pancreas | 11 | 3,012 |

## Discussion

v0.4 证明主要问题不是单纯阈值过宽，而是原始候选中存在大量“高 HU、非管状或弱管状、但和血管不连续”的结构。扩展器官/骨骼/胸腔结构排除后，胃、胰腺、十二指肠等明确脏器误分割显著下降。

但最大 Hausdorff 仍然很高，且未标注预测比例仍高，说明仍有远离血管主干的残留片段。这类误差不适合继续只靠 HU 阈值解决，因为过度收紧阈值会牺牲门静脉、肝静脉或小分支召回。

下一版应从“候选像不像血管”转向“候选是否属于目标血管拓扑网络”：

- 基于 aorta、IVC、PV/SV 预训练血管锚点做连通域保留
- 对未连接到锚点、且距离锚点过远的组件进行剔除
- 用 3D spacing-aware Hessian/vesselness 替代当前 slice-wise Frangi
- 对主干和 2-3 mm 小分支使用不同尺度的 component/radius 规则
- 对融合后的 mask 做骨架化和端点桥接，避免单纯过滤造成断链

## Appendix

v0.4 推荐在 3D Slicer 中优先查看：

- `masks/reference_ct_portal.nrrd`: 对齐参考 CT
- `masks/vessel_fused_multilabel.nrrd`: 三期融合多标签体
- `masks/vessel_fused_binary.nrrd`: 三期融合二值血管候选
- `mesh/vessel_fused_binary.stl`: 三期融合表面网格
- `mesh/artery_candidate.stl`: 动脉期候选网格
- `mesh/portal_vein_candidate.stl`: 门静脉期候选网格
- `mesh/systemic_vein_candidate.stl`: 静脉期候选网格
- `qc/evaluation_against_pseudo_label.json`: 仅评价用指标
- `qc/reconstruction_summary.json`: 重建参数与输出摘要
