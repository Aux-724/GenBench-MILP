# ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation

## 基本信息

- **作者**: Ziao Guo, Yang Li, Chang Liu, Wenli Ouyang, Junchi Yan
- **机构**: 上海交通大学
- **代码**: https://github.com/Thinklab-SJTU/ACM-MILP

## 摘要

数据在混合整数线性规划（MILP）的经典方法和基于学习的方法的发展中起着关键作用。然而，现实应用中数据的稀缺性凸显了MILP实例生成方法的必要性。目前，这些方法主要依赖于迭代随机单约束修改，忽视了具有约束相互关系的基本问题结构，从而导致质量和可解性受损。本文提出ACM-MILP，一个MILP实例生成框架，实现自适应约束修改和约束相互关系建模。它采用基于潜在空间中概率估计的自适应约束选择机制来保留实例特征。同时，它通过社区检测检测和分组强相关约束，实现考虑约束依赖性的集体修改。实验结果显示，该框架在问题求解难度相似性方面取得了显著改进。此外，在下游任务中，展示了生成实例用于超参数调优的有效性。

## 核心动机

1. **现有方法的局限性**:
   - 主要依赖迭代随机单约束修改
   - 忽视约束之间的相互关系
   - 导致生成质量和可解性受损

2. **关键洞察**: 约束之间存在相互关系，集体修改比单约束修改更能保留问题结构

3. **方法核心**:
   - 自适应约束选择机制（基于潜在空间概率估计）
   - 社区检测来分组强相关约束
   - 集体修改考虑约束依赖性

## 方法

### 1. MILP 二分图表示

- 将MILP实例表示为加权二分图
- 约束顶点和变量顶点
- 非零约束系数作为边

### 2. ACM-MILP 框架

**编码器**: 学习约束和变量的潜在表示
- 约束首先被分组并划分为社区
- 编码器产生潜在空间表示

**解码器**: 生成新的约束顶点
- 在选定的社区中生成新的约束顶点
- 以及连接到该社区的边

**关键创新**:
- 利用VAE（变分自编码器）编码约束分布
- 通过社区检测识别强相关约束组
- 自适应约束选择保留实例特征

## 实验结果

- 在问题求解难度相似性方面取得显著改进
- 生成实例在下游任务（超参数调优）中展示有效性

## 引用

```bibtex
@article{guo2024acmmilp,
  title={ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation},
  author={Guo, Ziao and Li, Yang and Liu, Chang and Ouyang, Wenli and Yan, Junchi},
  journal={},
  year={2024}
}
```
