# DIG-MILP: A Deep Instance Generator for Mixed-Integer Linear Programming with Feasibility Guarantee

## 基本信息

- **作者**: Haoyu Wang, Jialin Liu, Xiaohan Chen, Xinshang Wang, Pan Li, Wotao Yin
- **机构**: Georgia Tech, Damo Academy (Alibaba US)
- **arXiv ID**: [2310.13261](https://arxiv.org/abs/2310.13261)
- **代码**: https://github.com/Graph-COM/DIG_MILP.git

## 摘要

混合整数线性规划（MILP）是一个著名的NP难问题，对众多关键工业应用至关重要。有效的算法开发、求解器调优以及用于MILP求解的机器学习模型训练都依赖于广泛、多样且具有代表性的数据。然而，与图像和文本领域中丰富的自然数据相比，MILP明显数据匮乏，这凸显了合成数据生成的重要角色。本文提出DIG-MILP，一个基于变分自编码器（VAE）的生成框架，从有限的MILP数据中提取深层结构特征并生成反映目标数据的实例。它利用MILP对偶性来保证正确且完整的生成空间，并确保有界性和可行性。在两个下游任务上验证：数据共享用于求解器调优和数据增强用于ML模型。

## 核心动机

1. **数据匮乏**: 与图像和文本领域相比，MILP数据明显不足
2. **现有方法的局限**: 现有合成方法要么依赖专家设计的公式，要么难以捕捉真实世界实例的丰富特征
3. **可行性保证的需求**: 生成实例需要保证可行性和有界性

## 方法

### 1. MILP 表示为图

DIG-MILP 将MILP实例编码为图结构:
- **约束矩阵 A**: 表示约束系数
- **变量向量 x**: 决策变量
- **目标系数 y**: 目标函数系数
- **约束边界 s**: 约束的右端项
- **变量边界 r**: 变量的上下界

### 2. VAE 生成框架

**编码器** $q_\phi(\mathbf{z}|G)$:
- 使用图神经网络获取节点表示
- 学习约束和变量的潜在分布

**解码器** $p_\theta(G|\tilde{G}, \mathbf{z})$:
- 从潜在变量和完整图重建图结构
- 生成新的约束顶点及其连接边

### 3. 可行性保证

利用MILP对偶理论:
- **正确且完整的生成空间**: 确保生成空间覆盖所有可能的可行实例
- **有界性保证**: 确保生成实例的目标函数有界
- **可行性保证**: 确保生成实例至少存在一个可行解

## 关键创新

1. **基于VAE的深度生成框架**: 首次将VAE应用于MILP实例生成
2. **MILP对偶性利用**: 通过对偶理论保证生成实例的可行性和有界性
3. **深层结构特征提取**: 从有限数据中学习和生成反映目标分布的实例

## 实验

在两个下游任务上验证:
1. **数据共享用于求解器调优**: 生成实例用于调整求解器参数
2. **数据增强用于ML模型**: 使用生成实例增强训练数据

## 引用

```bibtex
@article{wang2023dig,
  title={DIG-MILP: A Deep Instance Generator for Mixed-Integer Linear Programming with Feasibility Guarantee},
  author={Wang, Haoyu and Liu, Jialin and Chen, Xiaohan and Wang, Xinshang and Li, Pan and Yin, Wotao},
  journal={arXiv preprint arXiv:2310.13261},
  year={2023}
}
```
