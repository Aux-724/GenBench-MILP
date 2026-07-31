# MILPGen: Learning to Generate Scalable MILP Instances

## 基本信息

- **作者**: Tianxing Yang, Huigen Ye, Hua Xu
- **会议**: GECCO 2024 Companion
- **DOI**: [10.1145/3638530.3661481](https://doi.org/10.1145/3638530.3661481)
- **论文来源**: https://huigenye.com/PDF/yang2024learning.pdf
- **代码**: https://github.com/thuiar/MILPGen

## 摘要

MILPGen 是一个面向可扩展 MILP 实例的生成框架。它将 MILP 表示为变量—约束二分图，先把图拆解为树状结构，再训练判别模型完成节点合并和结构重建。生成阶段可以拼接来自同一问题类别的树结构，从而产生规模大于训练样本的新实例。

## 方法要点

1. **节点拆分与合并**：把复杂二分图分解为树结构，并学习哪些同侧节点应当重新合并。
2. **可扩展结构构造**：拼接同一类别的多个树结构，再执行节点合并，生成更大规模实例。
3. **结构保持评测**：在 MIS、组合拍卖和最小顶点覆盖上比较生成实例与原始实例的结构和求解难度。
4. **下游数据增强**：验证生成实例对学习型优化框架的训练价值。

## 与 GenBench-MILP 的关系

- 属于从原始实例学习并生成同类别 MILP 的直接生成方法。
- 支持同规模生成和规模外推，适合评测结构保真度、数学有效性和求解难度。
- 官方仓库存在，但 README 标注部分生成脚本仍在建设中，复现前需要单独审计代码完整性。

## 引用

```bibtex
@inproceedings{yang2024learning,
  title={Learning to Generate Scalable MILP Instances},
  author={Yang, Tianxing and Ye, Huigen and Xu, Hua},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  pages={159--162},
  year={2024},
  doi={10.1145/3638530.3661481}
}
```
