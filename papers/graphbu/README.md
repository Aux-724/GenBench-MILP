# GraphBU: MILP Instance Generation with Graph-Native Block Units

## 基本信息

- **作者**: Xiaolei Guo, Chenyu Zhou, Jianghao Lin, Dongdong Ge
- **机构**: 上海交通大学
- **arXiv ID**: [2607.06532](https://arxiv.org/abs/2607.06532)
- **版本**: arXiv v1，2026-07-07
- **代码**: 论文当前未给出公开代码地址

## 摘要

GraphBU 使用带显式耦合接口的图原生块单元生成 MILP 实例。方法在变量—约束二分图中识别主约束和边界变量，把局部子问题及其与全局模型的接口共同保存为 block unit，再通过形状、接口维度、变量域和行元数据兼容性检查进行块替换。

## 方法要点

1. **图原生分解**：从二分图中检测耦合节点，并将剩余部分分解为局部模块。
2. **显式接口**：每个块同时保存局部结构、主约束接口和边界变量接口。
3. **兼容替换**：仅在局部形状、接口维度、约束方向和变量元数据兼容时替换块。
4. **理论性质**：讨论接口分离、接口松弛条件下的可行替换，以及对行列置换的不变性。
5. **下游验证**：评测图统计相似度、可行率、非平凡求解行为及 Predict-and-Search 数据增强效果。

## 与 GenBench-MILP 的关系

- 是直接面向目标 MILP family 的通用实例生成方法。
- 与 MILP-StuDio 都使用块结构，但 GraphBU 把耦合接口作为块的一部分，避免仅依赖矩阵行列布局。
- 论文使用修改比例控制替换预算，与当前统一 eta 网格具有较好的概念对应关系。

## 引用

```bibtex
@article{guo2026graphbu,
  title={GraphBU: MILP Instance Generation with Graph-Native Block Units},
  author={Guo, Xiaolei and Zhou, Chenyu and Lin, Jianghao and Ge, Dongdong},
  journal={arXiv preprint arXiv:2607.06532},
  year={2026}
}
```
