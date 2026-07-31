# Generation Techniques for Linear Programming Instances with Controllable Properties

## 基本信息

- **作者**: Simon Bowly, Kate Smith-Miles, Davaatseren Baatar, Hans Mittelmann
- **期刊**: Mathematical Programming Computation, 12(3), 389--415, 2020
- **DOI**: [10.1007/s12532-019-00170-6](https://doi.org/10.1007/s12532-019-00170-6)
- **期刊全文**: https://link.springer.com/content/pdf/10.1007/s12532-019-00170-6.pdf
- **预印本记录**: https://optimization-online.org/2017/04/5976/（预印本标题包含 “linear and integer programming”）
- **代码**: https://github.com/simonbowly/lp-generators

## 摘要

该工作把线性规划实例生成和实例空间搜索映射到一个替代编码空间，从而构造可行、有界且属性可控的实例。参数化生成器与搜索方法只产生可行有界实例，并可针对密度、系数、退化性和求解难度等特征覆盖更丰富的实例空间。

## 方法要点

1. **可行有界编码**：通过已知原始—对偶解构造松弛，避免先随机生成再丢弃无效实例。
2. **可控统计属性**：支持控制规模、密度、系数分布、变量界和与求解行为相关的特征。
3. **实例空间搜索**：使用搜索过程生成随机采样难以覆盖、或对求解器更困难的实例。
4. **MIP 扩展**：公开代码将其描述为随机 MIP 实例生成包，并支持添加整数和二元变量限制。

## 与 GenBench-MILP 的关系

- 是 G2MILP、MILPGen 等方法采用的传统生成基线。
- 与现代分布学习方法相比，它保证基本数学有效性，但主要匹配手工指定的统计属性。
- 可作为非深度学习基线，检验复杂生成模型相对传统可控采样的增益。

## 引用

```bibtex
@article{bowly2020generation,
  title={Generation Techniques for Linear Programming Instances with Controllable Properties},
  author={Bowly, Simon and Smith-Miles, Kate and Baatar, Davaatseren and Mittelmann, Hans},
  journal={Mathematical Programming Computation},
  volume={12},
  number={3},
  pages={389--415},
  year={2020},
  doi={10.1007/s12532-019-00170-6}
}
```
