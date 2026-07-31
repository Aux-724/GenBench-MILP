# G2MILP: A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability

## 基本信息

- **作者**: Zijie Geng, Xijun Li, Jie Wang, Xiao Li, Yongdong Zhang, Feng Wu
- **机构**: 中国科学技术大学, 华为诺亚方舟实验室
- **发表会议**: NeurIPS 2023
- **arXiv ID**: [2310.02807](https://arxiv.org/abs/2310.02807)
- **代码**: https://github.com/MIRALab-USTC/L2O-G2MILP
- **项目页面**: https://miralab-ustc.github.io/L2O-G2MILP

## 摘要

G2MILP是**第一个针对MILP实例的深度学习生成框架**。它将MILP实例表示为二分图，并应用**掩码变分自编码器（Masked VAE）**来迭代地损坏和替换原始图的部分以生成新的实例。G2MILP的吸引人之处在于，它可以在没有先验专家设计公式的情况下学习生成新颖且逼真的MILP实例，同时保留真实世界数据集的**结构和计算难度**。生成的实例可以促进在有限数据可用性下增强MILP求解器的下游任务。

## 核心动机

1. **真实世界MILP实例的有限可用性**: 由于劳动密集型的数据收集和专有性问题，获取大量真实MILP实例仍然是一个关键挑战
2. **现有方法的局限性**: 现有方法要么严重依赖专家设计的公式（如TSP、集合覆盖），要么难以捕捉真实世界实例的丰富特征
3. **机器学习方法的训练需求**: 改进MILP求解器的机器学习方法需要大量i.i.d.训练数据

## 方法

### 1. MILP表示为二分图

G2MILP将每个MILP实例构建为一个加权二分图 $\mathcal{G} = (\mathcal{V} \cup \mathcal{W}, \mathcal{E})$:

- **约束顶点集** $\mathcal{V} = \{v_1, \cdots, v_m\}$: 每个 $v_i$ 对应第 $i$ 个约束
  - 顶点特征 $v_i$ 由偏置项描述: $v_i = (b_i)$
- **变量顶点集** $\mathcal{W} = \{w_1, \cdots, w_n\}$: 每个 $w_j$ 对应第 $j$ 个变量
  - 顶点特征 $w_j$ 是9维向量，包含目标系数 $c_j$、变量类型和边界 $l_j, u_j$
- **边集** $\mathcal{E} = \{e_{ij}\}$: 边 $e_{ij}$ 连接约束顶点 $v_i$ 和变量顶点 $w_j$
  - 边特征 $e_{ij} = (a_{ij})$，即系数。如果 $a_{ij} = 0$，则无边

### 2. 掩码VAE范式

G2MILP的核心是一个**掩码变分自编码器**，灵感来自掩码语言模型和VAE:

**掩码过程**: 给定图 $\mathcal{G}$，通过掩码过程 $\tilde{\mathcal{G}} \sim \tilde{p}(\tilde{\mathcal{G}}|\mathcal{G})$ 进行损坏:
- 均匀采样一个约束顶点 $\tilde{v} \sim \mathcal{U}(\mathcal{V})$
- 掩码它及其相邻边，同时保持变量顶点不变
- 用特殊的[mask]标记标记顶点 $\tilde{v}$
- 添加连接 $\tilde{v}$ 与每个变量顶点的虚拟边

**编码器** $q_\phi(\mathbf{Z}|\mathcal{G})$:
- 使用GNN获取节点表示
- 通过两个网络 $\mu_\phi$ 和 $\Sigma_\phi$ 获取潜在向量

**解码器** $p_\theta(\mathcal{G}|\tilde{\mathcal{G}}, \mathbf{Z})$:
解码器由四个协同模块组成，用于重建被掩码的约束顶点:
1. **Bias Predictor**: 预测偏置项 $b_{\tilde{v}}$
2. **Degree Predictor**: 预测约束的度 $d_{\tilde{v}}$（涉及的变量数）
3. **Logits Predictor**: 预测变量顶点是否与约束相连
4. **Weights Predictor**: 预测边的权重（系数）

### 3. 训练目标

使用**证据下界（ELBO）**进行优化:

$$\mathcal{L} = \mathbb{E}_{\mathcal{G} \sim \mathcal{D}} \mathbb{E}_{\tilde{\mathcal{G}} \sim \tilde{p}(\tilde{\mathcal{G}}|\mathcal{G})} \left[ \underbrace{\mathbb{E}_{\mathbf{Z} \sim q_\phi(\mathbf{Z}|\mathcal{G})} \left[ -\log p_\theta(\mathcal{G}|\tilde{\mathcal{G}}, \mathbf{Z}) \right]}_{\mathcal{L}_{\text{rec}}} + \beta \cdot \underbrace{D_{\text{KL}} \left[ q_\phi(\mathbf{Z}|\mathcal{G}) \| \mathcal{N}(\mathbf{0}, \mathbf{I}) \right]}_{\mathcal{L}_{\text{prior}}} \right]$$

### 4. 推理

- 丢弃编码器，从标准高斯分布中采样 $\mathbf{Z}$
- 迭代地掩码一个约束顶点并用生成的顶点替换
- 超参数 $\eta$ 控制迭代次数与约束数的比例
- 较大的 $\eta$ 产生更新颖的实例，较小的 $\eta$ 产生更好的相似性

## 评估指标

设计了三个评估基准:

1. **结构分布相似性**: 使用11个经典统计量（系数密度、节点度、图聚类、图模块度等），计算Jensen-Shannon散度
2. **计算难度**: 使用Gurobi求解实例，报告求解时间和分支节点数
3. **下游任务**:
   - 最优值预测: 使用GNN预测最优值
   - 预测-搜索框架: 用于求解MILP

## 实验结果

### 数据集
- **大规模**: Maximum Independent Set (MIS), Set Covering
- **中等规模**: Mixed-integer Knapsack (MIK)
- **小规模**: MIPLIB 2017中的Nurse Scheduling问题

### 主要结果

| 数据集 | 方法 | 结构相似性 | 求解时间 | 可行性 |
|--------|------|-----------|---------|--------|
| MIS | G2MILP | 0.997 | 0.354s (1.5%) | 100% |
| SetCover | G2MILP | 0.835 | 2.360s (0.8%) | 100% |
| MIK | G2MILP | 0.991 | 0.169s (14.7%) | 100% |

- G2MILP生成的实例在**结构和计算难度**方面都与训练集非常相似
- 使用生成的实例扩充训练数据，最优值预测的MSE降低了**73.7%**（MIK）和**24.3%**（Nurse Scheduling）
- G2MILP自动学习产生可行和有界实例，无需施加显式规则

## 关键贡献

1. **首个深度生成框架**: G2MILP是第一个针对通用MILP实例的深度学习生成框架
2. **无需专家知识**: 不需要先验专家设计的公式即可生成实例
3. **结构和难度保持**: 同时保持原始数据集的图结构和计算难度
4. **下游任务增强**: 生成的实例可以显著提高下游任务的性能

## 局限性与未来方向

- 仅迭代地损坏和替换约束顶点，更多样化的掩码方案值得探索
- 更复杂的设计可以控制实例的关键属性（如可行性）
- 可以开发基于生成模型的预训练模型，有益于跨领域的下游任务
