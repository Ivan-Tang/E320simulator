# E320 ML Seeding 研究提案

## 研究背景与动机

### 粒子径迹重建的挑战

在高能物理实验中，粒子径迹重建是从探测器击中点（hits）中重建带电粒子轨迹的关键步骤。随着大型强子对撞机（LHC）升级至高亮度LHC（HL-LHC），数据率将提高一个数量级，传统基于 combinatorial Kalman filter 的径迹重建方法面临计算复杂度爆炸的挑战。机器学习方法，尤其是图神经网络（GNN）和 Transformer，已在大型对撞机实验中展示出显著潜力。

### E320实验的特殊需求

E320 是一个原型追踪器实验，旨在测量电子-激光碰撞产生的单个正电子（Borysov et al., "Preliminary experience with the E320 Prototype Tracker"）。该实验具有以下特点：
- **低信噪比**：需要从背景中识别稀疏的正电子信号
- **高精度要求**：精确测量正电子的能量和角度分布
- **实时处理需求**：实验数据需要快速重建以指导实验运行
- **探测器几何特殊性**：5层 ALPIDE 传感器，与 LHC 大型探测器在几何结构和物理过程上均有显著差异

现有 ML 方法主要针对大型对撞机实验（如 ATLAS、CMS）设计，需要针对 E320 进行专门优化。

---

## 问题定义

给定一组探测器击中点 H = {h₁, h₂, ..., h_N}，每个击中点包含位置 (x, y, z)、时间 t、能量沉积等信息，目标是将击中点分组为**种子**（初始径迹段）S = {s₁, s₂, ..., s_K}，每个种子包含少量（通常3-5个）属于同一粒子径迹的击中点。

核心科学问题：如何设计高效、鲁棒的 ML seeding 算法，使其能够：
1. 适应 E320 实验的特定探测器几何和物理过程
2. 在低信噪比条件下保持高效率和纯度
3. 实现快速推理以满足实时处理需求（<10 ms/event）
4. 具备良好的泛化能力以适应实验条件变化

---

## 方向一：基于 GNN 的 Seeding（EvoHierGNN）

### 架构设计

提出 **EvoHierGNN**（Evolving Hierarchical Graph Neural Network）架构，将击中点建模为图节点，融合三个核心模块：

**1. 动态图构建模块**
- 基于 k 最近邻（k-NN）的初始图构建
- 利用注意力机制学习边权重，动态修剪低权重边
- 适应 E320 探测器的几何约束（5层结构，层间连接优先）

**2. 分层图学习模块**（借鉴 Liu et al., 2023 的 GMPool 机制）
- 生成不同层次的抽象表示
- 底层表示捕获局部击中点关系
- 高层表示对应完整的径迹模式
- 允许不连通的击中点分配给同一径迹

**3. 演化图更新模块**（借鉴 Calafiura et al., 2024 的 EggNet）
- 采用递归图注意力网络与演化图结构
- 每次迭代后基于节点嵌入重新计算边连接
- 实现图结构与节点表示的共同演化
- 直接从击中点点云重建粒子径迹，无需预构建固定图

### 训练策略
- **多任务学习**：同时优化种子分类、径迹参数回归和噪声抑制
- **课程学习**：从简单场景逐渐过渡到复杂场景
- **自监督预训练**：利用无标签数据学习通用几何和物理特征
- **损失函数**：Focal Loss（类别不平衡）+ Huber Loss（参数回归）+ 对比损失（嵌入相似性）+ 稀疏损失（图稀疏性）

### 技术优势
| 特性 | 说明 |
|------|------|
| 计算复杂度 | O(E)，E 为边数，随击中点数线性增长 |
| 局部几何 | 消息传递机制天然捕捉局部空间关系，适合探测器层间结构 |
| 物理先验 | 图结构可直接编码 E320 几何约束 |
| 可扩展性 | 适合大规模并行计算 |

### 里程碑

| 阶段 | 时间 | 主要目标 |
|------|------|----------|
| 第一阶段：基础研究 | 第1-3个月 | 深入分析 E320 数据；复现 Choma et al.、Liu et al.、Calafiura et al. 方法；建立评估框架 |
| 第二阶段：算法开发 | 第4-8个月 | 设计 EvoHierGNN；实现动态图构建和分层学习模块；开发训练评估流程 |
| 第三阶段：实验验证 | 第9-12个月 | 系统测试；消融研究；性能与计算效率优化 |
| 第四阶段：部署应用 | 第13-15个月 | 集成到 E320 重建软件栈；在真实数据上验证；撰写论文 |
| 第五阶段：总结拓展 | 第16-18个月 | 总结成果；探索在 LHCb、Belle II 的应用；完成报告 |

---

## 方向二：基于 Transformer 的 Seeding（TrackFormer-Seed）

### 架构设计

受 Stroud et al. (2024) 启发，提出 **TrackFormer-Seed** 架构，采用编码器-解码器结构，直接将击中点集合映射为种子候选：

```
┌─────────────────────────────────────────────────────────────┐
│                    TrackFormer-Seed Architecture            │
├─────────────────────────────────────────────────────────────┤
│  Input: Hit Cloud (N hits)                                  │
│                                                             │
│  Stage 1: Hit Embedding & Encoding                          │
│  ├─ Hit Feature Extraction (MLP)                            │
│  ├─ Positional Encoding (3D sinusoidal)                     │
│  └─ Transformer Encoder (L₁ layers)                         │
│                                                             │
│  Stage 2: Seed Proposal Generation                          │
│  ├─ Learnable Seed Queries (K queries)                      │
│  ├─ Transformer Decoder (L₂ layers)                         │
│  └─ Seed Prediction Heads                                   │
│     ├─ Seed Confidence (sigmoid)                            │
│     ├─ Seed Parameters (position, direction, curvature)     │
│     └─ Hit Assignment Masks (N×K attention weights)         │
│                                                             │
│  Stage 3: Seed Refinement & Filtering                       │
│  ├─ Non-Maximum Suppression (NMS)                           │
│  ├─ Confidence Thresholding                                 │
│  └─ Hit Clustering (based on assignment masks)              │
│                                                             │
│  Output: K' seed candidates with associated hits            │
└─────────────────────────────────────────────────────────────┘
```

**Stage 1 – Hit Embedding & Encoding**
- MLP 将原始特征映射到 d_model 维
- 3D 正弦位置编码 PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- L₁ 层 Transformer Encoder（多头自注意力 + FFN + LayerNorm）

**Stage 2 – Seed Proposal Generation**（借鉴 DETR / MaskFormer）
- K 个可学习种子查询 Q_seed ∈ ℝ^(K×d_model)
- L₂ 层 Transformer Decoder（交叉注意力关注编码后击中点 + 自注意力）
- 并行预测头：种子置信度（sigmoid）、种子参数（位置/方向/曲率）、击中点分配矩阵 A ∈ ℝ^(K×N)

**Stage 3 – 细化与过滤**
- NMS 去除冗余种子
- 置信度阈值过滤（p_conf > τ）
- 基于注意力矩阵 A 聚类分配击中点

### E320-specific 优化
- **背景查询**：引入特殊"背景查询"专门吸收噪声击中点，应对低信噪比
- **相对位置编码**：编码击中点间相对位置关系，增强几何感知
- **曲率感知注意力**：注意力机制融入曲率先验，偏好符合螺旋轨迹的击中点
- **磁场约束**：损失函数中加入磁场约束项，确保种子参数符合洛伦兹力方程
- **稀疏注意力**：降低 O(N²) 复杂度，满足实时性要求

### 训练策略
多任务损失：`L_total = λ_cls·L_cls + λ_param·L_param + λ_assign·L_assign + λ_reg·L_reg`
- L_cls：Focal Loss，处理正负样本不平衡
- L_param：Huber Loss，回归种子参数
- L_assign：基于匈牙利匹配的分配损失
- L_reg：注意力矩阵稀疏性正则化

推荐超参数起点：d_model=256，L₁=6，L₂=6，8 attention heads，FFN dim=1024，K=100，lr=1e-4 with warmup。

### 技术优势
| 特性 | 说明 |
|------|------|
| 全局上下文 | 自注意力机制捕捉整个事件中的全局模式，有助于识别稀疏正电子信号 |
| 并行化 | 完全并行，无消息传递顺序依赖，GPU 加速效率高 |
| 端到端 | 无需手工设计图结构，直接从点云生成种子 |
| 噪声抑制 | 注意力权重自动抑制无关击中点 |

### 里程碑

| 阶段 | 时间 | 主要目标 |
|------|------|----------|
| 第一阶段：环境与基准 | 第1-2个月 | 数据准备；复现 Stroud et al. (2024) 和 Choma et al. (2020)；建立评估流水线 |
| 第二阶段：原型开发 | 第3-5个月 | 实现 TrackFormer-Seed 核心模块；小规模数据训练与消融研究 |
| 第三阶段：E320优化 | 第6-8个月 | 噪声抑制、几何约束、稀疏注意力、模型量化 |
| 第四阶段：系统集成 | 第9-10个月 | 集成到 E320 重建框架；全面测试（信噪比、实时性、鲁棒性） |
| 第五阶段：总结拓展 | 第11-12个月 | 性能分析；知识迁移到其他实验；撰写论文，开源代码 |

---

## 两方向对比与选择策略

| 维度 | EvoHierGNN（方向一） | TrackFormer-Seed（方向二） |
|------|----------------------|--------------------------|
| **计算复杂度** | O(E)，随击中点数近线性 | O(N²)，击中点数较多时需稀疏注意力 |
| **局部/全局建模** | 优于局部几何关系 | 优于全局模式捕捉 |
| **图结构依赖** | 需要初始图构建（k-NN） | 无需预构建图，直接处理点云 |
| **物理先验融入** | 图结构可直接编码几何约束 | 通过位置编码和损失函数融入 |
| **并行化效率** | 消息传递存在顺序依赖 | 完全并行，GPU 友好 |
| **训练数据需求** | 中等 | 较高（端到端需要更多数据） |
| **已有代码基础** | `ResGNN`、`EggNet`、`HierarchicalGNN` 已实现 | `TransformerEdgeClassifier`、`train_trackformer.py` 已实现 |
| **预期开发周期** | 18 个月 | 12 个月 |

**选择策略**：两个方向并行探索，共用数据预处理流水线和评估框架。
- **短期（0-6 个月）**：优先推进 TrackFormer-Seed（周期更短，代码框架已有基础），同时在小规模实验中验证 EvoHierGNN 核心模块。
- **中期（6-12 个月）**：根据两个方向的初步实验结果，决定是否将资源集中于表现更优的方向，或探索两者的融合架构（如 GNN 特征提取 + Transformer 全局推理）。
- **融合探索**：可考虑以 GNN 输出的局部嵌入作为 Transformer 的输入，结合两者优势。

---

## 统一评估指标

所有方法在相同的 E320 模拟数据集上评估：

| 类别 | 指标 | 目标值 |
|------|------|--------|
| **效率** | 种子效率（Signal seeding efficiency） | ≥ 95% |
| **效率** | 径迹重建效率 | ≥ 95% |
| **纯度** | 误判率（Fake rate） | ≤ 5% |
| **纯度** | 假阳性率 | ≤ 4% |
| **质量** | 种子参数（动量、方向）相对误差 | ≤ 5% |
| **质量** | 击中点分配准确率 | ≥ 97% |
| **实时性** | 推理时间（V100/RTX 3090 GPU） | ≤ 10 ms/event |
| **内存** | GPU 内存占用 | ≤ 2 GB |

**基准方法**（共同比较对象）：
1. 传统方法：斜率窗口 + 链式种子（Baseline）
2. 传统方法：Hough 变换追踪器
3. 嵌入空间 GNN（Choma et al., 2020）
4. 分层 GNN（Liu et al., 2023）
5. 演化图注意力网络 EggNet（Calafiura et al., 2024）
6. Transformer 径迹重建（Stroud et al., 2024）

**消融研究**：通过系统移除或修改各架构组件，评估每个设计选择的贡献。

**泛化能力测试**：在不同信噪比、探测器效率、校准误差条件下测试鲁棒性。

---

## 参考文献

1. Choma, N., et al. (2020). "Track Seeding and Labelling with Embedded-space Graph Neural Networks." arXiv:2007.00149.

2. Liu, R., et al. (2023). "Hierarchical Graph Neural Networks for Particle Track Reconstruction." arXiv:2303.01640.

3. Calafiura, P., et al. (2024). "EggNet: An Evolving Graph-based Graph Attention Network for Particle Track Reconstruction." arXiv:2407.13925.

4. Stroud, A., et al. (2024). "Transformers for Charged Particle Track Reconstruction in High Energy Physics." (TrackML SOTA: 97% efficiency, 0.6% fake rate, 100 ms inference.)

5. Borysov, O., et al. "Preliminary experience with the E320 Prototype Tracker."

6. Ju, X., et al. (2021). "Graph Neural Networks for Particle Reconstruction in High Energy Physics." Nature Reviews Physics, 3(10), 676-688.

7. Qasim, S. R., et al. (2022). "Learning the Language of Particle Tracking." Machine Learning: Science and Technology, 3(1), 015003.

8. Exa.TrkX Collaboration. (2020). "TrackML Particle Tracking Challenge." https://www.kaggle.com/c/trackml-particle-identification

---

*本提案整合自 `research_proposal_GNN_seeding_E320.md`（EvoHierGNN）和 `transformer_seeding_E320_proposal.md`（TrackFormer-Seed），以两个方向并列探索为主线。具体技术细节需与 E320 实验组成员进一步讨论确定。*
