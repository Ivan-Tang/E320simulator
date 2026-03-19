# 实验进度记录

## 当前阶段

**阶段**：ML 模型预训练准备期
**时间**：2026年3月
**状态**：Baseline 和 Hough 算法已完成并可评估；ML 模型框架已搭建完毕；尚未开始系统性训练实验。

---

## 已完成工作（按时间倒序）

### 2026年3月（当前）
- 整合两份 ML seeding 研究提案（GNN + Transformer）为统一文档 `research_proposal_ML_seeding.md`
- 建立本进度记录文档和会话工作规范（CLAUDE.md）

### ML 框架搭建（已完成）
- `src/models.py`：实现 `EdgeMLP`、`InteractionNet`、`ResGNN`、`EggNet`、`HierarchicalGNN`、`TransformerEdgeClassifier`、`Embedder`、`TransformerEmbedder`
- `src/layers.py`：实现 `MLP` 工厂、`PositionalEncoding3D`、`MultiHeadAttention`、Transformer 层
- `src/losses.py`：实现 `FocalLoss`（类别不平衡）和 `HingeLoss`（度量学习）
- `src/train.py`：边分类模型统一训练循环
- `src/train_embedder.py`：Metric-learning 嵌入器训练 + KDTree 推理
- `src/train_trackformer.py`：Transformer 端到端轨迹寻找

### Hough 变换追踪器（已完成）
- `src/hough_baseline.py`：在 (θ, ρ) 空间的 Hough 变换追踪器
- 脚本：`scripts/run_hough.py`

### Baseline 算法（已完成）
- `src/baseline.py`：斜率窗口边构建 + 贪心链式种子生成 + 3D 直线拟合
- 脚本：`scripts/run_baseline.py`

### 数据与模拟（已完成）
- `src/simulator.py`：集群级快速模拟器（SimConfig 控制信号率、噪声、集群大小模型）
- `src/geometry.py`：ALPIDE 传感器参数，5层几何，坐标变换（像素→芯片局部→TRK→LAB）
- 数据路径：`/storage/agrp/yiwen/data_Run502`（Lustre，未备份）

---

## 关键实验结果（表格）

| 方法 | 种子效率 | 误判率 | 推理时间 | 备注 |
|------|----------|--------|----------|------|
| Baseline（斜率窗口+链式）| — | — | — | 待系统评估 |
| Hough 变换 | — | — | — | 待系统评估 |
| ML 模型（EdgeMLP 等）| — | — | — | 尚未训练 |
| EvoHierGNN | — | — | — | 尚未训练 |
| TrackFormer-Seed | — | — | — | 尚未训练 |

*目标值：种子效率 ≥95%，误判率 ≤5%，推理时间 ≤10 ms/event（V100/RTX 3090）*

---

## 已知问题 / Blockers

- ML 模型尚无系统性训练结果，缺乏与 baseline 的定量对比
- 需要确认 E320 模拟数据（GEANT4）的规模和质量，以支持 ML 训练
- `scripts/run_benchmark.py` 的完整基准测试尚未运行（计划提交 batch job）

---

## 下一步计划

1. **运行基准测试**：提交 `subs/benchmark.sh` batch job，获取 Baseline 和 Hough 的定量指标，填充上方结果表格
2. **启动 ML 训练**：
   - 先从 `EdgeMLP` / `InteractionNet` 等简单模型开始，验证训练流水线
   - 提交 `scripts/run_benchmark.py --device cuda --epochs 200` batch job
3. **选择探索方向**：根据初步 ML 结果，决定优先推进 TrackFormer-Seed 还是 EvoHierGNN（或两者并行）
4. **数据集构建**：为 ML 训练生成足够规模的模拟数据集（目标 ~10⁶ 事件）

---

*每次会话结束时在此文件记录：完成的工作、实验结果、遇到的问题、下一步调整。*
