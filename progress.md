# 实验进度记录

## 当前阶段

**阶段**：Benchmark 已部分完成，Scaling 仍在运行，ML 完整训练待重跑
**时间**：2026年3月19日
**状态**：Baseline 和 Hough 已得到定量结果；Benchmark 中 ML 模型训练因 OOM 被 kill，未获得完整 ML 指标；Scaling job 仍在运行中（已运行 ~4h）。

---

## 已完成工作（按时间倒序）

### 2026年3月19日
- Benchmark batch job 完成（部分）：获得 Baseline 和 Hough 定量结果（见下方表格）
- Benchmark 中 ML 部分（Embedder epoch 186/200）被 OOM kill，ML 模型评估结果未写入
- Scaling job（`3911702.pbs`）仍在运行，已运行约 4 小时
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

| 方法 | 径迹效率 | 误判率 | 推理时间 | 备注 |
|------|----------|--------|----------|------|
| Baseline（斜率窗口+链式）| **74.3%** | **42.5%** | 553 s/10k events | benchmark_output.log，10k 测试事件 |
| Hough 变换 | **82.7%** | **46.1%** | 4545 s/10k events | benchmark_output.log，10k 测试事件 |
| Embedder（metric-learning）| — | — | — | OOM killed @ epoch 186/200，未完成评估 |
| ML 模型（EdgeMLP 等）| — | — | — | Benchmark OOM，未完成 |
| EvoHierGNN | — | — | — | 尚未训练 |
| TrackFormer-Seed | — | — | — | 尚未训练 |

测试集：10,000 events，1,173 真实径迹，信号占比 0.02%（极低信噪比）
*目标值：径迹效率 ≥95%，误判率 ≤5%，推理时间 ≤10 ms/event（V100/RTX 3090）*

**注**：Hough 比 Baseline 效率高 8.4pp，但误判率更高（+3.6pp），且耗时是 Baseline 的 8 倍。两者误判率均远超目标，ML 方法的提升空间巨大。

---

## 已知问题 / Blockers

- **Benchmark OOM**：`run_benchmark.py --epochs 200 --workers 8` 在 Embedder 训练阶段（epoch 186）被 OOM kill（64 GB RAM 不够）。需要减少 workers 或分拆训练任务重跑 ML 部分。
- **Baseline/Hough 误判率极高（42-46%）**：在 0.02% 信号占比的极低信噪比下，传统方法表现差，凸显 ML 方法的必要性。
- **Scaling job 尚未完成**：`3911702.pbs` 仍在运行，结果待分析。
- `src/baseline.py:110-111`：存在 divide-by-zero RuntimeWarning（dz=0 时斜率计算），不影响结果但需修复。

---

## 下一步计划

1. **等待 Scaling job 完成**：分析 scaling 结果，了解不同事件规模下算法性能变化
2. **重跑 ML Benchmark**（解决 OOM）：
   - 减少 `--workers` 数量（如从 8 改为 4），或
   - 分拆：单独提交 embedder 训练 + 各 ML 模型评估的独立 job
3. **修复 baseline divide-by-zero**：在 `src/baseline.py` 中加入 dz ≈ 0 的守卫条件
4. **选择 ML 探索方向**：获得完整 ML benchmark 结果后，决定优先推进 TrackFormer-Seed 还是 EvoHierGNN

---

*每次会话结束时在此文件记录：完成的工作、实验结果、遇到的问题、下一步调整。*
