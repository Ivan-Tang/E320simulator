# 实验进度记录

## 当前阶段

**阶段**：Scaling 完成；Benchmark OOM 问题未解决，需降低 --workers
**时间**：2026年3月20日
**状态**：Scaling sweep 全部完成。Benchmark 256GB 版本仍在 epoch ~187 被 OOM kill，根因确认为 `--workers 8` 导致 8 个进程各持 35M clusters，需减少 workers 后重提交。

---

## 已完成工作（按时间倒序）

### 2026年3月20日
- Benchmark `3913186.pbs`（256GB）完成，但仍在 epoch 187/200 被 OOM kill
- 根因确认：`--workers 8` × 35M clusters = 单次作业需要 >256GB RAM
- 非 ML 结果再次确认：Baseline 74.3%/42.5%，Hough 82.7%/46.1%（与上次一致）
- Scaling job `3911702.pbs` 完成（耗时 ~5h，03月19日23:42落盘）
- 分析 Scaling 结果（两轮 sweep：背景扫描 + 信号密度扫描），见下方详细表格

### 2026年3月19日
- Benchmark job `3912419.pbs`（64GB）部分完成：Baseline 和 Hough 定量结果获得，ML 部分 OOM kill
- 重新提交 Benchmark job（256GB）：`3913186.pbs`
- 整合两份 ML seeding 研究提案为统一文档 `research_proposal_ML_seeding.md`
- 建立进度记录文档和会话工作规范（CLAUDE.md）

### ML 框架搭建（已完成）
- `src/models.py`：`EdgeMLP`、`InteractionNet`、`ResGNN`、`EggNet`、`HierarchicalGNN`、`TransformerEdgeClassifier`、`Embedder`、`TransformerEmbedder`
- `src/layers.py`：`MLP` 工厂、`PositionalEncoding3D`、`MultiHeadAttention`、Transformer 层
- `src/losses.py`：`FocalLoss`、`HingeLoss`
- `src/train.py` / `src/train_embedder.py` / `src/train_trackformer.py`

### Hough 变换 / Baseline 算法（已完成）
- `src/hough_baseline.py`、`src/baseline.py`
- 数据路径：`/storage/agrp/yiwen/data_Run502`（Lustre）

---

## 关键实验结果

### Benchmark（10k 测试事件，信号占比 0.02%）

| 方法 | 径迹效率 | 误判率 | 推理时间 | 备注 |
|------|----------|--------|----------|------|
| Baseline（斜率窗口+链式）| **74.3%** | **42.5%** | 553 s/10k events | |
| Hough 变换 | **82.7%** | **46.1%** | 4545 s/10k events | |
| ML 模型（EdgeMLP 等）| — | — | — | Benchmark 256GB 重跑中 |

### Scaling Sweep 1：背景强度扫描（mean_n_signal=0.5，1000 events）

| bg_per_layer | ~hits/event | Baseline eff / fake | Hough eff / fake | MLP eff / fake | InteractionNet eff / fake | Transformer eff / fake |
|---|---|---|---|---|---|---|
| 0 | 0 | 73.5% / 0% | 80.0% / 0% | 73.5% / 0% | 61.3% / 0% | 40.3% / 0% |
| 100 | 500 | 77.8% / 0% | 82.3% / 0% | 77.8% / 0% | 51.8% / 0% | 47.9% / **77.8%** |
| 300 | 1500 | 73.9% / 1.1% | 78.1% / 0% | 73.9% / 0.3% | 49.5% / 0.4% | **0% / 100%** |
| 500 | 2500 | 74.2% / 8.7% | 78.4% / 1.9% | 74.2% / 3.2% | 51.1% / 1.5% | **0% / 100%** |
| 700 | 3500 | 72.9% / 16.2% | 77.1% / 17.4% | 72.9% / 9.8% | 53.8% / **1.6%** | **0% / 100%** |
| 1000 | 5000 | 75.2% / 34.1% | 82.6% / 60.7% | 75.2% / 27.6% | 54.8% / **11.4%** | **0% / 100%** |

其他模型（GNN：15-16%效率；EggNet：~4%效率；HGNN：10-22%效率）略。

### Scaling Sweep 2：信号密度扫描（bg_per_layer=700，1000 events）

| mean_n_signal | ~真实径迹数 | Baseline eff / fake | Hough eff / fake | MLP eff / fake | InteractionNet eff / fake |
|---|---|---|---|---|---|
| 0.1 | 105 | 70.5% / 46.0% | 82.9% / 46.6% | 70.5% / 31.5% | 50.5% / 8.6% |
| 0.3 | 299 | 68.9% / 26.2% | 80.9% / 23.9% | 68.9% / 25.9% | 54.5% / 12.4% |
| 0.5 | 468 | 72.9% / 16.2% | 77.1% / 17.4% | 72.9% / 15.8% | 61.3% / 5.3% |
| 1.0 | 1029 | 73.5% / 9.0% | 83.3% / 8.6% | 73.5% / 9.0% | 58.9% / 2.6% |
| 2.0 | 2004 | 75.4% / 3.9% | 83.1% / 5.7% | 75.4% / 3.9% | 59.7% / 2.0% |
| 3.0 | 3041 | 74.7% / 3.2% | 82.7% / 3.8% | 74.7% / 3.2% | 59.1% / 1.3% |

Transformer 在 bg=700 下全线 0% 效率 / 100% 误判率，略。

*目标值：径迹效率 ≥95%，误判率 ≤5%，推理时间 ≤10 ms/event*

---

## 关键结论（Scaling 分析）

1. **Transformer 彻底失效**：bg≥300（~1500 hits/event）时效率归零、误判率100%。现有 checkpoint 完全不适用于 E320 条件，需从头重新设计训练。

2. **EggNet 也基本失效**：效率 ~4%，多个条件下出现 NaN，checkpoint 可能损坏或训练不收敛。

3. **InteractionNet 是现有 ML 模型最优**：效率 ~50-61%（低于 Baseline 的 73-75%），但误判率控制显著更好（bg=1000 时 11.4% vs Baseline 34.1%）。在高背景下是效率与纯度最佳权衡。

4. **MLP 与 Baseline 几乎等同**：效率相同，误判率略低，没有显著提升。

5. **Hough 效率最高但误判率不可控**：bg=1000 时误判率高达 60.7%，不适合高背景场景。

6. **所有现有方法效率上限约 83%，远低于 95% 目标**，且误判率在实际运行条件（bg~700）下普遍超标。→ 需要重新训练专门针对 E320 条件的模型。

---

## 已知问题 / Blockers

- **Transformer checkpoint 失效**：需要重新训练，或完全重新设计针对 E320 的 TrackFormer-Seed。
- **EggNet NaN/低效率**：checkpoint 疑似有问题，需重新训练。
- **Benchmark OOM 根因确认**：`--workers 8` 让 8 个进程各持 35M clusters，256GB 仍不够。需改用 `--workers 2` 或将 ML 训练从 benchmark 脚本中拆出单独提交。
- **所有方法效率上限 ~83%，误判率在实际背景下远超 5% 目标**：需要专门针对 E320 重新训练。
- `src/baseline.py:110-111`：divide-by-zero RuntimeWarning（dz=0），需修复。

---

## 下一步计划

1. **修复 Benchmark OOM，重新提交**：将 `--workers 8` 改为 `--workers 2`，或拆分任务——ML 训练单独跑，不与数据生成混跑
2. **重新训练 Transformer（TrackFormer-Seed）**：
   - 现有 checkpoint 完全失效，从头设计针对 E320 低信噪比的训练流程
   - 优先推进方向：TrackFormer-Seed（周期短，已有框架）
3. **重新训练 EggNet**：排查 NaN 问题，调整训练超参数
4. **针对 InteractionNet 优化**：在现有最优 ML 基础上，探索提升效率的方向（效率从 ~55% 提升到 ≥95% 需要根本性改进）
5. **修复 baseline divide-by-zero**：`src/baseline.py:110-111`

---

*每次会话结束时在此文件记录：完成的工作、实验结果、遇到的问题、下一步调整。*
