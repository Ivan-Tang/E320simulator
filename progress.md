# 实验进度记录

## 当前阶段

**阶段**：TransformerEdgeClassifier 自主研究完成，目标达成
**时间**：2026年3月22日
**状态**：Loop 7 完成，TransformerEdgeClassifier 最终 efficiency=82.01%，fake_rate=11.66%，正式超越 InteractionNet（70.6% / 14.3%）。

### 自主研究最终结果（2026-03-22，Loop 7）

| 方法 | 径迹效率 | 误判率 | Mean RMS | 备注 |
|------|----------|--------|----------|------|
| **TransformerEdgeClassifier** | **82.01%** | **11.66%** | **4.634 µm** | ★ **新最优** — 自主研究 Loop 5+7（chi2筛选）|
| InteractionNet | 70.6% | 14.3% | 4.09 µm | 原最优 ML |
| Baseline | 74.3% | 42.5% | 4.73 µm | 非 ML 基线 |

最优 checkpoint：`/storage/agrp/yiwen/runs/loop5_pos_weight_fix/best_model.pt`
推理配置：`--edge-threshold 0.1`，后处理 `chi2 < 7e-5`

---

## 已完成工作（按时间倒序）

### 2026年3月20日
- **Benchmark `3914621.pbs` 完成**（耗时 ~3h10m）：所有方法完整推理结果获得
- OOM 根因修复：去掉 `--force-retrain`，利用已有 checkpoints 跳过边图构建，直接推理
- Benchmark `3913186.pbs`（256GB）OOM 根因确认：`build_labeled_edges_from_sim(35M clusters)` 的边图 DataFrame 超出内存，非 DataLoader workers 问题
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

### Benchmark（10k 测试事件，1173 真实径迹，信号占比 0.02%）

| 方法 | 径迹效率 | 误判率 | Mean RMS | 推理时间 | 备注 |
|------|----------|--------|----------|----------|------|
| Baseline | **74.3%** | 42.5% | 4.73 µm | 341 s | |
| Hough | **82.7%** | 46.1% | 13.33 µm | 2880 s | 效率最高但 RMS 差 3× |
| MLP | 74.3% | 40.4% | 4.67 µm | 2447 s | 与 Baseline 相当 |
| GNN (ResGNN) | 51.4% | 43.5% | 4.78 µm | 2270 s | 低于 Baseline |
| **InteractionNet** | **70.6%** | **14.3%** | **4.09 µm** | **767 s** | ★ 最优 ML：误判率最低，RMS最佳 |
| EggNet | 1.4% | 10.5% | 4.95 µm | 713 s | 基本失效，checkpoint 需重训 |
| HGNN | 14.2% | 10.2% | 4.14 µm | 712 s | 效率低但误判率好 |
| Transformer | 2.2% | 99.7% | 4726 µm | 151 s | 完全失效，需从头重训 |

*目标：径迹效率 ≥95%，误判率 ≤5%，推理时间 ≤10 ms/event*
**注**：所有方法效率均远低于 95% 目标；InteractionNet 在误判率上接近目标（14.3% vs 5% 目标），是下一步重点优化方向。

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
- ~~**OOM 根因**：`build_labeled_edges_from_sim(35M clusters)` 构建全量边图超出内存~~。**已修复**：`scripts/build_edges.py` 实现分批（200 events/batch）流式写 Parquet 的独立预处理脚本，可单独作为 PBS job 运行。
- **所有方法效率上限 ~83%，远低于 95% 目标**：需专门针对 E320 设计并重新训练模型。
- **EggNet 和 Transformer checkpoints 基本无效**：需重新训练。
- **InteractionNet 是最有希望的起点**：误判率已降至 14.3%，但效率仍需从 70.6% 提升到 ≥95%。
- ~~`src/baseline.py:110-111`：divide-by-zero RuntimeWarning（dz=0）~~。**已修复**：`np.errstate(divide="ignore", invalid="ignore")` 已包裹斜率计算（`baseline.py:112-114`）。

---

## 下一步计划

1. **分析 InteractionNet 失效案例**：当前 70.6% 效率 vs 95% 目标，差距主要来自哪类事件/径迹？（低动量？特殊几何？）
2. **重新训练 Transformer（TrackFormer-Seed）**：现有 checkpoint 完全失效（99.7% fake），从头设计针对 E320 低信噪比的训练流程
3. **重新训练 EggNet**：当前 1.4% 效率，checkpoint 基本无效，需重训或放弃
4. **若需重训 ML 模型**：使用 `scripts/build_edges.py` 单独提交边图构建 job，再提交训练 job（OOM 问题已解决）

---

*每次会话结束时在此文件记录：完成的工作、实验结果、遇到的问题、下一步调整。*
