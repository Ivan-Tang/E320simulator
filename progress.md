# 实验进度记录

## 最近变更（新会话速览）

| 日期 | 做了什么 | 结果 | 关键文件 |
|------|---------|------|---------|
| 04-30 | Benchmark v6（A5000 gwn243，workers=4 并行，24h 完成全部 5 模型） | 结果与 v4 完全一致（deterministic 三次验证通过）；MLP 79.45%/31.72%，InteractionNet 77.15%/22.12%/F1=77.52%，HGNN 56.95%/20.48%；PBS 又落在 A5000，A6000 速度对比仍待测 | `logs/benchmark_a6000_run.log` |
| 04-28 | Benchmark v5（A6000, 24h walltime 被杀） | MLP **79.5%/31.7%/F1=73.5%**（首个干净 MLP 结果）；GNN 78.3%/80.3%（与 v4 完全一致，deterministic 验证）；walltime 不足，InteractionNet+ 未跑完 | `logs/benchmark_a6000_run.log` |
| 04-27 | Benchmark v4（A5000，deterministic+seed=42） + GNN 阈值扫描 | InteractionNet **77.15%/22.12%/F1=77.52%**；EggNet 71.61%/44%；HGNN 56.95%/20.48%；GNN 78.35%/80.31%（阈值无法解决 fake_rate，是结构性问题）；MLP SIGSEGV | `logs/benchmark_run.log`, `logs/threshold_sweep_gnn_run.log` |
| 04-25 | HGNN deterministic v4 完成（Job 4158623） | **56.95%/20.48%/F1=66.37%**；v3 的 96.68% 是侥幸——HGNN 对初始化极敏感，三次结果：46%/57%/97%；已分析根因见下方 | `logs/train_hgnn_run.log` |
| 04-23 | Benchmark v3 完成（Job 4123847） | MLP 完全一致（GPU seed 未控制！）；HGNN **96.68%/35.86%** F1=77%（本轮最高）；GNN 99.23%/87.88%；InteractionNet 34.78%（三轮下跌，放弃） | `logs/benchmark_run.log` |
| 04-24 | **DDP 路线关闭** | 瓶颈=Polars CPU 加载（POLARS_MAX_THREADS=1 为 Lustre SIGSEGV 必要限制），加 GPU/CPU 均无效；单卡 165s/epoch，DDP 519s/epoch（3.1x 慢）；后续一律单卡 | — |
| 04-22 | DDP no_sync() 修复提交 | AllReduce 次数 4000→40/epoch；加 --log-every CLI；提交修复验证 job 4135811 | `src/train.py`, `scripts/run_benchmark.py` |
| 04-21 | DDP 2-GPU smoke test 通过 | gwn243 A5000×2，10 epochs，无 NCCL deadlock | `subs/ddp_test_2gpu.sh` |
| 04-16 | 失效诊断分析完成 | ①图覆盖率100%；②model_miss=100%失效根因；GNN@0.1 in-graph TPR=97.1%，效率96.2%（超95%目标） | `scripts/analyze_diagnosis.py` |
| 04-16 | 单卡 benchmark 完成（Job 4040917） | GNN **92.84%/79.48%**；MLP 79.45%/31.72%；EggNet 67.95%（从1.4%恢复）；InteractionNet 52%（↓，待排查） | `logs/benchmark_run.log` |
| 04-13 | balanced_sampling 默认 True | 修复 GNN/EggNet 梯度崩溃根因（pos_frac激进10倍）；A6000节点全部禁止 | `src/train.py`, `subs/benchmark.sh` |
| 03-31 | 合并 feature/ddp → master | 7 文件冲突解决，86 测试通过，DDP 支持就绪 | `src/train.py`, `src/ddp.py` |
| 03-22 | Auto-research Loop 7 完成 | TransformerEdgeClassifier **82.01% / 11.66%**，超越 InteractionNet 70.6% / 14.3% | checkpoint: `runs/loop5_pos_weight_fix/best_model.pt` |

**当前最优模型（F1 受控）**：TransformerEdgeClassifier（82.01% / 11.66%，来自 auto-research）；在最新 benchmark 中 InteractionNet F1=77.52% 是次优
**HGNN 结论**：初始化极敏感（三次：46%/57%/97%），不稳定，不作为主要方向
**GNN 结论**：fake_rate 结构性问题（80%+），阈值扫描（0.01→0.85，1000 events）无效，fake_rate 仅从 89.5%→83.8%
**可重复性结论**：deterministic fix 有效；GNN A5000 vs A6000 结果完全一致（78.3%/80.3%）
**目标**：≥95% eff / ≤5% fake（F1≥80%）
**下一步**：① 重提 benchmark_a6000.sh（改 walltime=72h）完成所有模型 ② 或直接聚焦 InteractionNet/Transformer 优化（诊断失效根因）

---

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

### HGNN Deterministic v4（Job 4158623，2026-04-25，gwn244 A5000，200 epochs）

**目的**：用 deterministic fix 重现 v3 的 96.68%，确认是否稳定。

| 版本 | 条件 | Efficiency | Fake rate | F1 |
|------|------|-----------|-----------|-----|
| v2 | 无 GPU seed 控制 | 46.04% | 22.86% | 57.66% |
| v3 | 无 GPU seed 控制 | **96.68%** | 35.86% | 77.12% |
| **v4** | **deterministic fix** | **56.95%** | **20.48%** | **66.37%** |

**结论：v3 的 96.68% 是偶然，HGNN 对初始化极度敏感，不稳定，不作为主要优化方向。**

#### HGNN 高方差根因分析

HGNN 方差远大于其他模型（GNN/MLP），根因在于**两阶段架构的连锁脆弱性**：

**1. `index_add_` 调用次数远多于其他模型**

每次 forward 包含 ~13 次 `index_add_`（GNN 仅 3 次）：
- `_build_super_structure()` 中 4 次（supernode/embedding 聚合）
- 每个 `HierarchicalGNNCell` 3 次 × 3 cells = 9 次

每次 `index_add_` 在 CUDA 上做非确定性原子加法，误差随深度和轮数指数级累积。deterministic fix 解决了不可复现问题，但不能改变对初始化的敏感性。

**2. bipartite_weights 软门控放大初始化差异**

`_build_super_structure()` 中（`models.py:965`）：
```python
cos_sim = (embeddings * emb_layer[layer_ids]).sum(dim=-1, keepdim=True)
bipartite_weights = cos_sim / cos_mean.clamp(min=1e-8)
```
这个权重由第一阶段（InteractionGNN）的中间 embedding 动态计算。若初始化使 embedding 空间处于"好"的状态，bipartite_weights 有信息量，第二阶段能有效学习；若 embedding 在初期几乎均匀，权重退化为常数，hierarchical 部分无法获得有效监督。结果就是两个截然不同的吸引域：**激进态**（高效率/高 fake）或**保守态**（低效率/低 fake）。

**3. 损失面双峰**

三次结果（46%/57%/97%）分属两个吸引域：
- 保守态：效率 46-57%，fake 20-23%（模型默认打分低）
- 激进态：效率 96%，fake 36%（偶然找到好初始化）

**修复方向（如需探索）**：
- 两阶段训练：先单独训练 InteractionGNN 阶段固定 embedding，再训练 hierarchical 部分
- 或将 bipartite_weights 改为基于 layer_id 的固定均匀权重（去掉 cosine 动态计算），降低对 embedding 质量的依赖

---

### Benchmark v6（Job 4175662，2026-04-28~30，gwn243 A5000，workers=4 并行）

**目的**：完整运行所有 5 个 ML 模型，确认 deterministic 可重复性，同时尝试在 A6000 上跑（实际仍落在 A5000）。

**结果（全部完成，与 v4 完全一致）：**

| 方法 | 效率 | fake率 | F1 | 训练时间 |
|------|------|--------|----|---------|
| MLP | **79.45%** | **31.72%** | **73.44%** | 8.5h（30675s）|
| GNN | 78.35% | 80.31% | 31.47% | 9.4h（33885s）|
| **InteractionNet** | **77.15%** | **22.12%** | **77.52%** | 9.1h（32866s）|
| EggNet | 71.61% | 44.00% | 62.85% | 10.2h（36768s）|
| HGNN | 56.95% | 20.48% | 66.37% | 12.6h（45208s）|

**workers=4 效果**：4 个模型同时并行跑（MLP/GNN/InteractionNet/EggNet 并行，HGNN 单独），总 wall time ≈ max(10.2h, 9.4h, 9.1h, 8.5h) + 12.6h ≈ 22.8h，实际 24h 完成全部。

**结论**：3 次 benchmark（v4/v5/v6）结果完全一致，deterministic fix（seed=42 + use_deterministic_algorithms）在 A5000 上完全可重复。A6000 速度对比因 PBS 调度仍未测到。

---

### Benchmark v5（Job 4171289，2026-04-28，gwn243 A5000，24h walltime 限制）

**结果（24h 内完成两个模型后被 PBS 杀死）：**

| 方法 | 效率 | fake率 | F1 | 备注 |
|------|------|--------|----|------|
| MLP | 79.5% | 31.7% | 73.5% | 首个干净 MLP 结果（v4 SIGSEGV 后）|
| GNN | 78.3% | 80.3% | 31.4% | 与 v4 完全一致 |
| InteractionNet+ | — | — | — | walltime=24h 不足，未启动 |

---

### GNN v4 阈值扫描（Job ~4170xxx，2026-04-27，1000 events，27 阈值 0.01→0.85）

**目的**：通过调整 edge-threshold 降低 GNN 的 fake_rate（结构性 80%+）。

**结论：阈值无效，GNN fake_rate 是结构性问题。**

- 阈值范围 0.01→0.85，fake_rate 仅从 89.5% 降至 83.8%（↓仅 5.7 pp）
- 高阈值下 efficiency 急剧下降，无任何可用工作点
- 根因：GNN 对 fake edges 打分与 true edges 打分重叠严重，分离度不足

---

### Benchmark v4（Job 4167031，2026-04-27，gwn243 A5000，deterministic fix + seed=42）

**目的**：在 deterministic 条件下建立全量模型干净基线。

| 方法 | 径迹效率 | 误判率 | F1 | 备注 |
|------|----------|--------|-----|------|
| MLP | SIGSEGV | — | — | A5000 OOM（CUBLAS workspace 4GiB × workers + Polars）|
| GNN | 78.35% | 80.31% | 31.47% | 独立 job 完成（确定性验证通过）|
| **InteractionNet** | **77.15%** | **22.12%** | **77.52%** | ★ 当前 benchmark 最优 F1 |
| EggNet | 71.61% | 44.00% | 62.85% | 效率回升但 fake 偏高 |
| HGNN | 56.95% | 20.48% | 66.37% | 保守吸引域，与单独训练结果一致 |

---

### Benchmark v3（Job 4123847，2026-04-21~23，gwn244 A5000，10k 测试事件，1173 真实径迹，200 epochs + balanced_sampling）

**目的**：重复 v2 检验可重复性，发现 GPU 端随机变量未被控制。

| 方法 | 径迹效率 | 误判率 | F1 | Mean RMS | 训练时间 | vs v2 |
|------|----------|--------|-----|----------|----------|-------|
| MLP | 79.45% | 31.72% | 73.44% | 6.76 µm | ~32190s | **完全一致**（CPU seed 有效）|
| GNN | 99.23% | 87.88% | 21.60% | 11.11 µm | ~34020s | eff ↑+6.4%，fake ↑+8.4% |
| InteractionNet | 34.78% | 19.53% | 48.57% | 6.10 µm | ~33379s | eff ↓-17%（三轮连续下跌，放弃）|
| EggNet | 54.90% | 32.49% | 60.55% | 6.95 µm | ~35578s | eff ↓-13% |
| **HGNN** | **96.68%** | **35.86%** | **77.12%** | 7.12 µm | ~37692s | **eff ↑+50%！本轮 F1 最高** |

**可重复性结论**：GPU 端随机性未控制（缺少 `torch.cuda.manual_seed_all()` + `cudnn.deterministic=True`）。各模型波动巨大（HGNN 46%→96%，InteractionNet 70%→52%→35%）。MLP 因主要受 CPU 操作主导而完全一致。

---

### Benchmark v2（Job 4040917，2026-04-14~16，gwn244 A5000，10k 测试事件，1173 真实径迹，200 epochs + balanced_sampling）

| 方法 | 径迹效率 | 误判率 | F1 | Mean RMS | 推理时间 | vs 上次 |
|------|----------|--------|-----|----------|----------|---------|
| MLP | 79.45% | 31.72% | 73.44% | 6.76 µm | ~117 ms/evt | ↑ 74.3%→79.5% |
| **GNN (ResGNN)** | **92.84%** | **79.48%** | 33.61% | 10.49 µm | ~281 ms/evt | ↑↑ 51.4%→92.8%（fake 率过高）|
| InteractionNet | 52.00% | 20.68% | 62.82% | 6.07 µm | ~115 ms/evt | ↓ 70.6%→52%（待排查）|
| EggNet | 67.95% | 46.97% | 59.57% | 8.05 µm | ~153 ms/evt | ↑↑ 1.4%→68%（完全恢复）|
| HGNN | 46.04% | 22.86% | 57.66% | 6.44 µm | ~123 ms/evt | ↑ 14.2%→46%（改善）|

*Baseline/Hough 未在此次 benchmark 中运行（脚本配置跳过）*
**注**：GNN 效率接近目标 95%，但 fake_rate 79% 需要大幅降低；balanced_sampling 修复使 EggNet 完全复活。F1 最优为 MLP（73.44%）。

### Benchmark v1（Job 3914621，2026-03-20，10k 测试事件，1173 真实径迹）

| 方法 | 径迹效率 | 误判率 | Mean RMS | 推理时间 | 备注 |
|------|----------|--------|----------|----------|------|
| Baseline | **74.3%** | 42.5% | 4.73 µm | 341 s | |
| Hough | **82.7%** | 46.1% | 13.33 µm | 2880 s | 效率最高但 RMS 差 3× |
| MLP | 74.3% | 40.4% | 4.67 µm | 2447 s | 与 Baseline 相当 |
| GNN (ResGNN) | 51.4% | 43.5% | 4.78 µm | 2270 s | 低于 Baseline |
| **InteractionNet** | **70.6%** | **14.3%** | **4.09 µm** | **767 s** | ★ 最优 ML：误判率最低，RMS最佳 |
| EggNet | 1.4% | 10.5% | 4.95 µm | 713 s | 基本失效（balanced_sampling 关闭所致）|
| HGNN | 14.2% | 10.2% | 4.14 µm | 712 s | 效率低但误判率好 |
| Transformer | 2.2% | 99.7% | 4726 µm | 151 s | 完全失效，需从头重训 |

*目标：径迹效率 ≥95%，误判率 ≤5%，推理时间 ≤10 ms/event*

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

## 2026年3月31日 — 合并冲突解决

### 完成工作

解决了 `feature/ddp` 分支与 `master` 之间的 7 个文件合并冲突（共 20+ 个冲突块）：

- **策略**：对 `src/train*.py`、`scripts/run_benchmark.py` 取 DDP 侧（`src/ddp.py` 提供 backward-compatible 单卡降级）；对 `src/utils.py`、`cluster_agent/agent.py` 取 HEAD 侧
- **额外修复**：`src/train.py` 中 `_evaluate` 函数签名从 `(model, device, ...)` 改为 `(model, df, device, ...)` 以匹配 DDP 版调用约定
- **关键改进**（来自 DDP 侧）：Polars `group_by` → sort-based slicing，修复大 DataFrame 上的 Rust thread panic（segfault in PBS）
- **测试**：全部 86 个 pytest 测试通过
- **推送**：已 push 至 remote，PBS 作业可以看到最新代码

### 单卡 benchmark 使用方式

```bash
# 提交 PBS 作业
qsub subs/benchmark.sh
# 或交互式（小规模测试）
conda run -n e320root python scripts/run_benchmark.py --device cuda --epochs 200 --workers 8
```

DDP 参数 `--ddp-nproc` 默认为 1（单卡），无需特殊配置。

---

## 下一步计划（2026-04-28 更新）

**已确认结论：**
- GNN fake_rate 结构性问题，阈值无法解决 → 放弃 GNN 作为优化方向
- HGNN 初始化极敏感，不稳定 → 不作为主要方向
- InteractionNet（77.15%/22.12%/F1=77.52%）和 Transformer（82.01%/11.66%，历史最优）是最值得投入的方向

**待办：**
1. **【可选】A6000 速度测试**：需在 `benchmark_a6000.sh` 加 `#PBS -l gputype=RTX6000Ada`（或正确的 A6000 gputype 字符串）才能真正路由到 A6000 节点
2. **【优先】失效案例诊断**：运行 `scripts/diagnose_failures.py`，拆解 InteractionNet/Transformer 效率损失三类根因（图覆盖率不足 / 模型打分低 / 后处理丢失）
3. **TransformerEdgeClassifier 优化**：在诊断结论指导下，针对性改进（hard negative mining / 更多物理特征 / 更大模型）

---

*每次会话结束时在此文件记录：完成的工作、实验结果、遇到的问题、下一步调整。*
