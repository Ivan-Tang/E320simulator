# 自主研究日志

记录每轮循环的假设、代码变更、实验结果和评估。由 Claude 自动追加，供人类审计。

---

<!-- Claude 从这里开始追加条目，最新的在最下面 -->

## Loop 1 (已作废，状态被重置) — 2026-03-21T20:00

*注：此条目对应的 PBS 作业从未成功执行，experiment_state.json 手动重置为 loop_count=0 重新开始。*

---

## Loop 1 — 2026-03-21T21:00

### 上轮结果回顾
- 首次运行，无上轮结果
- 基线：InteractionNet: efficiency=70.6%, fake_rate=14.3%；TransformerEdgeClassifier: efficiency=2.2%, fake_rate=99.7%（完全失效）

### 当前假设
如果以小容量 Transformer（d_model=64, 2层 encoder, dim_feedforward=256）加 warmup（10 epochs，线性升温）和适度 focal_alpha=0.99 训练 100 epochs，efficiency 应从 2.2% 提升到 >30%，因为旧 Benchmark checkpoint 的根因失效是大模型（d_model=256, 6层）在 E320 数据量上不收敛、且缺少 warmup 导致 Transformer 初期训练不稳定。
**风险**：O(N²) attention 在 ~3500 hits/event 下显存约 200–400MB，训练较慢；小容量模型可能表达能力不足。

### 代码修改
- 无新修改。前一 session 已完成的代码修改（CLI "transformer" 选项、warmup scheduler、TrainConfig 字段）均已在当前代码中生效：
  - `src/train.py`: `model_type` Literal 含 "transformer"、warmup+cosine SequentialLR、全部 transformer CLI 参数
  - 默认超参已调整为 d_model=64, n_encoder_layers=2

### 预期结果
- efficiency 应从 2.2% 提升到 >30%（首次以正确小容量配置训练）
- fake_rate 应从 99.7% 降到 <50%
- 训练 loss 应在 warmup 后单调下降

### PBS 作业
脚本: ~/subs/auto_loop1_small_warmup_focal.sh（修复 PBS 绝对路径 bug）
训练参数: d_model=64, n_heads=4, n_encoder_layers=2, dim_feedforward=256, dropout=0.0, focal_alpha=0.99, warmup_epochs=10, epochs=100, lr=1e-3
提交时间: 2026-03-21T21:00

---
### 实际结果（下轮填入）
此作业结果未记录（state 被重置）。

---

## Loop 1 (最终版) — 2026-03-21T22:00

### 上轮结果回顾
- 首次运行（前两次 Loop 1 均因 state 手动重置未留下结果）
- 基线：InteractionNet: efficiency=70.6%, fake_rate=14.3%；TransformerEdgeClassifier: efficiency=2.2%, fake_rate=99.7%（完全失效）

### 当前假设
如果将 TransformerEdgeClassifier 的全局自注意力（O(N²)，N~3500 hits/event）改为**探测器层内分组注意力**（按 layer_id 分 5 组，每组 ~700 hits，各组独立 self-attention），efficiency 应从 2.2% 提升到 >40%，因为：1）层内相同 track 的 hits 具有强局部相关性，2）每层的信噪比远好于全局（700 hits 中 5 个真实 hit vs 3500 hits 中 25 个真实 hit），3）消除了 O(N²) 梯度稀释问题。

**风险**：层内 attention 可能无法捕获跨层信息（edge classification 使用 src/dst 来自不同层），但 edge feature（dx/dy/dz/slope）已编码跨层信息。

### 代码修改
- `src/models.py` 第 297-361 行：重写 `TransformerEdgeClassifier`
  - 替换 `TransformerEmbedder`（全局 N 节点注意力）为直接 `input_proj` + `encoder_layers` + `norm`
  - `forward()` 新增层内分组循环：按 `layer_id` 分 5 组，各组独立 self-attention
  - 保持 `edge_encoder` 和 `classifier` 不变，外部接口不变

### 预期结果
- efficiency: >40%（从 2.2% 大幅提升）
- fake_rate: <50%（从 99.7% 改善）
- 训练 loss 应在 warmup 后单调下降，收敛比全局注意力快

### PBS 作业
脚本: ~/subs/auto_loop1_layer_attn.sh
训练参数: d_model=64, n_heads=4, n_encoder_layers=2, dim_feedforward=256, dropout=0.0, focal_alpha=0.95, warmup_epochs=10, epochs=100, lr=1e-3
提交时间: 2026-03-21T22:00

---
### 实际结果（下轮填入）
（待填）
