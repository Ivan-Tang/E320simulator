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
（待填）
