# 自主研究日志

记录每轮循环的假设、代码变更、实验结果和评估。由 Claude 自动追加，供人类审计。

---

<!-- Claude 从这里开始追加条目，最新的在最下面 -->

## Loop 1 — 2026-03-21T20:00

### 上轮结果回顾
- 首次运行，无上轮结果
- 基线：InteractionNet: efficiency=70.6%, fake_rate=14.3%，TransformerEdgeClassifier: efficiency=2.2%, fake_rate=99.7%（完全失效）

### 当前假设
如果我修复 CLI 中缺失 "transformer" 选项的 bug（导致 Transformer 从未被正确训练），并添加所有 Transformer 超参的 CLI 控制，再以小容量配置（d_model=64, 2层, focal_alpha=0.90, warmup=5 epochs）训练，efficiency 应从 2.2% 提升到 >30%，因为当前模型根本没有通过正确参数训练过。
**风险**：O(N²) attention 在 3500 hits/event 下可能 OOM；小容量模型可能欠拟合。

### 代码修改
- `src/train.py` 第 54 行: `model_type` Literal 类型中加入 "transformer"（TrainConfig 已有字段，CLI choices 遗漏）
- `src/train.py` CLI `--model` choices: 添加 "transformer"
- `src/train.py` CLI: 新增 `--d-model`, `--n-heads`, `--n-encoder-layers`, `--dim-feedforward`, `--dropout`, `--warmup-epochs`, `--focal-alpha`, `--focal-gamma`, `--grad-clip`, `--weight-decay` 参数
- `src/train.py` `_cli()`: TrainConfig 构建加入上述新参数
- `src/train.py` `train()`: 添加 warmup + cosine decay 的 SequentialLR 调度器（warmup_epochs > 0 时）
- `src/train.py` TrainConfig: 添加 `warmup_epochs: int = 0` 字段；TransformerEdgeClassifier 默认参数调整为 d_model=64, n_heads=4, n_encoder_layers=2, dim_feedforward=256

### 预期结果
- efficiency 应从 2.2% 提升到 >30%（因为首次正确训练）
- fake_rate 应从 99.7% 降到 <50%
- loss 应在 warmup 后单调下降

### PBS 作业
脚本: ~/subs/auto_loop1_fix_cli_small.sh
提交时间: 2026-03-21

---
### 实际结果（下轮填入）
（待填）
