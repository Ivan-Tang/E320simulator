# 自主研究状态

**更新时间**: 2026-03-21T19:55
**循环进度**: 1 / 15
**状态**: 🚀 PBS 作业已提交（重新提交，修复路径错误）

---

## 当前假设
d_model=64/2层 Transformer + warmup(10ep) + focal_alpha=0.99 应让 TransformerEdgeClassifier 从完全失效（2.2%）收敛到 efficiency ≥30%

## 当前 PBS 作业
- **脚本**: ~/subs/auto_loop1_small_warmup_focal.sh
- **作业 ID**: （提交后更新）
- **训练参数**: d_model=64, n_heads=4, 2层encoder, dim_feedforward=256, dropout=0.0, focal_alpha=0.99, warmup=10ep, 100ep total, lr=1e-3

## 上轮结果（基线）
| 指标 | 值 |
|------|-----|
| efficiency | 2.2% （完全失效） |
| fake_rate | 99.7% |
| mean_rms | 4726 µm |

**对比目标**: efficiency ≥ 60%, fake_rate ≤ 20%

## 代码变更摘要（Loop 1）
- `src/models.py`: TransformerEdgeClassifier/Embedder 默认值缩小（d_model 256→64, 6层→2层）
- `src/train.py`: TrainConfig 新增 warmup_epochs 字段；CLI 修复（添加 transformer 选项）；SequentialLR warmup 调度器

## 研究进展
| Loop | 假设 | efficiency | fake_rate | 结论 |
|------|------|-----------|-----------|------|
| 基线 | — | 2.2% | 99.7% | 完全失效 |
| 1 | 小容量+warmup+focal_alpha=0.99 | 待填 | 待填 | 进行中 |
