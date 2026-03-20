# 研究目标

<!-- 每次新研究 session 前，更新此文件，然后启动自主循环 -->

## 当前目标

（请在此填写本次研究目标）

例如：提升 InteractionNet 的 track efficiency 到 ≥ 80%，同时保持 fake rate ≤ 12%。
当前最优：efficiency 70.6%，fake rate 14.3%（InteractionNet，见 progress.md）。

## 允许的改动范围

- 可以改 `src/models.py` 中现有模型类的架构（InteractionNet、ResGNN 等）
- 可以改 `src/layers.py` 中的 MLP、消息传递、注意力等模块
- 可以改 `src/train.py` 中的训练策略（调度器、warmup、早停等）
- 可以改 `src/losses.py` 中的 loss 函数和参数
- **不要改** `geometry.py`、`simulator.py`、`utils.py`、`config.py`

## 成功判断标准

```
track_efficiency >= 0.80 AND fake_rate <= 0.12
（在 10k 测试事件上评估）
```

## 最大循环次数

15

## 优先方向

优先探索架构改动而非只调超参。当前 InteractionNet 的聚合方式（mean pooling）
可能是瓶颈，考虑注意力机制、多尺度特征、更深的消息传递等方向。

## 备注

- 每次循环的假设和结果记录在 `research_log.md`
- 所有代码改动在 `auto-research-*` 分支，不直接改 master
- 研究进展随时查看 `status.md`
