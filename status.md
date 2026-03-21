# 自主研究状态

**更新时间**: 2026-03-21T21:00
**当前循环**: 1 / 15
**状态**: submitted（PBS 作业提交中）

---

## 当前假设

小容量 Transformer（d_model=64, 2层 encoder, dim_ff=256, warmup=10 epochs, focal_alpha=0.99）首次以正确配置训练，efficiency 应从 2.2% 提升到 >30%。

## 上轮结果

（首次运行，无上轮结果）
- 基线参考：InteractionNet efficiency=70.6%, fake_rate=14.3%
- Transformer 旧 checkpoint：efficiency=2.2%, fake_rate=99.7%（完全失效）

## 当前 PBS 作业

| 字段 | 值 |
|------|-----|
| 作业 ID | （提交后更新）|
| 脚本 | ~/subs/auto_loop1_small_warmup_focal.sh |
| 训练参数 | d_model=64, n_heads=4, 2层, dim_ff=256, warmup=10, epochs=100, lr=1e-3 |
| 输出目录 | /storage/agrp/yiwen/runs/loop1_small_warmup_focal/ |

## 研究进展

| Loop | 假设 | 结果 |
|------|------|------|
| 1 | 小容量+warmup 首次收敛 | 进行中 |

## 成功标准

`track_efficiency >= 0.60 AND fake_rate <= 0.20`（10k 测试事件）
