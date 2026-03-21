# 自主研究状态

**更新时间**: 2026-03-21T21:10
**当前循环**: 1 / 15
**状态**: submitted（PBS 作业 3924983.pbs 已提交）

---

## 当前假设

探测器层内分组自注意力（d_model=64, 2层, warmup=10, focal_alpha=0.99）首次以正确配置训练。
- 按 layer_id 分5组做 self-attention（~700 hits/层），比全局 O(N²) 更高效且有空间归纳偏置
- Pre-LN + 输出 bias 初始化(-3.9) 防止初始全假率 99%

## 上轮结果

（首次运行，无上轮结果）
- 基线参考：InteractionNet efficiency=70.6%, fake_rate=14.3%
- Transformer 旧 checkpoint：efficiency=2.2%, fake_rate=99.7%（完全失效）

## 当前 PBS 作业

| 字段 | 值 |
|------|-----|
| 作业 ID | 3924983.pbs |
| 脚本 | ~/subs/auto_loop1_layer_attn.sh |
| 训练参数 | d_model=64, n_heads=4, 2层, dim_ff=256, warmup=10, epochs=100, lr=1e-3, focal_alpha=0.95 |
| 输出目录 | /storage/agrp/yiwen/runs/loop1_layer_attn/ |

## 架构变更摘要

| 变更 | 文件 | 效果 |
|------|------|------|
| Pre-LN (Pre-Norm) | src/layers.py | 更好梯度流，收敛更稳定 |
| 层内分组注意力 | src/models.py | O(N²/5)，空间归纳偏置 |
| 输出 bias=-3.9 | src/models.py | 初始预测~2% 正例，防99%假率 |

## 研究进展

| Loop | 假设 | 结果 |
|------|------|------|
| 1 | 层内分组attn+Pre-LN+bias_init | 进行中（3924983.pbs）|

## 成功标准

`track_efficiency >= 0.60 AND fake_rate <= 0.20`（10k 测试事件）
