# 自主研究状态

**更新时间**: 2026-03-22T01:05
**当前循环**: Loop 2 / 15
**分支**: auto-research-transformer-convergence

---

## 当前假设

限制训练到 2000 事件（--max-events 2000）避免 OOM，用层内分组注意力 TransformerEdgeClassifier
首次成功完成训练，预期 efficiency >30%。

## 上轮结果（Loop 1 最终版）

| 指标 | 值 |
|------|----|
| track_efficiency | N/A（OOM killed） |
| fake_rate | N/A |
| mean_rms | N/A |
| PBS job | 3924983.pbs |
| 失败原因 | build_labeled_edges_from_sim 全量 10k 事件超出内存 |

## 代码修改（Loop 2）

- `src/train.py`: 添加 `--max-events` CLI 参数，限制训练事件数（default=0 即不限制）
- `src/models.py`: TransformerEdgeClassifier 层内分组注意力（Loop 1 已实现，本轮首次训练）

## 当前 PBS 作业

| 字段 | 值 |
|------|-----|
| 作业 ID | 3925203.pbs |
| 脚本 | ~/subs/auto_loop2_fix_oom.sh |
| 训练参数 | max_events=2000, d_model=64, n_heads=4, 2层, dim_ff=256, warmup=10, epochs=100, lr=1e-3, focal_alpha=0.95 |
| 输出目录 | /storage/agrp/yiwen/runs/loop2_fix_oom/ |

## 架构变更摘要

| 变更 | 文件 | 效果 |
|------|------|------|
| Pre-LN (Pre-Norm) | src/layers.py | 更好梯度流，收敛更稳定 |
| 层内分组注意力 | src/models.py | O(N²/5)，空间归纳偏置 |
| 输出 bias=-3.9 | src/models.py | 初始预测~2% 正例，防99%假率 |
| --max-events 限制 | src/train.py | **修复 OOM**，限制训练事件数 |

## 研究进展

| Loop | 假设 | 结果 |
|------|------|------|
| 1 (最终) | 层内分组 attn | OOM killed（基础设施问题）|
| **2** | 限制 2000 事件 + 层内分组 attn | **进行中** |

## 成功标准

`track_efficiency >= 0.60 AND fake_rate <= 0.20`（10k 测试事件）
