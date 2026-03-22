# 自主研究状态

**最后更新**: 2026-03-22T07:30
**当前循环**: Loop 4 / 15
**研究分支**: auto-research-transformer-convergence

---

## 当前假设（一句话）

修复 Loop 3 的 3 个 PBS 脚本 bug（--clusters 缺失、--checkpoint 参数名错误、eval 路径问题），重试 balanced sampling（BCELoss + 1:100 比例），验证正边得分 > 0.5。

---

## 上轮结果（Loop 3: loop3_balanced_sampling）

| 指标 | 数值 |
|------|------|
| track_efficiency | **N/A** ❌ |
| fake_rate | N/A |
| n_kept | N/A |
| 根因 | PBS 脚本缺 `--clusters`，训练立即退出（argparse error） |

**额外发现（Loop 3 脚本审查）**：
- eval 步骤也有 bug：`--checkpoint` 应为 `--edge-checkpoint`
- eval 步骤缺少 `--tracks`，且使用了相对路径（文件不存在）
- 所有 3 个 bug 已在 Loop 4 脚本中修复

---

## 当前 PBS 作业（Loop 4）

| 字段 | 值 |
|------|-----|
| 作业 ID | （等待 qsub） |
| 脚本 | ~/subs/auto_loop4_fix_clusters.sh |
| 训练参数 | clusters=sim_clusters_train.parquet, max_events=2000, balanced_sampling=True, neg_pos_ratio=100, d_model=64, 2层, dim_ff=256, warmup=10, epochs=200, lr=1e-3, BCELoss |
| 输出目录 | /storage/agrp/yiwen/runs/loop4_fix_clusters/ |

## Loop 4 代码修改

- 无 src/ 代码修改（所有代码已在 Loop 3 实现且正确）
- PBS 脚本修复：
  1. 训练命令添加 `--clusters .../sim_clusters_train.parquet`
  2. eval 改为 `--edge-checkpoint` + 绝对路径 + `--tracks` + 内联 Python 生成 JSON

## 研究进展

| Loop | 假设 | 结果 | 关键发现 |
|------|------|------|---------|
| 1 | 层内分组注意力架构 | ❌ OOM | 代码正确，内存超限 |
| 2 | --max-events 2000 修复 OOM | ❌ n_kept=0 | AUC=0.976 但得分全 < 0.5（threshold collapse）|
| 3 | 平衡采样 + BCELoss | ❌ PBS bug | PBS 脚本缺 --clusters，训练从未运行 |
| **4** | 修复 3 个 PBS bug，重试 Loop 3 | ⏳ **等待** | 预期 efficiency >30% |

## 成功标准

`track_efficiency >= 0.60 AND fake_rate <= 0.20`（10k 测试事件）
