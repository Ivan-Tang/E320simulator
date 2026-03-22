# 自主研究状态

**最后更新**: 2026-03-22T03:00
**当前循环**: Loop 3 / 15
**研究分支**: auto-research-transformer-convergence

---

## 当前假设（一句话）

平衡小批次采样（1:100 正:负，BCELoss）迫使模型输出正边得分 > 0.5，修复阈值崩溃问题。

---

## 上轮结果（Loop 2: loop2_fix_oom）

| 指标 | 数值 |
|------|------|
| track_efficiency | **0.0%** ❌ |
| fake_rate | 0.0% |
| n_kept | 0 |
| AUC | 0.976 ✓ |
| AP | 0.0006 |
| mean_rms | NaN |

**根因**: 模型学到了好的排序（AUC=0.97）但所有边得分 << 0.5 推理阈值。
极端类别不平衡（1:50,000 = pos_frac=0.00002）导致 FocalLoss 收敛到全零输出局部最优。

---

## 当前 PBS 作业（Loop 3）

| 字段 | 值 |
|------|-----|
| 作业 ID | 3925300.pbs |
| 脚本 | ~/subs/auto_loop3_balanced_sampling.sh |
| 训练参数 | max_events=2000, balanced_sampling=True, neg_pos_ratio=100, d_model=64, 2层, dim_ff=256, warmup=10, epochs=200, lr=1e-3, BCELoss |
| 输出目录 | /storage/agrp/yiwen/runs/loop3_balanced_sampling/ |

## Loop 3 代码修改

- `src/train.py`: 新增 `balanced_sampling`、`neg_pos_ratio` 字段
- `src/train.py`: 训练循环中每事件采样 pos + 100×neg，改用 `nn.BCELoss()`
- `src/train.py`: balanced+transformer 时重置 classifier 偏置为 0.0（中性初始化）
- `src/train.py` CLI: 新增 `--balanced-sampling`、`--neg-pos-ratio`

## 研究进展

| Loop | 假设 | 结果 | 关键发现 |
|------|------|------|---------|
| 1 | 层内分组注意力架构 | ❌ OOM | 代码正确，内存超限 |
| 2 | --max-events 2000 修复 OOM | ❌ n_kept=0 | 训练成功但得分全 < 0.5 |
| **3** | 平衡采样 + BCELoss | ⏳ **进行中** | 预期 efficiency >30% |

## 成功标准

`track_efficiency >= 0.60 AND fake_rate <= 0.20`（10k 测试事件）
