# 自主研究状态

**最后更新**: 2026-03-22T08:30
**当前循环**: Loop 5 / 15
**研究分支**: auto-research-transformer-convergence

---

## 当前假设（一句话）

添加 `pos_weight=100` 修正 balanced BCELoss 的梯度不对称，使正边得分从 ~0.24 提升到 > 0.5，从而打破 efficiency=0% 的僵局。

---

## 上轮结果（Loop 4: loop4_fix_clusters）

| 指标 | 数值 |
|------|------|
| track_efficiency | **0.0%** ❌ |
| fake_rate | 0.0% |
| n_kept | 0 |
| 训练 AUC | **0.977** ✓（模型有判别力！）|
| 训练 loss | 0.015（收敛）|
| 根因 | BCELoss 梯度不对称：balanced 1:100 批次均衡点对应 p_pos≈0.24，远低于 0.5 阈值 |

**数学确诊**：在 1:100 balanced batch 中：
- 正样本梯度权重 = 1/101 × (-1/p_pos)
- 100个负样本总梯度 = 100/101 × (1/(1-p_neg))
- 当 loss≈0.015 时：−log(p_pos)/101 ≈ 0.014 → p_pos ≈ 0.24（不是 0.5！）

---

## 当前 PBS 作业（Loop 5）

| 字段 | 值 |
|------|-----|
| 作业 ID | **3925521.pbs** |
| 脚本 | ~/subs/auto_loop5_pos_weight_fix.sh |
| 训练参数 | 同 Loop 4 + **pos_weight=100**（关键新增）|
| 推理阈值 | **0.1**（安全网；修复后 0.5 也应工作）|
| 输出目录 | /storage/agrp/yiwen/runs/loop5_pos_weight_fix/ |

## Loop 5 代码修改

- `src/train.py` 第 324 行：`criterion = None`（balanced 时不用全局 criterion）
- `src/train.py` 第 390-402 行：balanced_sampling 时用加权 BCE：
  ```python
  sample_weight = where(lab==1, neg_pos_ratio, 1.0)
  loss = F.binary_cross_entropy(pred, lab.float(), weight=sample_weight)
  ```

## 研究进展

| Loop | 假设 | 结果 | 关键发现 |
|------|------|------|---------|
| 1 | 层内分组注意力架构 | ❌ OOM | 代码正确，内存超限 |
| 2 | --max-events 2000 修复 OOM | ❌ n_kept=0 | AUC=0.976，FocalLoss 得分全<0.5 |
| 3 | 平衡采样 + BCELoss | ❌ PBS bug | 缺 --clusters，训练从未运行 |
| 4 | 修复 3 个 PBS bug | ❌ n_kept=0 | AUC=0.977！但 BCELoss 梯度不对称 |
| **5** | **pos_weight=100 修正梯度** | ⏳ **进行中** | 预期 efficiency >30% |

## 成功标准

`track_efficiency >= 0.60 AND fake_rate <= 0.20`（10k 测试事件）
