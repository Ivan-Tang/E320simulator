# 研究洞见记录
_由 Claude 自动维护，每轮提炼新规律，跨 session 积累。_
_格式：**结论**（Loop 编号/来源）_

---

## 模型架构

- **层内分组注意力（Layer-Grouped Attention）是 TransformerEdgeClassifier 必须的设计**：按 layer_id 分 5 组（每组 ~700 hits），各组独立 self-attention。全局 O(N²) attention（N~3500 hits/event）存在梯度稀释且 OOM。（Loop 1，session transformer-convergence）

## 数据处理

- **`build_labeled_edges_from_sim` 在 10k 事件上 OOM**：训练必须用 `--max-events 2000`（或分批处理）。全量边图构建超出 GPU 节点（RTX 3090, 24GB）内存上限。（Loop 1-2，session transformer-convergence）

## 损失函数与类别不平衡

- **FocalLoss 在 1:50000 极端不平衡下失效**：AUC 可达 0.97，但所有边得分趋近 0（n_kept=0）。FocalLoss 无法解决 pos_frac=0.00002 级别的不平衡。（Loop 2，session transformer-convergence）
- **BCELoss + balanced_sampling(1:100) 的梯度均衡点是 0.24，不是 0.5**：数学推导：-log(p_pos) ≈ 1.41 → p_pos ≈ 0.24。不加 pos_weight 时，threshold=0.5 永远不会触发（n_kept=0），即使训练收敛、AUC=0.977。（Loop 4，session transformer-convergence）
- **修复方案：pos_weight=neg_pos_ratio（100）**：每正样本梯度贡献 100× 于负样本，均衡点推向 1.0，配合 threshold=0.1 即可触发正边得分 > 0.5。（Loop 5，session transformer-convergence）

## 后处理

- **chi2 < 7e-5 可将 fake_rate 从 37% 降至 11.66%，同时只损失 0.2% 真实轨迹**：
  - 真实 4-layer track：chi2 均值=2.1e-5，std=1.6e-5
  - 假 track：chi2 均值=1.56e-4，std=1.09e-4（均值 7× 差异）
  - 7e-5 = 真实轨迹的 99.8th 百分位，筛除 78.5% 假轨迹
  - （Loop 5-7，session transformer-convergence）

## PBS 脚本规范（已确认的 Bug 源）

- **`run_model.py` 推理输出用 `--output <file_path>`**，不存在 `--output-dir` 参数（Loop 6 bug）
- **eval 步骤用 `--edge-checkpoint`**，不是 `--checkpoint`（Loop 3-4 bug）
- **训练命令必须包含 `--clusters <绝对路径>`**（Loop 3 bug）
- **PBS 日志路径必须用绝对路径**：`/srv01/agrp/yiwen/logs/auto_LABEL.out`（相对路径找不到目录）

## 已达到最优配置（session transformer-convergence，2026-03-22）

| 项目 | 值 |
|------|---|
| 架构 | TransformerEdgeClassifier，层内分组注意力 |
| d_model / n_heads / n_encoder_layers | 64 / 4 / 2 |
| 训练策略 | balanced_sampling，neg_pos_ratio=100，pos_weight=100，BCELoss |
| epochs / lr / warmup | 200 / 1e-3 / 10 |
| 推理阈值 | edge_threshold=0.1，chi2_cut=7e-5 |
| **track_efficiency** | **82.01%** |
| **fake_rate** | **11.66%** |
| mean_rms | 4.634 µm |
| checkpoint | `/storage/agrp/yiwen/runs/loop5_pos_weight_fix/best_model.pt` |

---
_最后更新：2026-03-22（session transformer-convergence 研究目标达成后）_
