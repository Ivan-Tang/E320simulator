# 研究目标

## 当前目标

**让 TransformerEdgeClassifier 从完全失效状态收敛到可用水平。**

当前状态（来自 progress.md Benchmark 结果）：
- TransformerEdgeClassifier: efficiency **2.2%**，fake_rate **99.7%**，基本等同于随机输出
- 根因推测：`max_len=1000` 硬编码上限，而 bg=700 条件下每事件约 3500 hits，序列直接越界；此外模型容量（d_model=256，6层）可能对 E320 数据量来说过大导致不收敛

对比基准：
- InteractionNet: efficiency 70.6%，fake_rate 14.3%（目前最优 ML 模型）
- Baseline: efficiency 74.3%，fake_rate 42.5%

## 成功判断标准

```
track_efficiency >= 0.60 AND fake_rate <= 0.20
（在 10k 测试事件上评估，与 Benchmark 条件一致）
```

达到此标准即视为本阶段目标完成（`goal_achieved = true`）。

## 允许的改动范围（白名单）

- `src/models.py` — 修改 `TransformerEdgeClassifier` 架构、超参默认值
- `src/layers.py` — 修改 `TransformerEmbedder` 及底层模块（注意力、位置编码等）
- `src/train.py` — 修改训练策略、超参、调度器、warmup；可新增/修改 transformer 相关的 `TrainConfig` 字段
- `src/losses.py` — 修改 FocalLoss 参数或添加新 loss

**绝对禁止修改**：`geometry.py`、`simulator.py`、`utils.py`、`config.py`、`CLAUDE.md`、`autonomous_loop_prompt.md`、`autonomous_watcher.sh`

## 优先探索方向（按预期收益排序）

### 第一优先：修复已知硬 bug
1. **`max_len` 越界**：`TransformerEmbedder` 的 `max_len=1000` 在 bg=700（~3500 hits）时直接越界。需动态适配序列长度，或将 `max_len` 改为按实际输入动态设置。
2. **位置编码维度**：确认 `PositionalEncoding3D` 在长序列下不产生越界或 NaN。

### 第二优先：模型容量下调，先让模型收敛
- 当前默认 d_model=256，6层 encoder，参数量对 E320 数据量可能过大
- 建议先尝试 d_model=64~128，2~3层，验证收敛后再逐步放大
- 减小 dropout（0.1 → 0.05）或移除 dropout 先看收敛

### 第三优先：训练策略调整
- 加 warmup（前几百 steps 线性升温），Transformer 对初始 lr 敏感
- focal_alpha/gamma 重新调参（当前 alpha=0.995 可能对 Transformer 过于激进）
- 考虑梯度裁剪值（当前 grad_clip=1.0）

### 第四优先：架构改进（收敛后再做）
- 在节点编码后引入边特征的交叉注意力，而非简单拼接
- 稀疏注意力（只关注 k-NN 邻居），降低 O(N²) 开销并提升归纳偏置

## 训练数据规模

- **默认使用 10k 测试事件**（与 Benchmark 一致）
- 仅当 10k 事件在单 GPU（RTX 3090，24GB VRAM）上 OOM 时，才缩减到更小规模（如 2k 或 5k）
- Claude 可自行决定每轮具体的 epochs、batch_size、数据量，无需额外确认

## 最大循环次数

15

## 分支规则

所有代码改动必须在 `auto-research-transformer` 分支，不得直接 commit 到 master。

## 备注

- 本阶段不做 TrackFormer-Seed（DETR 端到端方向），不涉及 `train_trackformer.py`
- `TransformerEdgeClassifier` 通过 `train.py` 的 `model_type="transformer"` 训练，与 InteractionNet 同一训练框架
- 每次循环的假设和结果记录在 `research_log.md`
- 研究进展随时查看 `status.md`
