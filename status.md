# 自主研究状态

## 🎉 研究目标已达成！

**更新时间**: 2026-03-22T12:00:00
**当前循环**: 7/15
**状态**: GOAL ACHIEVED ✓

---

## 最终结果（Loop 7，PBS 3925801.pbs）

| 指标 | 结果 | 目标 | 达标 |
|------|------|------|------|
| track_efficiency | **82.01%** | ≥ 60% | ✓ |
| fake_rate | **11.66%** | ≤ 20% | ✓ |
| mean_rms | **4.634 µm** | — | — |

**测试集**：10k 事件，1173 真实径迹，n_matched=962，n_kept=1089（chi2 < 7e-5 筛选后）

---

## 与 InteractionNet 对比（原最优 ML 模型）

| 指标 | TransformerEdgeClassifier | InteractionNet | 结论 |
|------|---|---|---|
| track_efficiency | **82.01%** | 70.6% | Transformer **优胜 +11.4pp** |
| fake_rate | **11.66%** | 14.3% | Transformer **更低 -2.6pp** |
| mean_rms | 4.634 µm | 4.09 µm | InteractionNet 略优 |

---

## 最优模型位置

- **Checkpoint**: `/storage/agrp/yiwen/runs/loop5_pos_weight_fix/best_model.pt`
- **推理参数**: `--edge-threshold 0.1`，后处理 `chi2 < 7e-5`
- **架构**: TransformerEdgeClassifier，层内分组注意力，d_model=64，n_heads=4，n_encoder_layers=2

---

## 关键突破历程（7轮迭代）

| Loop | 关键改动 | efficiency | fake_rate |
|------|---------|-----------|----------|
| 基线 | TransformerEdgeClassifier（旧） | 2.2% | 99.7% |
| 1 | 层内分组注意力 + OOM保护 | OOM/0% | — |
| 2 | max_events=2000（解决OOM） | 0% | 0% |
| 3-4 | balanced_sampling + BCELoss | 0% | 0% |
| **5** | **pos_weight=100（修复分数校准）** | **82.95%** | **37.1%** |
| 6 | chi2质量筛选（PBS脚本bug） | FAILED | FAILED |
| **7** | **修复PBS脚本 → 正式验证** | **82.01%** | **11.66%** ✓ |

---

## 下一步建议（人工决策）

1. **Merge 到 master**：本分支 `auto-research-transformer-convergence` 可合并
2. **提升效率上限**：目标 ≥95%，需要更大训练集（解决 OOM 后全量 10k 训练）
3. **减小 mean_rms**：Transformer 的 4.634 µm vs InteractionNet 4.09 µm，可优化位置编码
4. **调低 fake_rate**：chi2 阈值可进一步收紧（如 5e-5），权衡 eff 损失
