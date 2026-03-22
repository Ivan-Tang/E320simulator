# 自主研究状态

**更新时间**: 2026-03-22T11:00
**循环进度**: Loop 7 / 15
**研究分支**: auto-research-transformer-convergence

---

## 当前状态

**Loop 7 已提交**：修复 loop6 PBS 脚本 `--output-dir` bug，重新运行 chi2 质量筛选评估

**关键发现（本地验证）**：
- Loop 5 模型 + chi2 < 7e-5 筛选：efficiency=**82.01%**，fake_rate=**11.66%**
- **研究目标已满足！** (eff ≥ 60%, fake_rate ≤ 20%)
- Loop 7 PBS 作业目的：正式记录并确认这个结果

---

## 上轮结果（Loop 6）

| 指标 | 值 |
|------|-----|
| 状态 | **FAILED**（PBS 脚本 bug） |
| 失败原因 | `run_model.py` 不接受 `--output-dir` 参数 |
| Loop 5 本地验证 | efficiency=82.01%，fake_rate=11.66% |

---

## 历史结果摘要

| Loop | 关键变更 | Efficiency | Fake Rate | 状态 |
|------|---------|-----------|----------|------|
| 1 | 层内分组注意力 | N/A | N/A | OOM |
| 2 | max_events=2000 | 0% | 0% | 得分<0.5（FocalLoss校准失效）|
| 3 | balanced sampling | N/A | N/A | PBS脚本缺--clusters |
| 4 | 修复PBS脚本 | 0% | 0% | BCELoss均衡点=0.24 |
| 5 | pos_weight=100 | **82.95%** | 37.14% | ✓ 效率突破！但fake_rate超标 |
| 6 | chi2 < 7e-5筛选 | N/A | N/A | PBS bug: --output-dir |
| **7** | **修复--output bug** | **~82.01%** | **~11.66%** | **待确认（预期目标达成）** |

---

## 当前假设

修复 PBS 脚本参数名错误，用 loop5 模型推理 + chi2 < 7e-5 质量筛选，
正式确认 efficiency=82.01%、fake_rate=11.66%，满足研究目标。

---

## PBS 作业信息

- **脚本**: `~/subs/auto_loop7_fix_output_arg.sh`
- **模型**: loop5 checkpoint（无需重训）
- **Job ID**: 3925801.pbs

---

## 研究目标

```
track_efficiency >= 0.60 AND fake_rate <= 0.20
```
预期达成状态：✅（本地验证通过）
