# 自主研究状态

**更新时间**: 2026-03-21T20:00
**循环**: 1 / 15
**状态**: submitted — PBS 作业已提交

---

## 当前假设（Loop 1）

修复 CLI 中缺失 "transformer" 选项的 bug，用小容量参数（d_model=64, 2层encoder, focal_alpha=0.90, warmup 5 epochs）首次正确训练 TransformerEdgeClassifier。

## 上轮结果

| 指标 | 值 |
|------|----|
| track_efficiency | 2.2%（完全失效，Benchmark checkpoint） |
| fake_rate | 99.7% |
| mean_rms | 4726 µm |

## 本轮训练配置

| 参数 | 值 |
|------|----|
| model | transformer |
| d_model | 64 |
| n_heads | 4 |
| n_encoder_layers | 2 |
| dim_feedforward | 128 |
| dropout | 0.05 |
| focal_alpha | 0.90 |
| warmup_epochs | 5 |
| epochs | 30 |
| lr | 1e-4 |

## PBS 作业

| 项目 | 值 |
|------|----|
| 脚本 | ~/subs/auto_loop1_fix_cli_small.sh |
| Job ID | 3924893.pbs |
| 输出目录 | /storage/agrp/yiwen/runs/loop1_fix_cli_small/ |

## 研究进展简表

| Loop | 假设 | 结果 |
|------|------|------|
| 1 | 修复 CLI + 小容量 Transformer | 运行中 |

## 干预命令

**优雅停止**：
```bash
cd ~/E320simulator
python3 -c "
import json
with open('experiment_state.json') as f: s = json.load(f)
s['stop_requested'] = True
with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
"
git add experiment_state.json && git commit -m "manual: request graceful stop" && git push
touch ~/E320simulator/.stop_watcher
```

**紧急停止**：
```bash
touch ~/E320simulator/.stop_watcher
kill $(cat ~/E320simulator/.watcher.pid 2>/dev/null) 2>/dev/null
CURRENT_JOB=$(python3 -c "import json; print(json.load(open('experiment_state.json')).get('current_pbs_job_id') or '')" 2>/dev/null)
[ -n "$CURRENT_JOB" ] && qdel "$CURRENT_JOB" && echo "Cancelled: $CURRENT_JOB"
```
