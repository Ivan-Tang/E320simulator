# 自主研究状态

**系统状态**: IDLE（尚未启动）

启动自主循环前，请先：
1. 填写 `research_goal.md`
2. 创建研究分支：`git checkout -b auto-research-{目标描述}`
3. 编辑 `experiment_state.json`：填入 `loop_start_time`、`research_branch`、`max_loops`
4. 启动 watcher：`nohup bash ~/subs/autonomous_watcher.sh > ~/logs/watcher.log 2>&1 &`
5. 手动触发第一次：`claude --print --dangerously-skip-permissions "$(cat autonomous_loop_prompt.md)"`

---

## 干预命令

**优雅停止**（等当前作业完成后停止）：
```bash
cd ~/E320simulator
python3 -c "
import json
with open('experiment_state.json') as f: s = json.load(f)
s['stop_requested'] = True
with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
print('stop_requested = True')
"
git add experiment_state.json && git commit -m "manual: request graceful stop" && git push
touch ~/E320simulator/.stop_watcher
```

**紧急停止**（立即终止一切）：
```bash
touch ~/E320simulator/.stop_watcher
kill $(cat ~/E320simulator/.watcher.pid 2>/dev/null) 2>/dev/null
CURRENT_JOB=$(python3 -c "import json; print(json.load(open('experiment_state.json')).get('current_pbs_job_id') or '')" 2>/dev/null)
[ -n "$CURRENT_JOB" ] && qdel "$CURRENT_JOB" && echo "Cancelled: $CURRENT_JOB"
```

**恢复**（清除错误状态后重启）：
```bash
# 先处理问题，然后：
python3 -c "
import json
with open('experiment_state.json') as f: s = json.load(f)
s['error_state'] = None; s['stop_requested'] = False; s['loop_status'] = 'idle'
with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
"
rm -f ~/E320simulator/.stop_watcher
git add experiment_state.json && git commit -m "manual: clear error, resuming" && git push
nohup bash ~/subs/autonomous_watcher.sh > ~/logs/watcher.log 2>&1 &
```
