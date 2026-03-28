# autonomous_watcher.sh 通知集成补丁

在 `~/subs/autonomous_watcher.sh` 中，找到作业完成/失败的判断处，加入以下调用。

## 在文件开头（变量定义区）追加

```bash
PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
NOTIFY_CMD="conda run -n e320root python ${PROJ_DIR}/cluster_agent/notifier.py"
```

## 在 PBS 作业完成检测处追加通知

找到检测 qstat 确认作业完成的代码块，在其后添加：

```bash
# ── 作业完成通知 ──
${NOTIFY_CMD} --slack "✅ PBS 作业 ${JOB_ID} 完成 (Loop ${LOOP_COUNT})，正在触发 Claude 分析…" || true
```

## 在 goal_achieved=true 处追加通知

```bash
# ── 目标达成通知 ──
RESULTS=$(python3 -c "
import json
s = json.load(open('${PROJ_DIR}/experiment_state.json'))
r = s.get('last_eval_results', {})
if r:
    key = list(r.keys())[-1]
    v = r[key]
    print(f\"eff={v.get('track_efficiency',0):.1%}  fake={v.get('fake_rate',0):.1%}\")
" 2>/dev/null || echo "见 experiment_state.json")
${NOTIFY_CMD} \
    --slack "🎉 *研究目标已达成！* Loop ${LOOP_COUNT}  ${RESULTS}" \
    --email "研究目标达成 ${RESULTS}" || true
```

## 在 error_state 设置处追加通知

```bash
# ── 错误通知 ──
ERROR_MSG=$(python3 -c "
import json
s = json.load(open('${PROJ_DIR}/experiment_state.json'))
print(s.get('error_state','未知错误'))
" 2>/dev/null || echo "见 experiment_state.json")
${NOTIFY_CMD} \
    --slack "❌ *错误，需人工干预*: ${ERROR_MSG}" \
    --email "E320 Agent 错误: ${ERROR_MSG}" || true
```

## 在 max_loops 达到时追加通知

```bash
# ── 达到最大循环次数通知 ──
${NOTIFY_CMD} \
    --slack "🔚 已达到最大循环次数 ${MAX_LOOPS}，研究会话结束。" \
    --email "E320 research session 结束（max_loops=${MAX_LOOPS}）" || true
```

---

**注意**：所有调用末尾都有 `|| true`，确保通知失败不影响主流程。
