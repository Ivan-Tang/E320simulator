#!/bin/bash
# ============================================================
# stop_research.sh
# 紧急停止自主研究循环（停止 watcher + 取消 PBS 作业）
#
# 用法:
#   bash stop_research.sh        # 紧急停止（立即）
#   bash stop_research.sh grace  # 优雅停止（等当前作业完成）
# ============================================================

PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
MODE="${1:-emergency}"

cd "${PROJ_DIR}"

echo "========================================================"
echo "  停止自主研究循环 (模式: ${MODE})"
echo "========================================================"

if [ "${MODE}" = "grace" ]; then
    echo "[1] 设置 stop_requested 标志（等当前 PBS 作业完成后停止）..."
    python3 -c "
import json
with open('experiment_state.json') as f: s = json.load(f)
s['stop_requested'] = True
with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
print('  stop_requested = True')
"
    git add experiment_state.json
    git commit -m "manual: graceful stop requested"
    git push
    touch "${PROJ_DIR}/.stop_watcher"
    echo "  watcher 将在当前作业完成后退出"
else
    # 紧急停止
    echo "[1] 停止 watcher..."
    touch "${PROJ_DIR}/.stop_watcher"
    if [ -f "${PROJ_DIR}/.watcher.pid" ]; then
        WP=$(cat "${PROJ_DIR}/.watcher.pid")
        kill "${WP}" 2>/dev/null && echo "  watcher PID=${WP} killed" || echo "  watcher 已不在运行"
    fi

    echo "[2] 取消当前 PBS 作业..."
    CURRENT_JOB=$(python3 -c "
import json
try:
    s = json.load(open('experiment_state.json'))
    print(s.get('current_pbs_job_id') or '')
except: print('')
" 2>/dev/null)
    if [ -n "${CURRENT_JOB}" ]; then
        qdel "${CURRENT_JOB}" 2>/dev/null && echo "  qdel ${CURRENT_JOB}" || echo "  作业已结束或不存在"
    else
        echo "  没有正在运行的 PBS 作业"
    fi

    echo "[3] 更新状态..."
    python3 -c "
import json
from datetime import datetime
with open('experiment_state.json') as f: s = json.load(f)
s['loop_status'] = 'stopped'
s['stop_requested'] = True
s['error_state'] = 'Emergency stop by human at $(date)'
s['last_updated'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
print('  state updated')
"
    git add experiment_state.json
    git commit -m "manual: emergency stop at $(date)"
    git push
fi

echo ""
echo "  恢复方法: bash ${PROJ_DIR}/start_research.sh \"{目标描述}\""
echo "========================================================"
