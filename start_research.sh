#!/bin/bash
# ============================================================
# start_research.sh
# 一键启动自主研究循环
#
# 用法:
#   bash start_research.sh "inet efficiency"  # 目标描述（用于分支名）
#   bash start_research.sh "inet efficiency" 20  # 目标描述 + 最大循环次数（默认 15）
#
# 前提: 先填好 research_goal.md
# ============================================================

set -euo pipefail

PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
LOGS_DIR="/srv01/agrp/yiwen/logs"

# --- 参数处理 ---
GOAL_SLUG="${1:-research}"
MAX_LOOPS="${2:-15}"

# 分支名：小写，空格→连字符
BRANCH_SLUG=$(echo "${GOAL_SLUG}" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr '_' '-')
BRANCH_NAME="auto-research-${BRANCH_SLUG}"
NOW=$(date -u +"%Y-%m-%dT%H:%M:%S")

echo "========================================================"
echo "  自主研究循环启动"
echo "  目标描述: ${GOAL_SLUG}"
echo "  分支: ${BRANCH_NAME}"
echo "  最大循环次数: ${MAX_LOOPS}"
echo "  启动时间: ${NOW}"
echo "========================================================"

cd "${PROJ_DIR}"

# --- 检查 research_goal.md 已填写 ---
if grep -q "请在此填写本次研究目标" research_goal.md 2>/dev/null; then
    echo "ERROR: research_goal.md 还未填写，请先编辑它："
    echo "  vim ${PROJ_DIR}/research_goal.md"
    exit 1
fi

# --- 检查 watcher 是否已在运行 ---
if [ -f "${PROJ_DIR}/.watcher.pid" ]; then
    OLD_PID=$(cat "${PROJ_DIR}/.watcher.pid")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "WARNING: watcher 已在运行 (PID=${OLD_PID})，先停止旧的..."
        touch "${PROJ_DIR}/.stop_watcher"
        sleep 3
        kill "${OLD_PID}" 2>/dev/null || true
        rm -f "${PROJ_DIR}/.stop_watcher"
    fi
fi

# --- Git: 拉取最新，切换到研究分支 ---
echo ""
echo "[1/5] Git 同步..."
git pull

# 切换或创建研究分支
if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
    echo "  分支已存在，切换到 ${BRANCH_NAME}"
    git checkout "${BRANCH_NAME}"
    git pull origin "${BRANCH_NAME}" 2>/dev/null || true
else
    echo "  创建新分支: ${BRANCH_NAME}"
    git checkout -b "${BRANCH_NAME}"
    git push -u origin "${BRANCH_NAME}"
fi

# --- 初始化 experiment_state.json ---
echo ""
echo "[2/5] 初始化状态文件..."
python3 -c "
import json, os
state_file = '${PROJ_DIR}/experiment_state.json'
# 读取现有状态（如果有）
if os.path.exists(state_file):
    with open(state_file) as f:
        s = json.load(f)
else:
    s = {}

# 重置为新 session 状态
s.update({
    'schema_version': '2.0',
    'loop_status': 'idle',
    'loop_count': 0,
    'max_loops': ${MAX_LOOPS},
    'loop_start_time': '${NOW}',
    'current_pbs_job_id': None,
    'current_loop_label': None,
    'current_loop_script': None,
    'research_branch': '${BRANCH_NAME}',
    'last_eval_results': None,
    'stop_requested': False,
    'goal_achieved': False,
    'error_state': None,
    'last_updated': '${NOW}',
    'last_updated_by': 'start_research.sh'
})
with open(state_file, 'w') as f:
    json.dump(s, f, indent=2)
print('  experiment_state.json initialized')
"

# --- 清理遗留文件 ---
rm -f "${PROJ_DIR}/.job_done_trigger"
rm -f "${PROJ_DIR}/.stop_watcher"

# --- Git: 提交初始化状态 ---
echo ""
echo "[3/5] 提交初始状态..."
git add experiment_state.json research_goal.md research_log.md status.md autonomous_loop_prompt.md
git diff --cached --quiet || git commit -m "auto: start research session '${GOAL_SLUG}' on ${NOW}"
git push origin "${BRANCH_NAME}"

# --- 启动 watcher ---
echo ""
echo "[4/5] 启动 watcher 守护进程..."
mkdir -p "${LOGS_DIR}"
nohup bash ~/subs/autonomous_watcher.sh > "${LOGS_DIR}/watcher.log" 2>&1 &
WATCHER_PID=$!
sleep 1
if kill -0 "${WATCHER_PID}" 2>/dev/null; then
    echo "  watcher 启动成功 (PID=${WATCHER_PID})"
    echo "  日志: tail -f ${LOGS_DIR}/watcher.log"
else
    echo "ERROR: watcher 启动失败，检查 ${LOGS_DIR}/watcher.log"
    exit 1
fi

# --- 触发第一次 Claude 会话 ---
echo ""
echo "[5/5] 触发首次研究循环..."
SESSION_LOG="${LOGS_DIR}/claude_init_$(date +%Y%m%d_%H%M%S).log"
echo "  Claude 会话日志: ${SESSION_LOG}"
echo "  （在后台运行，可用 tail -f ${SESSION_LOG} 跟踪）"

nohup claude \
    --print \
    --dangerously-skip-permissions \
    "$(cat ${PROJ_DIR}/autonomous_loop_prompt.md)" \
    > "${SESSION_LOG}" 2>&1 &

echo ""
echo "========================================================"
echo "  自主研究循环已启动！"
echo ""
echo "  查看进度:  cat ${PROJ_DIR}/status.md"
echo "  查看日志:  tail -f ${LOGS_DIR}/watcher.log"
echo "  查看研究:  cat ${PROJ_DIR}/research_log.md"
echo ""
echo "  优雅停止:  touch ${PROJ_DIR}/.stop_watcher"
echo "  紧急停止:  bash ${PROJ_DIR}/stop_research.sh"
echo "========================================================"
