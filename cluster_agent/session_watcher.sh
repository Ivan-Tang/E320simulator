#!/bin/bash
# ============================================================
# session_watcher.sh — Per-session 研究循环 watcher 守护进程
#
# 每个研究 session 独立一个 watcher，互不干扰。
# 主仓库始终保持在 master，watcher 在各自的 worktree 中运行。
#
# 启动方式（由 start_research.sh 调用，不要手动启动）:
#   PROJ_DIR=/srv01/agrp/yiwen/research/<session> \
#   nohup bash cluster_agent/session_watcher.sh \
#       > /srv01/agrp/yiwen/logs/watcher_<session>.log 2>&1 &
#
# 环境变量:
#   PROJ_DIR  — session worktree 目录（必须设置）
# ============================================================

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-}"
if [ -z "${PROJ_DIR}" ]; then
    echo "[session_watcher] ERROR: PROJ_DIR 未设置，退出"
    exit 1
fi

if [ ! -d "${PROJ_DIR}" ]; then
    echo "[session_watcher] ERROR: PROJ_DIR 不存在: ${PROJ_DIR}"
    exit 1
fi

LOGS_DIR="/srv01/agrp/yiwen/logs"
POLL_INTERVAL=30  # 轮询间隔（秒）

SESSION_NAME=$(basename "${PROJ_DIR}")
PID_FILE="${PROJ_DIR}/.watcher.pid"
STOP_FILE="${PROJ_DIR}/.stop_watcher"
TRIGGER_FILE="${PROJ_DIR}/.job_done_trigger"

mkdir -p "${LOGS_DIR}"

echo "[watcher:${SESSION_NAME}] 启动 PID=$$"
echo "[watcher:${SESSION_NAME}] PROJ_DIR=${PROJ_DIR}"
echo "[watcher:${SESSION_NAME}] 轮询间隔: ${POLL_INTERVAL}s"
echo "$$" > "${PID_FILE}"

# ── Conda 环境 ──
set +u
source /usr/wipp/conda/24.5.0/etc/profile.d/conda.sh
conda activate e320root
set -u

# ── 主轮询循环 ──
while true; do
    # 停止信号
    if [ -f "${STOP_FILE}" ]; then
        echo "[watcher:${SESSION_NAME}] $(date -u +%Y-%m-%dT%H:%M:%S) 收到停止信号，退出"
        rm -f "${PID_FILE}"
        exit 0
    fi

    # 检查实验状态：若 loop_status=completed 或存在触发文件，则运行 Claude
    SHOULD_RUN=false
    if [ -f "${TRIGGER_FILE}" ]; then
        SHOULD_RUN=true
        rm -f "${TRIGGER_FILE}"
        echo "[watcher:${SESSION_NAME}] $(date -u +%Y-%m-%dT%H:%M:%S) 检测到 .job_done_trigger"
    elif [ -f "${PROJ_DIR}/experiment_state.json" ]; then
        LOOP_STATUS=$(python3 -c "
import json, sys
try:
    s = json.load(open('${PROJ_DIR}/experiment_state.json'))
    print(s.get('loop_status', 'idle'))
except Exception as e:
    print('idle')
" 2>/dev/null)
        if [ "${LOOP_STATUS}" = "completed" ]; then
            SHOULD_RUN=true
            echo "[watcher:${SESSION_NAME}] $(date -u +%Y-%m-%dT%H:%M:%S) loop_status=completed，触发 Claude"
        fi
    fi

    if [ "${SHOULD_RUN}" = "true" ]; then
        LOOP_LOG="${LOGS_DIR}/claude_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"
        echo "[watcher:${SESSION_NAME}] 运行 Claude，日志: ${LOOP_LOG}"

        cd "${PROJ_DIR}"

        # 读取 prompt 并运行 Claude
        if [ ! -f "${PROJ_DIR}/autonomous_loop_prompt.md" ]; then
            echo "[watcher:${SESSION_NAME}] ERROR: 找不到 autonomous_loop_prompt.md"
        else
            PROMPT=$(cat "${PROJ_DIR}/autonomous_loop_prompt.md")
            claude \
                --print \
                --dangerously-skip-permissions \
                "${PROMPT}" \
                > "${LOOP_LOG}" 2>&1
            EXIT_CODE=$?
            echo "[watcher:${SESSION_NAME}] Claude 完成 (exit=${EXIT_CODE})"

            # 通知 Slack（静默失败）
            python3 "${PROJ_DIR}/cluster_agent/notifier.py" \
                --slack "[${SESSION_NAME}] Loop 完成 (exit=${EXIT_CODE})，查看 research_log.md" \
                2>/dev/null || true
        fi
    fi

    sleep "${POLL_INTERVAL}"
done
