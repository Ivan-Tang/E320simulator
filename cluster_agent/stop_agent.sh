#!/bin/bash
# ============================================================
# stop_agent.sh — 停止 E320 Research Agent 守护进程
# ============================================================

PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
PID_FILE="${PROJ_DIR}/.agent.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "未找到 PID 文件，Agent 可能未在运行"
    exit 0
fi

PID=$(cat "${PID_FILE}")

if ! kill -0 "${PID}" 2>/dev/null; then
    echo "PID ${PID} 已不存在，清理文件"
    rm -f "${PID_FILE}"
    exit 0
fi

echo "正在停止 E320 Research Agent (PID=${PID})..."
kill "${PID}"
sleep 2

if kill -0 "${PID}" 2>/dev/null; then
    echo "进程未响应，强制终止..."
    kill -9 "${PID}" 2>/dev/null || true
fi

rm -f "${PID_FILE}"
echo "✓ Agent 已停止"
