#!/bin/bash
# ============================================================
# start_agent.sh — 在集群分析节点上启动 E320 Research Agent
#
# 用法: bash cluster_agent/start_agent.sh
#
# 前提:
#   1. 已安装依赖: conda run -n e320root pip install slack-bolt python-dotenv requests
#   2. 已在 .env 中填写 SLACK_BOT_TOKEN / SLACK_APP_TOKEN / SLACK_CHANNEL_ID
# ============================================================

set -eo pipefail

PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
LOGS_DIR="/srv01/agrp/yiwen/logs"
PID_FILE="${PROJ_DIR}/.agent.pid"
LOG_FILE="${LOGS_DIR}/agent.log"

# ── 检查是否已在运行 ──
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "E320 Agent 已在运行 (PID=${OLD_PID})"
        echo "如需重启，先运行: bash ${PROJ_DIR}/cluster_agent/stop_agent.sh"
        exit 1
    else
        echo "发现过期 PID 文件，清理中..."
        rm -f "${PID_FILE}"
    fi
fi

# ── 检查 .env ──
if [ ! -f "${PROJ_DIR}/.env" ]; then
    echo "ERROR: 找不到 ${PROJ_DIR}/.env"
    echo "请参考 .env.example 创建并填入 Slack token"
    exit 1
fi

if ! grep -q "SLACK_BOT_TOKEN=xoxb-" "${PROJ_DIR}/.env" 2>/dev/null; then
    echo "ERROR: .env 中未找到有效的 SLACK_BOT_TOKEN（应以 xoxb- 开头）"
    exit 1
fi

# ── 激活 conda 并启动 ──
echo "正在启动 E320 Research Agent..."

set +u  # conda 内部脚本引用未定义变量，临时关闭 -u
source /usr/wipp/conda/24.5.0/etc/profile.d/conda.sh
conda activate e320root
set -u

mkdir -p "${LOGS_DIR}"
cd "${PROJ_DIR}"

nohup env PYTHONUNBUFFERED=1 python cluster_agent/agent.py > "${LOG_FILE}" 2>&1 &
AGENT_PID=$!
echo "${AGENT_PID}" > "${PID_FILE}"

# 等待一秒确认进程存活
sleep 2
if kill -0 "${AGENT_PID}" 2>/dev/null; then
    echo "✓ E320 Research Agent 启动成功 (PID=${AGENT_PID})"
    echo "  日志: tail -f ${LOG_FILE}"
    echo "  停止: bash ${PROJ_DIR}/cluster_agent/stop_agent.sh"
else
    echo "✗ Agent 启动失败，请检查日志:"
    echo "  tail -20 ${LOG_FILE}"
    rm -f "${PID_FILE}"
    exit 1
fi
