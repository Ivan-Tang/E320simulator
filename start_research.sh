#!/bin/bash
# ============================================================
# start_research.sh — 一键启动自主研究循环（worktree 模式）
#
# 用法:
#   bash start_research.sh "目标描述"       # 默认最大 15 轮
#   bash start_research.sh "目标描述" 20    # 指定最大循环次数
#
# 前提: 先填好 research_goal.md（或用 !goal "..." 通过 Slack 填写）
#
# 架构:
#   主仓库 /srv01/agrp/yiwen/E320simulator 永远在 master
#   每个 session 在 /srv01/agrp/yiwen/research/<slug>/ 独立 worktree
#   多个 session 可同时运行，互不干扰
# ============================================================

set -euo pipefail

PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
RESEARCH_DIR="/srv01/agrp/yiwen/research"   # worktree 根目录
LOGS_DIR="/srv01/agrp/yiwen/logs"

# ── 参数处理 ──
GOAL_SLUG="${1:-research}"
MAX_LOOPS="${2:-15}"

# 分支名/目录名：小写，空格→连字符
BRANCH_SLUG=$(echo "${GOAL_SLUG}" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr '_' '-' | tr -cd '[:alnum:]-')
BRANCH_NAME="auto-research-${BRANCH_SLUG}"
WORKTREE_DIR="${RESEARCH_DIR}/${BRANCH_SLUG}"
NOW=$(date -u +"%Y-%m-%dT%H:%M:%S")

echo "========================================================"
echo "  自主研究循环启动（worktree 模式）"
echo "  目标描述  : ${GOAL_SLUG}"
echo "  分支      : ${BRANCH_NAME}"
echo "  Worktree  : ${WORKTREE_DIR}"
echo "  最大循环  : ${MAX_LOOPS}"
echo "  启动时间  : ${NOW}"
echo "========================================================"

# 确认在主仓库目录
cd "${PROJ_DIR}"

# ── 检查 research_goal.md 已填写 ──
if grep -q "请在此填写本次研究目标" research_goal.md 2>/dev/null; then
    echo "ERROR: research_goal.md 还未填写，请先编辑："
    echo "  vim ${PROJ_DIR}/research_goal.md"
    exit 1
fi

# ── 停止已有同名 session 的 watcher（如果在运行）──
if [ -f "${WORKTREE_DIR}/.watcher.pid" ]; then
    OLD_PID=$(cat "${WORKTREE_DIR}/.watcher.pid" 2>/dev/null || echo "")
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "WARNING: 该 session watcher 已在运行 (PID=${OLD_PID})，先停止旧的..."
        touch "${WORKTREE_DIR}/.stop_watcher"
        sleep 3
        kill "${OLD_PID}" 2>/dev/null || true
        rm -f "${WORKTREE_DIR}/.stop_watcher"
    fi
fi

# ── [1/5] Git 同步（主仓库 master）──
echo ""
echo "[1/5] Git 同步..."
git checkout master
git pull

mkdir -p "${RESEARCH_DIR}"

# 创建或复用 worktree
if [ -d "${WORKTREE_DIR}" ]; then
    # worktree 已存在：验证分支一致
    WT_BRANCH=$(git -C "${WORKTREE_DIR}" branch --show-current 2>/dev/null || echo "unknown")
    if [ "${WT_BRANCH}" != "${BRANCH_NAME}" ]; then
        echo "ERROR: worktree 已存在但分支不匹配 (${WT_BRANCH} != ${BRANCH_NAME})"
        exit 1
    fi
    echo "  复用已有 worktree: ${WORKTREE_DIR} (branch: ${WT_BRANCH})"
    git -C "${WORKTREE_DIR}" pull origin "${BRANCH_NAME}" 2>/dev/null || true
else
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        echo "  分支已存在，创建 worktree"
        git worktree add "${WORKTREE_DIR}" "${BRANCH_NAME}"
        git -C "${WORKTREE_DIR}" pull origin "${BRANCH_NAME}" 2>/dev/null || true
    else
        echo "  创建新分支 + worktree: ${BRANCH_NAME}"
        git worktree add -b "${BRANCH_NAME}" "${WORKTREE_DIR}"
        git -C "${WORKTREE_DIR}" push -u origin "${BRANCH_NAME}"
    fi
fi

# 同步 research_goal.md（主仓库 → worktree，不修改 master 的 git 状态）
cp "${PROJ_DIR}/research_goal.md" "${WORKTREE_DIR}/research_goal.md"

# ── [2/5] 初始化 experiment_state.json（在 worktree 中）──
echo ""
echo "[2/5] 初始化状态文件..."
python3 - <<PYEOF
import json, os
state_file = "${WORKTREE_DIR}/experiment_state.json"
s = {}
if os.path.exists(state_file):
    try:
        with open(state_file) as f:
            s = json.load(f)
    except Exception:
        s = {}

s.update({
    "schema_version": "2.0",
    "loop_status": "idle",
    "loop_count": 0,
    "max_loops": ${MAX_LOOPS},
    "loop_start_time": "${NOW}",
    "current_pbs_job_id": None,
    "current_loop_label": None,
    "current_loop_script": None,
    "research_branch": "${BRANCH_NAME}",
    "worktree_dir": "${WORKTREE_DIR}",
    "last_eval_results": None,
    "stop_requested": False,
    "goal_achieved": False,
    "error_state": None,
    "last_updated": "${NOW}",
    "last_updated_by": "start_research.sh",
})
with open(state_file, "w") as f:
    json.dump(s, f, indent=2)
print("  experiment_state.json 已初始化")
PYEOF

# 清理遗留触发文件
rm -f "${WORKTREE_DIR}/.job_done_trigger"
rm -f "${WORKTREE_DIR}/.stop_watcher"

# ── [3/5] Git commit + push（在 worktree 中）──
echo ""
echo "[3/5] 提交初始状态到 ${BRANCH_NAME}..."
cd "${WORKTREE_DIR}"

git add experiment_state.json research_goal.md
for f in research_log.md status.md autonomous_loop_prompt.md; do
    [ -f "${f}" ] && git add "${f}" || true
done
git diff --cached --quiet || \
    git commit -m "auto: start research session '${GOAL_SLUG}' at ${NOW}"
git push origin "${BRANCH_NAME}"

# ── [4/5] 启动 per-session watcher ──
echo ""
echo "[4/5] 启动 session watcher..."
mkdir -p "${LOGS_DIR}"
WATCHER_LOG="${LOGS_DIR}/watcher_${BRANCH_SLUG}.log"

nohup env PROJ_DIR="${WORKTREE_DIR}" \
    bash "${PROJ_DIR}/cluster_agent/session_watcher.sh" \
    > "${WATCHER_LOG}" 2>&1 &
WATCHER_PID=$!
sleep 1

if kill -0 "${WATCHER_PID}" 2>/dev/null; then
    echo "  watcher 启动成功 (PID=${WATCHER_PID})"
    echo "  日志: tail -f ${WATCHER_LOG}"
else
    echo "ERROR: watcher 启动失败，检查: ${WATCHER_LOG}"
    exit 1
fi

# ── [5/5] 触发首次 Claude 循环（在 worktree 中运行）──
echo ""
echo "[5/5] 触发首次研究循环..."
INIT_LOG="${LOGS_DIR}/claude_init_${BRANCH_SLUG}_$(date +%Y%m%d_%H%M%S).log"
echo "  Claude 日志: ${INIT_LOG}（后台运行）"

nohup bash -c "
    cd '${WORKTREE_DIR}'
    claude --print --dangerously-skip-permissions \
        \"\$(cat '${WORKTREE_DIR}/autonomous_loop_prompt.md')\"
" > "${INIT_LOG}" 2>&1 &

echo ""
echo "========================================================"
echo "  自主研究循环已启动！"
echo ""
echo "  Session   : ${BRANCH_SLUG}"
echo "  分支      : ${BRANCH_NAME}"
echo "  Worktree  : ${WORKTREE_DIR}"
echo ""
echo "  查看进度  : cat ${WORKTREE_DIR}/status.md"
echo "  查看日志  : tail -f ${WATCHER_LOG}"
echo "  查看研究  : cat ${WORKTREE_DIR}/research_log.md"
echo ""
echo "  优雅停止  : touch ${WORKTREE_DIR}/.stop_watcher"
echo "  紧急停止  : bash ${PROJ_DIR}/stop_research.sh ${BRANCH_SLUG}"
echo "========================================================"
