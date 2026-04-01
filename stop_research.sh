#!/bin/bash
# ============================================================
# stop_research.sh — 停止指定研究 session
#
# 用法:
#   bash stop_research.sh <session_slug>           # 紧急停止（立即）
#   bash stop_research.sh <session_slug> grace     # 优雅停止（等当前作业完成）
#   bash stop_research.sh                          # 列出所有活跃 session
#
# session_slug: RESEARCH_DIR 下的子目录名，例如 "gnn-v2"
#   对应 worktree: /srv01/agrp/yiwen/research/gnn-v2
#   对应分支    : auto-research-gnn-v2
# ============================================================

PROJ_DIR="/srv01/agrp/yiwen/E320simulator"
RESEARCH_DIR="/srv01/agrp/yiwen/research"

SESSION_SLUG="${1:-}"
MODE="${2:-emergency}"

# ── 不传参数：列出活跃 session ──
if [ -z "${SESSION_SLUG}" ]; then
    echo "用法: bash stop_research.sh <session_slug> [grace|emergency]"
    echo ""
    echo "活跃 sessions (存在 experiment_state.json 的 worktree):"
    if [ -d "${RESEARCH_DIR}" ]; then
        FOUND=false
        for d in "${RESEARCH_DIR}"/*/; do
            if [ -f "${d}experiment_state.json" ]; then
                NAME=$(basename "${d}")
                STATUS=$(python3 -c "
import json
try:
    s = json.load(open('${d}experiment_state.json'))
    print(s.get('loop_status','?'), 'loop', s.get('loop_count',0), '/', s.get('max_loops','?'))
except: print('?')
" 2>/dev/null)
                WATCHER=""
                if [ -f "${d}.watcher.pid" ]; then
                    WP=$(cat "${d}.watcher.pid")
                    kill -0 "${WP}" 2>/dev/null && WATCHER=" [watcher running]" || WATCHER=" [watcher dead]"
                fi
                echo "  ${NAME}: ${STATUS}${WATCHER}"
                FOUND=true
            fi
        done
        ${FOUND} || echo "  (无)"
    else
        echo "  (RESEARCH_DIR 不存在: ${RESEARCH_DIR})"
    fi
    exit 0
fi

# ── 前缀匹配 session ──
WORKTREE_DIR=""
if [ -d "${RESEARCH_DIR}/${SESSION_SLUG}" ]; then
    WORKTREE_DIR="${RESEARCH_DIR}/${SESSION_SLUG}"
else
    # 前缀匹配
    MATCHES=()
    if [ -d "${RESEARCH_DIR}" ]; then
        for d in "${RESEARCH_DIR}"/*/; do
            NAME=$(basename "${d}")
            if [[ "${NAME}" == "${SESSION_SLUG}"* ]]; then
                MATCHES+=("${d%/}")
            fi
        done
    fi
    if [ "${#MATCHES[@]}" -eq 1 ]; then
        WORKTREE_DIR="${MATCHES[0]}"
    elif [ "${#MATCHES[@]}" -gt 1 ]; then
        echo "ERROR: 前缀 '${SESSION_SLUG}' 匹配多个 session:"
        for m in "${MATCHES[@]}"; do echo "  $(basename ${m})"; done
        exit 1
    else
        echo "ERROR: 找不到 session '${SESSION_SLUG}'（在 ${RESEARCH_DIR} 中）"
        bash "${0}"  # 显示列表
        exit 1
    fi
fi

SESSION_NAME=$(basename "${WORKTREE_DIR}")
BRANCH_NAME=$(git -C "${WORKTREE_DIR}" branch --show-current 2>/dev/null || echo "unknown")

echo "========================================================"
echo "  停止研究 session: ${SESSION_NAME}  (模式: ${MODE})"
echo "  Worktree: ${WORKTREE_DIR}"
echo "  分支    : ${BRANCH_NAME}"
echo "========================================================"

cd "${WORKTREE_DIR}"

if [ "${MODE}" = "grace" ]; then
    # ── 优雅停止 ──
    echo "[1] 设置 stop_requested 标志..."
    python3 -c "
import json
with open('experiment_state.json') as f: s = json.load(f)
s['stop_requested'] = True
with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
print('  stop_requested = True')
"
    touch "${WORKTREE_DIR}/.stop_watcher"

    git add experiment_state.json
    git commit -m "manual: graceful stop requested" 2>/dev/null || true
    git push origin "${BRANCH_NAME}" 2>/dev/null || true

    echo "  watcher 将在当前 PBS 作业完成后退出"

else
    # ── 紧急停止 ──
    echo "[1] 停止 watcher..."
    touch "${WORKTREE_DIR}/.stop_watcher"
    if [ -f "${WORKTREE_DIR}/.watcher.pid" ]; then
        WP=$(cat "${WORKTREE_DIR}/.watcher.pid")
        kill "${WP}" 2>/dev/null && echo "  watcher PID=${WP} 已终止" || echo "  watcher 已不在运行"
        rm -f "${WORKTREE_DIR}/.watcher.pid"
    else
        echo "  未找到 watcher PID 文件"
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
try:
    with open('experiment_state.json') as f: s = json.load(f)
    s['loop_status'] = 'stopped'
    s['stop_requested'] = True
    s['error_state'] = 'Emergency stop by human at $(date -u +%Y-%m-%dT%H:%M:%SZ)'
    s['last_updated'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    with open('experiment_state.json', 'w') as f: json.dump(s, f, indent=2)
    print('  状态已更新')
except Exception as e:
    print(f'  WARNING: {e}')
"
    git add experiment_state.json
    git commit -m "manual: emergency stop at $(date -u +%Y-%m-%dT%H:%M:%SZ)" 2>/dev/null || true
    git push origin "${BRANCH_NAME}" 2>/dev/null || true
fi

echo ""
echo "恢复方法:"
echo "  bash ${PROJ_DIR}/start_research.sh '${SESSION_NAME}' [N]"
echo ""
echo "清理 worktree（删除本地分支副本）:"
echo "  cd ${PROJ_DIR} && git worktree remove ${WORKTREE_DIR} --force"
echo "========================================================"
