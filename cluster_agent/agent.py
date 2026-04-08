#!/usr/bin/env python3
"""
E320 Research Cluster Agent
持久化 Slack bot，运行在集群分析节点 (wipp-an1/wipp-an2) 上。

功能:
  - !sessions / !status / !log / !state / !result / !wlog 等带 session 参数的状态命令
  - !start / !stop / !kill 控制研究循环（支持多 session 并发）
  - 自然语言 → claude --print --dangerously-skip-permissions
  - 后台监控所有 session 的 experiment_state.json，变化时主动通知 Slack
  - PBS 作业监控（关联到具体 session）

启动:
  bash cluster_agent/start_agent.sh

架构:
  主仓库 PROJ_DIR 永远在 master
  研究 sessions 在 RESEARCH_DIR/<slug>/ 独立 worktree 中运行

依赖 .env:
  SLACK_BOT_TOKEN=xoxb-...
  SLACK_APP_TOKEN=xapp-...
  SLACK_CHANNEL_ID=C...
  NOTIFY_EMAIL=user@domain.com  (可选)
"""
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ── 配置 ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path("/srv01/agrp/yiwen/E320simulator")       # 主仓库（永远 master）
RESEARCH_DIR = Path("/srv01/agrp/yiwen/research")         # worktree 根目录
load_dotenv(PROJ_DIR / ".env")

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "")
LOGS_DIR = Path("/srv01/agrp/yiwen/logs")

# Claude session 文件目录（由 cwd 决定，cwd=PROJ_DIR）
_proj_hash = str(PROJ_DIR).replace("/", "-")
CLAUDE_SESSIONS_DIR = Path.home() / ".claude" / "projects" / _proj_hash

# 用户白名单（留空则不限制）
_raw_ids = os.environ.get("ALLOWED_USER_IDS", "").strip()
ALLOWED_USER_IDS: set[str] = {uid.strip() for uid in _raw_ids.split(",") if uid.strip()}

MAX_CHUNK = 3800
MONITOR_INTERVAL = 60   # 状态轮询间隔（秒）
AGENT_MEMORY_FILE = PROJ_DIR / "cluster_agent" / "agent_memory.md"
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "3600"))
MAX_HISTORY_TURNS = 10
PBS_POLL_INTERVAL = 60
PBS_USER = "yiwen"

CLAUDE_MODEL: str | None = os.environ.get("CLAUDE_MODEL") or None
CLAUDE_EFFORT: str | None = os.environ.get("CLAUDE_EFFORT") or None

MODEL_ALIASES: dict[str, str] = {
    "opus":    "claude-opus-4-6",
    "sonnet":  "claude-sonnet-4-6",
    "haiku":   "claude-haiku-4-5-20251001",
    "opus4":   "claude-opus-4-6",
    "sonnet4": "claude-sonnet-4-6",
    "haiku4":  "claude-haiku-4-5-20251001",
}
VALID_EFFORTS = {"low", "medium", "high", "max"}

app = App(token=SLACK_BOT_TOKEN)


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    return [text[i:i + MAX_CHUNK] for i in range(0, max(len(text), 1), MAX_CHUNK)]


def say_long(say, text: str):
    for chunk in chunk_text(text):
        say(chunk)


def post_to_channel(text: str):
    if not SLACK_CHANNEL_ID:
        return
    for chunk in chunk_text(text):
        try:
            app.client.chat_postMessage(channel=SLACK_CHANNEL_ID, text=chunk)
        except Exception as e:
            print(f"[agent] post_to_channel error: {e}", file=sys.stderr)


def run_shell(cmd: str, cwd=None, timeout=30) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=str(cwd or PROJ_DIR), timeout=timeout,
        )
        out = (result.stdout + result.stderr).strip()
        return out if out else "(无输出)"
    except subprocess.TimeoutExpired:
        return f"(超时 {timeout}s)"
    except Exception as e:
        return f"(错误: {e})"


# ── Session 管理 ──────────────────────────────────────────────────────────────

def _list_sessions() -> list[Path]:
    """列出所有存在 experiment_state.json 的 worktree（按修改时间倒序）。"""
    if not RESEARCH_DIR.exists():
        return []
    return sorted(
        [d for d in RESEARCH_DIR.iterdir()
         if d.is_dir() and (d / "experiment_state.json").exists()],
        key=lambda d: (d / "experiment_state.json").stat().st_mtime,
        reverse=True,
    )


def _resolve_session(name: str | None) -> tuple[Path | None, str]:
    """
    根据名称（支持前缀匹配）找到 session 目录。
    返回 (Path, error_msg)，找到时 error_msg 为空字符串。
    """
    sessions = _list_sessions()
    if not sessions:
        return None, "没有活跃的研究 session（先用 `!start \"目标\" [N]` 启动）"
    if name is None:
        if len(sessions) == 1:
            return sessions[0], ""
        names = ", ".join(f"`{s.name}`" for s in sessions)
        return None, f"有多个 session，请指定名称（支持前缀）: {names}"
    # 精确匹配或前缀匹配
    matches = [
        s for s in sessions
        if s.name == name or s.name.startswith(name)
    ]
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return None, f"找不到 session `{name}`，用 `!sessions` 查看列表"
    names = ", ".join(f"`{s.name}`" for s in matches)
    return None, f"前缀 `{name}` 匹配多个 session: {names}"


def _session_summary(sess_dir: Path) -> str:
    """返回 session 的单行状态摘要。"""
    try:
        state = json.loads((sess_dir / "experiment_state.json").read_text())
        status = state.get("loop_status", "?")
        loop = state.get("loop_count", 0)
        max_l = state.get("max_loops", "?")
        goal_ok = "🎉" if state.get("goal_achieved") else ""
        err = "❌" if state.get("error_state") else ""
        # watcher 状态
        pid_file = sess_dir / ".watcher.pid"
        watcher = ""
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                import signal
                os.kill(pid, 0)
                watcher = "▶watcher"
            except (ValueError, ProcessLookupError, PermissionError):
                watcher = "⚠watcher-dead"
        return f"`{sess_dir.name}` — {status} Loop{loop}/{max_l} {goal_ok}{err} {watcher}"
    except Exception as e:
        return f"`{sess_dir.name}` — (读取失败: {e})"


def _read_session_state(sess_dir: Path) -> dict:
    try:
        return json.loads((sess_dir / "experiment_state.json").read_text())
    except Exception:
        return {}


def _thread_session_id(thread_ts: str) -> str:
    """从 Slack thread_ts 派生确定性 UUID，用于 claude --session-id / --resume。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slack-thread-{thread_ts}"))


def _session_exists(session_id: str) -> bool:
    """检查对应的 Claude session 文件是否已存在。"""
    return (CLAUDE_SESSIONS_DIR / f"{session_id}.jsonl").exists()


# ── Agent 记忆 ───────────────────────────────────────────────────────────────

def _read_memory() -> str:
    try:
        return AGENT_MEMORY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _append_memory(fact: str):
    """在 agent_memory.md 末尾追加一条有时间戳的记忆。"""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d")
    line = f"- [{ts}] {fact.strip()}"
    try:
        existing = AGENT_MEMORY_FILE.read_text(encoding="utf-8") if AGENT_MEMORY_FILE.exists() else ""
        AGENT_MEMORY_FILE.write_text(existing.rstrip() + f"\n{line}\n", encoding="utf-8")
    except OSError as e:
        print(f"[agent] 写记忆失败: {e}", file=sys.stderr)


def _forget_memory(keyword: str) -> tuple[int, list[str]]:
    """删除 agent_memory.md 中包含 keyword 的行，返回 (删除行数, 删除的行列表)。"""
    if not AGENT_MEMORY_FILE.exists():
        return 0, []
    lines = AGENT_MEMORY_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    removed, kept = [], []
    for line in lines:
        if keyword.lower() in line.lower() and line.strip().startswith("-"):
            removed.append(line.rstrip())
        else:
            kept.append(line)
    if removed:
        AGENT_MEMORY_FILE.write_text("".join(kept), encoding="utf-8")
    return len(removed), removed


def _extract_save_tags(response: str) -> tuple[str, list[str]]:
    """
    从 Claude 输出中提取 [SAVE]: <fact> 行。
    返回 (去掉标签后的输出, 提取到的 facts 列表)。
    """
    facts, clean_lines = [], []
    for line in response.splitlines():
        m = re.match(r"^\[SAVE\]:\s*(.+)$", line.strip())
        if m:
            facts.append(m.group(1).strip())
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines).strip(), facts


# ── Slack 历史 / Claude 调用 ─────────────────────────────────────────────────

def _fetch_channel_history(channel: str) -> list[dict]:
    try:
        result = app.client.conversations_history(
            channel=channel, limit=MAX_HISTORY_TURNS * 2 + 1,
        )
        return list(reversed(result.get("messages", [])))
    except Exception as e:
        print(f"[agent] 拉取历史失败: {e}", file=sys.stderr)
        return []


def _build_prompt_from_slack(channel: str, current_text: str) -> str:
    msgs = _fetch_channel_history(channel)

    history_lines = []
    for msg in msgs:
        text = msg.get("text", "").strip()
        if not text or text.startswith("!"):
            continue
        is_bot = bool(msg.get("bot_id"))
        # 跳过 bot 的占位消息和代码块输出（避免污染 prompt）
        if is_bot and (text.startswith("⏳") or text.startswith("```")):
            continue
        role = "助手" if is_bot else "用户"
        history_lines.append(f"{role}: {text}")

    if history_lines and history_lines[-1] == f"用户: {current_text}":
        history_lines = history_lines[:-1]

    parts: list[str] = []

    # 注入长期记忆
    memory = _read_memory()
    if memory:
        parts += [
            "[长期记忆 — 跨会话持久，请优先参考]",
            memory,
            "",
            "如果本次对话中发现了值得长期记住的事实、用户偏好或重要决定，",
            "请在回复末尾另起一行输出：[SAVE]: <一句话描述>（可多行）",
            "",
        ]

    if history_lines:
        parts += ["以下是我们最近的对话历史：", ""] + history_lines + [""]

    parts += ["请基于以上上下文回复用户最新消息：", current_text]
    return "\n".join(parts)


# 并发保护：同一时刻只允许一个 Claude 调用
_claude_lock = threading.Lock()


def run_claude_async(message: str, say, thread_ts: str = "", channel: str = ""):
    def _worker():
        if not _claude_lock.acquire(blocking=False):
            say("⏳ 已排队，等待当前处理完成…")
            _claude_lock.acquire(blocking=True)
        try:
            labels = []
            if CLAUDE_MODEL:
                labels.append(CLAUDE_MODEL)
            if CLAUDE_EFFORT:
                labels.append(f"effort={CLAUDE_EFFORT}")
            label_str = f" [{', '.join(labels)}]" if labels else ""

            cmd = ["claude", "--print", "--dangerously-skip-permissions"]
            if CLAUDE_MODEL:
                cmd += ["--model", CLAUDE_MODEL]
            if CLAUDE_EFFORT:
                cmd += ["--effort", CLAUDE_EFFORT]

            # Session 管理：同一 Slack thread 复用同一个 Claude session
            if thread_ts:
                session_id = _thread_session_id(thread_ts)
                if _session_exists(session_id):
                    cmd += ["--resume", session_id]
                else:
                    cmd += ["--session-id", session_id]

            # 构建 prompt：注入长期记忆（resume 时 Claude 已有对话上下文，只补充记忆）
            memory = _read_memory()
            if memory:
                prompt = (
                    f"[长期记忆 — 跨会话持久，请优先参考]\n{memory}\n\n"
                    f"如果本次对话发现值得长期记住的事实，请在回复末尾输出：[SAVE]: <一句话描述>\n\n"
                    f"{message}"
                )
            else:
                prompt = message
            cmd.append(prompt)

            # 发初始消息，捕获 ts 以便后续原地更新
            init_result = say(f"⏳ Claude{label_str} 处理中…")
            msg_ts = None
            msg_channel = None
            if isinstance(init_result, dict) and init_result.get("ok"):
                msg_ts = init_result.get("ts")
                msg_channel = init_result.get("channel")

            response = "(Claude 返回空输出)"
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(PROJ_DIR),
                )

                start_time = time.time()
                progress = {"lines": 0, "last": ""}
                output_parts: list[str] = []
                done = threading.Event()

                def _reader():
                    for line in proc.stdout:
                        output_parts.append(line)
                        stripped = line.strip()
                        if stripped:
                            progress["lines"] += 1
                            progress["last"] = stripped
                    done.set()

                def _progress_updater():
                    """每 20 秒原地编辑初始消息，显示进度。"""
                    while not done.wait(timeout=20):
                        elapsed = int(time.time() - start_time)
                        m, s = divmod(elapsed, 60)
                        n = progress["lines"]
                        last = progress["last"]
                        # 截取最后一行，去掉可能的 ANSI 控制符，最多 120 字符
                        last_clean = re.sub(r"\x1b\[[0-9;]*m", "", last)[:120]

                        text = f"⏳ Claude{label_str} 处理中… `{m:02d}:{s:02d}`"
                        if n > 0:
                            text += f"  已输出 {n} 行"
                        if last_clean:
                            text += f"\n> _{last_clean}_"

                        if msg_ts and msg_channel:
                            try:
                                app.client.chat_update(
                                    channel=msg_channel, ts=msg_ts, text=text,
                                )
                            except Exception:
                                pass

                threading.Thread(target=_reader, daemon=True).start()
                threading.Thread(target=_progress_updater, daemon=True).start()

                proc.wait(timeout=CLAUDE_TIMEOUT)
                done.wait(timeout=5)
                response = "".join(output_parts).strip() or "(Claude 返回空输出)"

            except subprocess.TimeoutExpired:
                proc.kill()
                response = f"⚠️ Claude 超时（{CLAUDE_TIMEOUT}s）"
            except FileNotFoundError:
                response = "❌ 找不到 `claude` 命令，请确认 Claude CLI 已安装并在 PATH 中"
            except Exception as e:
                response = f"❌ 执行异常: {e}"

            # 解析 [SAVE]: 标签并写入长期记忆
            response, facts = _extract_save_tags(response)
            for fact in facts:
                _append_memory(fact)
                print(f"[agent] 已保存记忆: {fact}", file=sys.stderr)

            say_long(say, response)
        finally:
            _claude_lock.release()

    threading.Thread(target=_worker, daemon=True).start()


# ── Help Text ─────────────────────────────────────────────────────────────────

HELP_TEXT = """*E320 Research Agent — 命令列表*

*查看 Session 状态*（`[session]` 为 session 名前缀，可省略时自动选唯一 session）
`!sessions`                  列出所有活跃研究 session 及简要状态
`!status [session]`          查看 status.md
`!log [session] [N]`         最近 N 行 research_log.md（默认 80）
`!state [session]`           查看 experiment_state.json
`!result [session]`          最新 eval 指标（eff / fake / rms）
`!wlog [session] [N]`        最近 N 行 watcher 日志（默认 50）
`!qstat`                     PBS 作业队列（所有 session）
`!jobs`                      最近生成的 PBS 脚本

*研究目标*
`!goal`                      查看主仓库 research_goal.md（新 session 模板）
`!goal "内容"`               更新主仓库 research_goal.md 模板
`!goal [session]`            查看指定 session 的 research_goal.md
`!goal [session] "内容"`     更新指定 session 的 research_goal.md

*控制研究循环*
`!start "目标" [N]`          在新 worktree 启动 autoresearch（最大 N 轮，默认 15）
`!stop [session]`            优雅停止（当前作业完成后停止）
`!kill [session]`            紧急停止（立即终止 watcher + 取消 PBS 作业）

*Agent 记忆*（跨重启持久）
`!memories`                  查看所有长期记忆
`!remember "内容"`           手动添加一条记忆
`!forget "关键词"`           删除包含关键词的记忆条目

*系统管理*
`!shell "cmd"`               直接执行 shell 命令（谨慎使用）
`!model [name]`              查看/切换 Claude 模型（opus/sonnet/haiku）
`!effort [level]`            查看/切换 effort（low/medium/high/max）
`!update`                    git pull + 自动重启 agent
`!help`                      显示此帮助

*Claude CLI*
其他任何文本 → 交给 Claude 处理（讨论目标、分析结果、修改代码等）"""


# ── 命令处理 ──────────────────────────────────────────────────────────────────

def _parse_session_and_rest(arg_str: str) -> tuple[str | None, str]:
    """
    从参数字符串中解析可选的 session 名和剩余内容。
    规则：第一个 token 若不以引号开头，则视为 session 名。
    返回 (session_name_or_None, rest_str)
    """
    arg_str = arg_str.strip()
    if not arg_str:
        return None, ""
    # 若以引号开头，整体视为内容（session=None）
    if arg_str[0] in ('"', "'", "\u201c", "\u2018"):
        return None, arg_str
    # 否则第一个 token 是 session
    parts = arg_str.split(None, 1)
    session = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    return session, rest


def _extract_content(s: str) -> str:
    """从带引号的字符串中提取内容。"""
    s = s.strip()
    try:
        parts = shlex.split(s)
        return parts[0] if parts else s
    except ValueError:
        return s.strip('"\'\u201c\u201d\u2018\u2019')


def handle_command(text: str, say, channel: str = "", thread_ts: str = ""):
    text = text.strip()
    if not text:
        return

    # ── !help ──────────────────────────────────────────────────────────────
    if text == "!help":
        say(HELP_TEXT)

    # ── !sessions ──────────────────────────────────────────────────────────
    elif text == "!sessions":
        sessions = _list_sessions()
        if not sessions:
            say("没有活跃的研究 session。用 `!start \"目标\" [N]` 启动新 session。")
            return
        lines = [f"*活跃研究 Sessions* ({len(sessions)} 个)\n"]
        for s in sessions:
            lines.append(_session_summary(s))
        say("\n".join(lines))

    # ── !status [session] ──────────────────────────────────────────────────
    elif text.startswith("!status"):
        arg = text[7:].strip()
        sess_dir, err = _resolve_session(arg if arg else None)
        if err:
            say(f"❌ {err}")
            return
        out = run_shell(f"cat {sess_dir}/status.md", timeout=5)
        say_long(say, f"*{sess_dir.name}* — status.md\n```\n{out}\n```")

    # ── !log [session] [N] ─────────────────────────────────────────────────
    elif text.startswith("!log"):
        parts = text.split()
        # !log [session] [N]
        session_arg = None
        n = 80
        if len(parts) >= 2:
            if parts[1].isdigit():
                n = int(parts[1])
            else:
                session_arg = parts[1]
                if len(parts) >= 3 and parts[2].isdigit():
                    n = int(parts[2])
        sess_dir, err = _resolve_session(session_arg)
        if err:
            say(f"❌ {err}")
            return
        out = run_shell(f"tail -{n} {sess_dir}/research_log.md", timeout=5)
        say_long(say, f"*{sess_dir.name}* — research_log.md (最后 {n} 行)\n```\n{out}\n```")

    # ── !state [session] ───────────────────────────────────────────────────
    elif text.startswith("!state"):
        arg = text[6:].strip()
        sess_dir, err = _resolve_session(arg if arg else None)
        if err:
            say(f"❌ {err}")
            return
        out = run_shell(f"cat {sess_dir}/experiment_state.json", timeout=5)
        say(f"*{sess_dir.name}* — experiment_state.json\n```json\n{out}\n```")

    # ── !result [session] ──────────────────────────────────────────────────
    elif text.startswith("!result"):
        arg = text[7:].strip()
        sess_dir, err = _resolve_session(arg if arg else None)
        if err:
            say(f"❌ {err}")
            return
        state = _read_session_state(sess_dir)
        results = state.get("last_eval_results") or {}
        if not results:
            say(f"*{sess_dir.name}* — 暂无 eval 结果")
            return
        lines = [f"*{sess_dir.name}* — 最新 eval 结果 (Loop {state.get('loop_count', '?')})"]
        for label, r in results.items():
            eff = r.get("track_efficiency", "?")
            fake = r.get("fake_rate", "?")
            rms = r.get("mean_rms", "?")
            if isinstance(eff, float):
                lines.append(f"`{label}`: eff={eff:.1%}  fake={fake:.1%}  rms={rms:.4f}")
            else:
                lines.append(f"`{label}`: {r}")
        say("\n".join(lines))

    # ── !wlog [session] [N] ────────────────────────────────────────────────
    elif text.startswith("!wlog"):
        parts = text.split()
        session_arg = None
        n = 50
        if len(parts) >= 2:
            if parts[1].isdigit():
                n = int(parts[1])
            else:
                session_arg = parts[1]
                if len(parts) >= 3 and parts[2].isdigit():
                    n = int(parts[2])
        sess_dir, err = _resolve_session(session_arg)
        if err:
            say(f"❌ {err}")
            return
        log_path = LOGS_DIR / f"watcher_{sess_dir.name}.log"
        if not log_path.exists():
            say(f"*{sess_dir.name}* — watcher 日志不存在: {log_path}")
            return
        out = run_shell(f"tail -{n} {log_path}", timeout=5)
        say_long(say, f"*{sess_dir.name}* — watcher log (最后 {n} 行)\n```\n{out}\n```")

    # ── !qstat ─────────────────────────────────────────────────────────────
    elif text == "!qstat":
        out = run_shell("qstat -u yiwen", cwd="/tmp", timeout=15)
        say(f"```\n{out}\n```")

    # ── !jobs ──────────────────────────────────────────────────────────────
    elif text == "!jobs":
        out = run_shell("ls -lt ~/subs/auto_loop*.sh 2>/dev/null | head -15", cwd="/tmp", timeout=10)
        say(f"```\n{out}\n```")

    # ── !goal [[session]] ["内容"] ─────────────────────────────────────────
    elif text.startswith("!goal"):
        arg_str = text[5:].strip()

        if not arg_str:
            # !goal — 显示主仓库模板
            out = run_shell(f"cat {PROJ_DIR}/research_goal.md", timeout=5)
            sessions = _list_sessions()
            header = f"*主仓库 research_goal.md* (新 session 模板):\n```\n{out}\n```"
            if sessions:
                header += f"\n\n*活跃 sessions*: {', '.join(f'`{s.name}`' for s in sessions)}"
                header += "\n用 `!goal <session>` 查看指定 session 的目标"
            say_long(say, header)
            return

        session_arg, rest = _parse_session_and_rest(arg_str)

        if session_arg is None:
            # !goal "内容" — 更新主仓库模板
            content = _extract_content(rest)
            try:
                (PROJ_DIR / "research_goal.md").write_text(content + "\n", encoding="utf-8")
                say(f"✅ 主仓库 research_goal.md 已更新。\n```\n{content[:300]}\n```\n用 `!start \"{content[:40]}\" [N]` 启动新研究。")
            except Exception as e:
                say(f"❌ 写入失败: {e}")
            return

        # 有 session_arg — 先尝试解析为 session
        sess_dir, sess_err = _resolve_session(session_arg)

        if sess_err and not rest:
            # session 解析失败且没有内容 → 可能是 !goal "单词内容" 没加引号
            say(f"❌ {sess_err}")
            return

        if sess_dir is None and rest:
            # session 解析失败但有 rest → 整体视为内容（session=None）
            content = _extract_content(arg_str)
            try:
                (PROJ_DIR / "research_goal.md").write_text(content + "\n", encoding="utf-8")
                say(f"✅ 主仓库 research_goal.md 已更新。\n```\n{content[:300]}\n```")
            except Exception as e:
                say(f"❌ 写入失败: {e}")
            return

        if not rest:
            # !goal <session> — 查看该 session 的目标
            goal_file = sess_dir / "research_goal.md"
            if goal_file.exists():
                out = goal_file.read_text(encoding="utf-8")
                say_long(say, f"*{sess_dir.name}* — research_goal.md:\n```\n{out}\n```")
            else:
                say(f"*{sess_dir.name}* — research_goal.md 不存在")
            return

        # !goal <session> "内容" — 更新该 session 的目标
        content = _extract_content(rest)
        try:
            (sess_dir / "research_goal.md").write_text(content + "\n", encoding="utf-8")
            say(f"✅ *{sess_dir.name}* research_goal.md 已更新。\n```\n{content[:300]}\n```")
        except Exception as e:
            say(f"❌ 写入失败: {e}")

    # ── !start "目标" [N] ──────────────────────────────────────────────────
    elif text.startswith("!start"):
        arg_str = text[6:].strip()
        try:
            parts = shlex.split(arg_str)
            goal = parts[0] if parts else "research"
            max_loops = parts[1] if len(parts) > 1 else "15"
        except ValueError:
            say("❌ 参数解析失败。用法：`!start \"目标描述\" [最大轮数]`")
            return

        say(f"🚀 启动 autoresearch: *{goal}* (最大 {max_loops} 轮)\n（后台执行，完成后通知）")

        def _run_start():
            out = run_shell(
                f"bash {PROJ_DIR}/start_research.sh {shlex.quote(goal)} {max_loops}",
                cwd=str(PROJ_DIR), timeout=180,
            )
            say_long(say, f"```\n{out[-MAX_CHUNK:]}\n```")

        threading.Thread(target=_run_start, daemon=True).start()

    # ── !stop [session] ────────────────────────────────────────────────────
    elif text.startswith("!stop"):
        arg = text[5:].strip()
        sess_dir, err = _resolve_session(arg if arg else None)
        if err:
            say(f"❌ {err}")
            return
        run_shell(f"touch {sess_dir}/.stop_watcher", timeout=5)
        state = _read_session_state(sess_dir)
        job_id = state.get("current_pbs_job_id") or "无"
        say(f"🛑 *{sess_dir.name}*: 已发送优雅停止信号。\n当前 PBS 作业: `{job_id}`\n作业完成后 watcher 将退出。")

    # ── !kill [session] ────────────────────────────────────────────────────
    elif text.startswith("!kill"):
        arg = text[5:].strip()
        sess_dir, err = _resolve_session(arg if arg else None)
        if err:
            say(f"❌ {err}")
            return
        say(f"💥 *{sess_dir.name}*: 执行紧急停止…")
        out = run_shell(
            f"bash {PROJ_DIR}/stop_research.sh {shlex.quote(sess_dir.name)} emergency",
            timeout=30,
        )
        say_long(say, f"```\n{out}\n```")

    # ── !shell ─────────────────────────────────────────────────────────────
    elif text.startswith("!shell"):
        cmd = text[6:].strip().strip('"\'\u201c\u201d\u2018\u2019')
        if not cmd:
            say("用法：`!shell 命令`（不需要加引号）")
        else:
            say(f"▶ `{cmd}`")
            out = run_shell(cmd, cwd=str(PROJ_DIR), timeout=60)
            say_long(say, f"```\n{out}\n```")

    # ── !effort ────────────────────────────────────────────────────────────
    elif text.startswith("!effort"):
        global CLAUDE_EFFORT
        arg = text[7:].strip().strip('"\'\u201c\u201d\u2018\u2019').lower()
        if not arg:
            current = CLAUDE_EFFORT or "(默认，由模型决定)"
            say(f"*当前 effort*: `{current}`\n可选: `low` | `medium` | `high` | `max`\n`!effort default` 恢复默认")
        elif arg == "default":
            CLAUDE_EFFORT = None
            say("✅ 已恢复默认 effort")
        elif arg in VALID_EFFORTS:
            CLAUDE_EFFORT = arg
            say(f"✅ effort 已切换为 `{arg}`")
        else:
            say(f"❌ 无效值 `{arg}`，可选: low / medium / high / max")

    # ── !model ─────────────────────────────────────────────────────────────
    elif text.startswith("!model"):
        global CLAUDE_MODEL
        arg = text[6:].strip().strip('"\'\u201c\u201d\u2018\u2019')
        if not arg:
            current = CLAUDE_MODEL or "(默认)"
            aliases_str = "\n".join(f"  `{k}` → `{v}`" for k, v in MODEL_ALIASES.items())
            say(f"*当前模型*: `{current}`\n\n*可用简称*:\n{aliases_str}\n\n用法: `!model sonnet`\n`!model default` 恢复默认")
        elif arg.lower() == "default":
            CLAUDE_MODEL = None
            say("✅ 已恢复默认模型")
        else:
            resolved = MODEL_ALIASES.get(arg.lower(), arg)
            CLAUDE_MODEL = resolved
            say(f"✅ 模型已切换为 `{resolved}`")

    # ── !memories ──────────────────────────────────────────────────────────────
    elif text == "!memories":
        mem = _read_memory()
        if mem:
            say_long(say, f"*Agent 长期记忆*\n```\n{mem}\n```")
        else:
            say("记忆为空。用 `!remember \"内容\"` 手动添加，或直接和 Claude 对话让它自动记录。")

    # ── !remember ──────────────────────────────────────────────────────────────
    elif text.startswith("!remember"):
        content = text[9:].strip().strip('"\'\u201c\u201d\u2018\u2019')
        if not content:
            say("用法：`!remember \"要记住的内容\"`")
        else:
            _append_memory(content)
            say(f"✅ 已记住：_{content}_")

    # ── !forget ────────────────────────────────────────────────────────────────
    elif text.startswith("!forget"):
        keyword = text[7:].strip().strip('"\'\u201c\u201d\u2018\u2019')
        if not keyword:
            say("用法：`!forget \"关键词\"` — 删除记忆中包含该关键词的条目")
        else:
            n, removed = _forget_memory(keyword)
            if n == 0:
                say(f"没有找到包含 `{keyword}` 的记忆条目。")
            else:
                removed_str = "\n".join(f"  {r}" for r in removed)
                say(f"🗑 已删除 {n} 条记忆：\n```\n{removed_str}\n```")

    # ── !update ────────────────────────────────────────────────────────────
    elif text == "!update":
        say("🔄 正在后台拉取最新代码，约 10 秒后重启…")
        log_file = str(LOGS_DIR / "agent_update.log")
        script = (
            f"exec >{log_file} 2>&1; set -x; "   # 脚本内重定向，不依赖继承的 fd
            f"sleep 3 && "
            f"cd {PROJ_DIR} && git pull && "
            f"bash {PROJ_DIR}/cluster_agent/stop_agent.sh ; "  # ; 而非 &&，确保即使 stop 失败也继续
            f"sleep 1 && "
            f"bash {PROJ_DIR}/cluster_agent/start_agent.sh"
        )
        subprocess.Popen(
            ["bash", "-c", script],
            start_new_session=True,
        )

    # ── 自然语言 → Claude ──────────────────────────────────────────────────
    else:
        run_claude_async(text, say, thread_ts=thread_ts)


# ── Slack 事件处理 ────────────────────────────────────────────────────────────

def is_allowed(user_id: str) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def _threaded_say(say, thread_ts: str):
    def _say(text, **kwargs):
        return say(text, thread_ts=thread_ts, **kwargs)
    return _say


@app.message(re.compile(r".*"))
def handle_message(message, say):
    if message.get("bot_id"):
        return
    text = message.get("text", "").strip()
    if re.search(r"<@[A-Z0-9]+>", text):
        return
    user_id = message.get("user", "")
    if not is_allowed(user_id):
        say(f"⛔ 用户 `{user_id}` 无权限。请联系管理员加入白名单。")
        return
    channel = message.get("channel", "")
    thread_ts = message.get("thread_ts") or message.get("ts", "")
    handle_command(text, _threaded_say(say, thread_ts), channel=channel, thread_ts=thread_ts)


@app.event("app_mention")
def handle_mention(event, say):
    user_id = event.get("user", "")
    if not is_allowed(user_id):
        say(f"⛔ 用户 `{user_id}` 无权限。")
        return
    text = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
    channel = event.get("channel", "")
    thread_ts = event.get("thread_ts") or event.get("ts", "")
    handle_command(text, _threaded_say(say, thread_ts), channel=channel, thread_ts=thread_ts)


# ── PBS 作业监控 ──────────────────────────────────────────────────────────────

def _parse_qstat(output: str) -> dict[str, dict]:
    jobs = {}
    lines = output.strip().splitlines()
    for line in lines[5:]:
        parts = line.split()
        if len(parts) < 11:
            continue
        job_id = parts[0]
        queue  = parts[2]
        name   = parts[3]
        status = parts[-2]
        jobs[job_id] = {"name": name, "status": status, "queue": queue}
    return jobs


def _job_to_session(job_id: str) -> str:
    """从所有 session 的 experiment_state.json 中找到对应的 session 名。"""
    for sess_dir in _list_sessions():
        state = _read_session_state(sess_dir)
        if state.get("current_pbs_job_id") == job_id:
            return sess_dir.name
    return ""


def _job_log_tail(job_name: str, lines: int = 5) -> str:
    for candidate in [
        LOGS_DIR / f"auto_{job_name}.out",
        LOGS_DIR / f"{job_name}.out",
    ]:
        if candidate.exists():
            try:
                result = subprocess.run(
                    ["tail", f"-{lines}", str(candidate)],
                    capture_output=True, text=True, timeout=5,
                )
                return result.stdout.strip()
            except Exception:
                pass
    return ""


def monitor_pbs_jobs():
    known: dict[str, dict] = {}
    first_run = True

    while True:
        try:
            result = subprocess.run(
                ["qstat", "-u", PBS_USER],
                capture_output=True, text=True, timeout=15,
            )
            current = _parse_qstat(result.stdout)

            if first_run:
                known = current
                first_run = False
            else:
                for jid, info in current.items():
                    prev = known.get(jid)
                    sess = _job_to_session(jid)
                    sess_tag = f" [{sess}]" if sess else ""
                    if prev is None:
                        post_to_channel(
                            f"📋 PBS 作业 `{jid}` *{info['name']}*{sess_tag} 已入队 (queue: {info['queue']})"
                        )
                    elif prev["status"] != info["status"] and info["status"] == "R":
                        post_to_channel(
                            f"▶️ PBS 作业 `{jid}` *{info['name']}*{sess_tag} 开始运行"
                        )

                for jid, info in known.items():
                    if jid not in current:
                        sess = _job_to_session(jid)
                        sess_tag = f" [{sess}]" if sess else ""
                        tail = _job_log_tail(info["name"])
                        # 尝试判断成功/失败（通过日志末尾关键字）
                        status_icon = "✅"
                        if tail and any(kw in tail.lower() for kw in ["error", "failed", "traceback", "killed"]):
                            status_icon = "⚠️"
                        msg = f"{status_icon} PBS 作业 `{jid}` *{info['name']}*{sess_tag} 已结束"
                        if tail:
                            msg += f"\n```\n{tail[-600:]}\n```"
                        post_to_channel(msg)

                known = current

        except Exception as e:
            print(f"[pbs-monitor] 异常: {e}", file=sys.stderr)

        time.sleep(PBS_POLL_INTERVAL)


# ── 实验状态监控（多 session）────────────────────────────────────────────────

def monitor_experiment_state():
    """每隔 MONITOR_INTERVAL 秒检查所有 session 的状态，变化时通知 Slack。"""
    # per-session 的上次状态 {session_name: {"status": ..., "loop_count": ...}}
    last_states: dict[str, dict] = {}

    while True:
        try:
            for sess_dir in _list_sessions():
                name = sess_dir.name
                state_file = sess_dir / "experiment_state.json"
                try:
                    state = json.loads(state_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

                status = state.get("loop_status")
                loop_count = state.get("loop_count", 0)
                goal_achieved = state.get("goal_achieved", False)
                error_state = state.get("error_state")

                prev = last_states.get(name)

                if prev is None:
                    # 首次见到此 session，静默记录
                    last_states[name] = {"status": status, "loop_count": loop_count}
                    continue

                status_changed = status != prev["status"]
                loop_changed = loop_count != prev["loop_count"]

                if not status_changed and not loop_changed:
                    continue

                # 有变化 — 更新记录
                last_states[name] = {"status": status, "loop_count": loop_count}

                tag = f"[{name}]"

                if goal_achieved and loop_changed:
                    results = state.get("last_eval_results") or {}
                    if results:
                        last_key = list(results.keys())[-1]
                        r = results[last_key]
                        summary = (
                            f"eff={r.get('track_efficiency', '?'):.1%}  "
                            f"fake={r.get('fake_rate', '?'):.1%}  "
                            f"rms={r.get('mean_rms', '?'):.4f}"
                        )
                    else:
                        summary = "(无结果数据)"
                    msg = f"🎉 *{tag} 研究目标达成！* Loop {loop_count}\n{summary}"
                    post_to_channel(msg)
                    _send_email_notify(f"{tag} 研究目标达成！{summary}", msg)

                elif error_state and status_changed:
                    msg = f"❌ *{tag} 错误，需人工干预*\n```{error_state}```"
                    post_to_channel(msg)
                    _send_email_notify(f"{tag} Agent 错误: {error_state[:80]}", msg)

                elif status == "submitted" and loop_changed:
                    label = state.get("current_loop_label", "?")
                    job_id = state.get("current_pbs_job_id", "?")
                    post_to_channel(f"📤 *{tag}* Loop {loop_count} 已提交 PBS `{job_id}` ({label})")

                elif status == "completed" and loop_changed:
                    post_to_channel(f"✅ *{tag}* Loop {loop_count} PBS 作业完成，等待 Claude 分析…")

                elif status == "stopped" and status_changed:
                    post_to_channel(f"🛑 *{tag}* 研究循环已停止")

        except Exception as e:
            print(f"[monitor] 异常: {e}", file=sys.stderr)

        time.sleep(MONITOR_INTERVAL)


def _send_email_notify(subject: str, body: str = ""):
    if not NOTIFY_EMAIL:
        return
    try:
        import smtplib
        import socket
        from email.mime.text import MIMEText
        msg = MIMEText(body or subject, "plain", "utf-8")
        msg["Subject"] = f"[E320 Agent] {subject}"
        msg["From"] = f"e320agent@{socket.gethostname()}"
        msg["To"] = NOTIFY_EMAIL
        with smtplib.SMTP("localhost", timeout=10) as s:
            s.send_message(msg)
    except Exception as e:
        print(f"[agent] email 发送失败: {e}", file=sys.stderr)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    print(f"[agent] 启动 E320 Research Agent")
    print(f"[agent] 主仓库: {PROJ_DIR}")
    print(f"[agent] Worktree 根目录: {RESEARCH_DIR}")

    threading.Thread(target=monitor_experiment_state, daemon=True, name="state-monitor").start()
    threading.Thread(target=monitor_pbs_jobs, daemon=True, name="pbs-monitor").start()
    print(f"[agent] 监控线程已启动（state: {MONITOR_INTERVAL}s, PBS: {PBS_POLL_INTERVAL}s）")

    if SLACK_CHANNEL_ID:
        try:
            sessions = _list_sessions()
            sess_info = f"，{len(sessions)} 个活跃 session" if sessions else ""
            app.client.chat_postMessage(
                channel=SLACK_CHANNEL_ID,
                text=f"🤖 *E320 Research Agent 已上线*{sess_info}\n发送 `!help` 查看可用命令，`!sessions` 查看研究状态。",
            )
        except Exception as e:
            print(f"[agent] 上线通知发送失败: {e}", file=sys.stderr)

    print("[agent] 正在连接 Slack Socket Mode…")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()


if __name__ == "__main__":
    main()
