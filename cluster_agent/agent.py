#!/usr/bin/env python3
"""
E320 Research Cluster Agent
持久化 Slack bot，运行在集群分析节点 (wipp-an1/wipp-an2) 上。

功能:
  - !status / !log / !state / !qstat / !start / !stop 等内置命令
  - 自然语言 → claude --print --dangerously-skip-permissions
  - 后台监控 experiment_state.json，变化时主动通知 Slack
  - 重要事件同时发 email 通知

启动:
  bash cluster_agent/start_agent.sh

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
from pathlib import Path

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ── 配置 ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path("/srv01/agrp/yiwen/E320simulator")
load_dotenv(PROJ_DIR / ".env")

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "")
LOGS_DIR = Path("/srv01/agrp/yiwen/logs")

# 用户白名单：只有列出的 Slack User ID 可以使用 agent
# 格式：ALLOWED_USER_IDS=U123456,U789ABC（逗号分隔，留空则不限制）
_raw_ids = os.environ.get("ALLOWED_USER_IDS", "").strip()
ALLOWED_USER_IDS: set[str] = {uid.strip() for uid in _raw_ids.split(",") if uid.strip()}

MAX_CHUNK = 3800       # Slack 单条消息字符上限（留缓冲）
MONITOR_INTERVAL = 60  # 状态轮询间隔（秒）
CLAUDE_TIMEOUT = 600   # claude --print 超时（秒）
MAX_HISTORY_TURNS = 10 # 每个 session 最多保留的对话轮数

app = App(token=SLACK_BOT_TOKEN)

# ── 对话历史 (session_key → [{"role": "user"|"assistant", "text": "..."}])
# session_key：thread_ts（thread 内继续对话）或 channel+ts（新消息开新会话）
_history_lock = threading.Lock()
conversation_history: dict[str, list] = {}


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """将长文本按 MAX_CHUNK 字符分段。"""
    return [text[i:i + MAX_CHUNK] for i in range(0, max(len(text), 1), MAX_CHUNK)]


def say_long(say, text: str):
    """分段发送长消息。"""
    for chunk in chunk_text(text):
        say(chunk)


def post_to_channel(text: str):
    """主动发消息到默认频道（不依赖 say 回调）。"""
    if not SLACK_CHANNEL_ID:
        return
    for chunk in chunk_text(text):
        try:
            app.client.chat_postMessage(channel=SLACK_CHANNEL_ID, text=chunk)
        except Exception as e:
            print(f"[agent] post_to_channel error: {e}", file=sys.stderr)


def run_shell(cmd: str, cwd=None, timeout=30) -> str:
    """执行 shell 命令，返回合并后的输出字符串。"""
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


def _build_prompt(session_key: str, message: str) -> str:
    """把历史对话拼成 Claude prompt。无历史时直接返回原消息。"""
    with _history_lock:
        history = conversation_history.get(session_key, [])
    if not history:
        return message
    lines = ["以下是我们本次对话的历史记录：", ""]
    for turn in history:
        role = "用户" if turn["role"] == "user" else "助手"
        lines.append(f"{role}: {turn['text']}")
    lines += ["", "请基于以上上下文回复用户最新消息：", message]
    return "\n".join(lines)


def _append_history(session_key: str, role: str, text: str):
    """向 session 历史追加一条记录，超出上限时丢弃最早的。"""
    with _history_lock:
        hist = conversation_history.setdefault(session_key, [])
        hist.append({"role": role, "text": text})
        # 按"轮"裁剪（一轮 = user + assistant），保留最近 MAX_HISTORY_TURNS 轮
        if len(hist) > MAX_HISTORY_TURNS * 2:
            conversation_history[session_key] = hist[-(MAX_HISTORY_TURNS * 2):]


def run_claude_async(message: str, say, session_key: str = ""):
    """在后台线程运行 claude --print，完成后将输出发回 Slack，并维护对话历史。"""
    def _worker():
        say("⏳ Claude 处理中，请稍候…")
        prompt = _build_prompt(session_key, message)
        if session_key:
            _append_history(session_key, "user", message)
        try:
            result = subprocess.run(
                ["claude", "--print", "--dangerously-skip-permissions", prompt],
                capture_output=True, text=True,
                cwd=str(PROJ_DIR), timeout=CLAUDE_TIMEOUT,
            )
            response = (result.stdout + result.stderr).strip()
            if not response:
                response = "(Claude 返回空输出)"
        except subprocess.TimeoutExpired:
            response = f"⚠️ Claude 超时（{CLAUDE_TIMEOUT}s）"
        except FileNotFoundError:
            response = "❌ 找不到 `claude` 命令，请确认 Claude CLI 已安装并在 PATH 中"
        except Exception as e:
            response = f"❌ 执行异常: {e}"

        if session_key:
            _append_history(session_key, "assistant", response)
        say_long(say, response)

    threading.Thread(target=_worker, daemon=True).start()


# ── 内置命令处理 ──────────────────────────────────────────────────────────────

HELP_TEXT = """*E320 Research Agent — 命令列表*

*查看状态*
`!status`             查看 status.md（实时研究状态）
`!log [N]`            最近 N 行 research_log.md（默认 80）
`!state`              查看 experiment_state.json
`!qstat`              PBS 作业队列状态
`!jobs`               最近生成的 PBS 脚本

*研究目标*
`!goal`               查看当前 research_goal.md
`!goal "内容"`        覆盖写入 research_goal.md（然后用 !start 启动）

*控制研究循环*
`!start "目标" [N]`   写入目标并启动 autoresearch（最大 N 轮，默认 15）
`!stop`               优雅停止 watcher（当前作业完成后停止）
`!kill`               紧急停止（终止 watcher + 取消 PBS 作业）

*系统管理*
`!shell "cmd"`        直接执行 shell 命令
`!update`             git pull + 自动重启 agent
`!help`               显示此帮助

*Claude CLI*
其他任何文本 → 交给 Claude 处理（可讨论目标、分析结果、修改代码等）"""


def handle_command(text: str, say, session_key: str = ""):
    """根据文本路由到对应处理逻辑。"""
    text = text.strip()

    if not text:
        return

    # ── 内置命令 ──
    if text == "!help":
        say(HELP_TEXT)

    elif text == "!status":
        out = run_shell(f"cat {PROJ_DIR}/status.md", timeout=5)
        say_long(say, f"```\n{out}\n```")

    elif text.startswith("!log"):
        parts = text.split()
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 80
        out = run_shell(f"tail -{n} {PROJ_DIR}/research_log.md", timeout=5)
        say_long(say, f"```\n{out}\n```")

    elif text == "!state":
        out = run_shell(f"cat {PROJ_DIR}/experiment_state.json", timeout=5)
        say(f"```json\n{out}\n```")

    elif text == "!qstat":
        out = run_shell("qstat -u yiwen", cwd="/tmp", timeout=15)
        say(f"```\n{out}\n```")

    elif text == "!jobs":
        out = run_shell("ls -lt ~/subs/auto_loop*.sh 2>/dev/null | head -15", cwd="/tmp", timeout=10)
        say(f"```\n{out}\n```")

    elif text.startswith("!goal"):
        arg_str = text[5:].strip()
        goal_file = PROJ_DIR / "research_goal.md"

        if not arg_str:
            # !goal — 查看当前内容
            branch = run_shell("git branch --show-current", timeout=5)
            out = run_shell(f"cat {goal_file}", timeout=5)
            say_long(say, f"*当前 research_goal.md* (branch: `{branch}`):\n```\n{out}\n```")
        else:
            # !goal "内容" — 切回 master 再写入
            try:
                content = shlex.split(arg_str)[0]
            except ValueError:
                content = arg_str.strip('"\'')
            try:
                checkout_out = run_shell("git checkout master", timeout=10)
                goal_file.write_text(content + "\n", encoding="utf-8")
                say(f"✅ 已切到 master，research_goal.md 已更新。\n```\n{content}\n```\n用 `!start \"{content[:40]}\" [N]` 启动研究。")
            except Exception as e:
                say(f"❌ 写入失败: {e}")

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
                cwd=str(PROJ_DIR), timeout=120,
            )
            say_long(say, f"```\n{out[-MAX_CHUNK:]}\n```")

        threading.Thread(target=_run_start, daemon=True).start()

    elif text == "!stop":
        run_shell(f"touch {PROJ_DIR}/.stop_watcher", timeout=5)
        say("🛑 已发送优雅停止信号。当前 PBS 作业完成后 watcher 将退出。")

    elif text == "!kill":
        say("💥 执行紧急停止…")
        out = run_shell(f"bash {PROJ_DIR}/stop_research.sh", timeout=30)
        say_long(say, f"```\n{out}\n```")

    elif text.startswith("!shell"):
        cmd = text[6:].strip().strip('"\'')
        if not cmd:
            say("用法：`!shell \"命令\"`")
        else:
            say(f"▶ `{cmd}`")
            out = run_shell(cmd, cwd=str(PROJ_DIR), timeout=60)
            say_long(say, f"```\n{out}\n```")

    elif text == "!update":
        say("🔄 正在后台拉取最新代码，约 10 秒后重启…")
        script = (
            f"sleep 3 && "
            f"cd {PROJ_DIR} && git pull && "
            f"pkill -f 'python.*agent.py' || true && sleep 2 && "
            f"bash {PROJ_DIR}/cluster_agent/start_agent.sh"
        )
        subprocess.Popen(
            ["bash", "-c", script],
            start_new_session=True,
            stdout=open(LOGS_DIR / "agent_update.log", "w"),
            stderr=subprocess.STDOUT,
        )

    else:
        # ── 自然语言 → Claude CLI（带对话历史）──
        run_claude_async(text, say, session_key=session_key)


# ── Slack 事件处理 ────────────────────────────────────────────────────────────

def is_allowed(user_id: str) -> bool:
    """检查用户是否在白名单中（白名单为空则允许所有人）。"""
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def _session_key(message: dict) -> str:
    """
    确定 session key：
    - 若消息在某个 thread 里（有 thread_ts），用 thread_ts → 同 thread 共享历史
    - 否则用 channel + ts → 每条顶层消息开一个新会话
    """
    thread_ts = message.get("thread_ts")
    if thread_ts:
        return thread_ts
    return f"{message.get('channel', '')}:{message.get('ts', '')}"


def _threaded_say(say, thread_ts: str):
    """
    返回一个始终在指定 thread 内回复的 say 函数。
    - 对于顶层消息：thread_ts = 消息自身的 ts，回复会创建新 thread
    - 对于 thread 内消息：thread_ts = 所在 thread 的 ts，回复保持在同一 thread
    """
    def _say(text, **kwargs):
        say(text, thread_ts=thread_ts, **kwargs)
    return _say


@app.message(re.compile(r".*"))
def handle_message(message, say):
    """处理所有直接消息（DM 或频道消息）。"""
    if message.get("bot_id"):
        return  # 忽略 bot 自己的消息
    text = message.get("text", "").strip()
    if re.search(r"<@[A-Z0-9]+>", text):
        return  # 有@提及，交给 app_mention handler 处理，避免重复回复
    user_id = message.get("user", "")
    if not is_allowed(user_id):
        say(f"⛔ 用户 `{user_id}` 无权限使用此 agent。请联系管理员将你的 User ID 加入白名单。")
        return
    # thread_ts: 已在 thread 中则用 thread_ts，否则用自身 ts 开新 thread
    thread_ts = message.get("thread_ts") or message.get("ts", "")
    handle_command(text, _threaded_say(say, thread_ts), session_key=_session_key(message))


@app.event("app_mention")
def handle_mention(event, say):
    """处理 @提及。"""
    user_id = event.get("user", "")
    if not is_allowed(user_id):
        say(f"⛔ 用户 `{user_id}` 无权限。")
        return
    text = event.get("text", "")
    text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
    thread_ts = event.get("thread_ts") or event.get("ts", "")
    handle_command(text, _threaded_say(say, thread_ts), session_key=_session_key(event))


# ── 后台状态监控 ──────────────────────────────────────────────────────────────

def monitor_experiment_state():
    """每隔 MONITOR_INTERVAL 秒检查 experiment_state.json，变化时通知 Slack。"""
    state_file = PROJ_DIR / "experiment_state.json"
    last_status = None
    last_loop_count = -1

    while True:
        try:
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)

                status = state.get("loop_status")
                loop_count = state.get("loop_count", 0)
                goal_achieved = state.get("goal_achieved", False)
                error_state = state.get("error_state")

                # 仅在 loop_count 增加或 status 变化时通知（避免刷屏）
                changed = (status != last_status and last_status is not None) or \
                          (loop_count != last_loop_count and last_loop_count >= 0)

                if changed:
                    if goal_achieved:
                        results = state.get("last_eval_results", {})
                        # 提取最新结果
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
                        msg = f"🎉 *研究目标已达成！* Loop {loop_count}\n{summary}"
                        post_to_channel(msg)
                        _send_email_notify(f"研究目标达成！{summary}", msg)

                    elif error_state:
                        msg = f"❌ *错误，需人工干预*\n```{error_state}```"
                        post_to_channel(msg)
                        _send_email_notify(f"E320 Agent 错误: {error_state[:80]}", msg)

                    elif status == "submitted" and loop_count != last_loop_count:
                        label = state.get("current_loop_label", "?")
                        job_id = state.get("current_pbs_job_id", "?")
                        post_to_channel(f"📤 Loop {loop_count} 已提交 PBS 作业 `{job_id}` ({label})")

                    elif status == "completed":
                        post_to_channel(f"✅ Loop {loop_count} PBS 作业完成，等待 Claude 分析…")

                last_status = status
                last_loop_count = loop_count

        except json.JSONDecodeError:
            pass  # 文件写入中，忽略
        except Exception as e:
            print(f"[monitor] 异常: {e}", file=sys.stderr)

        time.sleep(MONITOR_INTERVAL)


def _send_email_notify(subject: str, body: str = ""):
    """发送 email 通知（静默失败）。"""
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
    print(f"[agent] 启动 E320 Research Agent，项目目录: {PROJ_DIR}")

    # 启动后台监控线程
    threading.Thread(target=monitor_experiment_state, daemon=True, name="state-monitor").start()
    print(f"[agent] 状态监控线程已启动（每 {MONITOR_INTERVAL}s 轮询）")

    # 发送上线通知
    if SLACK_CHANNEL_ID:
        try:
            app.client.chat_postMessage(
                channel=SLACK_CHANNEL_ID,
                text="🤖 *E320 Research Agent 已上线*\n发送 `!help` 查看可用命令。",
            )
        except Exception as e:
            print(f"[agent] 上线通知发送失败: {e}", file=sys.stderr)

    # 启动 Socket Mode（阻塞）
    print("[agent] 正在连接 Slack Socket Mode…")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()


if __name__ == "__main__":
    main()
