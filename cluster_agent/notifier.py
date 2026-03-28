#!/usr/bin/env python3
"""
轻量级通知脚本 — 被 autonomous_watcher.sh 在作业完成/失败时调用。

用法:
  python cluster_agent/notifier.py "消息内容"              # 同时发 Slack + email
  python cluster_agent/notifier.py --slack "Slack 消息"
  python cluster_agent/notifier.py --email "邮件主题" --body "邮件正文"
  python cluster_agent/notifier.py --slack "消息" --email "邮件主题"

依赖 .env 中的:
  SLACK_BOT_TOKEN=xoxb-...
  SLACK_CHANNEL_ID=C...
  NOTIFY_EMAIL=user@domain.com  (可选)
"""
import os
import sys
import argparse
from pathlib import Path


def load_env():
    """从项目根目录的 .env 加载环境变量（不覆盖已有变量）。"""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def send_slack(message: str) -> bool:
    """通过 Slack API 发送消息，返回是否成功。"""
    import requests

    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        print("[notifier] SLACK_BOT_TOKEN 或 SLACK_CHANNEL_ID 未设置，跳过 Slack 通知", file=sys.stderr)
        return False

    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            json={"channel": channel, "text": message},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        data = resp.json()
        if not data.get("ok"):
            print(f"[notifier] Slack 发送失败: {data.get('error', data)}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[notifier] Slack 请求异常: {e}", file=sys.stderr)
        return False


def send_email(subject: str, body: str = "") -> bool:
    """通过本地 sendmail 发送邮件，返回是否成功。"""
    import smtplib
    from email.mime.text import MIMEText
    import socket

    to_addr = os.environ.get("NOTIFY_EMAIL")
    if not to_addr:
        return False  # 未配置则静默跳过

    try:
        msg = MIMEText(body or subject, "plain", "utf-8")
        msg["Subject"] = f"[E320 Agent] {subject}"
        msg["From"] = f"e320agent@{socket.gethostname()}"
        msg["To"] = to_addr
        with smtplib.SMTP("localhost", timeout=10) as s:
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"[notifier] 邮件发送失败: {e}", file=sys.stderr)
        return False


def main():
    load_env()

    parser = argparse.ArgumentParser(description="E320 集群通知脚本")
    parser.add_argument("message", nargs="?", default=None, help="同时发送到 Slack 和 email（简单用法）")
    parser.add_argument("--slack", help="Slack 消息内容")
    parser.add_argument("--email", help="邮件主题")
    parser.add_argument("--body", default="", help="邮件正文（默认与主题相同）")
    args = parser.parse_args()

    if args.message:
        # 简单用法：positional arg 同时发两个
        send_slack(args.message)
        send_email(args.message, args.body or args.message)
    else:
        if args.slack:
            send_slack(args.slack)
        if args.email:
            send_email(args.email, args.body or args.email)
        if not args.slack and not args.email:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
