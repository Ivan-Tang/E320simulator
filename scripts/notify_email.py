"""
Email notification utility for E320 batch jobs.

Usage (standalone):
    python scripts/notify_email.py \
        --subject "Benchmark done" \
        --log ~/logs/benchmark_output.log \
        --to yiwen.tang@smail.nju.edu.cn

Usage (from Python):
    from scripts.notify_email import send_notification
    send_notification(
        subject="Benchmark complete",
        summary="InteractionNet best: eff=74%, fake=14%",
        log_path="~/logs/benchmark_output.log",
        tail_lines=80,
    )
"""
from __future__ import annotations

import argparse
import smtplib
import socket
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Credentials ───────────────────────────────────────────────────────────────
SMTP_HOST = "smtp.exmail.qq.com"
SMTP_PORT = 465
SMTP_USER = "yiwen.tang@smail.nju.edu.cn"
SMTP_PASS = "ohn2EHrk7joGqHDV"
SMTP_FROM = "yiwen.tang@smail.nju.edu.cn"


# ── Core send function ────────────────────────────────────────────────────────

def send_notification(
    subject: str,
    summary: str = "",
    log_path: str | Path | None = None,
    tail_lines: int = 100,
    to: str | list[str] = SMTP_USER,
) -> None:
    """Send a job-completion email.

    Parameters
    ----------
    subject:    Email subject line.
    summary:    Short plain-text summary shown at the top of the email body.
    log_path:   Path to a log file; last `tail_lines` lines are appended.
    tail_lines: How many lines from the end of the log to include.
    to:         Recipient address(es).
    """
    if isinstance(to, str):
        to = [to]

    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Build plain-text body ─────────────────────────────────────────────────
    body_parts = [
        f"Time    : {timestamp}",
        f"Host    : {hostname}",
    ]

    if summary:
        body_parts += ["", "── Summary ──────────────────────────────────────────────", summary]

    if log_path is not None:
        p = Path(log_path).expanduser()
        if p.exists():
            lines = p.read_text(errors="replace").splitlines()
            excerpt = "\n".join(lines[-tail_lines:])
            body_parts += [
                "",
                f"── Log tail ({p.name}, last {tail_lines} lines) ──────────────────",
                excerpt,
            ]
        else:
            body_parts += ["", f"[log not found: {p}]"]

    body = "\n".join(body_parts)

    # ── Build HTML body (monospace pre block for the log) ────────────────────
    import html as html_lib

    def section(title: str, content: str, mono: bool = False) -> str:
        style = "font-family:monospace;white-space:pre;font-size:12px;" if mono else ""
        return (
            f"<h3 style='margin-bottom:4px'>{title}</h3>"
            f"<div style='{style}'>{html_lib.escape(content)}</div>"
        )

    html_parts = [
        "<html><body>",
        f"<p><b>Time:</b> {timestamp} &nbsp; <b>Host:</b> {hostname}</p>",
    ]
    if summary:
        html_parts.append(section("Summary", summary, mono=True))
    if log_path is not None:
        p = Path(log_path).expanduser()
        if p.exists():
            lines = p.read_text(errors="replace").splitlines()
            excerpt = "\n".join(lines[-tail_lines:])
            html_parts.append(section(f"Log tail — {p.name} (last {tail_lines} lines)", excerpt, mono=True))
    html_parts.append("</body></html>")
    html_body = "\n".join(html_parts)

    # ── Assemble MIME message ─────────────────────────────────────────────────
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(to)
    msg.attach(MIMEText(body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # ── Send via SSL ──────────────────────────────────────────────────────────
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_FROM, to, msg.as_bytes())

    print(f"[notify] email sent → {', '.join(to)}  subject: {subject!r}")


# ── Helpers for benchmark results ─────────────────────────────────────────────

def extract_benchmark_table(log_path: str | Path) -> str:
    """Extract the comparison table block from a benchmark output log."""
    p = Path(log_path).expanduser()
    if not p.exists():
        return "(log not found)"
    lines = p.read_text(errors="replace").splitlines()
    # Find the COMPARISON TABLE section
    start = None
    for i, line in enumerate(lines):
        if "COMPARISON TABLE" in line or "computing metrics" in line.lower():
            start = i
    if start is None:
        return "(comparison table not found in log)"
    return "\n".join(lines[start:])


def notify_benchmark_done(
    log_path: str | Path = "~/logs/benchmark_output.log",
    job_id: str = "",
    to: str | list[str] = SMTP_USER,
) -> None:
    """Convenience wrapper: send benchmark-complete notification with table."""
    p = Path(log_path).expanduser()
    table = extract_benchmark_table(p)
    subject = f"[E320] Benchmark complete" + (f" — job {job_id}" if job_id else "")
    send_notification(
        subject=subject,
        summary=table,
        log_path=p,
        tail_lines=60,
        to=to,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Send job-completion email notification.")
    parser.add_argument("--subject", default="[E320] Job complete")
    parser.add_argument("--summary", default="")
    parser.add_argument("--log", default=None, help="Path to log file")
    parser.add_argument("--tail", type=int, default=100, help="Lines of log to include")
    parser.add_argument("--to", default=SMTP_USER, help="Recipient email")
    parser.add_argument("--benchmark", action="store_true",
                        help="Auto-extract benchmark table from --log")
    parser.add_argument("--job-id", default="")
    args = parser.parse_args()

    if args.benchmark and args.log:
        notify_benchmark_done(log_path=args.log, job_id=args.job_id, to=args.to)
    else:
        send_notification(
            subject=args.subject,
            summary=args.summary,
            log_path=args.log,
            tail_lines=args.tail,
            to=args.to,
        )


if __name__ == "__main__":
    _cli()
