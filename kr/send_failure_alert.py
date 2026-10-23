"""
Send a failure-alert email for the KR scheduled job.

Used by the Task Scheduler wrapper (kr/run_scheduled_kr.ps1) when the
daily KR dry-run/live script exits non-zero - a separate, outer path from
the script's own success-path email, since a crash before that point
would never reach it.

Usage:
    .venv/Scripts/python.exe kr/send_failure_alert.py "<subject>" "<message>"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from notifications import create_email_notifier

DEFAULT_EMAIL = "joshuaJang89@gmail.com"


def main():
    if len(sys.argv) < 3:
        print("usage: send_failure_alert.py <subject> <message>", file=sys.stderr)
        sys.exit(2)
    subject, message = sys.argv[1], sys.argv[2]

    notifier = create_email_notifier(DEFAULT_EMAIL)
    if not notifier.is_configured():
        print("SMTP not configured - cannot send failure alert (this itself should be rare/caught separately).",
              file=sys.stderr)
        sys.exit(1)

    ok = notifier.send_alert(subject, message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
