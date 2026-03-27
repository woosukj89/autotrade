"""
Refresh Robinhood Session
=========================
Connects to Robinhood to extend the session token, then disconnects.
Intended to be run daily via GitHub Actions (including weekends) to
prevent the ~7-day session expiry from causing trade failures.

Sends a warning email if the session was dangerously old before this
refresh — a sign that the automated job has been silently failing.
"""

import os
import sys
import json
import base64
import pickle
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, '.env'))
except ImportError:
    pass

from notifications import create_email_notifier

# Warn if the session pickle was this many days old before refresh.
# Robinhood sessions last ~7 days; 4 days old means 4 consecutive
# refresh failures and only 3 days of buffer remaining.
WARN_AGE_DAYS = 4

DEFAULT_EMAIL = os.environ.get('REPORT_EMAIL', 'joshuaJang89@gmail.com')


# ── Session helpers ────────────────────────────────────────────────

def _pickle_path(pickle_dir: str) -> str:
    return os.path.join(pickle_dir or '.session', 'robinhood.pickle')


def get_pickle_age_days(pickle_dir: str) -> Optional[float]:
    """Return how many days old the pickle file is, or None if not found."""
    path = _pickle_path(pickle_dir)
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 86400


def get_token_expiry(pickle_dir: str) -> Optional[datetime]:
    """
    Decode the JWT access_token from the pickle to get its expiry time.
    Falls back to (mtime + 7 days) if the token is not a JWT.
    """
    path = _pickle_path(pickle_dir)
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        access_token = data.get('access_token', '')
        parts = access_token.split('.')
        if len(parts) == 3:
            # JWT: base64url-decode the payload segment
            payload_b64 = parts[1]
            pad = 4 - len(payload_b64) % 4
            if pad != 4:
                payload_b64 += '=' * pad
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get('exp')
            if exp:
                return datetime.fromtimestamp(exp)

        # Fallback: assume 7-day session from last file modification
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime + 7 * 86400)

    except Exception as e:
        print(f"[refresh_session] Could not determine token expiry: {e}")
        return None


# ── Email helpers ──────────────────────────────────────────────────

def send_failure_email(error_hint: str) -> None:
    """Send an alert email when the session refresh job itself fails."""
    notifier = create_email_notifier(DEFAULT_EMAIL)
    if not notifier.is_configured():
        print("[refresh_session] Email not configured — skipping failure alert.")
        return

    subject = "[AutoTrade] ERROR: Robinhood session refresh failed"
    body_html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px; max-width: 600px;">
        <h2 style="color: #dc3545;">&#10060; Session Refresh Failed</h2>

        <p>The automated Robinhood session refresh job failed on
        <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong>.</p>

        <div style="background: #f8d7da; border: 1px solid #f5c6cb;
                    padding: 15px; border-radius: 4px; margin: 15px 0;">
            <strong>Error:</strong><br>
            <code>{error_hint}</code>
        </div>

        <p>If this keeps failing, the Robinhood session will expire within ~7 days
        and live trading will stop. To fix, re-authenticate manually:</p>
        <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px;">
python scripts/bootstrap_session.py</pre>
        <p>Then update the <code>ROBINHOOD_SESSION</code> GitHub Actions secret.</p>

        <hr style="border: 1px solid #eee; margin-top: 20px;">
        <p style="font-size: 12px; color: #999;">
            AutoTrade &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </body>
    </html>
    """

    notifier.send_email(subject, body_html)
    print(f"[refresh_session] Failure alert email sent to {DEFAULT_EMAIL}")


def send_expiry_warning(age_days: float, expiry: Optional[datetime]) -> None:
    """Send a warning email that the session was dangerously close to expiry."""
    notifier = create_email_notifier(DEFAULT_EMAIL)
    if not notifier.is_configured():
        print("[refresh_session] Email not configured — skipping warning.")
        return

    expiry_str = expiry.strftime('%Y-%m-%d %H:%M UTC') if expiry else 'unknown'
    if expiry:
        days_left = (expiry - datetime.now()).total_seconds() / 86400
        days_left_str = f"{days_left:.1f} days"
    else:
        days_left_str = 'unknown'

    buffer_days = 7 - age_days
    subject = "[AutoTrade] WARNING: Robinhood session was close to expiring"
    body_html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px; max-width: 600px;">
        <h2 style="color: #dc3545;">&#9888; Robinhood Session Warning</h2>

        <p>Today's session refresh found that the stored session was
        <strong>{age_days:.1f} days old</strong> — only {buffer_days:.1f} days
        before the ~7-day expiry. This means the automated refresh job had
        been silently failing for several days.</p>

        <div style="background: #fff3cd; border: 1px solid #ffc107;
                    padding: 15px; border-radius: 4px; margin: 15px 0;">
            <strong>Session details</strong><br>
            Age before today's refresh: <strong>{age_days:.1f} days</strong><br>
            New token expires: <strong>{expiry_str}</strong>
            ({days_left_str} from now)<br>
        </div>

        <p>The session has been refreshed successfully today. However, if
        the refresh job fails again for <strong>3 or more consecutive days</strong>,
        the session will expire and live trading will stop.</p>

        <p>If that happens, re-authenticate manually:</p>
        <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px;">
python scripts/bootstrap_session.py</pre>
        <p>Then update the <code>ROBINHOOD_SESSION</code> GitHub Actions secret.</p>

        <hr style="border: 1px solid #eee; margin-top: 20px;">
        <p style="font-size: 12px; color: #999;">
            AutoTrade &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </body>
    </html>
    """

    notifier.send_email(subject, body_html)
    print(f"[refresh_session] Warning email sent to {DEFAULT_EMAIL}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("ROBINHOOD SESSION REFRESH")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    pickle_dir = os.environ.get('ROBINHOOD_PICKLE_PATH', '.session')

    # ── 1. Check session age BEFORE connecting ──────────────────────
    age_days = get_pickle_age_days(pickle_dir)
    if age_days is not None:
        print(f"\nSession age before refresh: {age_days:.1f} days")
        if age_days >= WARN_AGE_DAYS:
            print(f"WARNING: session is {age_days:.1f}d old (limit ~7d, warn at {WARN_AGE_DAYS}d)")
    else:
        print("\nNo existing session pickle found.")

    # ── 2. Connect (robin_stocks re-authenticates and writes new pickle)
    print("\nConnecting to Robinhood...")
    try:
        from connectors import create_robinhood_connector
        connector = create_robinhood_connector()
    except Exception as e:
        print(f"ERROR: Could not create connector: {e}")
        send_failure_email(str(e))
        sys.exit(1)

    if not connector.connect():
        print("ERROR: Robinhood connection failed.")
        send_failure_email("connector.connect() returned False — check credentials or MFA challenge")
        sys.exit(1)

    print("Connected — session token refreshed.")

    # ── 3. Report new token expiry ──────────────────────────────────
    expiry = get_token_expiry(pickle_dir)
    if expiry:
        days_remaining = (expiry - datetime.now()).total_seconds() / 86400
        print(f"New token expires: {expiry.strftime('%Y-%m-%d %H:%M')} "
              f"({days_remaining:.1f} days from now)")
    else:
        print("Could not determine token expiry.")

    connector.disconnect()

    # ── 4. Send warning email if session was dangerously old ────────
    if age_days is not None and age_days >= WARN_AGE_DAYS:
        send_expiry_warning(age_days, expiry)

    print("\nSession refresh complete.")


if __name__ == "__main__":
    main()
