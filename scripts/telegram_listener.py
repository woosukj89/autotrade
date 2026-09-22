"""
Telegram Listener — one-tap Robinhood session re-trigger
==========================================================

Polls the Telegram Bot API's getUpdates endpoint for new messages, and
when it sees a /refresh command from the authorized chat, fires a
GitHub repository_dispatch event to re-run scripts/refresh_session.py
— optionally with a manually-relayed MFA/SMS code.

This exists because Robinhood's programmatic MFA is documented as
unreliable (broke for many robin_stocks users in a Dec 2024 incident),
so full closed-loop automation of re-authentication isn't a safe bet.
This does not try to automate Robinhood's own approval step. It
minimizes time-to-notice (via refresh_session.py's ntfy.sh push) and
lets the user re-trigger the automated refresh from their phone in one
message, after they've done whatever Robinhood itself requires
(opening the app, approving a push, reading an SMS code).

Designed to run as a scheduled GitHub Actions job (no always-on server
needed) — see .github/workflows/telegram_listener.yml. Each run:
  1. Loads the last processed Telegram update_id from
     live/telegram_offset.json (committed to the repo, same pattern as
     live/trader_state.json).
  2. Calls getUpdates with offset=last_id+1 (Telegram's own
     at-most-once delivery semantics — no update is redelivered once
     acknowledged via a higher offset).
  3. For each new message FROM THE AUTHORIZED CHAT ONLY matching
     /refresh or /refresh <code>, fires repository_dispatch and replies
     in Telegram confirming it did so.
  4. Saves the new offset.

Security: TELEGRAM_CHAT_ID is a hard allowlist, not a hint — any
message from a different chat is logged and silently ignored. Telegram
bots are discoverable by username search; without this check, anyone
who finds the bot could trigger a repository_dispatch (cheap to spam,
but still not something to leave open).

Environment Variables Required:
    TELEGRAM_BOT_TOKEN: from @BotFather
    TELEGRAM_CHAT_ID: the one chat ID allowed to issue commands (get
        this once by messaging the bot, then checking
        https://api.telegram.org/bot<token>/getUpdates)
    GH_PAT: GitHub PAT with repo scope, to fire repository_dispatch and
        commit the updated offset file (same PAT already used for
        `gh secret set` elsewhere in this project)
    GITHUB_REPOSITORY: "owner/repo" (set automatically by GitHub Actions)
"""

import os
import sys
import json
import requests
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, '.env'))
except ImportError:
    pass

BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
GH_PAT = os.environ.get('GH_PAT')
GITHUB_REPOSITORY = os.environ.get('GITHUB_REPOSITORY')  # "owner/repo"

OFFSET_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'live', 'telegram_offset.json')

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def _load_offset() -> int:
    if not os.path.exists(OFFSET_FILE):
        return 0
    try:
        with open(OFFSET_FILE, 'r') as f:
            return json.load(f).get('last_update_id', 0)
    except (json.JSONDecodeError, IOError):
        return 0


def _save_offset(update_id: int) -> None:
    os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
    with open(OFFSET_FILE, 'w') as f:
        json.dump({'last_update_id': update_id}, f, indent=2)


def _reply(chat_id: str, text: str) -> None:
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception as e:
        print(f"[telegram_listener] Reply failed (non-fatal): {e}")


def _trigger_refresh(mfa_code: Optional[str]) -> bool:
    """Fire a repository_dispatch event that re-runs refresh_session.py."""
    if not GH_PAT or not GITHUB_REPOSITORY:
        print("[telegram_listener] GH_PAT or GITHUB_REPOSITORY not set — cannot dispatch.")
        return False
    try:
        resp = requests.post(
            f"https://api.github.com/repos/{GITHUB_REPOSITORY}/dispatches",
            headers={
                "Authorization": f"Bearer {GH_PAT}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "event_type": "manual-session-refresh",
                "client_payload": {"mfa_code": mfa_code or ""},
            },
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[telegram_listener] Dispatch failed: {e}")
        return False


def main():
    if not BOT_TOKEN or not CHAT_ID:
        print("[telegram_listener] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — nothing to do.")
        return

    offset = _load_offset()
    print(f"[telegram_listener] Polling from update_id={offset + 1}...")

    try:
        resp = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset + 1, "timeout": 0},
            timeout=15,
        )
        resp.raise_for_status()
        updates = resp.json().get("result", [])
    except Exception as e:
        print(f"[telegram_listener] getUpdates failed: {e}")
        return

    if not updates:
        print("[telegram_listener] No new messages.")
        _save_offset(offset)  # ensure the state file exists even on an empty poll
        return

    highest_id = offset
    for update in updates:
        highest_id = max(highest_id, update.get("update_id", highest_id))

        message = update.get("message") or {}
        text = (message.get("text") or "").strip()
        sender_chat_id = str(message.get("chat", {}).get("id", ""))

        if not text:
            continue

        if sender_chat_id != str(CHAT_ID):
            print(f"[telegram_listener] Ignoring message from unauthorized chat {sender_chat_id}.")
            continue

        if text.startswith("/refresh"):
            parts = text.split(maxsplit=1)
            mfa_code = parts[1].strip() if len(parts) > 1 else None

            print(f"[telegram_listener] /refresh command received (mfa_code={'<provided>' if mfa_code else 'none'}).")
            if _trigger_refresh(mfa_code):
                _reply(CHAT_ID, "Session refresh triggered" + (" with your code." if mfa_code else " (TOTP/no manual code)."))
            else:
                _reply(CHAT_ID, "Could not trigger session refresh - check GH_PAT/repo config.")
        elif text.startswith("/start") or text.startswith("/help"):
            _reply(CHAT_ID, "Commands:\n/refresh - retry Robinhood session refresh\n/refresh <code> - retry with a manual MFA/SMS code")

    _save_offset(highest_id)
    print(f"[telegram_listener] Processed {len(updates)} update(s), new offset={highest_id}.")


if __name__ == "__main__":
    main()
