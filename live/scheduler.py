"""
Live Trading Scheduler
======================

Runs the live regime trader on a configurable schedule.

Usage:
    # Run once immediately
    python scheduler.py --once

    # Run on schedule (default: daily at 9:45 AM ET)
    python scheduler.py

    # Run on custom schedule
    python scheduler.py --time 10:00 --timezone US/Eastern

Requirements:
    pip install schedule pytz
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import schedule
    import pytz
except ImportError:
    print("Missing required packages. Install with:")
    print("  pip install schedule pytz")
    sys.exit(1)

from live.live_regime_trader import LiveRegimeTrader
from connectors import create_robinhood_connector
from notifications import create_email_notifier


# Default settings
DEFAULT_EMAIL = "joshuaJang89@gmail.com"
DEFAULT_TIME = "09:45"  # 9:45 AM
DEFAULT_TIMEZONE = "US/Eastern"


def run_trading_job(email: str, dry_run: bool):
    """Execute a single trading job."""
    print("\n" + "=" * 70)
    print(f"SCHEDULED TRADING JOB - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        # Create components
        connector = create_robinhood_connector()
        notifier = create_email_notifier(email)

        # Create and run trader
        trader = LiveRegimeTrader(
            connector=connector,
            email_notifier=notifier,
            dry_run=dry_run,
        )

        success = trader.run()

        if success:
            print("\n[Scheduler] Job completed successfully.")
        else:
            print("\n[Scheduler] Job failed.")

    except Exception as e:
        print(f"\n[Scheduler] Error: {e}")
        import traceback
        traceback.print_exc()

        # Try to send error notification
        try:
            notifier = create_email_notifier(email)
            if notifier.is_configured():
                notifier.send_alert(
                    "Scheduled Trading Job Failed",
                    f"Error: {str(e)}\n\nTime: {datetime.now()}"
                )
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Live Trading Scheduler")

    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once immediately and exit'
    )
    parser.add_argument(
        '--time',
        type=str,
        default=DEFAULT_TIME,
        help=f'Time to run daily (HH:MM format, default: {DEFAULT_TIME})'
    )
    parser.add_argument(
        '--timezone',
        type=str,
        default=DEFAULT_TIMEZONE,
        help=f'Timezone (default: {DEFAULT_TIMEZONE})'
    )
    parser.add_argument(
        '--email',
        type=str,
        default=DEFAULT_EMAIL,
        help=f'Email for reports (default: {DEFAULT_EMAIL})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Simulate trades without executing'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Execute real trades'
    )

    args = parser.parse_args()

    dry_run = not args.live

    if args.live:
        print("!" * 70)
        print("WARNING: LIVE TRADING MODE")
        print("Real trades will be executed!")
        print("!" * 70)
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Aborted.")
            return

    # Set timezone
    try:
        tz = pytz.timezone(args.timezone)
    except Exception:
        print(f"Invalid timezone: {args.timezone}")
        return

    print(f"\n[Scheduler] Configuration:")
    print(f"  Email: {args.email}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Time: {args.time} {args.timezone}")

    if args.once:
        # Run once immediately
        print("\n[Scheduler] Running once...")
        run_trading_job(args.email, dry_run)
    else:
        # Schedule daily run
        print(f"\n[Scheduler] Scheduling daily run at {args.time} {args.timezone}")

        # Schedule the job
        schedule.every().day.at(args.time).do(
            run_trading_job,
            email=args.email,
            dry_run=dry_run
        )

        print("[Scheduler] Waiting for scheduled time...")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n[Scheduler] Stopped by user.")


if __name__ == "__main__":
    main()
