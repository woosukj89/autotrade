"""
kr/live_dry_run_kr.py — KR paper-trading dry-run driver.

*** PAPER TRADING ONLY. SIMULATED NUMBERS ONLY. NO REAL MONEY. ***
*** NO BROKER CONNECTION OF ANY KIND.                          ***

This script computes what round 7's strategy WOULD buy/sell TODAY against
REAL KOSPI/KOSDAQ closing prices (pulled live via FinanceDataReader, the
same source kr/data_kr.py uses) and updates a simulated position book
(kr/dry_run_state_kr.json) accordingly. It never places, or assumes the
existence of, a real brokerage order. No KR brokerage API has been
selected or validated for this project yet — see kr/RESEARCH_REPORT.md
("acquiring this account was explicitly out of scope for this task") and
connectors/toss.py's own "UNVALIDATED SKELETON" docstring (written for a
different, not-yet-exercised purpose: a hypothetical future Toss
Securities integration, not anything this script touches or depends on).

── What's deployed ──────────────────────────────────────────────────
RelativeStrengthRotationStrategy (kr/strategies_kr.py, reused UNCHANGED)
+ BacktestKR's continuous Barroso & Santa-Clara vol-targeting mechanism
(kr/backtest_kr.py, reused UNCHANGED), N=8 positions, on kr/data_kr.py's
286-name KOSPI/KOSDAQ universe (reused UNCHANGED). Parameters are the set
that won 3 of round 7's 4 walk-forward folds (B, C, D — see
kr/RESEARCH_REPORT_ROUND7.md section "Per-fold parameter stability"):

    lookback=10, stop_loss_pct=0.10, vol_target=0.20,
    vol_target_lookback_days=60, vol_target_min_exposure=0.3,
    vol_target_max_exposure=1.0, vol_target_tolerance=0.10

Round 9's blend was NOT selected — the user explicitly chose round 7's
strategy alone for this dry run.

── Design note: how this gets cross-run state persistence WITHOUT
   modifying kr/backtest_kr.py ──────────────────────────────────────
live/live_momentum_trader.py's pattern (the one this file was asked to
mirror) works by hydrating/serializing MomentumStrategy's own INSTANCE
ATTRIBUTES across independent daily processes, because that strategy
owns its rebalance-cadence and vol-exposure state internally by design.
kr/backtest_kr.py::BacktestKR is architecturally different: vol-targeting
state (`implemented_exposure`, `value_history`) lives as LOCAL VARIABLES
inside `BacktestKR.run()`'s loop, not on any object that survives past a
single call — there is no hook to pull that state out mid-run or inject
it back in without editing that file, which the task says not to do.

Instead, this driver gets the exact same cross-run-persistence *effect*
via a different, lower-risk mechanism: BacktestKR.run() is a pure,
deterministic function of (price_data, strategy_params, start, end,
capital). So instead of hand-rolling a second, parallel implementation
of the engine's per-day stepping (rescale-on-drift, limit-lock retry,
rank-exit, etc. — all intricate, already-validated logic that would be
easy to subtly reimplement incorrectly), this script just RE-RUNS the
SAME unmodified engine, every day, over the ENTIRE window from a fixed
`inception_date` (chosen once, on the very first run, and persisted
forever after) through TODAY. Today's holdings/cash/exposure are
whatever the engine's own internal state is at the end of that run —
derived fresh each time, not accumulated by hand. This is provably
equivalent to true incremental state-hydration for a deterministic
engine over an unchanging price history, and it means the vol-target
window, the stop-loss/rank-exit bookkeeping, and the limit-lock retry
queue are all exercised through the REAL, validated engine code path
every single day, not a second hand-written approximation of it.
`kr/dry_run_state_kr.json` still persists everything the task asked for
(current holdings, cash, the vol-targeting window's effective state) —
it's just a MATERIALIZED CACHE of the engine's output rather than an
input fed back into a resumable step function. Re-running on the same
calendar day is therefore always safe and idempotent (produces identical
output), which also sidesteps any double-processing risk from running
this script more than once on the same trading day.

── Cold-start bootstrap ──────────────────────────────────────────────
A freshly-started dry run has no "yesterday's returns" to compute a
60-day realized-vol reading from. Per the task, this is solved by
running the actual portfolio-construction logic backward over the most
recent ~65 trading days of already-available historical KR data — i.e.
exactly the "re-run from inception_date" mechanism above, with
`inception_date` chosen (on first run only) as the trading day
`BOOTSTRAP_TRADING_DAYS` (65) before yesterday. By the time "today" is
reached, the engine's internal value_history already has >60 days of
real engine-simulated history behind it, so vol-targeting is active
(not defaulting to max exposure / assuming vol=0) and current holdings
reflect actual signals generated over the bootstrap window, not an
arbitrary empty-cash start. Price history is fetched much further back
than the bootstrap window itself (see FETCH_LOOKBACK_CALENDAR_DAYS)
purely so the strategy's own 60-day moving averages are already warmed
up (non-NaN) BEFORE the bootstrap window begins — `inception_date`
governs what's SIMULATED, not what's DOWNLOADED.

── Environment note: KS11/KQ11 index data lag in this environment ────
Verified by hand before writing this script: FinanceDataReader's
individual-equity endpoint returns genuinely current data (through
today), but its KS11/KQ11 (KOSPI/KOSDAQ composite index) endpoint was
observed stuck several weeks in the past regardless of the date range
requested. Since round 7's deployed configuration uses no regime filter
and no fast breaker (both of which are the only KR strategy components
that depend on the index series), this doesn't affect the strategy
itself at all. It only affects (a) trading-day detection, for which this
script uses a large, liquid universe constituent (005930 / Samsung
Electronics, confirmed fresh) as a proxy instead, and (b) the optional
KOSPI buy-and-hold comparison line in the report, which is clearly
labeled as stale if the index feed hasn't caught up.

── CLI ─────────────────────────────────────────────────────────────
    .venv/Scripts/python.exe kr/live_dry_run_kr.py --dry-run
        [--state-file kr/dry_run_state_kr.json]
        [--starting-cash 30000000]
        [--email | --no-email]            (default: no-email)
        [--email-to address@example.com]

`--dry-run` is accepted for symmetry with live/live_momentum_trader.py's
CLI shape but is not really a "mode switch" — paper trading with no
broker is the ONLY thing this script ever does.
"""

from __future__ import annotations

import os
import sys
import json
import html
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import FinanceDataReader as fdr

KR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(KR_DIR)
sys.path.insert(0, KR_DIR)
sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass

import data_kr
from costs_kr import CostModel
from backtest_kr import BacktestKR, BacktestResultKR, TradeKR
import strategies_kr as sk

try:
    from notifications import create_email_notifier
except Exception:  # pragma: no cover - notifications package always present in this repo
    create_email_notifier = None


# ── deployed strategy config — round 7, the fold B/C/D winner ─────────
# (RESEARCH_REPORT_ROUND7.md: "lookback=10, stop_loss_pct=0.10,
#  vol_target=0.20, vol_target_lookback_days=60 won 3 of 4 folds".)
N_POSITIONS = 8
ROTATION_PARAMS = dict(lookback=10, stop_loss_pct=0.10, n_positions=N_POSITIONS)
VOL_TARGET = 0.20
VOL_TARGET_LOOKBACK_DAYS = 60
VOL_TARGET_MIN_EXPOSURE = 0.3
VOL_TARGET_MAX_EXPOSURE = 1.0
VOL_TARGET_TOLERANCE = 0.10

STARTING_CASH_DEFAULT = 30_000_000.0  # KRW, same figure used throughout this project
STRATEGY_LABEL = "RelativeStrengthRotationStrategy + continuous vol-target (round 7, fold B/C/D params)"

BOOTSTRAP_TRADING_DAYS = 72            # "~65 trading days ending yesterday" per task, + margin.
# Margin note (found empirically on the first real run of this script,
# 2026-10-07): the engine only recomputes realized_vol on REBALANCE days
# (Mondays), and only once value_history has >= vol_target_lookback_days+1
# (61) entries. With bootstrap_trading_days=65 exactly, the most recent
# Monday before "today" landed just ONE trading day short of that
# threshold (and the Monday that would have cleared it, 2026-10-05, was a
# KRX holiday and never simulated at all) — so vol-targeting never
# actually activated once in the whole bootstrap window, defeating the
# point of bootstrapping. 72 gives ~11 trading days of slack above the
# 61-day threshold to absorb the Monday-only cadence + holiday skips.
FETCH_LOOKBACK_CALENDAR_DAYS = 320     # raw price HISTORY fetched (covers MA-60 warmup + bootstrap + margin)

CALENDAR_PROXY_CODE = "005930"         # Samsung Electronics — see module docstring (KS11/KQ11 lag workaround)

DRY_RUN_PRICE_CACHE_DIR = os.path.join(data_kr.CACHE_DIR, "dry_run_prices")
DEFAULT_STATE_FILE = os.path.join(KR_DIR, "dry_run_state_kr.json")
DEFAULT_REPORT_DIR = KR_DIR
DEFAULT_EMAIL = "joshuaJang89@gmail.com"


def log(msg: str) -> None:
    print(f"[live_dry_run_kr] {msg}", flush=True)


# ── non-invasive engine extension (subclass, not a modification) ─────
class _ReasonTrackingEngine(BacktestKR):
    """BacktestKR (kr/backtest_kr.py) is reused UNCHANGED — this
    subclass only ADDS an observability side-channel via inheritance.
    TradeKR itself has no `reason` field (see backtest_kr.py), so
    without this, today's trade list in the report would have no way to
    say WHY each trade happened (entry rank, stop-loss, rank-exit,
    vol-target trim/top-up, ...). Captures PendingOrder.reason alongside
    each TradeKR it produces, keyed by object identity (safe: this
    engine run's trade list is never copied/deduplicated mid-run)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trade_reasons: dict[int, str] = {}

    def _execute(self, order, row, date, cash, positions, result):
        before = len(result.trades)
        super()._execute(order, row, date, cash, positions, result)
        if len(result.trades) > before:
            self.trade_reasons[id(result.trades[-1])] = order.reason


REASON_LABELS = {
    "entry": "new entry (relative-strength rank)",
    "stop_loss": "stop-loss triggered (-10% from entry)",
    "rank_exit": "rank exit (dropped out of weekly top-8)",
    "regime_exit": "regime force-exit",
    "fast_breaker_exit": "fast-breaker force-exit",
    "vol_target_trim": "vol-target rebalance: trim toward lower exposure",
    "vol_target_topup": "vol-target rebalance: top-up toward higher exposure",
}


def reason_label(reason: str) -> str:
    return REASON_LABELS.get(reason, reason or "unknown")


# ── data fetch ──────────────────────────────────────────────────────
def fetch_universe_price_data(codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """Fresh FinanceDataReader pull for [start_date, end_date] for every
    code, cached under kr/data_cache/dry_run_prices/ (gitignored — same
    convention as kr/data_cache/prices/). Deliberately NOT
    kr/data_kr.py's own `_download_and_cache`: that cache's "stale if
    within 5 days of end_date" rule is tuned for the research backtests'
    huge fixed-history pulls, and would happily serve a day-OLD cached
    file on a day we specifically need TODAY's new bar. The cache here
    exists only so repeated same-day invocations (e.g. manual testing)
    don't re-hit the network — not as a correctness mechanism."""
    os.makedirs(DRY_RUN_PRICE_CACHE_DIR, exist_ok=True)
    price_data: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    end_ts = pd.Timestamp(end_date)
    for i, code in enumerate(codes):
        cache_path = os.path.join(DRY_RUN_PRICE_CACHE_DIR, f"{code}.csv")
        df = None
        if os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not cached.empty and cached.index[-1] >= end_ts:
                    df = cached
            except Exception:
                df = None
        if df is None:
            try:
                df = fdr.DataReader(code, start_date, end_date)
            except Exception as e:
                failures.append(f"{code} ({e})")
                continue
            if df is None or df.empty:
                failures.append(f"{code} (empty)")
                continue
            df.to_csv(cache_path)
        price_data[code] = df
        if (i + 1) % 100 == 0:
            log(f"fetched {i + 1}/{len(codes)}")
    if failures:
        shown = failures[:10]
        more = f" (+{len(failures) - 10} more)" if len(failures) > 10 else ""
        log(f"WARNING: {len(failures)} codes failed to fetch and were dropped: {shown}{more}")
    return price_data


def fetch_kospi_benchmark(inception_date: pd.Timestamp, today_date: pd.Timestamp, starting_cash: float):
    """Best-effort KS11 buy-and-hold over the dry run's window, for
    report context. Returns (total_return_or_None, note_or_None). See
    module docstring re: KS11 lag in this environment — this must never
    raise or block the rest of the report.

    NOTE: run_buy_and_hold sizes an INTEGER share count via
    `int(capital // fill_price)` (kr/backtest_kr.py, reused unchanged) —
    passing a tiny "ratio basis" capital like 1.0 would floor to 0
    shares and silently report a fake 0% return regardless of the
    index's actual move. Must pass a realistic KRW capital figure (the
    dry run's own starting_cash) for this to mean anything."""
    try:
        ks11 = fdr.DataReader("KS11", inception_date.strftime("%Y-%m-%d"), today_date.strftime("%Y-%m-%d"))
    except Exception as e:
        return None, f"KOSPI benchmark unavailable (fetch error: {e})"
    if ks11 is None or ks11.empty:
        return None, "KOSPI benchmark unavailable (no data returned)"
    last_date = ks11.index[-1]
    engine = BacktestKR(cost_model=CostModel())
    bh = engine.run_buy_and_hold(ks11, inception_date.strftime("%Y-%m-%d"), today_date.strftime("%Y-%m-%d"),
                                  starting_cash)
    note = None
    if last_date < today_date - pd.Timedelta(days=3):
        note = (f"KOSPI index data source is lagging in this environment (last available "
                f"{last_date.date()}, not {today_date.date()}) — this return is ONLY through "
                f"{last_date.date()}, not a like-for-like same-window comparison.")
    return bh.total_return, note


# ── state file ──────────────────────────────────────────────────────
def load_state(state_file: str) -> dict | None:
    if not state_file or not os.path.exists(state_file):
        return None
    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log(f"WARNING: could not load state file ({e}) — treating as a fresh dry run.")
        return None


def save_state(state_file: str, state: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(state_file)), exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log(f"state saved to {state_file}")


def determine_inception_date(calendar: pd.DatetimeIndex, today: pd.Timestamp,
                              bootstrap_trading_days: int = BOOTSTRAP_TRADING_DAYS) -> pd.Timestamp:
    """First-ever run only: choose inception_date = the trading day that
    is `bootstrap_trading_days` trading days before YESTERDAY (the most
    recently closed session strictly before today) — see module
    docstring's "Cold-start bootstrap" section."""
    past = calendar[calendar < today]
    if len(past) == 0:
        raise RuntimeError("no historical trading days available before today in the fetched "
                            "price data — cannot choose a bootstrap inception date")
    prior_through_yesterday = calendar[calendar <= past[-1]]
    if len(prior_through_yesterday) <= bootstrap_trading_days:
        return prior_through_yesterday[0]
    return prior_through_yesterday[-(bootstrap_trading_days + 1)]


# ── simulation ──────────────────────────────────────────────────────
def run_simulation(price_data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex,
                    inception_date: pd.Timestamp, today: pd.Timestamp, starting_cash: float):
    strat = sk.RelativeStrengthRotationStrategy(**ROTATION_PARAMS)
    strat.prepare(price_data)
    strat._price_data_ref = price_data
    engine = _ReasonTrackingEngine(
        cost_model=CostModel(),
        vol_target=VOL_TARGET,
        vol_target_lookback_days=VOL_TARGET_LOOKBACK_DAYS,
        vol_target_min_exposure=VOL_TARGET_MIN_EXPOSURE,
        vol_target_max_exposure=VOL_TARGET_MAX_EXPOSURE,
        vol_target_tolerance=VOL_TARGET_TOLERANCE,
    )
    result = engine.run(price_data, strat, calendar, inception_date, today, starting_cash,
                         n_positions=N_POSITIONS)
    return result, engine, strat


def reconstruct_positions(trades: list[TradeKR]) -> dict[str, dict]:
    """Replays the trade log into a current-holdings dict, using the
    EXACT SAME weighted-avg-cost / entry-date-stays-at-first-open
    arithmetic as BacktestKR._execute (kr/backtest_kr.py) — see that
    file's PositionKR handling. Needed because BacktestResultKR doesn't
    expose the engine's internal live `positions` dict (it's a local
    variable in run()), only the flat trade log."""
    positions: dict[str, dict] = {}
    for t in trades:
        if t.action == "buy":
            if t.code in positions:
                p = positions[t.code]
                new_shares = p["shares"] + t.shares
                p["avg_cost"] = (p["avg_cost"] * p["shares"] + t.fill_price * t.shares) / new_shares
                p["shares"] = new_shares
            else:
                positions[t.code] = {"shares": t.shares, "avg_cost": t.fill_price,
                                      "entry_date": pd.Timestamp(t.date).strftime("%Y-%m-%d")}
        else:  # sell
            p = positions.get(t.code)
            if p is None:
                continue
            p["shares"] -= t.shares
            if p["shares"] <= 0:
                del positions[t.code]
    return positions


def rank_of(strat, date, price_data, code) -> int | None:
    try:
        ranked = strat._compute_rank(date, price_data)
        return ranked.index(code) + 1
    except Exception:
        return None


# ── report ──────────────────────────────────────────────────────────
def build_report_text(*, today_date, inception_date, is_first_run, result, positions, price_data,
                       today_trades, engine, strat, starting_cash, kospi_return, kospi_note) -> str:
    lines: list[str] = []
    W = 78
    lines.append("=" * W)
    lines.append("KR STRATEGY — PAPER-TRADING DRY RUN (SIMULATED ONLY — NO REAL MONEY, NO BROKER)")
    lines.append("=" * W)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Trading date processed: {today_date.date()}")
    lines.append(f"Strategy: {STRATEGY_LABEL}")
    lines.append(f"Params: {ROTATION_PARAMS}, vol_target={VOL_TARGET}, "
                  f"vol_target_lookback_days={VOL_TARGET_LOOKBACK_DAYS}, "
                  f"vol_target_min/max_exposure={VOL_TARGET_MIN_EXPOSURE}/{VOL_TARGET_MAX_EXPOSURE}")
    lines.append(f"Dry run inception date: {inception_date.date()}"
                 + ("  (chosen today — cold-start bootstrap)" if is_first_run else ""))
    lines.append("")

    snap = result.snapshots[-1]
    lines.append("-" * W)
    lines.append("PORTFOLIO STATE (simulated)")
    lines.append("-" * W)
    lines.append(f"Total value:  {snap.total_value:>18,.0f} KRW")
    lines.append(f"Cash:         {snap.cash:>18,.0f} KRW")
    lines.append(f"Invested:     {snap.positions_value:>18,.0f} KRW  ({snap.n_positions} positions)")
    lines.append(f"Starting cash (at inception): {starting_cash:,.0f} KRW")
    if snap.cash < 0:
        lines.append(f"  note: cash is temporarily negative. This is a known, pre-existing property of "
                      f"kr/backtest_kr.py's limit-lock retry queue (reused UNCHANGED here, per the task) — "
                      f"a buy order sized against the cash available WHEN IT WAS QUEUED can fill a day or "
                      f"more later (once its limit-up lock clears) WITHOUT being re-checked against cash "
                      f"that other same-window trades have since consumed. total_value (cash + positions_value) "
                      f"is still fully consistent with the engine's own validated accounting; this is a "
                      f"cosmetic quirk of the cash/positions split, not a solvency problem or a bug "
                      f"introduced by this driver.")
    lines.append(f"Cumulative return since inception: {result.total_return * 100:+.2f}%")
    if kospi_return is not None:
        lines.append(f"KOSPI buy-and-hold, same window:    {kospi_return * 100:+.2f}%")
    if kospi_note:
        lines.append(f"  note: {kospi_note}")
    lines.append("")

    lines.append("-" * W)
    lines.append("CURRENT HOLDINGS")
    lines.append("-" * W)
    if positions:
        lines.append(f"{'Code':<8}{'Shares':>10}{'AvgCost':>12}{'Price':>12}{'Value':>16}{'Weight':>8}{'P&L%':>9}{'Entry':>12}")
        rows = []
        for code, p in positions.items():
            df = price_data.get(code)
            if df is not None and today_date in df.index:
                price = float(df.loc[today_date, "Close"])
            elif df is not None and len(df):
                price = float(df["Close"].iloc[-1])
            else:
                price = p["avg_cost"]
            value = p["shares"] * price
            weight = value / snap.total_value if snap.total_value > 0 else 0.0
            pnl_pct = (price / p["avg_cost"] - 1.0) * 100 if p["avg_cost"] > 0 else 0.0
            rows.append((code, p["shares"], p["avg_cost"], price, value, weight, pnl_pct, p["entry_date"]))
        rows.sort(key=lambda r: r[4], reverse=True)
        for code, shares, avg_cost, price, value, weight, pnl_pct, entry_date in rows:
            lines.append(f"{code:<8}{shares:>10,d}{avg_cost:>12,.0f}{price:>12,.0f}{value:>16,.0f}"
                          f"{weight * 100:>7.1f}%{pnl_pct:>+8.1f}%{entry_date:>12}")
    else:
        lines.append("(no open positions)")
    lines.append("")

    lines.append("-" * W)
    lines.append(f"TODAY'S SIMULATED TRADES ({today_date.date()})")
    lines.append("-" * W)
    if today_trades:
        for t in today_trades:
            reason = reason_label(engine.trade_reasons.get(id(t), ""))
            extra = ""
            if t.action == "buy" and (engine.trade_reasons.get(id(t)) == "entry"):
                r = rank_of(strat, today_date, price_data, t.code)
                if r is not None:
                    extra = f" (rank #{r})"
            lines.append(f"{t.action.upper():<5}{t.code:<8}{t.shares:>8,d} sh @ {t.fill_price:>10,.0f} "
                         f"= {t.gross_value:>14,.0f} KRW   reason: {reason}{extra}")
    else:
        lines.append("(no trades today)")
    lines.append("")

    lines.append("-" * W)
    lines.append("VOL-TARGETING EXPOSURE")
    lines.append("-" * W)
    if result.exposure_history:
        date, target_exp, realized_vol, implemented = result.exposure_history[-1]
        rv_str = f"{realized_vol * 100:.1f}%" if realized_vol is not None else "n/a (insufficient history)"
        as_of_note = "" if pd.Timestamp(date).normalize() == today_date else f"  (as of last rebalance day {pd.Timestamp(date).date()}, not updated daily)"
        lines.append(f"Implemented exposure: {implemented * 100:.1f}%   Target exposure: {target_exp * 100:.1f}%"
                      f"{as_of_note}")
        lines.append(f"Realized volatility (trailing {VOL_TARGET_LOOKBACK_DAYS}d, annualized): {rv_str}")
        lines.append(f"Vol target (annualized): {VOL_TARGET * 100:.0f}%   "
                      f"Bounds: [{VOL_TARGET_MIN_EXPOSURE * 100:.0f}%, {VOL_TARGET_MAX_EXPOSURE * 100:.0f}%]")
    else:
        lines.append("No rebalance-day exposure reading yet (still within bootstrap warmup).")
    lines.append("")

    lines.append("=" * W)
    lines.append("This is a SIMULATED paper-trading report. No real money. No broker is")
    lines.append("connected — nothing above was, or could have been, sent as a real order.")
    lines.append("=" * W)
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KR strategy paper-trading dry run (round 7 strategy)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                         help="Paper trading is the only mode this script supports; kept for "
                              "CLI symmetry with live/live_momentum_trader.py.")
    parser.add_argument("--state-file", type=str, default=DEFAULT_STATE_FILE,
                         help=f"Path to the JSON state file (default: {DEFAULT_STATE_FILE})")
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH_DEFAULT,
                         help=f"Starting notional capital, KRW (default: {STARTING_CASH_DEFAULT:,.0f})")
    parser.add_argument("--email", action="store_true", help="Send the report via email (also requires "
                                                               "SMTP_USERNAME/SMTP_PASSWORD env vars).")
    parser.add_argument("--no-email", action="store_true", help="Explicitly disable email (default behavior).")
    parser.add_argument("--email-to", type=str, default=DEFAULT_EMAIL, help="Report recipient address.")
    args = parser.parse_args()

    print("=" * 78)
    print("KR LIVE DRY RUN — paper trading only, no broker")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    today_date = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    state = load_state(args.state_file)

    log("loading universe (reusing kr/data_cache/universe.csv if present, unchanged kr/data_kr.py)...")
    try:
        universe = data_kr.load_universe()
    except FileNotFoundError:
        log("no cached universe found — building it now via data_kr.build_universe() "
            "(slow: full structural/liquidity/cap scan over the current KRX listing).")
        universe = data_kr.build_universe()
    codes = list(universe.index)
    if CALENDAR_PROXY_CODE not in codes:
        codes = [CALENDAR_PROXY_CODE] + codes
    log(f"universe: {len(codes)} codes")

    fetch_start = (today_date - pd.Timedelta(days=FETCH_LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    today_str = today_date.strftime("%Y-%m-%d")
    log(f"fetching fresh price data [{fetch_start} .. {today_str}] via FinanceDataReader...")
    price_data = fetch_universe_price_data(codes, fetch_start, today_str)

    if CALENDAR_PROXY_CODE not in price_data:
        log(f"FATAL: calendar-proxy ticker {CALENDAR_PROXY_CODE} could not be fetched — cannot "
            f"determine whether today is a trading day. Aborting without writing state.")
        sys.exit(1)

    calendar = price_data[CALENDAR_PROXY_CODE].index
    if today_date not in calendar:
        log(f"{today_str} has no new row in the calendar-proxy data ({CALENDAR_PROXY_CODE}) — "
            f"not a KRX trading day (weekend/holiday) or today's bar isn't published yet. "
            f"No-op: nothing to process, state file left untouched.")
        return

    is_first_run = state is None or not state.get("inception_date")
    if is_first_run:
        inception_date = determine_inception_date(calendar, today_date)
        starting_cash = args.starting_cash
        created_at = datetime.now().isoformat()
        log(f"no prior state — bootstrapping a new dry run. inception_date={inception_date.date()} "
            f"(~{BOOTSTRAP_TRADING_DAYS} trading days before yesterday), starting_cash={starting_cash:,.0f} KRW")
    else:
        inception_date = pd.Timestamp(state["inception_date"])
        starting_cash = float(state.get("starting_cash", args.starting_cash))
        created_at = state.get("created_at", datetime.now().isoformat())
        log(f"resuming existing dry run: inception_date={inception_date.date()}, "
            f"started={created_at}, starting_cash={starting_cash:,.0f} KRW")

    log(f"running strategy simulation [{inception_date.date()} .. {today_date.date()}] "
        f"({len(calendar[(calendar >= inception_date) & (calendar <= today_date)])} trading days)...")
    result, engine, strat = run_simulation(price_data, calendar, inception_date, today_date, starting_cash)

    positions = reconstruct_positions(result.trades)
    today_trades = [t for t in result.trades if pd.Timestamp(t.date).normalize() == today_date]

    log("fetching KOSPI buy-and-hold benchmark for context...")
    kospi_return, kospi_note = fetch_kospi_benchmark(inception_date, today_date, starting_cash)

    report_text = build_report_text(
        today_date=today_date, inception_date=inception_date, is_first_run=is_first_run,
        result=result, positions=positions, price_data=price_data, today_trades=today_trades,
        engine=engine, strat=strat, starting_cash=starting_cash,
        kospi_return=kospi_return, kospi_note=kospi_note,
    )
    print("\n" + report_text + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(DEFAULT_REPORT_DIR, f"dry_run_report_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log(f"report saved to {report_path}")

    snap = result.snapshots[-1]
    exposure_entry = None
    if result.exposure_history:
        d, target_exp, realized_vol, implemented = result.exposure_history[-1]
        exposure_entry = {
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "target_exposure": target_exp,
            "realized_vol": realized_vol,
            "implemented_exposure": implemented,
        }

    new_state = {
        "version": 1,
        "paper_trading_only": True,
        "no_broker_connected": True,
        "strategy": STRATEGY_LABEL,
        "params": {**ROTATION_PARAMS, "vol_target": VOL_TARGET,
                   "vol_target_lookback_days": VOL_TARGET_LOOKBACK_DAYS,
                   "vol_target_min_exposure": VOL_TARGET_MIN_EXPOSURE,
                   "vol_target_max_exposure": VOL_TARGET_MAX_EXPOSURE,
                   "vol_target_tolerance": VOL_TARGET_TOLERANCE},
        "starting_cash": starting_cash,
        "inception_date": inception_date.strftime("%Y-%m-%d"),
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "last_processed_date": today_str,
        "cash": snap.cash,
        "positions_value": snap.positions_value,
        "total_value": snap.total_value,
        "cumulative_return_since_inception": result.total_return,
        "holdings": positions,
        "exposure": exposure_entry,
        "last_report_file": os.path.basename(report_path),
    }
    save_state(args.state_file, new_state)

    send_email = args.email and not args.no_email and os.environ.get("SMTP_USERNAME") and os.environ.get("SMTP_PASSWORD")
    if send_email:
        if create_email_notifier is None:
            log("email requested but notifications package could not be imported — skipping.")
        else:
            notifier = create_email_notifier(args.email_to)
            if notifier.is_configured():
                subject = f"KR Dry Run (Paper Trading) — {today_date.date()}"
                body = (f'<pre style="white-space:pre-wrap;font-family:monospace;font-size:12px;">'
                        f'{html.escape(report_text)}</pre>')
                notifier.send_alert(subject, body)
            else:
                log("email requested but SMTP not configured — skipping.")
    else:
        log("email disabled (default) — report written to file only.")

    print("\n" + "=" * 78)
    print("SUMMARY (paper trading — simulated only)")
    print("=" * 78)
    print(f"Total value: {snap.total_value:,.0f} KRW   Cash: {snap.cash:,.0f} KRW   "
          f"Positions: {snap.n_positions}")
    print(f"Cumulative return since inception ({inception_date.date()}): {result.total_return * 100:+.2f}%")
    print(f"Trades today: {len(today_trades)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
