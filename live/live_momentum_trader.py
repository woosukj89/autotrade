"""
Live Momentum Trading Runner
=============================

Executes MomentumStrategy (improvement/momentum_strategy.py) with real
money using Robinhood.

This is the best fully-validated result from this session's research:
23.1% CAGR / 33.9% MaxDD / 0.79 Sharpe vs. the previously-live
RegimeAdaptiveStrategy's 14.2% / 58.0% / 0.50, measured with identical,
honest methodology (point-in-time-clean, daily cadence matching this
same live cron, real slippage/fees/tax). See improvement/PROGRESS.md
Iterations 24-29 for the full research trail.

A separate driver file from live_regime_trader.py rather than modifying
it in place - this trades real money, so rollback-ability and blast-
radius isolation are weighted over avoiding duplication. Deliberately
duplicates the trading-logic methods (get_current_portfolio,
calculate_trades, execute_trades, send_report, run/main scaffolding);
imports the pure-utility pieces that carry zero strategy-specific
content (MockConnector, TradeOrder, parse_positions_file,
write_positions_file) directly from live_regime_trader.py rather than
re-duplicating ~400 lines of identical infrastructure.

Critical difference from live_regime_trader.py's design: MomentumStrategy
owns its rebalance-cadence and market-filter hysteresis state INTERNALLY
via instance attributes, not via an external driver-level gate. Without
persisting that state across the fresh-process-per-cron-run pattern,
the market-filter hysteresis (the single biggest driver of the validated
MaxDD improvement) would silently never activate. See _load_momentum_state
/_hydrate_strategy_state/_save_momentum_state below - this is the load-
bearing addition this driver exists to get right.

Usage:
    # Dry run (uses mock connector, no Robinhood credentials needed)
    python live_momentum_trader.py --dry-run

    # Live trading (requires Robinhood credentials)
    python live_momentum_trader.py --live --email recipient@email.com

Environment Variables Required (for live mode only):
    - ROBINHOOD_USERNAME: Robinhood login email
    - ROBINHOOD_PASSWORD: Robinhood password
    - ROBINHOOD_TOTP_SECRET: (optional) TOTP secret for 2FA

For email notifications:
    - SMTP_USERNAME: Email account for sending reports
    - SMTP_PASSWORD: Email password or app password

For the macro-context report panel (optional, informational only):
    - FRED_API_KEY: FRED API key

Safety Features:
    - Dry-run mode uses mock connector (no real trading)
    - Order validation before submission
    - Market hours check
    - Reports saved to file AND sent via email
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import yfinance as yf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, '.env'))
except ImportError:
    pass  # python-dotenv not installed

from connectors.base import (
    ExchangeConnector,
    Position as ConnectorPosition,
    Order,
    AccountInfo,
    OrderSide,
)
from notifications import (
    EmailNotifier,
    RebalanceAction,
    RebalanceReport,
    create_email_notifier,
)
from strategies.strategy import Portfolio, Position
from live.execution_context import DryRunExecutionContext, LiveExecutionContext
from live.macro_snapshot import fetch_raw_macro_metrics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'improvement'))
from momentum_strategy import MomentumStrategy

# Pure-utility pieces with zero strategy-specific content - reused as-is
# rather than re-duplicated. This is an import, not a modification, of
# the currently-live regime-trader file: zero risk to that file.
from live.live_regime_trader import (
    MockConnector,
    TradeOrder,
    parse_positions_file,
    write_positions_file,
)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*YFPricesMissingError.*')


# Default email recipient
DEFAULT_EMAIL = "joshuaJang89@gmail.com"

# Strategy parameters
STRATEGY_NAME = "Momentum Strategy (12-1 Dual Momentum + Market Filter)"

# Validated configuration (improvement/PROGRESS.md Iteration 29):
# 23.1% CAGR / 33.9% MaxDD / 0.79 Sharpe, full 20yr daily cadence, real costs.
MOMENTUM_CONFIG = dict(
    max_positions=10,
    lookback_days=189,
    rebalance_days=30,
    trend_sma_days=200,
    market_filter=True,
    market_filter_sma_days=200,
    market_filter_defensive_weight=0.30,
    market_filter_buffer_pct=0.02,
    market_filter_confirm_days=5,
)

# Minimum trade value to avoid tiny orders
MIN_TRADE_VALUE = 50.0

# Default starting cash for dry-run mock portfolio
DEFAULT_STARTING_CASH = 0.0


# ── MomentumStrategy internal-state persistence ─────────────────────────
#
# Unlike RegimeAdaptiveStrategy (externally gated by trader_state.json,
# tolerant of its own state resetting every process), MomentumStrategy
# owns its rebalance-cadence and market-filter hysteresis state
# internally via instance attributes. GitHub Actions runs a fresh
# process every day - without this, market_filter_confirm_days (5) can
# never accumulate confirmation days across independent runs, and the
# market filter - the single biggest driver of the validated MaxDD
# improvement - silently stays permanently disabled while appearing
# configured. Kept in a separate file from trader_state.json: different
# schema, independent rollback, no collisions during parallel dry-run.

def _load_momentum_state(state_file: str) -> Optional[Dict]:
    """Load MomentumStrategy's persisted internal state."""
    if not state_file or not os.path.exists(state_file):
        return None
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[LiveMomentumTrader] Warning: could not load momentum state file: {e}")
        return None


def _hydrate_strategy_state(strategy: MomentumStrategy, state: Optional[Dict]) -> None:
    """Apply persisted state onto a freshly-constructed MomentumStrategy,
    before anything else touches it. Every field defaults to the
    strategy's own fresh-instance defaults when state is missing/partial,
    so a missing or corrupt state file degrades to "acts like day one"
    rather than crashing.
    """
    state = state or {}

    last_rebalance_str = state.get('last_rebalance')
    strategy._last_rebalance = datetime.fromisoformat(last_rebalance_str) if last_rebalance_str else None

    strategy._holdings = state.get('holdings', {}) or {}
    strategy._market_state_healthy = state.get('market_state_healthy', True)
    strategy._market_pending_flip = state.get('market_pending_flip')
    strategy._market_pending_days = state.get('market_pending_days', 0)
    strategy._value_history = state.get('value_history', []) or []
    strategy._implemented_vol_exposure = state.get('implemented_vol_exposure', 1.0)

    if state:
        print(f"[LiveMomentumTrader] Hydrated strategy state: "
              f"last_rebalance={strategy._last_rebalance}, "
              f"{len(strategy._holdings)} holdings, "
              f"market_healthy={strategy._market_state_healthy}, "
              f"pending_flip={strategy._market_pending_flip} "
              f"({strategy._market_pending_days} days)")
    else:
        print("[LiveMomentumTrader] No prior momentum state found - starting fresh "
              "(market filter hysteresis will need market_filter_confirm_days "
              "consecutive runs before it can act on a flip).")


def _save_momentum_state(strategy: MomentumStrategy, state_file: str) -> None:
    """Persist MomentumStrategy's internal state. Called UNCONDITIONALLY
    after every strategy.execute() - not gated behind "a trade happened".
    The hysteresis counters mutate on every call including no-op days;
    skipping the save on quiet days reproduces the exact bug this
    function exists to prevent.
    """
    if not state_file:
        return
    state = {
        'last_rebalance': strategy._last_rebalance.isoformat() if strategy._last_rebalance else None,
        'holdings': strategy._holdings,
        'market_state_healthy': strategy._market_state_healthy,
        'market_pending_flip': strategy._market_pending_flip,
        'market_pending_days': strategy._market_pending_days,
        'value_history': strategy._value_history,
        'implemented_vol_exposure': strategy._implemented_vol_exposure,
        'saved_at': datetime.now().isoformat(),
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(state_file)), exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"[LiveMomentumTrader] Momentum state saved to {state_file}")
    except IOError as e:
        print(f"[LiveMomentumTrader] Warning: could not save momentum state file: {e}")


class LiveMomentumTrader:
    """
    Live trading implementation using MomentumStrategy.

    Uses the same strategy logic as backtesting for consistency.
    In dry-run mode, uses MockConnector instead of real Robinhood connection.
    """

    def __init__(
        self,
        connector: ExchangeConnector,
        email_notifier: Optional[EmailNotifier] = None,
        dry_run: bool = True,
        report_dir: str = None,
        available_cash: float = None,
        positions_file: str = None,
        state_file: str = "live/momentum_trader_state.json",
        fred_api_key: Optional[str] = None,
    ):
        self.connector = connector
        self.notifier = email_notifier
        self.dry_run = dry_run
        self.report_dir = report_dir or os.path.dirname(os.path.abspath(__file__))
        self.available_cash = available_cash
        self.positions_file = positions_file
        self.state_file = state_file
        self.fred_api_key = fred_api_key

        self.previous_positions: Dict[str, Dict] = {}
        if positions_file and os.path.exists(positions_file):
            self.previous_positions = parse_positions_file(positions_file)
            if self.previous_positions:
                print(f"[LiveMomentumTrader] Loaded {len(self.previous_positions)} previous positions from {positions_file}")

        self._strategy: Optional[MomentumStrategy] = None
        self._actions_taken: List[RebalanceAction] = []
        self._exposure: float = 1.0
        self._market_healthy: bool = True
        self._last_rebalance_date: Optional[datetime] = None
        self._holdings_snapshot: Dict[str, dict] = {}
        self._top_candidates: List[dict] = []
        self._days_since_rebalance: Optional[int] = None
        self._session_failed: bool = False

    def _get_strategy(self) -> MomentumStrategy:
        """Lazy-construct the strategy and hydrate persisted state onto it."""
        if self._strategy is None:
            self._strategy = MomentumStrategy(**MOMENTUM_CONFIG)
            state = _load_momentum_state(self.state_file)
            _hydrate_strategy_state(self._strategy, state)
        return self._strategy

    def get_current_portfolio(self) -> Tuple[AccountInfo, Portfolio, float]:
        """Get current portfolio state from connector and convert to strategy Portfolio."""
        print("\n[LiveMomentumTrader] Fetching current portfolio...")

        account = self.connector.get_account_info()
        connector_positions = self.connector.get_positions()

        positions = {}
        total_invested = 0.0

        for pos in connector_positions:
            if pos.shares > 0:
                positions[pos.ticker] = Position(
                    ticker=pos.ticker,
                    shares=pos.shares,
                    avg_cost=pos.avg_cost,
                )
                total_invested += pos.market_value

        portfolio = Portfolio(cash=account.cash, positions=positions)
        total_value = account.cash + total_invested

        print(f"  Account: {account.account_id}")
        print(f"  Cash: ${account.cash:,.2f}")
        print(f"  Invested: ${total_invested:,.2f}")
        print(f"  Total Value: ${total_value:,.2f}")
        print(f"  Positions: {len(positions)}")

        return account, portfolio, total_value

    def run_strategy(self, portfolio: Portfolio) -> Portfolio:
        """Run MomentumStrategy to get the target portfolio. Saves the
        strategy's internal state unconditionally afterward - see the
        module docstring on why this must not be skipped."""
        print("\n[LiveMomentumTrader] Running MomentumStrategy...")

        strategy = self._get_strategy()

        if self.dry_run:
            context = DryRunExecutionContext(portfolio=portfolio, date=datetime.now())
        else:
            context = LiveExecutionContext(connector=self.connector, portfolio=portfolio, date=datetime.now())

        context._get_price = lambda ticker, date: context.get_price(ticker)
        context._get_historical = lambda ticker, date, periods: context.get_historical_prices(ticker, periods)
        context._get_fundamentals = lambda ticker, date, field: context.get_fundamentals(ticker, field)

        target_portfolio = strategy.execute(context)

        # Persist internal state immediately, regardless of what happens
        # to the returned portfolio afterward (trade execution, etc.) -
        # this is the strategy's own bookkeeping, not the driver's.
        _save_momentum_state(strategy, self.state_file)

        # Capture state for report content
        self._exposure = strategy._current_exposure
        self._market_healthy = strategy._market_state_healthy
        self._last_rebalance_date = strategy._last_rebalance
        self._holdings_snapshot = dict(strategy._holdings)
        self._top_candidates = list(strategy._last_signals[:15]) if strategy._last_signals else []
        self._days_since_rebalance = (
            (context.date - strategy._last_rebalance).days if strategy._last_rebalance else None
        )

        print(f"\n[LiveMomentumTrader] Strategy Result:")
        print(f"  Market Filter: {'HEALTHY' if self._market_healthy else 'DEFENSIVE'}")
        print(f"  Exposure: {self._exposure*100:.0f}%")
        print(f"  Holdings: {len(self._holdings_snapshot)}")
        print(f"  Days since rebalance: {self._days_since_rebalance}")
        print(f"  Target Cash: ${target_portfolio.cash:,.2f}")

        return target_portfolio

    def calculate_trades(
        self,
        current_portfolio: Portfolio,
        target_portfolio: Portfolio,
        total_value: float,
    ) -> List[TradeOrder]:
        """Diff current vs. target and produce the trade list. No external
        rebalance gate needed here (unlike RegimeAdaptiveStrategy) - once
        MomentumStrategy's own _last_rebalance persists correctly (see
        state persistence above), execute() only reselects holdings every
        rebalance_days days, so this diff naturally produces zero trades
        on a no-op day without any additional gating."""
        print("\n[LiveMomentumTrader] Calculating trades...")

        trades: List[TradeOrder] = []
        skipped_tickers: List[str] = []

        all_tickers = set(current_portfolio.positions.keys()) | set(target_portfolio.positions.keys())

        quotes = {}
        for ticker in all_tickers:
            quote = self.connector.get_quote(ticker)
            if quote and quote.last > 0:
                quotes[ticker] = quote.last
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data = yf.download(ticker, period='5d', progress=False)
                        if data is not None and len(data) > 0:
                            quotes[ticker] = float(data['Close'].iloc[-1])
                except Exception:
                    pass

        for ticker in current_portfolio.positions:
            current_shares = current_portfolio.positions[ticker].shares
            target_shares = target_portfolio.positions.get(ticker, Position(ticker, 0, 0)).shares

            if current_shares > target_shares:
                shares_to_sell = int(current_shares - target_shares)
                if shares_to_sell > 0:
                    price = quotes.get(ticker, current_portfolio.positions[ticker].avg_cost)
                    trade_value = shares_to_sell * price
                    if trade_value >= MIN_TRADE_VALUE:
                        trades.append(TradeOrder(
                            ticker=ticker, side="SELL", shares=shares_to_sell,
                            estimated_price=price, estimated_value=trade_value,
                            reason="Reduce position per strategy rebalance",
                        ))

        for ticker in target_portfolio.positions:
            current_shares = current_portfolio.positions.get(ticker, Position(ticker, 0, 0)).shares
            target_shares = target_portfolio.positions[ticker].shares

            if target_shares > current_shares:
                shares_to_buy = int(target_shares - current_shares)
                if shares_to_buy > 0:
                    price = quotes.get(ticker)
                    if price is None:
                        skipped_tickers.append(ticker)
                        continue
                    trade_value = shares_to_buy * price
                    if trade_value >= MIN_TRADE_VALUE:
                        trades.append(TradeOrder(
                            ticker=ticker, side="BUY", shares=shares_to_buy,
                            estimated_price=price, estimated_value=trade_value,
                            reason="Increase position per strategy rebalance",
                        ))

        if skipped_tickers:
            print(f"\n  Skipped (no price available): {', '.join(skipped_tickers)}")

        print(f"\n  Planned Trades: {len(trades)}")
        for trade in trades:
            print(f"    {trade.side:4s} {trade.shares:4d} {trade.ticker:6s} @ ${trade.estimated_price:8.2f} = ${trade.estimated_value:10,.2f}")

        return trades

    def execute_trades(self, trades: List[TradeOrder]) -> List[RebalanceAction]:
        """Execute the planned trades."""
        actions = []

        if not trades:
            print("\n[LiveMomentumTrader] No trades to execute.")
            return actions

        print(f"\n[LiveMomentumTrader] Executing {len(trades)} trades...")

        if self.dry_run:
            print("  *** DRY RUN MODE - Simulating trades ***")

        sell_trades = [t for t in trades if t.side == "SELL"]
        buy_trades = [t for t in trades if t.side == "BUY"]

        for trade in sell_trades + buy_trades:
            print(f"\n  {trade.side} {trade.shares} {trade.ticker}...")
            try:
                side = OrderSide.SELL if trade.side == "SELL" else OrderSide.BUY
                order = self.connector.place_market_order(ticker=trade.ticker, side=side, quantity=trade.shares)
                if order:
                    success = True
                    filled_price = order.filled_price or trade.estimated_price
                    print(f"    Order placed: {order.order_id}")
                    print(f"    Status: {order.status.value}")
                    print(f"    Filled at: ${filled_price:.2f}")
                else:
                    success = False
                    filled_price = trade.estimated_price
                    print(f"    Order failed!")
            except Exception as e:
                success = False
                filled_price = trade.estimated_price
                print(f"    Error: {e}")

            if success:
                actions.append(RebalanceAction(
                    ticker=trade.ticker, action=trade.side, shares=trade.shares,
                    price=filled_price, value=trade.shares * filled_price, reason=trade.reason,
                ))

        self._actions_taken = actions
        return actions

    def build_report(
        self,
        account: Optional[AccountInfo],
        portfolio: Portfolio,
        total_value: float,
        actions: List[RebalanceAction],
        session_failed: bool = False,
        previous_portfolio_value: Optional[float] = None,
    ) -> RebalanceReport:
        """Build a rebalance report with momentum-specific content in
        place of the bear-score/factor-score fields that have no
        momentum equivalent."""
        positions_dict = {}
        for ticker, pos in portfolio.positions.items():
            quote = self.connector.get_quote(ticker) if not session_failed else None
            current_price = quote.last if quote else pos.avg_cost
            market_value = pos.shares * current_price
            pnl_pct = ((current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
            positions_dict[ticker] = {
                'shares': pos.shares, 'value': market_value,
                'weight': market_value / total_value if total_value > 0 else 0,
                'avg_cost': pos.avg_cost, 'pnl_pct': pnl_pct,
            }

        filter_label = "HEALTHY" if self._market_healthy else "DEFENSIVE"
        if actions:
            reason = (f"Rebalanced momentum holdings - market filter {filter_label} "
                      f"({self._exposure*100:.0f}% exposure).")
            hold_reason = None
        else:
            rebalance_days = MOMENTUM_CONFIG['rebalance_days']
            cadence = (f"{self._days_since_rebalance}/{rebalance_days} days since last rebalance"
                       if self._days_since_rebalance is not None else "cadence unknown")
            hold_reason = f"No rebalance due ({cadence}); market filter unchanged ({filter_label}, {self._exposure*100:.0f}% exposure)."
            reason = "Portfolio already aligned with target — no trades needed. " + hold_reason

        holdings_list = []
        for ticker, h in self._holdings_snapshot.items():
            weight = positions_dict.get(ticker, {}).get('weight', 0.0)
            holdings_list.append({'ticker': ticker, 'momentum': h.get('momentum', 0.0), 'weight': weight})
        holdings_list.sort(key=lambda h: h['momentum'], reverse=True)

        selected_tickers = set(self._holdings_snapshot.keys())
        candidates_list = [
            {'ticker': s['ticker'], 'momentum': s['momentum'], 'selected': s['ticker'] in selected_tickers}
            for s in self._top_candidates
        ]

        momentum_summary = {
            'exposure': self._exposure,
            'market_healthy': self._market_healthy,
            'holdings': holdings_list,
            'top_candidates': candidates_list,
            'days_since_rebalance': self._days_since_rebalance,
            'rebalance_days': MOMENTUM_CONFIG['rebalance_days'],
        }

        macro_metrics = fetch_raw_macro_metrics(fred_api_key=self.fred_api_key)

        return RebalanceReport(
            timestamp=datetime.now(),
            strategy_name=STRATEGY_NAME,
            # Required fields with no momentum equivalent - populated with
            # a reasonable semantic fit (exposure/cash split) rather than
            # left meaningless, per the report-design plan.
            bear_score=0.0,
            risk_level="N/A (momentum strategy)",
            allocation_aggressive=self._exposure,
            allocation_defensive=1.0 - self._exposure,
            actions=actions,
            portfolio_value=total_value,
            cash=account.cash if account is not None else 0.0,
            positions=positions_dict,
            rebalance_reason=reason,
            regime_change=bool(actions),
            hold_reason=hold_reason,
            factor_scores=None,
            session_failed=session_failed,
            previous_portfolio_value=previous_portfolio_value,
            strategy_type="momentum",
            macro_metrics=macro_metrics,
            bear_magnitude_pct=None,
            time_to_correction=None,
            momentum_summary=momentum_summary,
        )

    def save_report_to_file(self, report: RebalanceReport) -> str:
        """Save the report to a readable text file."""
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"report_{timestamp}.txt"
        filepath = os.path.join(self.report_dir, filename)

        total_invested = sum(p['value'] for p in report.positions.values())
        portfolio_total = report.portfolio_value
        manual_cash = self.available_cash if self.available_cash else portfolio_total

        positions_data = []
        for ticker, pos in report.positions.items():
            value = float(pos['value'])
            weight_pct = (value / total_invested * 100) if total_invested > 0 else 0
            price = float(pos['value']) / float(pos['shares']) if pos['shares'] > 0 else 0
            scaled_value = manual_cash * (weight_pct / 100)
            scaled_shares = int(scaled_value / price) if price > 0 else 0
            positions_data.append({
                'ticker': ticker, 'weight_pct': weight_pct, 'price': price,
                'original_shares': int(pos['shares']), 'original_value': value,
                'scaled_shares': scaled_shares, 'scaled_value': scaled_shares * price,
            })
        positions_data.sort(key=lambda x: x['weight_pct'], reverse=True)

        lines = []
        lines.append("=" * 70)
        lines.append("MOMENTUM STRATEGY REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Strategy:  {report.strategy_name}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("MOMENTUM SIGNALS")
        lines.append("-" * 70)
        ms = report.momentum_summary or {}
        filter_label = "HEALTHY" if ms.get('market_healthy', True) else "DEFENSIVE"
        lines.append(f"Market Filter: {filter_label}  ({ms.get('exposure', 1.0)*100:.0f}% exposure)")
        days_since = ms.get('days_since_rebalance')
        rebalance_days = ms.get('rebalance_days')
        if days_since is not None and rebalance_days:
            lines.append(f"Rebalance Cadence: {days_since}/{rebalance_days} days since last rebalance")
        lines.append("")
        lines.append(f"{'Ticker':<6} {'Momentum':>10} {'Weight':>8}")
        lines.append("-" * 26)
        for h in sorted(ms.get('holdings', []), key=lambda h: h['momentum'], reverse=True):
            lines.append(f"{h['ticker']:<6} {h['momentum']*100:>9.1f}% {h['weight']*100:>7.1f}%")
        lines.append("")
        lines.append(f"Reason: {report.rebalance_reason}")
        lines.append("")

        from live.macro_snapshot import format_macro_text_block
        lines.extend(format_macro_text_block(report.macro_metrics))

        lines.append("-" * 70)
        lines.append("MANUAL TRADING GUIDE")
        lines.append("-" * 70)
        if self.available_cash:
            lines.append(f"Available Cash: ${manual_cash:,.2f}")
        else:
            lines.append(f"Portfolio Value: ${manual_cash:,.2f}")
        lines.append(f"Number of Positions: {len(positions_data)}")
        lines.append("")

        total_scaled = sum(p['scaled_value'] for p in positions_data)
        lines.append(f"{'Ticker':<6} {'Weight':>8} {'Price':>10} {'Shares':>8} {'Amount':>12}")
        lines.append("-" * 46)
        for pos in positions_data:
            lines.append(
                f"{pos['ticker']:<6} {pos['weight_pct']:>7.1f}% ${pos['price']:>8.2f} "
                f"{pos['scaled_shares']:>8} ${pos['scaled_value']:>10,.2f}"
            )
        lines.append("-" * 46)
        lines.append(f"{'TOTAL':<6} {'100.0%':>8} {'':<10} {'':<8} ${total_scaled:>10,.2f}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("QUICK REFERENCE (Copy-Paste Format)")
        lines.append("-" * 70)
        lines.append("")
        for pos in positions_data:
            lines.append(f"{pos['ticker']}: Buy {pos['scaled_shares']} shares @ ${pos['price']:.2f} = ${pos['scaled_value']:,.2f}")
        lines.append("")

        if report.actions:
            lines.append("-" * 70)
            lines.append("ACTIONS TAKEN")
            lines.append("-" * 70)
            for action in report.actions:
                lines.append(
                    f"{action.action:4} {int(action.shares):>4} {action.ticker:<6} "
                    f"@ ${action.price:>8.2f} = ${action.value:>10,.2f}"
                )
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)
        with open(filepath, 'w') as f:
            f.write(report_text)

        print(f"\n[LiveMomentumTrader] Report saved to: {filepath}")
        print("")
        print(report_text)

        positions_changed = self._check_positions_changed(positions_data)
        if positions_changed or self.positions_file:
            positions_filepath = os.path.join(self.report_dir, f"positions_{timestamp}.txt")
            write_positions_file(
                filepath=positions_filepath,
                positions=positions_data,
                available_cash=manual_cash,
                previous_positions=self.previous_positions if self.previous_positions else None,
                bear_score=0.0,
                risk_level=filter_label,
                allocation=(self._exposure, 1.0 - self._exposure),
            )
            print(f"[LiveMomentumTrader] Positions file saved to: {positions_filepath}")

        return filepath

    def _check_positions_changed(self, new_positions: List[Dict]) -> bool:
        if not self.previous_positions:
            return True
        new_tickers = {p['ticker'] for p in new_positions}
        prev_tickers = set(self.previous_positions.keys())
        if new_tickers != prev_tickers:
            return True
        for pos in new_positions:
            ticker = pos['ticker']
            if ticker in self.previous_positions:
                if pos['scaled_shares'] != self.previous_positions[ticker]['shares']:
                    return True
        return False

    def send_report(self, report: RebalanceReport) -> bool:
        if not self.notifier:
            print("\n[LiveMomentumTrader] No email notifier configured, skipping email.")
            return False
        if not self.notifier.is_configured():
            print("\n[LiveMomentumTrader] Email not configured, skipping email.")
            return False
        print(f"\n[LiveMomentumTrader] Sending email report...")
        success = self.notifier.send_rebalance_report(report)
        print("  Email sent successfully!" if success else "  Failed to send email.")
        return success

    def run(self) -> bool:
        """Run the complete trading cycle."""
        print("=" * 70)
        print(f"LIVE MOMENTUM TRADER")
        print(f"Using: {STRATEGY_NAME}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dry_run:
            print("*** DRY RUN MODE - Using mock connector ***")
        print("=" * 70)

        try:
            print("\n[Step 1/5] Connecting to exchange...")
            session_failed = False
            if not self.connector.is_connected():
                if not self.connector.connect():
                    print("WARNING: Failed to connect to Robinhood. Running signal check only.")
                    session_failed = True
                    self._session_failed = True
                else:
                    print("Connected successfully.")
            else:
                print("Already connected.")

            previous_momentum_state = _load_momentum_state(self.state_file)
            previous_portfolio_value = None  # momentum state doesn't track portfolio_value; report delta uses report files, not this state

            if not session_failed:
                print("\n[Step 2/5] Checking market hours...")
                is_open = self.connector.is_market_open()
                if is_open:
                    print("Market is OPEN.")
                else:
                    print("Market is CLOSED.")
            else:
                print("\n[Step 2/5] Skipping market hours check (session unavailable).")

            if not session_failed:
                print("\n[Step 3/5] Fetching current portfolio...")
                account, portfolio, total_value = self.get_current_portfolio()
            else:
                print("\n[Step 3/5] Skipping portfolio fetch (session unavailable).")
                account = None
                portfolio = Portfolio(cash=0.0, positions={})
                total_value = 0.0

            # Step 4: Run strategy — always runs (uses yfinance, no Robinhood needed).
            # Persists momentum state internally regardless of session_failed,
            # since the market-filter/rebalance-cadence signal itself is
            # still worth tracking even on a day trades can't execute.
            print("\n[Step 4/5] Running strategy...")
            target_portfolio = self.run_strategy(portfolio)

            # Step 5: Calculate and execute trades — no external gate (see
            # calculate_trades docstring); the diff naturally no-ops.
            actions = []
            if session_failed:
                print("\n[Step 5/5] Skipping trades — Robinhood session unavailable.")
            else:
                print("\n[Step 5/5] Calculating and executing trades...")
                trades = self.calculate_trades(portfolio, target_portfolio, total_value)
                if trades:
                    actions = self.execute_trades(trades)
                    account, portfolio, total_value = self.get_current_portfolio()

            print("\nBuilding and saving report...")
            report = self.build_report(
                account, portfolio, total_value, actions,
                session_failed=session_failed,
                previous_portfolio_value=previous_portfolio_value,
            )
            self.save_report_to_file(report)
            self.send_report(report)

            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Market Filter: {'HEALTHY' if self._market_healthy else 'DEFENSIVE'}")
            print(f"Exposure: {self._exposure*100:.0f}%")
            print(f"Trades Executed: {len(actions)}")
            if not session_failed:
                print(f"Portfolio Value: ${total_value:,.2f}")
                print(f"Positions: {len(portfolio.positions)}")
            else:
                print("Portfolio: N/A (Robinhood session failed)")
            print("=" * 70)

            return not session_failed

        except Exception as e:
            print(f"\n[LiveMomentumTrader] ERROR: {e}")
            import traceback
            traceback.print_exc()
            if self.notifier and self.notifier.is_configured():
                self.notifier.send_alert(
                    "Trading Error",
                    f"An error occurred during live momentum trading: {str(e)}"
                )
            return False

        finally:
            if self.connector.is_connected():
                self.connector.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Live Momentum Trading Runner")
    parser.add_argument('--dry-run', action='store_true', default=True,
                         help='Simulate trades with mock connector (default: True)')
    parser.add_argument('--live', action='store_true',
                         help='Execute real trades with Robinhood (use with caution!)')
    parser.add_argument('--email', type=str, default=DEFAULT_EMAIL,
                         help=f'Email address for reports (default: {DEFAULT_EMAIL})')
    parser.add_argument('--no-email', action='store_true', help='Disable email notifications')
    parser.add_argument('--starting-cash', type=float, default=DEFAULT_STARTING_CASH,
                         help=f'Starting cash for dry-run mode (default: ${DEFAULT_STARTING_CASH:,.0f})')
    parser.add_argument('--available-cash', type=float, default=None,
                         help='Your available cash for manual trading')
    parser.add_argument('--positions-file', type=str, default=None,
                         help='Path to positions file to track current holdings and show diffs')
    parser.add_argument('--state-file', type=str, default='live/momentum_trader_state.json',
                         help='Path to JSON state file for MomentumStrategy internal state '
                              '(default: live/momentum_trader_state.json)')
    parser.add_argument('--no-confirm', action='store_true',
                         help='Skip interactive CONFIRM prompt (for CI/automated runs)')

    args = parser.parse_args()
    dry_run = not args.live

    initial_positions = None
    if args.positions_file and os.path.exists(args.positions_file):
        initial_positions = parse_positions_file(args.positions_file)
        if initial_positions:
            print(f"[Setup] Loaded {len(initial_positions)} positions from {args.positions_file}")

    if args.live:
        print("\n" + "!" * 70)
        print("WARNING: LIVE TRADING MODE")
        print("Real trades will be executed with real money!")
        print("!" * 70)
        if not args.no_confirm:
            confirm = input("\nType 'CONFIRM' to proceed: ")
            if confirm != 'CONFIRM':
                print("Aborted.")
                return
        else:
            print("\n[--no-confirm] Skipping interactive confirmation.")

        from connectors import create_robinhood_connector
        connector = create_robinhood_connector()
    else:
        print(f"\n[DryRun] Using MockConnector with ${args.starting_cash:,.0f} starting cash")
        connector = MockConnector(starting_cash=args.starting_cash, initial_positions=initial_positions)

    notifier = None
    if not args.no_email:
        notifier = create_email_notifier(args.email)
        if notifier.is_configured():
            print(f"Email notifications enabled -> {args.email}")
        else:
            print("Email not configured (missing SMTP credentials)")

    trader = LiveMomentumTrader(
        connector=connector,
        email_notifier=notifier,
        dry_run=dry_run,
        available_cash=args.available_cash,
        positions_file=args.positions_file,
        state_file=args.state_file,
        fred_api_key=os.environ.get('FRED_API_KEY'),
    )

    success = trader.run()

    if success:
        print("\nTrading cycle completed successfully.")
    else:
        print("\nTrading cycle failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
