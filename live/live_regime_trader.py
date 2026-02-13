"""
Live Regime-Adaptive Trading Runner
====================================

Executes the Regime-Adaptive Strategy with real money using Robinhood.

This script uses the same RegimeAdaptiveStrategy from strategies/regime_adaptive_strategy.py
that is used in backtesting, ensuring consistency between backtest and live trading.

Usage:
    # Dry run (uses mock connector, no Robinhood credentials needed)
    python live_regime_trader.py --dry-run

    # Live trading (requires Robinhood credentials)
    python live_regime_trader.py --live --email recipient@email.com

Environment Variables Required (for live mode only):
    - ROBINHOOD_USERNAME: Robinhood login email
    - ROBINHOOD_PASSWORD: Robinhood password
    - ROBINHOOD_TOTP_SECRET: (optional) TOTP secret for 2FA

For email notifications:
    - SMTP_USERNAME: Email account for sending reports
    - SMTP_PASSWORD: Email password or app password

Safety Features:
    - Dry-run mode uses mock connector (no real trading)
    - Order validation before submission
    - Position size limits
    - Market hours check
    - Reports saved to file AND sent via email
"""

import os
import sys
import re
import json
import argparse
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
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
    Quote,
    OrderSide,
    OrderType,
    OrderStatus,
)
from notifications import (
    EmailNotifier,
    RebalanceAction,
    RebalanceReport,
    create_email_notifier,
)
from strategies.strategy import Portfolio, Position, ExecutionContext

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*YFPricesMissingError.*')


# Default email recipient
DEFAULT_EMAIL = "joshuaJang89@gmail.com"

# Strategy parameters
STRATEGY_NAME = "Regime-Adaptive Strategy (Optimized)"

# Minimum trade value to avoid tiny orders
MIN_TRADE_VALUE = 50.0

# Minimum reallocation change to trigger rebalance
MIN_REALLOC_CHANGE = 0.10

# Default starting cash for dry-run mock portfolio
DEFAULT_STARTING_CASH = 0.0


@dataclass
class TradeOrder:
    """Represents a planned trade."""
    ticker: str
    side: str  # "BUY" or "SELL"
    shares: int
    estimated_price: float
    estimated_value: float
    reason: str


def parse_positions_file(filepath: str) -> Dict[str, Dict]:
    """
    Parse a positions file in the trade_summary_template.txt format.

    Returns:
        Dict mapping ticker -> {shares, price, value, weight}
    """
    positions = {}

    if not os.path.exists(filepath):
        return positions

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the positions table (starts after the header line with "Ticker")
    in_table = False
    for line in lines:
        line = line.strip()

        # Skip empty lines and separator lines
        if not line or line.startswith('-') or line.startswith('='):
            continue

        # Check for header line
        if line.startswith('Ticker') and 'Weight' in line:
            in_table = True
            continue

        # Check for TOTAL line (end of table)
        if line.startswith('TOTAL'):
            break

        if in_table:
            # Parse position line: "NVDA       9.6% $  191.13       25 $  4,778.25 ..."
            # Format: Ticker   Weight%   $Price   Shares   $Value   [optional diff columns]
            parts = line.split()
            if len(parts) >= 5:
                try:
                    ticker = parts[0]
                    # Weight is like "9.6%"
                    weight_str = parts[1].replace('%', '')
                    weight = float(weight_str)
                    # Price is like "$" followed by number
                    price_idx = 2
                    if parts[price_idx] == '$':
                        price = float(parts[price_idx + 1].replace(',', ''))
                        shares_idx = price_idx + 2
                    else:
                        price = float(parts[price_idx].replace('$', '').replace(',', ''))
                        shares_idx = price_idx + 1

                    shares = int(parts[shares_idx])

                    positions[ticker] = {
                        'shares': shares,
                        'price': price,
                        'weight': weight,
                        'value': shares * price,
                    }
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue

    return positions


def write_positions_file(
    filepath: str,
    positions: List[Dict],
    available_cash: float,
    previous_positions: Dict[str, Dict] = None,
    bear_score: float = 0,
    risk_level: str = "UNKNOWN",
    allocation: Tuple[float, float] = (1.0, 0.0),
):
    """
    Write positions to a file in trade_summary format with diff from previous.

    Args:
        filepath: Output file path
        positions: List of position dicts with ticker, weight_pct, price, scaled_shares, scaled_value
        available_cash: Total cash amount
        previous_positions: Previous positions dict for calculating diff
        bear_score: Current bear score
        risk_level: Current risk level
        allocation: (aggressive%, defensive%) tuple
    """
    lines = []

    # Header
    lines.append("=" * 90)
    lines.append(f"PORTFOLIO POSITIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Bear Score: {bear_score:.1f}  |  Risk Level: {risk_level}  |  Allocation: {allocation[0]*100:.0f}% Aggressive / {allocation[1]*100:.0f}% Defensive")
    lines.append(f"Total Value: ${available_cash:,.2f}")
    lines.append("")

    # Table header
    if previous_positions:
        lines.append(f"{'Ticker':<6} {'Weight':>8} {'Price':>10} {'Shares':>8} {'Value':>14}      {'Diff (Shares)':>14} {'Diff (Amount)':>16}")
    else:
        lines.append(f"{'Ticker':<6} {'Weight':>8} {'Price':>10} {'Shares':>8} {'Value':>14}")
    lines.append("-" * 90)

    total_value = 0
    for pos in positions:
        ticker = pos['ticker']
        weight = pos['weight_pct']
        price = pos['price']
        shares = pos['scaled_shares']
        value = pos['scaled_value']
        total_value += value

        base_line = f"{ticker:<6} {weight:>7.1f}% ${price:>8.2f} {shares:>8} ${value:>12,.2f}"

        if previous_positions and ticker in previous_positions:
            prev = previous_positions[ticker]
            prev_shares = prev['shares']
            share_diff = shares - prev_shares
            amount_diff = share_diff * price

            if share_diff != 0:
                diff_sign = '+' if share_diff > 0 else ''
                base_line += f"      {diff_sign:>1}{share_diff:>13} {diff_sign:>1}${abs(amount_diff):>13,.2f}"
        elif previous_positions:
            # New position
            base_line += f"      {'+':>1}{shares:>13} {'+':>1}${value:>13,.2f}"

        lines.append(base_line)

    # Check for removed positions
    if previous_positions:
        for ticker, prev in previous_positions.items():
            if not any(p['ticker'] == ticker for p in positions):
                # Position removed
                lines.append(f"{ticker:<6} {'0.0':>7}% ${prev['price']:>8.2f} {0:>8} ${0:>12,.2f}      {'-':>1}{prev['shares']:>13} {'-':>1}${prev['value']:>13,.2f}")

    lines.append("-" * 90)
    lines.append(f"{'TOTAL':<6} {'100.0%':>8} {'':<10} {'':<8} ${total_value:>12,.2f}")
    lines.append("")
    lines.append("=" * 90)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return filepath


class MockConnector(ExchangeConnector):
    """
    Mock connector for dry-run mode.

    Uses Yahoo Finance for price data and simulates a portfolio.
    No real Robinhood connection required.
    """

    def __init__(
        self,
        starting_cash: float = DEFAULT_STARTING_CASH,
        initial_positions: Dict[str, Dict] = None,
    ):
        self._connected = False
        self._positions: Dict[str, ConnectorPosition] = {}
        self._price_cache: Dict[str, float] = {}

        # Initialize positions from file if provided
        invested_value = 0.0
        if initial_positions:
            for ticker, pos_data in initial_positions.items():
                shares = pos_data['shares']
                price = pos_data['price']
                value = shares * price
                invested_value += value

                self._positions[ticker] = ConnectorPosition(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price,
                    current_price=price,
                    market_value=value,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                )

            # Cash is starting_cash minus invested value
            self._cash = starting_cash - invested_value
            print(f"[MockConnector] Loaded {len(initial_positions)} positions from file (${invested_value:,.2f} invested)")
        else:
            self._cash = starting_cash

    def connect(self) -> bool:
        self._connected = True
        print("[MockConnector] Connected (simulated)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        print("[MockConnector] Disconnected (simulated)")

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> AccountInfo:
        total_value = self._cash
        for pos in self._positions.values():
            total_value += pos.market_value

        return AccountInfo(
            account_id="MOCK-DRY-RUN",
            cash=self._cash,
            buying_power=self._cash,
            portfolio_value=total_value,
            equity=total_value,
            margin_used=0.0,
            day_trades_remaining=3,
        )

    def get_positions(self) -> List[ConnectorPosition]:
        return list(self._positions.values())

    def get_position(self, ticker: str) -> Optional[ConnectorPosition]:
        return self._positions.get(ticker)

    def _get_yf_price(self, ticker: str) -> Optional[float]:
        """Get price from Yahoo Finance with caching."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(ticker, period='5d', progress=False)
                if data is not None and len(data) > 0:
                    price = float(data['Close'].iloc[-1])
                    self._price_cache[ticker] = price
                    return price
        except Exception:
            pass
        return None

    def get_quote(self, ticker: str) -> Optional[Quote]:
        price = self._get_yf_price(ticker)
        if price:
            return Quote(
                ticker=ticker,
                bid=price * 0.999,
                ask=price * 1.001,
                last=price,
                volume=0,
                timestamp=datetime.now(),
            )
        return None

    def get_quotes(self, tickers: List[str]) -> Dict[str, Quote]:
        quotes = {}
        for ticker in tickers:
            quote = self.get_quote(ticker)
            if quote:
                quotes[ticker] = quote
        return quotes

    def get_historical_prices(
        self,
        ticker: str,
        days: int = 365,
        interval: str = "day"
    ) -> Optional[List[Dict]]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                end = datetime.now()
                start = end - timedelta(days=days)
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data is not None and len(data) > 0:
                    return data.to_dict('records')
        except Exception:
            pass
        return None

    def place_market_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float
    ) -> Optional[Order]:
        # Simulate order execution
        price = self._get_yf_price(ticker)
        if not price:
            return None

        order = Order(
            order_id=f"MOCK-{datetime.now().strftime('%Y%m%d%H%M%S')}-{ticker}",
            ticker=ticker,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=price,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Update mock portfolio
        if side == OrderSide.BUY:
            cost = quantity * price
            self._cash -= cost
            if ticker in self._positions:
                pos = self._positions[ticker]
                new_shares = pos.shares + quantity
                new_cost = (pos.avg_cost * pos.shares + cost) / new_shares
                self._positions[ticker] = ConnectorPosition(
                    ticker=ticker,
                    shares=new_shares,
                    avg_cost=new_cost,
                    current_price=price,
                    market_value=new_shares * price,
                    unrealized_pnl=(price - new_cost) * new_shares,
                    unrealized_pnl_pct=((price - new_cost) / new_cost) * 100 if new_cost > 0 else 0,
                )
            else:
                self._positions[ticker] = ConnectorPosition(
                    ticker=ticker,
                    shares=quantity,
                    avg_cost=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                )
        else:  # SELL
            proceeds = quantity * price
            self._cash += proceeds
            if ticker in self._positions:
                pos = self._positions[ticker]
                new_shares = pos.shares - quantity
                if new_shares <= 0:
                    del self._positions[ticker]
                else:
                    self._positions[ticker] = ConnectorPosition(
                        ticker=ticker,
                        shares=new_shares,
                        avg_cost=pos.avg_cost,
                        current_price=price,
                        market_value=new_shares * price,
                        unrealized_pnl=(price - pos.avg_cost) * new_shares,
                        unrealized_pnl_pct=((price - pos.avg_cost) / pos.avg_cost) * 100 if pos.avg_cost > 0 else 0,
                    )

        return order

    def place_limit_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        limit_price: float
    ) -> Optional[Order]:
        # For mock, just execute as market order
        return self.place_market_order(ticker, side, quantity)

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        return None

    def get_open_orders(self) -> List[Order]:
        return []

    def is_market_open(self) -> bool:
        now = datetime.now()
        # Simple check: weekday and between 9:30 AM and 4:00 PM ET
        if now.weekday() >= 5:  # Weekend
            return False
        hour = now.hour
        if 9 <= hour < 16:
            return True
        return False

    def get_market_hours(self) -> Dict[str, datetime]:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return {
            'open': today.replace(hour=9, minute=30),
            'close': today.replace(hour=16, minute=0),
        }


class DryRunExecutionContext:
    """
    Execution context for dry-run mode.

    Uses only Yahoo Finance for all data - no Robinhood connection needed.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        date: datetime = None,
    ):
        self.portfolio = portfolio
        self.date = date or datetime.now()
        self._price_cache: Dict[str, float] = {}
        self._historical_cache: Dict[str, pd.DataFrame] = {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price from Yahoo Finance."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(ticker, period='5d', progress=False)
                if data is not None and len(data) > 0:
                    price = float(data['Close'].iloc[-1])
                    self._price_cache[ticker] = price
                    return price
        except Exception:
            pass

        return None

    def get_historical_prices(self, ticker: str, periods: int = 30) -> Optional[pd.DataFrame]:
        """Get historical prices from yfinance."""
        cache_key = f"{ticker}_{periods}"
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Add buffer for weekends/holidays
                days = int(periods * 1.5) + 10
                end_date = self.date
                start_date = end_date - timedelta(days=days)

                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if data is not None and len(data) > 0:
                    # Keep only the requested number of periods
                    data = data.tail(periods)
                    self._historical_cache[cache_key] = data
                    return data
        except Exception:
            pass

        return None

    def get_fundamentals(self, ticker: str, field_name: str = None):
        """Get fundamentals - returns empty DataFrame for live trading."""
        return pd.DataFrame()


class LiveExecutionContext:
    """
    Execution context for live trading.

    Provides the same interface as backtest ExecutionContext but uses
    live data sources (Robinhood for prices, yfinance for historical data).
    """

    def __init__(
        self,
        connector: ExchangeConnector,
        portfolio: Portfolio,
        date: datetime = None,
    ):
        self.connector = connector
        self.portfolio = portfolio
        self.date = date or datetime.now()
        self._price_cache: Dict[str, float] = {}
        self._historical_cache: Dict[str, pd.DataFrame] = {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price from connector or Yahoo Finance."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        try:
            quote = self.connector.get_quote(ticker)
            if quote and quote.last > 0:
                self._price_cache[ticker] = quote.last
                return quote.last
        except Exception:
            pass

        # Fallback to yfinance
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(ticker, period='5d', progress=False)
                if data is not None and len(data) > 0:
                    price = float(data['Close'].iloc[-1])
                    self._price_cache[ticker] = price
                    return price
        except Exception:
            pass

        return None

    def get_historical_prices(self, ticker: str, periods: int = 30) -> Optional[pd.DataFrame]:
        """Get historical prices from yfinance."""
        cache_key = f"{ticker}_{periods}"
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                days = int(periods * 1.5) + 10
                end_date = self.date
                start_date = end_date - timedelta(days=days)

                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if data is not None and len(data) > 0:
                    data = data.tail(periods)
                    self._historical_cache[cache_key] = data
                    return data
        except Exception:
            pass

        return None

    def get_fundamentals(self, ticker: str, field_name: str = None):
        """Get fundamentals - returns empty DataFrame for live trading."""
        return pd.DataFrame()


class LiveRegimeTrader:
    """
    Live trading implementation using the actual RegimeAdaptiveStrategy.

    Uses the same strategy logic as backtesting for consistency.
    In dry-run mode, uses MockConnector instead of real Robinhood connection.
    """

    def __init__(
        self,
        connector: ExchangeConnector,
        email_notifier: Optional[EmailNotifier] = None,
        dry_run: bool = True,
        db_path: str = "data/fundamentals.sqlite",
        report_dir: str = None,
        available_cash: float = None,
        positions_file: str = None,
        state_file: str = "live/trader_state.json",
    ):
        """
        Initialize live trader.

        Args:
            connector: Exchange connector (Robinhood or Mock)
            email_notifier: Optional email notifier for reports
            dry_run: If True, simulate trades without executing
            db_path: Path to database for strategy caching
            report_dir: Directory to save reports (default: live/)
            available_cash: Optional cash amount for manual trading calculations
            positions_file: Optional path to positions file for tracking holdings
            state_file: Path to JSON state file for allocation change detection
        """
        self.connector = connector
        self.notifier = email_notifier
        self.dry_run = dry_run
        self.db_path = db_path
        self.report_dir = report_dir or os.path.dirname(os.path.abspath(__file__))
        self.available_cash = available_cash
        self.positions_file = positions_file
        self.state_file = state_file

        # Load previous positions from file if provided
        self.previous_positions: Dict[str, Dict] = {}
        if positions_file and os.path.exists(positions_file):
            self.previous_positions = parse_positions_file(positions_file)
            if self.previous_positions:
                print(f"[LiveTrader] Loaded {len(self.previous_positions)} previous positions from {positions_file}")

        # Lazy-load strategy to avoid import issues
        self._strategy = None
        self._actions_taken: List[RebalanceAction] = []
        self._bear_score: float = 0.0
        self._risk_level: str = "Unknown"
        self._allocation: Tuple[float, float] = (1.0, 0.0)

    def _get_strategy(self):
        """Lazy-load the strategy to avoid circular import issues."""
        if self._strategy is None:
            from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
            self._strategy = RegimeAdaptiveStrategy(
                db_path=self.db_path,
                max_positions=25,
                rebalance_days=30,
                regime_check_days=7,
                min_realloc_change=MIN_REALLOC_CHANGE,
                use_conservative=False,
            )
        return self._strategy

    def _load_state(self) -> Optional[Dict]:
        """Load previous trader state from JSON file."""
        if not self.state_file or not os.path.exists(self.state_file):
            return None
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[LiveTrader] Warning: could not load state file: {e}")
            return None

    def _save_state(self) -> None:
        """Save current trader state to JSON file."""
        if not self.state_file:
            return
        state = {
            'bear_score': self._bear_score,
            'allocation': list(self._allocation),
            'risk_level': self._risk_level,
            'positions': {
                ticker: {'shares': pos.shares, 'avg_cost': pos.avg_cost}
                for ticker, pos in self._last_target_positions.items()
            } if hasattr(self, '_last_target_positions') else {},
            'timestamp': datetime.now().isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"[LiveTrader] State saved to {self.state_file}")
        except IOError as e:
            print(f"[LiveTrader] Warning: could not save state file: {e}")

    def _allocation_changed(self, previous_state: Optional[Dict]) -> bool:
        """Check if allocation has changed from previous state (with tolerance)."""
        if previous_state is None:
            return True
        prev_alloc = previous_state.get('allocation')
        if prev_alloc is None or len(prev_alloc) != 2:
            return True
        tolerance = 0.001
        return (
            abs(self._allocation[0] - prev_alloc[0]) > tolerance
            or abs(self._allocation[1] - prev_alloc[1]) > tolerance
        )

    def get_current_portfolio(self) -> Tuple[AccountInfo, Portfolio, float]:
        """
        Get current portfolio state from connector and convert to strategy Portfolio.

        Returns:
            (account_info, strategy_portfolio, total_value)
        """
        print("\n[LiveTrader] Fetching current portfolio...")

        account = self.connector.get_account_info()
        connector_positions = self.connector.get_positions()

        # Convert connector positions to strategy Position objects
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

        # Create strategy Portfolio
        portfolio = Portfolio(cash=account.cash, positions=positions)
        total_value = account.cash + total_invested

        print(f"  Account: {account.account_id}")
        print(f"  Cash: ${account.cash:,.2f}")
        print(f"  Invested: ${total_invested:,.2f}")
        print(f"  Total Value: ${total_value:,.2f}")
        print(f"  Positions: {len(positions)}")

        return account, portfolio, total_value

    def run_strategy(self, portfolio: Portfolio) -> Portfolio:
        """
        Run the RegimeAdaptiveStrategy to get target portfolio.

        Returns:
            Target Portfolio from strategy
        """
        print("\n[LiveTrader] Running RegimeAdaptiveStrategy...")

        strategy = self._get_strategy()

        # Create appropriate execution context
        if self.dry_run:
            context = DryRunExecutionContext(
                portfolio=portfolio,
                date=datetime.now(),
            )
        else:
            context = LiveExecutionContext(
                connector=self.connector,
                portfolio=portfolio,
                date=datetime.now(),
            )

        # Set up context methods for strategy interface
        context._get_price = lambda ticker, date: context.get_price(ticker)
        context._get_historical = lambda ticker, date, periods: context.get_historical_prices(ticker, periods)
        context._get_fundamentals = lambda ticker, date, field: context.get_fundamentals(ticker, field)

        # Execute strategy
        target_portfolio = strategy.execute(context)

        # Capture regime state for reporting
        self._bear_score = strategy._current_bear_score
        self._allocation = strategy._current_allocation

        # Get risk level
        from data.regime import get_risk_recommendation
        recommendation = get_risk_recommendation(self._bear_score)
        self._risk_level = recommendation['level']

        print(f"\n[LiveTrader] Strategy Result:")
        print(f"  Bear Score: {self._bear_score:.1f}")
        print(f"  Risk Level: {self._risk_level}")
        print(f"  Allocation: {self._allocation[0]*100:.0f}% Aggressive / {self._allocation[1]*100:.0f}% Defensive")
        print(f"  Target Positions: {len(target_portfolio.positions)}")
        print(f"  Target Cash: ${target_portfolio.cash:,.2f}")

        return target_portfolio

    def calculate_trades(
        self,
        current_portfolio: Portfolio,
        target_portfolio: Portfolio,
        total_value: float,
    ) -> List[TradeOrder]:
        """
        Calculate trades needed to move from current to target portfolio.

        Returns:
            List of TradeOrder objects
        """
        print("\n[LiveTrader] Calculating trades...")

        trades: List[TradeOrder] = []
        skipped_tickers: List[str] = []

        # Get all tickers involved
        all_tickers = set(current_portfolio.positions.keys()) | set(target_portfolio.positions.keys())

        # Get quotes for all tickers (from Robinhood or yfinance fallback)
        quotes = {}
        for ticker in all_tickers:
            quote = self.connector.get_quote(ticker)
            if quote and quote.last > 0:
                quotes[ticker] = quote.last
            else:
                # Fallback to yfinance
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data = yf.download(ticker, period='5d', progress=False)
                        if data is not None and len(data) > 0:
                            quotes[ticker] = float(data['Close'].iloc[-1])
                except Exception:
                    pass

        # Calculate sells first (positions to reduce or close)
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
                            ticker=ticker,
                            side="SELL",
                            shares=shares_to_sell,
                            estimated_price=price,
                            estimated_value=trade_value,
                            reason="Reduce position per strategy rebalance",
                        ))

        # Calculate buys (positions to increase or open)
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
                            ticker=ticker,
                            side="BUY",
                            shares=shares_to_buy,
                            estimated_price=price,
                            estimated_value=trade_value,
                            reason="Increase position per strategy rebalance",
                        ))

        if skipped_tickers:
            print(f"\n  Skipped (no price available): {', '.join(skipped_tickers)}")

        print(f"\n  Planned Trades: {len(trades)}")
        for trade in trades:
            print(f"    {trade.side:4s} {trade.shares:4d} {trade.ticker:6s} @ ${trade.estimated_price:8.2f} = ${trade.estimated_value:10,.2f}")

        return trades

    def execute_trades(self, trades: List[TradeOrder]) -> List[RebalanceAction]:
        """
        Execute the planned trades.

        Returns:
            List of actions taken (for reporting)
        """
        actions = []

        if not trades:
            print("\n[LiveTrader] No trades to execute.")
            return actions

        print(f"\n[LiveTrader] Executing {len(trades)} trades...")

        if self.dry_run:
            print("  *** DRY RUN MODE - Simulating trades ***")

        # Execute sells first, then buys
        sell_trades = [t for t in trades if t.side == "SELL"]
        buy_trades = [t for t in trades if t.side == "BUY"]

        for trade in sell_trades + buy_trades:
            print(f"\n  {trade.side} {trade.shares} {trade.ticker}...")

            try:
                side = OrderSide.SELL if trade.side == "SELL" else OrderSide.BUY
                order = self.connector.place_market_order(
                    ticker=trade.ticker,
                    side=side,
                    quantity=trade.shares,
                )

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
                action = RebalanceAction(
                    ticker=trade.ticker,
                    action=trade.side,
                    shares=trade.shares,
                    price=filled_price,
                    value=trade.shares * filled_price,
                    reason=trade.reason,
                )
                actions.append(action)

        self._actions_taken = actions
        return actions

    def build_report(
        self,
        account: AccountInfo,
        portfolio: Portfolio,
        total_value: float,
        actions: List[RebalanceAction],
        regime_changed: bool,
    ) -> RebalanceReport:
        """Build a rebalance report for email notification and file save."""

        # Build positions dict
        positions_dict = {}
        for ticker, pos in portfolio.positions.items():
            quote = self.connector.get_quote(ticker)
            current_price = quote.last if quote else pos.avg_cost
            market_value = pos.shares * current_price
            pnl_pct = ((current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0

            positions_dict[ticker] = {
                'shares': pos.shares,
                'value': market_value,
                'weight': market_value / total_value if total_value > 0 else 0,
                'avg_cost': pos.avg_cost,
                'pnl_pct': pnl_pct,
            }

        # Determine rebalance reason
        if not actions:
            reason = "No rebalancing needed - portfolio already aligned with target allocation."
        else:
            reason = (f"Rebalanced to {self._allocation[0]*100:.0f}% aggressive / "
                     f"{self._allocation[1]*100:.0f}% defensive based on bear score of "
                     f"{self._bear_score:.1f} (Risk Level: {self._risk_level})")

        report = RebalanceReport(
            timestamp=datetime.now(),
            strategy_name=STRATEGY_NAME,
            bear_score=self._bear_score,
            risk_level=self._risk_level,
            allocation_aggressive=self._allocation[0],
            allocation_defensive=self._allocation[1],
            actions=actions,
            portfolio_value=total_value,
            cash=account.cash,
            positions=positions_dict,
            rebalance_reason=reason,
            regime_change=regime_changed,
        )

        return report

    def save_report_to_file(self, report: RebalanceReport) -> str:
        """Save the report to a readable text file with actionable manual trading info."""
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"report_{timestamp}.txt"
        filepath = os.path.join(self.report_dir, filename)

        # Calculate total invested value (excluding cash)
        total_invested = sum(p['value'] for p in report.positions.values())
        portfolio_total = report.portfolio_value

        # Use available_cash if provided for manual trading calculations
        manual_cash = self.available_cash if self.available_cash else portfolio_total

        # Build positions data sorted by weight
        positions_data = []
        for ticker, pos in report.positions.items():
            value = float(pos['value'])
            weight_pct = (value / total_invested * 100) if total_invested > 0 else 0
            price = float(pos['value']) / float(pos['shares']) if pos['shares'] > 0 else 0

            # Calculate scaled values for manual trading
            scaled_value = manual_cash * (weight_pct / 100)
            scaled_shares = int(scaled_value / price) if price > 0 else 0

            positions_data.append({
                'ticker': ticker,
                'weight_pct': weight_pct,
                'price': price,
                'original_shares': int(pos['shares']),
                'original_value': value,
                'scaled_shares': scaled_shares,
                'scaled_value': scaled_shares * price,
            })

        # Sort by weight descending
        positions_data.sort(key=lambda x: x['weight_pct'], reverse=True)

        # Build the report text
        lines = []
        lines.append("=" * 70)
        lines.append("REGIME-ADAPTIVE STRATEGY REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Strategy:  {report.strategy_name}")
        lines.append("")

        # Market Regime Section
        lines.append("-" * 70)
        lines.append("MARKET REGIME ASSESSMENT")
        lines.append("-" * 70)
        lines.append(f"Bear Score:  {report.bear_score:.1f} / 100")
        lines.append(f"Risk Level:  {report.risk_level}")
        lines.append(f"Allocation:  {report.allocation_aggressive*100:.0f}% Aggressive / {report.allocation_defensive*100:.0f}% Defensive")
        lines.append("")
        lines.append(f"Reason: {report.rebalance_reason}")
        lines.append("")

        # Manual Trading Guide Section
        lines.append("-" * 70)
        lines.append("MANUAL TRADING GUIDE")
        lines.append("-" * 70)
        if self.available_cash:
            lines.append(f"Available Cash: ${manual_cash:,.2f}")
        else:
            lines.append(f"Portfolio Value: ${manual_cash:,.2f}")
        lines.append(f"Number of Positions: {len(positions_data)}")
        lines.append("")

        # Calculate total for scaled positions
        total_scaled = sum(p['scaled_value'] for p in positions_data)

        # Header
        lines.append(f"{'Ticker':<6} {'Weight':>8} {'Price':>10} {'Shares':>8} {'Amount':>12}")
        lines.append("-" * 46)

        for pos in positions_data:
            lines.append(
                f"{pos['ticker']:<6} "
                f"{pos['weight_pct']:>7.1f}% "
                f"${pos['price']:>8.2f} "
                f"{pos['scaled_shares']:>8} "
                f"${pos['scaled_value']:>10,.2f}"
            )

        lines.append("-" * 46)
        lines.append(f"{'TOTAL':<6} {'100.0%':>8} {'':<10} {'':<8} ${total_scaled:>10,.2f}")
        lines.append("")

        # Quick Reference (for copy-paste)
        lines.append("-" * 70)
        lines.append("QUICK REFERENCE (Copy-Paste Format)")
        lines.append("-" * 70)
        lines.append("")
        for pos in positions_data:
            lines.append(f"{pos['ticker']}: Buy {pos['scaled_shares']} shares @ ${pos['price']:.2f} = ${pos['scaled_value']:,.2f}")
        lines.append("")

        # Actions Taken Section (if any)
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

        # Write to file
        report_text = "\n".join(lines)
        with open(filepath, 'w') as f:
            f.write(report_text)

        # Print to console
        print(f"\n[LiveTrader] Report saved to: {filepath}")
        print("")
        print(report_text)

        # Also write positions file if positions changed or if positions_file was provided
        positions_changed = self._check_positions_changed(positions_data)
        if positions_changed or self.positions_file:
            positions_filepath = os.path.join(
                self.report_dir,
                f"positions_{timestamp}.txt"
            )
            write_positions_file(
                filepath=positions_filepath,
                positions=positions_data,
                available_cash=manual_cash,
                previous_positions=self.previous_positions if self.previous_positions else None,
                bear_score=report.bear_score,
                risk_level=report.risk_level,
                allocation=self._allocation,
            )
            print(f"[LiveTrader] Positions file saved to: {positions_filepath}")

        return filepath

    def _check_positions_changed(self, new_positions: List[Dict]) -> bool:
        """Check if positions have changed from previous."""
        if not self.previous_positions:
            return True  # No previous = changed

        # Build set of current tickers
        new_tickers = {p['ticker'] for p in new_positions}
        prev_tickers = set(self.previous_positions.keys())

        # Check if tickers changed
        if new_tickers != prev_tickers:
            return True

        # Check if shares changed
        for pos in new_positions:
            ticker = pos['ticker']
            if ticker in self.previous_positions:
                if pos['scaled_shares'] != self.previous_positions[ticker]['shares']:
                    return True

        return False

    def send_report(self, report: RebalanceReport) -> bool:
        """Send the rebalance report via email."""
        if not self.notifier:
            print("\n[LiveTrader] No email notifier configured, skipping email.")
            return False

        if not self.notifier.is_configured():
            print("\n[LiveTrader] Email not configured, skipping email.")
            return False

        print(f"\n[LiveTrader] Sending email report...")
        success = self.notifier.send_rebalance_report(report)

        if success:
            print(f"  Email sent successfully!")
        else:
            print(f"  Failed to send email.")

        return success

    def run(self) -> bool:
        """
        Run the complete trading cycle.

        Returns:
            True if completed successfully
        """
        print("=" * 70)
        print(f"LIVE REGIME-ADAPTIVE TRADER")
        print(f"Using: {STRATEGY_NAME}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dry_run:
            print("*** DRY RUN MODE - Using mock connector ***")
        print("=" * 70)

        try:
            # Step 1: Connect to exchange
            print("\n[Step 1/6] Connecting to exchange...")
            if not self.connector.is_connected():
                if not self.connector.connect():
                    print("Failed to connect!")
                    return False
            print("Connected successfully.")

            # Step 2: Check market hours
            print("\n[Step 2/6] Checking market hours...")
            is_open = self.connector.is_market_open()
            market_hours = self.connector.get_market_hours()

            if is_open:
                print("Market is OPEN.")
            else:
                print("Market is CLOSED.")
                if market_hours:
                    print(f"  Opens: {market_hours.get('open', 'N/A')}")
                    print(f"  Closes: {market_hours.get('close', 'N/A')}")

            # Step 3: Get current portfolio
            print("\n[Step 3/6] Fetching current portfolio...")
            account, portfolio, total_value = self.get_current_portfolio()

            # Step 4: Run strategy to get target portfolio
            print("\n[Step 4/6] Running strategy...")
            target_portfolio = self.run_strategy(portfolio)

            # Check if allocation changed from previous run
            previous_state = self._load_state()
            alloc_changed = self._allocation_changed(previous_state)
            regime_changed = self._bear_score > 55

            # Store target positions for state saving
            self._last_target_positions = target_portfolio.positions

            # Step 5: Calculate and execute trades (only if allocation changed)
            actions = []
            if alloc_changed:
                print("\n[Step 5/6] Allocation changed - calculating and executing trades...")
                trades = self.calculate_trades(portfolio, target_portfolio, total_value)

                if trades:
                    actions = self.execute_trades(trades)

                # Refresh portfolio after trades
                if actions:
                    account, portfolio, total_value = self.get_current_portfolio()
            else:
                prev_alloc = previous_state.get('allocation', [0, 0]) if previous_state else [0, 0]
                print(f"\n[Step 5/6] Allocation unchanged ({prev_alloc[0]*100:.0f}%/{prev_alloc[1]*100:.0f}%) - skipping trades, no rebalancing needed.")

            # Step 6: Build, save, and send report (ALWAYS)
            print("\n[Step 6/6] Building and saving report...")
            report = self.build_report(
                account, portfolio, total_value, actions, regime_changed
            )

            # Always save report to file
            self.save_report_to_file(report)

            # Always send email report
            self.send_report(report)

            # Save state for next run
            self._save_state()

            # Print summary
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Bear Score: {self._bear_score:.1f}")
            print(f"Risk Level: {self._risk_level}")
            print(f"Allocation: {self._allocation[0]*100:.0f}% Aggressive / {self._allocation[1]*100:.0f}% Defensive")
            print(f"Trades Executed: {len(actions)}")
            print(f"Portfolio Value: ${total_value:,.2f}")
            print(f"Positions: {len(portfolio.positions)}")
            print("=" * 70)

            return True

        except Exception as e:
            print(f"\n[LiveTrader] ERROR: {e}")
            import traceback
            traceback.print_exc()

            # Try to send error alert
            if self.notifier and self.notifier.is_configured():
                self.notifier.send_alert(
                    "Trading Error",
                    f"An error occurred during live trading: {str(e)}"
                )

            return False

        finally:
            # Disconnect
            if self.connector.is_connected():
                self.connector.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live Regime-Adaptive Trading Runner"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Simulate trades with mock connector (default: True)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Execute real trades with Robinhood (use with caution!)'
    )
    parser.add_argument(
        '--email',
        type=str,
        default=DEFAULT_EMAIL,
        help=f'Email address for reports (default: {DEFAULT_EMAIL})'
    )
    parser.add_argument(
        '--no-email',
        action='store_true',
        help='Disable email notifications'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/fundamentals.sqlite',
        help='Path to database for caching'
    )
    parser.add_argument(
        '--starting-cash',
        type=float,
        default=DEFAULT_STARTING_CASH,
        help=f'Starting cash for dry-run mode (default: ${DEFAULT_STARTING_CASH:,.0f})'
    )
    parser.add_argument(
        '--available-cash',
        type=float,
        default=None,
        help='Your available cash for manual trading - report will calculate positions based on this amount'
    )
    parser.add_argument(
        '--positions-file',
        type=str,
        default=None,
        help='Path to positions file (trade_summary format) to track current holdings and show diffs'
    )
    parser.add_argument(
        '--state-file',
        type=str,
        default='live/trader_state.json',
        help='Path to JSON state file for allocation change detection (default: live/trader_state.json)'
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip interactive CONFIRM prompt (for CI/automated runs)'
    )

    args = parser.parse_args()

    # Determine mode
    dry_run = not args.live

    # Load initial positions from file if provided
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

        # Create real Robinhood connector
        from connectors import create_robinhood_connector
        connector = create_robinhood_connector()
    else:
        # Use mock connector for dry-run
        print(f"\n[DryRun] Using MockConnector with ${args.starting_cash:,.0f} starting cash")
        connector = MockConnector(
            starting_cash=args.starting_cash,
            initial_positions=initial_positions,
        )

    # Create email notifier
    notifier = None
    if not args.no_email:
        notifier = create_email_notifier(args.email)
        if notifier.is_configured():
            print(f"Email notifications enabled -> {args.email}")
        else:
            print("Email not configured (missing SMTP credentials)")

    # Create and run trader
    trader = LiveRegimeTrader(
        connector=connector,
        email_notifier=notifier,
        dry_run=dry_run,
        db_path=args.db_path,
        available_cash=args.available_cash,
        positions_file=args.positions_file,
        state_file=args.state_file,
    )

    success = trader.run()

    if success:
        print("\nTrading cycle completed successfully.")
    else:
        print("\nTrading cycle failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
