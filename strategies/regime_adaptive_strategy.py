"""
Regime-Adaptive Mixed Strategy
==============================

Dynamically blends High Beta Growth and Bear Beta Defensive strategies
based on the MacroMom bear market predictor.

Allocation Logic:
-----------------
Bear Score < 40 (LOW RISK):
    -> 100% High Beta (aggressive)

Bear Score 40-55 (ELEVATED):
    -> 75% High Beta + 25% Bear Beta

Bear Score 55-70 (HIGH):
    -> 40% High Beta + 60% Bear Beta

Bear Score > 70 (EXTREME):
    -> 10% High Beta + 90% Bear Beta

The strategy:
1. Fetches current regime data and computes bear score
2. Determines allocation weights between aggressive/defensive
3. Runs both sub-strategies with their allocated capital
4. Combines into a unified portfolio

Key Benefits:
- Captures upside during bull markets (high beta exposure)
- Rotates to defense before corrections (low bear beta stocks)
- Uses leading indicators (MacroMom gives ~173 days warning)
"""

import sqlite3
import warnings
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# Handle imports for both package and direct execution
try:
    from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext
    from strategies.high_beta_strategy import HighBetaGrowthStrategy
    from strategies.bear_beta_strategy import BearBetaStrategy
except ImportError:
    from strategy import Strategy, Portfolio, Position, ExecutionContext
    from high_beta_strategy import HighBetaGrowthStrategy
    from bear_beta_strategy import BearBetaStrategy

# Import regime module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from regime import (
    compute_bear_score,
    get_risk_recommendation,
    estimate_bear_magnitude,
    estimate_time_to_correction,
)

warnings.filterwarnings('ignore', category=FutureWarning)


class RegimeAdaptiveStrategy(Strategy):
    """
    Regime-Adaptive Mixed Strategy.

    Dynamically allocates between High Beta (aggressive) and
    Bear Beta (defensive) strategies based on market regime.
    """

    # CONSERVATIVE allocation weights (original) - good for risk reduction
    ALLOCATION_THRESHOLDS_CONSERVATIVE = [
        # (max_score, high_beta_weight, bear_beta_weight)
        (40, 1.00, 0.00),   # LOW risk: 100% aggressive
        (50, 0.80, 0.20),   # WATCH: 80% aggressive
        (55, 0.60, 0.40),   # CAUTION: 60% aggressive
        (65, 0.40, 0.60),   # ELEVATED: 40% aggressive
        (75, 0.20, 0.80),   # HIGH: 20% aggressive
        (100, 0.10, 0.90),  # EXTREME: 10% aggressive
    ]

    # AGGRESSIVE allocation weights (optimized for returns)
    # Key insight: Stay aggressive longer, only go defensive at extreme levels
    ALLOCATION_THRESHOLDS = [
        # (max_score, high_beta_weight, bear_beta_weight)
        (55, 1.00, 0.00),   # LOW-MODERATE: 100% aggressive (raised from 40)
        (65, 0.85, 0.15),   # ELEVATED: 85% aggressive (minimal hedge)
        (75, 0.60, 0.40),   # HIGH: 60% aggressive (moderate hedge)
        (100, 0.30, 0.70),  # EXTREME: 30% aggressive (major hedge, but not all-in defensive)
    ]

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 25,
        rebalance_days: int = 30,
        regime_check_days: int = 7,  # Check regime more frequently
        min_realloc_change: float = 0.10,  # Min 10% change to trigger realloc
        high_beta_params: dict = None,
        bear_beta_params: dict = None,
        use_conservative: bool = False,  # Use conservative thresholds
    ):
        self.db_path = db_path
        self.max_positions = max_positions
        self.rebalance_days = rebalance_days
        self.regime_check_days = regime_check_days
        self.min_realloc_change = min_realloc_change
        self.use_conservative = use_conservative

        # Initialize sub-strategies
        hb_params = high_beta_params or {}
        bb_params = bear_beta_params or {}

        self.high_beta_strategy = HighBetaGrowthStrategy(
            db_path=db_path,
            max_positions=max(10, max_positions // 2),
            rebalance_days=rebalance_days,
            **hb_params
        )

        self.bear_beta_strategy = BearBetaStrategy(
            db_path=db_path,
            max_positions=max(10, max_positions // 2),
            rebalance_days=rebalance_days,
            **bb_params
        )

        self._last_regime_check: Optional[datetime] = None
        self._last_rebalance: Optional[datetime] = None
        self._current_allocation: Tuple[float, float] = (1.0, 0.0)  # (high_beta, bear_beta)
        self._bear_score_history: List[float] = []
        self._current_bear_score: float = 0.0
        self._regime_inputs: Optional[dict] = None
        self._momentum_confirmed: bool = False  # Price below 200 DMA
        self._consecutive_high_scores: int = 0  # Require persistence

    def _fetch_regime_inputs_from_context(self, context: ExecutionContext) -> dict:
        """
        Fetch regime data using the backtest context's historical data.

        This ensures we only use data available at the current backtest date,
        not future data (look-ahead bias).
        """
        try:
            # Get historical prices from backtest context (756 days ~ 3 years)
            spy_hist = context.get_historical_prices('SPY', 756)
            vix_hist = context.get_historical_prices('^VIX', 756)

            if spy_hist is None or len(spy_hist) < 200:
                print(f"[RegimeAdaptive] Insufficient SPY data at {context.date}")
                return {}

            # Extract close prices
            sp500 = spy_hist['Close'].squeeze() if hasattr(spy_hist['Close'], 'squeeze') else spy_hist['Close']

            if vix_hist is not None and len(vix_hist) > 50:
                vix = vix_hist['Close'].squeeze() if hasattr(vix_hist['Close'], 'squeeze') else vix_hist['Close']
            else:
                # Fallback: estimate VIX from SPY volatility
                spy_returns = sp500.pct_change()
                vix = spy_returns.rolling(20).std() * np.sqrt(252) * 100
                vix = vix.fillna(20)  # Default VIX of 20

            # Try to get treasury yields
            y10_hist = context.get_historical_prices('^TNX', 756)
            y3m_hist = context.get_historical_prices('^IRX', 756)

            if y10_hist is not None and len(y10_hist) > 50:
                y10 = y10_hist['Close'].squeeze() if hasattr(y10_hist['Close'], 'squeeze') else y10_hist['Close']
            else:
                y10 = pd.Series(dtype=float)

            if y3m_hist is not None and len(y3m_hist) > 50:
                y3m = y3m_hist['Close'].squeeze() if hasattr(y3m_hist['Close'], 'squeeze') else y3m_hist['Close']
            else:
                y3m = pd.Series(dtype=float)

            # VIX 3-month approximation (use smoothed VIX)
            vix_3m = vix.rolling(20).mean() * 0.95

            # Market breadth proxy (SPY above its 200 DMA)
            spy_ma200 = sp500.rolling(200).mean()
            pct_above_200dma = (sp500 > spy_ma200).astype(float) * 100

            # Resample to weekly for regime calculation
            inputs = {
                'sp500': sp500.resample('W-FRI').last().ffill().dropna(),
                'vix': vix.resample('W-FRI').last().ffill().dropna(),
                'vix_3m': vix_3m.resample('W-FRI').last().ffill().dropna(),
                'pct_above_200dma': pct_above_200dma.resample('W-FRI').last().ffill().dropna(),
            }

            # Add yields if available
            if len(y10) > 50:
                inputs['y10'] = y10.resample('W-FRI').last().ffill().dropna()
            if len(y3m) > 50:
                inputs['y3m'] = y3m.resample('W-FRI').last().ffill().dropna()

            return inputs

        except Exception as e:
            print(f"[RegimeAdaptive] Warning: Could not get regime data: {e}")
            return {}

    def _get_allocation_weights(self, bear_score: float) -> Tuple[float, float]:
        """Determine allocation weights based on bear score."""
        thresholds = (self.ALLOCATION_THRESHOLDS_CONSERVATIVE
                      if self.use_conservative
                      else self.ALLOCATION_THRESHOLDS)
        for max_score, hb_weight, bb_weight in thresholds:
            if bear_score <= max_score:
                return (hb_weight, bb_weight)
        return (0.30, 0.70)  # Default (matches EXTREME level)

    def _should_check_regime(self, current_date: datetime) -> bool:
        """Check if regime should be re-evaluated."""
        if self._last_regime_check is None:
            return True
        return (current_date - self._last_regime_check).days >= self.regime_check_days

    def _update_regime(self, context: ExecutionContext) -> None:
        """Update regime assessment using historical data from backtest context."""
        self._regime_inputs = self._fetch_regime_inputs_from_context(context)

        if not self._regime_inputs:
            print(f"[RegimeAdaptive] {context.date.strftime('%Y-%m-%d')}: Using default allocation (insufficient data)")
            return

        # Compute bear score
        bear_score, factor_scores = compute_bear_score(
            self._regime_inputs,
            method="macro_momentum"
        )

        self._current_bear_score = bear_score
        self._bear_score_history.append(bear_score)
        if len(self._bear_score_history) > 52:
            self._bear_score_history = self._bear_score_history[-52:]

        # Get recommendation
        recommendation = get_risk_recommendation(bear_score)

        # ===== MOMENTUM CONFIRMATION =====
        # Only go defensive if price is actually weak (below 50 DMA)
        sp500 = self._regime_inputs.get('sp500')
        if sp500 is not None and len(sp500) >= 10:
            current_price = sp500.iloc[-1]
            ma_50 = sp500.iloc[-10:].mean()  # ~50 days in weekly data
            self._momentum_confirmed = current_price < ma_50
        else:
            self._momentum_confirmed = False

        # Track consecutive high scores for persistence
        if bear_score >= 60:
            self._consecutive_high_scores += 1
        else:
            self._consecutive_high_scores = 0

        # ===== ASYMMETRIC SWITCHING =====
        # Going defensive: Require high score + momentum confirmation + persistence
        # Going aggressive: Return quickly when score drops

        # Get raw allocation based on score
        raw_allocation = self._get_allocation_weights(bear_score)
        old_allocation = self._current_allocation

        # Apply asymmetric logic
        if raw_allocation[0] < old_allocation[0]:
            # Trying to go MORE defensive
            # Require: momentum confirmed AND at least 2 consecutive high scores
            if self._momentum_confirmed and self._consecutive_high_scores >= 2:
                new_allocation = raw_allocation
            else:
                # Stay at current allocation (don't go defensive yet)
                new_allocation = old_allocation
        else:
            # Going MORE aggressive - allow immediately
            new_allocation = raw_allocation

        # Check if allocation change is significant
        allocation_change = abs(new_allocation[0] - old_allocation[0])

        # Compact output for backtest
        date_str = context.date.strftime('%Y-%m-%d')
        momentum_flag = "WEAK" if self._momentum_confirmed else "OK"
        print(f"[Regime {date_str}] Score={bear_score:.1f} Level={recommendation['level']:8s} "
              f"Mom={momentum_flag} Alloc: {new_allocation[0]*100:.0f}%/{new_allocation[1]*100:.0f}% (HB/Def)")

        if allocation_change >= self.min_realloc_change:
            print(f"  -> REALLOCATION: {old_allocation[0]*100:.0f}% -> {new_allocation[0]*100:.0f}% High Beta")
            self._current_allocation = new_allocation

        self._last_regime_check = context.date

    def execute(self, context: ExecutionContext) -> Portfolio:
        """Execute the regime-adaptive strategy."""

        # Check and update regime if needed
        if self._should_check_regime(context.date):
            self._update_regime(context)

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        hb_weight, bb_weight = self._current_allocation

        # Allocate capital to each sub-strategy
        hb_capital = total_value * hb_weight
        bb_capital = total_value * bb_weight

        print(f"[RegimeAdaptive] Total: ${total_value:,.0f}")
        print(f"  High Beta: ${hb_capital:,.0f} ({hb_weight*100:.0f}%)")
        print(f"  Defensive: ${bb_capital:,.0f} ({bb_weight*100:.0f}%)")

        # Create sub-portfolios
        all_positions = {}
        remaining_cash = total_value

        # Execute High Beta strategy if allocated
        if hb_weight > 0.05:
            hb_portfolio = Portfolio(cash=hb_capital, positions={})
            hb_context = ExecutionContext(
                date=context.date,
                portfolio=hb_portfolio,
                get_price_fn=context._get_price,
                get_historical_fn=context._get_historical,
                get_fundamentals_fn=context._get_fundamentals,
            )
            hb_result = self.high_beta_strategy.execute(hb_context)

            for ticker, pos in hb_result.positions.items():
                if ticker in all_positions:
                    # Combine positions
                    existing = all_positions[ticker]
                    total_shares = existing.shares + pos.shares
                    avg_cost = (existing.shares * existing.avg_cost + pos.shares * pos.avg_cost) / total_shares
                    all_positions[ticker] = Position(ticker=ticker, shares=total_shares, avg_cost=avg_cost)
                else:
                    all_positions[ticker] = pos

            remaining_cash -= (hb_capital - hb_result.cash)

        # Execute Bear Beta strategy if allocated
        if bb_weight > 0.05:
            bb_portfolio = Portfolio(cash=bb_capital, positions={})
            bb_context = ExecutionContext(
                date=context.date,
                portfolio=bb_portfolio,
                get_price_fn=context._get_price,
                get_historical_fn=context._get_historical,
                get_fundamentals_fn=context._get_fundamentals,
            )
            bb_result = self.bear_beta_strategy.execute(bb_context)

            for ticker, pos in bb_result.positions.items():
                if ticker in all_positions:
                    existing = all_positions[ticker]
                    total_shares = existing.shares + pos.shares
                    avg_cost = (existing.shares * existing.avg_cost + pos.shares * pos.avg_cost) / total_shares
                    all_positions[ticker] = Position(ticker=ticker, shares=total_shares, avg_cost=avg_cost)
                else:
                    all_positions[ticker] = pos

            remaining_cash -= (bb_capital - bb_result.cash)

        # Ensure cash doesn't go negative due to rounding
        remaining_cash = max(0, remaining_cash)

        print(f"[RegimeAdaptive] Combined portfolio: {len(all_positions)} positions")

        return Portfolio(cash=remaining_cash, positions=all_positions)


class OptimizedRegimeStrategy(RegimeAdaptiveStrategy):
    """
    Optimized Regime Strategy with additional features:

    1. Momentum overlay on regime changes (gradual transitions)
    2. Volatility targeting (reduces exposure when vol spikes)
    3. Position-level optimization (mean-variance within constraints)
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 25,
        target_volatility: float = 0.15,  # 15% annual vol target
        transition_speed: float = 0.5,  # How fast to transition allocations
        **kwargs
    ):
        super().__init__(db_path=db_path, max_positions=max_positions, **kwargs)
        self.target_volatility = target_volatility
        self.transition_speed = transition_speed
        self._target_allocation: Tuple[float, float] = (1.0, 0.0)
        self._portfolio_volatility: float = 0.15

    def _get_allocation_weights(self, bear_score: float) -> Tuple[float, float]:
        """Get target allocation, then apply gradual transition."""
        # Get target allocation from parent
        target = super()._get_allocation_weights(bear_score)

        # Apply transition smoothing
        current = self._current_allocation
        new_hb = current[0] + (target[0] - current[0]) * self.transition_speed
        new_bb = current[1] + (target[1] - current[1]) * self.transition_speed

        return (new_hb, new_bb)

    def _estimate_portfolio_volatility(self, context: ExecutionContext) -> float:
        """Estimate current portfolio volatility."""
        try:
            # Use VIX as a proxy for now
            if self._regime_inputs and 'vix' in self._regime_inputs:
                vix = self._regime_inputs['vix'].iloc[-1]
                return vix / 100  # Convert to decimal
        except:
            pass
        return 0.15  # Default

    def execute(self, context: ExecutionContext) -> Portfolio:
        """Execute with volatility scaling."""
        # Get base portfolio
        portfolio = super().execute(context)

        # Estimate current portfolio volatility
        self._portfolio_volatility = self._estimate_portfolio_volatility(context)

        # Calculate volatility scaling factor
        vol_scale = min(1.0, self.target_volatility / max(0.05, self._portfolio_volatility))

        if vol_scale < 0.9:
            print(f"[OptimizedRegime] Vol scaling: {vol_scale:.2f} "
                  f"(current vol: {self._portfolio_volatility*100:.1f}%, "
                  f"target: {self.target_volatility*100:.1f}%)")

            # Scale down positions
            total_value = portfolio.cash
            for pos in portfolio.positions.values():
                price = context.get_price(pos.ticker)
                total_value += pos.shares * (price if price else pos.avg_cost)

            scaled_positions = {}
            for ticker, pos in portfolio.positions.items():
                scaled_shares = int(pos.shares * vol_scale)
                if scaled_shares > 0:
                    scaled_positions[ticker] = Position(
                        ticker=ticker,
                        shares=float(scaled_shares),
                        avg_cost=pos.avg_cost
                    )

            # Recalculate cash
            invested = sum(
                pos.shares * (context.get_price(pos.ticker) or pos.avg_cost)
                for pos in scaled_positions.values()
            )
            portfolio = Portfolio(cash=total_value - invested, positions=scaled_positions)

        return portfolio


def get_regime_strategies(db_path: str = "fundamentals.sqlite") -> dict:
    """Return all regime-adaptive strategies for testing."""
    return {
        "RegimeAdaptive": RegimeAdaptiveStrategy(db_path),
        "OptimizedRegime": OptimizedRegimeStrategy(db_path),
    }


def demo_regime_allocation():
    """Demonstrate the regime-based allocation logic."""
    import yfinance as yf

    print("=" * 70)
    print("REGIME-ADAPTIVE STRATEGY DEMO")
    print("=" * 70)

    # Fetch regime data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    print("\nFetching market data...")

    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Close']
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']

    if hasattr(sp500, 'squeeze'):
        sp500 = sp500.squeeze()
    if hasattr(vix, 'squeeze'):
        vix = vix.squeeze()

    try:
        vix_3m = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)['Close']
        if hasattr(vix_3m, 'squeeze'):
            vix_3m = vix_3m.squeeze()
    except:
        vix_3m = vix.rolling(20).mean() * 0.95

    try:
        y10 = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
        y3m = yf.download('^IRX', start=start_date, end=end_date, progress=False)['Close']
        if hasattr(y10, 'squeeze'):
            y10 = y10.squeeze()
        if hasattr(y3m, 'squeeze'):
            y3m = y3m.squeeze()
    except:
        y10 = pd.Series(dtype=float)
        y3m = pd.Series(dtype=float)

    spy_ma200 = sp500.rolling(200).mean()
    pct_above_200dma = (sp500 > spy_ma200).astype(float) * 100

    inputs = {
        'sp500': sp500.resample('W-FRI').last().ffill().dropna(),
        'vix': vix.resample('W-FRI').last().ffill().dropna(),
        'vix_3m': vix_3m.resample('W-FRI').last().ffill().dropna() if len(vix_3m) > 0 else vix.resample('W-FRI').last().ffill().dropna() * 0.95,
        'y10': y10.resample('W-FRI').last().ffill().dropna() if len(y10) > 0 else pd.Series(dtype=float),
        'y3m': y3m.resample('W-FRI').last().ffill().dropna() if len(y3m) > 0 else pd.Series(dtype=float),
        'pct_above_200dma': pct_above_200dma.resample('W-FRI').last().ffill().dropna(),
    }

    # Compute bear score
    bear_score, factor_scores = compute_bear_score(inputs, method="macro_momentum")
    recommendation = get_risk_recommendation(bear_score)
    magnitude = estimate_bear_magnitude(factor_scores)
    timing = estimate_time_to_correction(bear_score)

    print("\n" + "-" * 70)
    print("CURRENT REGIME ASSESSMENT")
    print("-" * 70)

    print(f"\nBear Score: {bear_score:.1f}/100")
    print(f"Risk Level: {recommendation['level']}")
    print(f"Expected Drawdown if Correction: {magnitude:.1f}%")
    print(f"Timing: {timing}")

    print(f"\nComponent Scores:")
    for factor, score in factor_scores.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        status = "HIGH" if score > 0.6 else "MED" if score > 0.4 else "LOW"
        print(f"  {factor:25s} [{bar}] {score:.2f} ({status})")

    # Determine allocation
    strategy = RegimeAdaptiveStrategy()
    hb_weight, bb_weight = strategy._get_allocation_weights(bear_score)

    print("\n" + "-" * 70)
    print("RECOMMENDED ALLOCATION")
    print("-" * 70)

    print(f"\n  High Beta (Aggressive):  {hb_weight*100:5.1f}%")
    print(f"  Bear Beta (Defensive):   {bb_weight*100:5.1f}%")

    print("\n  Allocation Thresholds:")
    for max_score, hb, bb in strategy.ALLOCATION_THRESHOLDS:
        marker = "  <-- CURRENT" if bear_score <= max_score and (
            max_score == strategy.ALLOCATION_THRESHOLDS[0][0] or
            bear_score > strategy.ALLOCATION_THRESHOLDS[strategy.ALLOCATION_THRESHOLDS.index((max_score, hb, bb)) - 1][0]
        ) else ""
        print(f"    Score <= {max_score:3.0f}: {hb*100:3.0f}% aggressive / {bb*100:3.0f}% defensive{marker}")

    print("\n" + "-" * 70)
    print("STRATEGY EXPLANATION")
    print("-" * 70)

    print("""
The Regime-Adaptive Strategy:

1. HIGH BETA COMPONENT (Aggressive)
   - Stocks with beta > 1.0, high growth, high profitability
   - Captures upside during bull markets
   - Tech, Consumer Discretionary, Financials

2. BEAR BETA COMPONENT (Defensive)
   - Stocks with bear beta < 0.5 (low correlation to down days)
   - Protects capital during corrections
   - Consumer Staples, Healthcare, Utilities, Gold miners

3. DYNAMIC ALLOCATION
   - Uses MacroMom predictor (173-day lead time on corrections)
   - Gradually shifts from aggressive to defensive as risk rises
   - Rebalances when allocation change > 10%

Example Transition:
   Bull Market (Score 30):  100% High Beta
   Warning (Score 50):       60% High Beta + 40% Defensive
   Correction (Score 70):    10% High Beta + 90% Defensive
""")


if __name__ == "__main__":
    demo_regime_allocation()
