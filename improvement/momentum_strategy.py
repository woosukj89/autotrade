"""
Redesigned stock-picker: cross-sectional price momentum with an absolute-
trend filter ("dual momentum" - Antonacci), instead of the beta/quality/
growth fundamentals screen used all prior session.

Why: every mechanism tried this session (binary timing, graduated
exposure, options collar, factor rotation, MacroMom's own overlay)
manages risk on top of the SAME underlying stock-picker - none of them
add return above its honest, point-in-time-corrected ceiling (16.4%
CAGR unhedged, see PROGRESS.md Iteration 22). Hitting 25%+ CAGR needs a
better return source, not a better way to time entry/exit around the
existing one. Momentum is a well-documented, historically strong
risk-adjusted-return factor, and unlike the fundamentals-based picker it
needs NO fundamentals data at all - purely price history via
context.get_historical_prices(), which is already point-in-time correct
(the backtest engine truncates to `date <= simulated_date`). This
sidesteps the entire yfinance-.info / SEC-EDGAR reliability problem this
session spent so much effort on - a genuine practical advantage, not
just a return-source change.

Selection each rebalance:
  1. Compute "12-1 month" momentum for each candidate: trailing 12-month
     return EXCLUDING the most recent month (standard momentum-literature
     construction - skips short-term reversal effects that plague a
     naive trailing-12-month return).
  2. Absolute trend filter: only keep candidates currently above their
     own `trend_sma_days`-day moving average (Antonacci's "dual
     momentum" - cross-sectional ranking alone still buys stocks in a
     downtrend if they're merely "the best of a bad bunch"; the absolute
     filter prevents that, and is the single most load-bearing change
     for controlling momentum's well-documented crash risk).
  3. Rank survivors by momentum score, keep the top `max_positions`.
  4. Optional inverse-volatility position sizing (risk-parity-style) in
     place of equal weighting - de-emphasizes the single most volatile
     names without excluding them outright.
"""
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from strategy import Strategy, Portfolio, Position, ExecutionContext
from yahoo_data import YahooDataProvider


class MomentumStrategy(Strategy):
    def __init__(
        self,
        max_positions: int = 15,
        lookback_days: int = 252,       # ~12 months
        skip_recent_days: int = 21,     # skip most recent ~1 month (12-1 momentum)
        trend_sma_days: int = 200,      # absolute-trend filter window
        rebalance_days: int = 30,
        max_sector_weight: float = 0.40,
        max_position_weight: float = 0.15,
        vol_scale_weighting: bool = False,
        vol_lookback_days: int = 60,
        min_price: float = 5.0,
    ):
        self.max_positions = max_positions
        self.lookback_days = lookback_days
        self.skip_recent_days = skip_recent_days
        self.trend_sma_days = trend_sma_days
        self.rebalance_days = rebalance_days
        self.max_sector_weight = max_sector_weight
        self.max_position_weight = max_position_weight
        self.vol_scale_weighting = vol_scale_weighting
        self.vol_lookback_days = vol_lookback_days
        self.min_price = min_price

        self._eligible_tickers: Optional[set] = None
        self._ticker_sectors: Dict[str, str] = {}
        self._holdings: Dict[str, dict] = {}
        self._last_rebalance = None

    def _load_eligible_tickers(self):
        provider = YahooDataProvider(cache_db=None)
        self._eligible_tickers = provider.get_high_beta_universe()

    def _compute_signal(self, ticker: str, context: ExecutionContext) -> Optional[dict]:
        needed = self.lookback_days + self.skip_recent_days + 5
        hist = context.get_historical_prices(ticker, max(needed, self.trend_sma_days + 5))
        if hist is None or len(hist) < max(self.lookback_days + self.skip_recent_days, self.trend_sma_days) * 0.9:
            return None
        closes = hist['Close'].dropna()
        if len(closes) < 30:
            return None
        current_price = float(closes.iloc[-1])
        if current_price < self.min_price:
            return None

        # 12-1 momentum: return from (end - lookback - skip) to (end - skip)
        skip = min(self.skip_recent_days, len(closes) - 2)
        lookback = min(self.lookback_days, len(closes) - skip - 1)
        if lookback < 20:
            return None
        p_end = float(closes.iloc[-1 - skip])
        p_start = float(closes.iloc[-1 - skip - lookback])
        if p_start <= 0:
            return None
        momentum = (p_end / p_start) - 1.0

        # Absolute trend filter
        sma_window = min(self.trend_sma_days, len(closes))
        sma = float(closes.tail(sma_window).mean())
        above_trend = current_price > sma

        # Volatility (for optional inverse-vol weighting)
        rets = closes.pct_change().dropna()
        vol_window = min(self.vol_lookback_days, len(rets))
        volatility = float(rets.tail(vol_window).std()) if vol_window > 5 else None

        return {
            'ticker': ticker, 'momentum': momentum, 'above_trend': above_trend,
            'price': current_price, 'volatility': volatility,
        }

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        needs_rebalance = (
            self._last_rebalance is None
            or (context.date - self._last_rebalance).days >= self.rebalance_days
        )

        if needs_rebalance:
            print(f"[Momentum] Rebalancing on {context.date.strftime('%Y-%m-%d')}...")
            candidates = list(self._eligible_tickers)
            signals = []
            for ticker in candidates:
                sig = self._compute_signal(ticker, context)
                if sig and sig['above_trend']:
                    signals.append(sig)

            signals.sort(key=lambda s: s['momentum'], reverse=True)
            print(f"[Momentum] {len(signals)}/{len(candidates)} candidates above trend, "
                  f"top momentum: {[round(s['momentum']*100,1) for s in signals[:5]]}")

            self._holdings = {}
            sector_weights: Dict[str, float] = {}
            selected = []
            for sig in signals:
                if len(selected) >= self.max_positions:
                    break
                selected.append(sig)

            if self.vol_scale_weighting and selected:
                inv_vols = [1.0 / s['volatility'] if s['volatility'] and s['volatility'] > 0 else 0.0 for s in selected]
                total_inv_vol = sum(inv_vols) or 1.0
                raw_weights = [iv / total_inv_vol for iv in inv_vols]
            else:
                raw_weights = [1.0 / len(selected)] * len(selected) if selected else []

            for sig, weight in zip(selected, raw_weights):
                weight = min(weight, self.max_position_weight)
                shares = int((weight * total_value) / sig['price']) if sig['price'] > 0 else 0
                if shares > 0:
                    self._holdings[sig['ticker']] = {
                        'shares': shares, 'entry_price': sig['price'], 'momentum': sig['momentum'],
                    }

            self._last_rebalance = context.date
            if self._holdings:
                print(f"[Momentum] Selected {len(self._holdings)} positions:")
                for t, h in list(self._holdings.items())[:10]:
                    print(f"  {t}: mom={h['momentum']*100:.1f}%")

        positions = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price and holding['shares'] > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(holding['shares']), avg_cost=holding['entry_price'])
                invested += holding['shares'] * price

        return Portfolio(cash=total_value - invested, positions=positions)
