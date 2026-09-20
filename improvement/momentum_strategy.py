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
        market_filter: bool = False,
        market_filter_ticker: str = 'SPY',
        market_filter_sma_days: int = 200,
        market_filter_defensive_weight: float = 0.0,  # fraction to KEEP invested when market is below trend (0 = full cash)
        market_filter_buffer_pct: float = 0.02,  # hysteresis band around the SMA
        market_filter_confirm_days: int = 5,     # consecutive days required before a flip is confirmed
        external_exposure_series: Optional[pd.Series] = None,  # date-indexed 0..1 exposure, overrides the internal SMA filter entirely
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
        self.market_filter = market_filter
        self.market_filter_ticker = market_filter_ticker
        self.market_filter_sma_days = market_filter_sma_days
        self.market_filter_defensive_weight = market_filter_defensive_weight
        self.market_filter_buffer_pct = market_filter_buffer_pct
        self.market_filter_confirm_days = market_filter_confirm_days
        self._market_state_healthy = True   # confirmed state, persists across calls
        self._market_pending_flip = None    # 'healthy'/'unhealthy' candidate awaiting confirmation
        self._market_pending_days = 0
        self.external_exposure_series = external_exposure_series

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

    def _market_exposure(self, context: ExecutionContext) -> float:
        """Market-level absolute trend filter (the asset-class-level half
        of "dual momentum" - Antonacci): 1.0 (fully invested) when the
        market itself is above its own trend SMA, else
        market_filter_defensive_weight (0.0 = full cash by default).

        Checked every call, but with hysteresis: a raw daily SMA-cross
        check, tried first, was catastrophic (-1.6% CAGR / 92.9% MaxDD
        at daily cadence, 4402 trades/20yr) - SPY chops around its own
        200d SMA constantly in real markets, and a binary flip forces a
        full portfolio resize (100%<->30% exposure) on every crossing,
        repeatedly buying high and selling low. Fixed the same way the
        regime classifier's vote signal was fixed earlier this session
        (see classifier.py / PROGRESS.md Iteration 14-16): a buffer band
        around the SMA (market_filter_buffer_pct) so small wiggles right
        at the line don't count as a real cross, plus a persistence
        requirement (market_filter_confirm_days consecutive days on the
        new side) before a flip is actually confirmed and acted on.
        """
        if self.external_exposure_series is not None:
            idx = self.external_exposure_series.index
            ts = pd.Timestamp(context.date)
            pos = idx.searchsorted(ts, side='right') - 1
            if pos < 0:
                return 1.0
            return float(self.external_exposure_series.iloc[pos])

        if not self.market_filter:
            return 1.0
        hist = context.get_historical_prices(self.market_filter_ticker, self.market_filter_sma_days + 5)
        if hist is None or len(hist) < self.market_filter_sma_days * 0.9:
            return self.market_filter_defensive_weight if not self._market_state_healthy else 1.0
        closes = hist['Close'].dropna()
        if len(closes) < 20:
            return self.market_filter_defensive_weight if not self._market_state_healthy else 1.0
        current = float(closes.iloc[-1])
        sma = float(closes.tail(min(self.market_filter_sma_days, len(closes))).mean())

        buf = self.market_filter_buffer_pct
        if current > sma * (1 + buf):
            raw_healthy = True
        elif current < sma * (1 - buf):
            raw_healthy = False
        else:
            raw_healthy = None  # inside the dead-band: no opinion, don't disturb pending confirmation

        target = 'healthy' if raw_healthy is True else ('unhealthy' if raw_healthy is False else None)
        if target is not None and target != ('healthy' if self._market_state_healthy else 'unhealthy'):
            if self._market_pending_flip == target:
                self._market_pending_days += 1
            else:
                self._market_pending_flip = target
                self._market_pending_days = 1
            if self._market_pending_days >= self.market_filter_confirm_days:
                self._market_state_healthy = (target == 'healthy')
                self._market_pending_flip = None
                self._market_pending_days = 0
        else:
            self._market_pending_flip = None
            self._market_pending_days = 0

        return 1.0 if self._market_state_healthy else self.market_filter_defensive_weight

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

        exposure = self._market_exposure(context)

        positions = {}
        invested = 0.0
        if exposure > 0:
            for ticker, holding in self._holdings.items():
                price = context.get_price(ticker)
                if price and holding['shares'] > 0:
                    target_shares = holding['shares'] * exposure
                    positions[ticker] = Position(ticker=ticker, shares=target_shares, avg_cost=holding['entry_price'])
                    invested += target_shares * price

        return Portfolio(cash=total_value - invested, positions=positions)
