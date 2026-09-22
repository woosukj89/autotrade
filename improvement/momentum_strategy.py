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
from sector_map import TICKER_SECTOR


class MomentumStrategy(Strategy):
    # Broad, liquid ETFs added to the candidate pool when include_etf_universe
    # is set - not a new switching mechanism (everything tried in Iteration
    # 25-27 that added a synchronized regime-switch failed on transaction
    # cost), just a wider, naturally less-correlated pool for the SAME
    # already-working momentum+trend-filter+ranking process to draw from.
    ETF_UNIVERSE = {
        # Broad US
        'SPY', 'QQQ', 'IWM', 'DIA', 'MDY',
        # US sector SPDRs
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLC', 'XLRE',
        # International / regional
        'EFA', 'EEM', 'VGK', 'VPL', 'EWJ', 'EWZ', 'INDA', 'FXI',
        # Real estate (equity REITs, trades like a stock)
        'VNQ',
    }

    def __init__(
        self,
        max_positions: int = 15,
        include_etf_universe: bool = False,
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
        stop_loss_pct: Optional[float] = None,  # trailing stop per position (e.g. 0.15 = exit if 15% below its peak since entry) - CANSLIM/turtle-trader style, bottom-up not portfolio-level
        atr_stop_multiplier: Optional[float] = None,  # volatility-ADJUSTED trailing stop: exit if price < peak - multiplier*ATR (the actual Turtle Trader technique). Takes precedence over stop_loss_pct if both set.
        vol_target: Optional[float] = None,  # Barroso & Santa-Clara style: target annualized portfolio vol (e.g. 0.20); exposure = clip(vol_target/realized_vol, min, max). Continuous/smooth, not a binary regime switch.
        vol_target_lookback_days: int = 20,
        vol_target_min_exposure: float = 0.3,
        vol_target_max_exposure: float = 1.0,  # capped at 1.0 - long-only, no leverage
        vol_target_tolerance: float = 0.10,  # only re-trade when target exposure drifts more than this from what's currently implemented
    ):
        self.max_positions = max_positions
        self.include_etf_universe = include_etf_universe
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
        self.stop_loss_pct = stop_loss_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.vol_target = vol_target
        self.vol_target_lookback_days = vol_target_lookback_days
        self.vol_target_min_exposure = vol_target_min_exposure
        self.vol_target_max_exposure = vol_target_max_exposure
        self.vol_target_tolerance = vol_target_tolerance
        self._value_history: List[float] = []
        self._implemented_vol_exposure = 1.0

        self._eligible_tickers: Optional[set] = None
        self._ticker_sectors: Dict[str, str] = {}
        self._holdings: Dict[str, dict] = {}
        self._last_rebalance = None
        self._last_signals: List[dict] = []   # full ranked candidate list from the last rebalance, for reporting/observability
        self._current_exposure: float = 1.0   # last computed exposure fraction (0..1), for reporting/observability

    def _load_eligible_tickers(self):
        provider = YahooDataProvider(cache_db=None)
        self._eligible_tickers = provider.get_high_beta_universe()
        if self.include_etf_universe:
            self._eligible_tickers = self._eligible_tickers | self.ETF_UNIVERSE

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

        # ATR (Average True Range) - for the volatility-ADJUSTED stop-loss.
        # A flat % stop tested worse than no stop at all (15% whipsawed
        # badly, 35% was too wide to ever matter) precisely because
        # momentum names have wildly different individual volatility - a
        # stop distance sized to each stock's OWN recent true range (the
        # actual Turtle Trader technique, not a flat %) is what the
        # trend-following literature actually uses.
        atr = None
        if 'High' in hist.columns and 'Low' in hist.columns:
            h, l, c = hist['High'], hist['Low'], hist['Close'].shift(1)
            tr = pd.concat([(h - l), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
            atr_window = min(20, len(tr.dropna()))
            if atr_window > 5:
                atr = float(tr.tail(atr_window).mean())

        return {
            'ticker': ticker, 'momentum': momentum, 'above_trend': above_trend,
            'price': current_price, 'volatility': volatility, 'atr': atr,
            'sector': TICKER_SECTOR.get(ticker, 'Unknown'),
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

    def _apply_stop_losses(self, context: ExecutionContext) -> None:
        """CANSLIM/turtle-trader style: a per-position trailing stop,
        checked every day regardless of the rebalance schedule. Stopped-
        out capital goes to cash until the next scheduled re-ranking
        (classic stop-loss discipline - get out and stay out, don't try
        to immediately redeploy). Deliberately bottom-up and independent
        per ticker - unlike every market-level exposure filter tried
        (and failed) in Iteration 25/26, there's no synchronized whole-
        portfolio action here, so it shouldn't reproduce the same
        whipsaw-driven turnover blowup.
        """
        if (not self.stop_loss_pct and not self.atr_stop_multiplier) or not self._holdings:
            return
        stopped = []
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if not price:
                continue
            if price > holding['peak_price']:
                holding['peak_price'] = price
            if self.atr_stop_multiplier and holding.get('atr'):
                stop_level = holding['peak_price'] - self.atr_stop_multiplier * holding['atr']
                if price < stop_level:
                    stopped.append(ticker)
            elif self.stop_loss_pct:
                if price < holding['peak_price'] * (1 - self.stop_loss_pct):
                    stopped.append(ticker)
        for ticker in stopped:
            del self._holdings[ticker]
        if stopped:
            print(f"[Momentum] Stopped out on {context.date.strftime('%Y-%m-%d')}: {stopped}")

    def _vol_target_exposure(self, total_value: float) -> float:
        """Barroso & Santa-Clara ("Momentum has its Moments", 2015):
        scale exposure by target_vol / realized_vol, using the
        STRATEGY'S OWN trailing realized volatility (tracked from this
        strategy's own portfolio value history, not an external market
        signal). Continuous and smooth - unlike every market-filter
        design tried in Iteration 25/26 (all binary/near-binary regime
        switches that whipsawed badly at daily cadence), this drifts
        gradually as realized vol changes, so it shouldn't reproduce the
        same synchronized-whole-portfolio-flip turnover blowup. Directly
        targets momentum's well-documented crash risk: momentum crashes
        cluster in exactly the high-realized-vol periods this scales
        away from.
        """
        self._value_history.append(total_value)
        if not self.vol_target:
            return 1.0
        if len(self._value_history) < self.vol_target_lookback_days + 1:
            return 1.0
        window = self._value_history[-(self.vol_target_lookback_days + 1):]
        rets = pd.Series(window).pct_change().dropna()
        realized_vol = float(rets.std() * (252 ** 0.5))
        if realized_vol <= 0:
            return 1.0
        raw = self.vol_target / realized_vol
        target = float(np.clip(raw, self.vol_target_min_exposure, self.vol_target_max_exposure))
        # Tolerance band: a "smooth" signal recomputed and RE-TRADED every
        # single day is not actually smooth in its trading impact - it
        # produces a small trade on every position every day (7,914
        # trades/5yr, -69% CAGR/99.9% MaxDD observed without this fix).
        # Only actually move exposure when it has drifted meaningfully
        # from what's currently implemented.
        if abs(target - self._implemented_vol_exposure) >= self.vol_target_tolerance:
            self._implemented_vol_exposure = target
        return self._implemented_vol_exposure

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
            self._last_signals = signals

            self._holdings = {}
            # Sector cap: previously declared (max_sector_weight,
            # sector_weights) but never actually enforced here - with as
            # few as 10 live positions an unchecked cap is a real
            # concentration risk (e.g. 4-6 of 10 slots landing in one
            # correlated sector during an AI/semis-momentum-driven
            # market), not a hypothetical one. Enforced via a static,
            # pre-committed ticker->sector map (data/sector_map.py) -
            # deliberately NOT a live yfinance `.info` lookup, since this
            # strategy is specifically designed to need no live
            # fundamentals/metadata fetch at all.
            sector_weights: Dict[str, float] = {}
            selected = []
            approx_weight = 1.0 / self.max_positions
            for sig in signals:
                if len(selected) >= self.max_positions:
                    break
                sector = sig.get('sector', 'Unknown')
                if sector_weights.get(sector, 0.0) + approx_weight > self.max_sector_weight:
                    continue
                selected.append(sig)
                sector_weights[sector] = sector_weights.get(sector, 0.0) + approx_weight

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
                        'peak_price': sig['price'], 'atr': sig.get('atr'),
                    }

            self._last_rebalance = context.date
            if self._holdings:
                print(f"[Momentum] Selected {len(self._holdings)} positions:")
                for t, h in list(self._holdings.items())[:10]:
                    print(f"  {t}: mom={h['momentum']*100:.1f}%")

        self._apply_stop_losses(context)
        exposure = self._market_exposure(context) * self._vol_target_exposure(total_value)
        self._current_exposure = exposure

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
