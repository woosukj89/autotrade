"""
Idea E (factor rotation) - the defensive sleeve: a low-beta, quality-tilted
stock-picker, NOT a cash/SH switch. Reuses HighBetaGrowthStrategy's data
pipeline (YahooDataProvider, beta calc, position sizing) but inverts the
selection criteria - low beta instead of high, defensive sectors (staples,
healthcare, utilities, gold miners) instead of tech/discretionary, and
requires real profitability/balance-sheet quality so this doesn't just
become "buy whatever has the lowest beta regardless of fundamentals."

Always fully invested in equities, same as the aggressive sleeve - this is
a factor tilt (WHICH stocks), not a market-timing exposure cut (HOW MUCH
is invested), which is what distinguishes idea E from ideas A/B already
tried this session.
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from high_beta_strategy import HighBetaGrowthStrategy
from yahoo_data import YahooDataProvider


class QualityDefensiveStrategy(HighBetaGrowthStrategy):
    # Override the parent's high-beta PRIORITY_TICKERS with the defensive
    # priority set so candidate evaluation order favors defensive names.
    PRIORITY_TICKERS = YahooDataProvider.DEFENSIVE_PRIORITY

    DEFENSIVE_SECTOR_SCORES = {
        'Consumer Staples': 15,
        'Consumer Defensive': 15,
        'Utilities': 15,
        'Health Care': 12,
        'Healthcare': 12,
        'Materials': 6,
        'Basic Materials': 6,
        'Communication Services': 3,
        'Financials': 3,
        'Financial Services': 3,
    }

    def __init__(
        self,
        db_path: str = "data/fundamentals.sqlite",
        max_positions: int = 15,
        max_beta: float = 0.85,
        rebalance_days: int = 90,
        max_sector_weight: float = 0.35,
        max_position_weight: float = 0.12,
        min_score: float = 35,
    ):
        # min_beta=0 disables the parent's high-beta gate entirely - this
        # subclass overrides _score_stock() with its own beta filter below.
        super().__init__(
            db_path=db_path, max_positions=max_positions, min_beta=0.0,
            min_beta_quality=0.0, min_quality_score=0, min_score=min_score,
            rebalance_days=rebalance_days, max_sector_weight=max_sector_weight,
            max_position_weight=max_position_weight,
        )
        self.max_beta = max_beta

    def _load_eligible_tickers(self) -> None:
        universe = self._yahoo_provider.get_defensive_universe()
        valid_tickers = [t for t in universe if self._is_valid_ticker(t)]
        self._eligible_tickers = set(valid_tickers)
        print(f"[QualityDefensive] Loaded {len(self._eligible_tickers)} tickers from Yahoo Finance")

    def _score_stock(self, ticker: str) -> Optional[dict]:
        beta = self._beta_cache.get(ticker)
        fund = self._fundamentals_cache.get(ticker)
        if beta is None or fund is None or beta <= 0 or beta > self.max_beta:
            return None

        roe = fund.roe or 0
        op_margin = fund.operating_margin or 0
        gross_margin = fund.gross_margin or 0
        debt_equity = fund.debt_to_equity or 999
        fcf = fund.free_cash_flow or 0
        div_yield = fund.dividend_yield or 0

        score = 0
        breakdown = {}

        # PROFITABILITY (max 25) - require real quality, not just "low beta"
        prof_score = 0
        if roe > 0.10:
            prof_score += 8
        if roe > 0.18:
            prof_score += 5
        if op_margin > 0.10:
            prof_score += 7
        if gross_margin > 0.30:
            prof_score += 5
        score += prof_score
        breakdown['profitability'] = prof_score

        # LOW BETA (max 25) - inverted vs. HighBetaGrowthStrategy: lower is better
        beta_score = 0
        if beta <= 0.85:
            beta_score += 10
        if beta <= 0.65:
            beta_score += 10
        if beta <= 0.45:
            beta_score += 5
        score += beta_score
        breakdown['low_beta'] = beta_score

        # BALANCE SHEET + CASH FLOW STRENGTH (max 25) - defensive names need
        # to actually survive a downturn, weighted higher than the
        # aggressive sleeve's balance-sheet component
        balance_score = 0
        if debt_equity < 100:
            balance_score += 5
        if debt_equity < 60:
            balance_score += 5
        if debt_equity < 30:
            balance_score += 5
        if fcf > 0:
            balance_score += 10
        score += balance_score
        breakdown['balance_sheet'] = balance_score

        # DIVIDEND (max 10) - stable payers are a defensive-quality signal
        div_score = 10 if div_yield and div_yield > 0.015 else 0
        score += div_score
        breakdown['dividend'] = div_score

        # SECTOR (max 15)
        sector = fund.sector or 'Unknown'
        sector_score = self.DEFENSIVE_SECTOR_SCORES.get(sector, 0)
        score += sector_score
        breakdown['sector'] = sector_score

        return {
            'ticker': ticker, 'score': score, 'breakdown': breakdown, 'beta': beta,
            'roe': roe, 'operating_margin': op_margin, 'gross_margin': gross_margin,
            'revenue_growth': fund.revenue_growth or 0, 'earnings_growth': fund.earnings_growth or 0,
            'debt_to_equity': debt_equity, 'sector': sector, 'market_cap': fund.market_cap or 0,
        }
