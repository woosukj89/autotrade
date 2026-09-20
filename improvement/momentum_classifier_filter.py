"""
Uses the already-calibrated 4-signal regime classifier (classifier.py,
90 transitions/20yr with proper entry/exit asymmetry) as the momentum
strategy's market-level exposure gate, instead of the naive single-SMA
crossing filter that failed twice at daily cadence (both 2%/5-day and
4%/15-day hysteresis made things worse than no filter at all - see
PROGRESS.md Iteration 25). Reuses proven infrastructure instead of
re-deriving persistence/hysteresis tuning from scratch.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from backtest import Backtest
from momentum_strategy import MomentumStrategy
from classifier import classify, ClassifierParams
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    with open(os.path.join(RESULTS_DIR, 'multi_def_summary.json')) as f:
        summary = json.load(f)
    params = ClassifierParams(**summary['alltime_15']['best_params'])
    data = load_all()
    defensive, transitions = classify(data, params)
    print(f'{len(transitions)} transitions over the classifier window')

    exposure_series = defensive.map(lambda d: 0.30 if d else 1.0)

    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)
    strategy = MomentumStrategy(max_positions=10, lookback_days=189, rebalance_days=30, trend_sma_days=200,
                                 external_exposure_series=exposure_series)
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    r = bt.backtest(lookback_years=years, end_year=2025, starting_fund=100000,
                     strategy=strategy, time_period='d', slippage_bps=5.0, regulatory_fees=True)
    print(f'{years}yr, classifier-gated momentum: CAGR', r.cagr, 'MaxDD', r.max_drawdown, 'Sharpe', r.sharpe_ratio)
    print('Trades', len(r.trades), 'Tax', r.tax_paid)


if __name__ == '__main__':
    main()
