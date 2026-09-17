"""
Validation follow-up: raw vote_threshold=1 whipsaws daily (616 transitions,
median 1-day episodes over 20yr - see PROGRESS.md Iteration 20). Since
every backtest this session rebalances on a fixed monthly grid (1st of
month), that result risked being a sampling-date artifact, not real
regime detection. This re-tests a set of PERSISTENCE-STABILIZED vote=1
variants (found via classify()-only pre-screening, much cheaper than a
full backtest) through the real 20yr portfolio backtest, to see whether
the "lower bar is fine for factor rotation" insight survives once the
signal is actually stable day-to-day, not just averaged out over a month.
"""
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from backtest import Backtest
from classifier import ClassifierParams
from factor_rotation_strategy import FactorRotationStrategy
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    with open(os.path.join(RESULTS_DIR, 'multi_def_summary.json')) as f:
        summary = json.load(f)
    base = ClassifierParams(**summary['alltime_15']['best_params'])

    variants = {
        'base (vote=2, reused alltime_15)': base,
        'vote=1, RAW (unvalidated headline)': replace(base, vote_threshold=1),
        'vote=1, combo_confirm=5': replace(base, vote_threshold=1, combo_confirm_days=5),
        'vote=1, combo_confirm=10': replace(base, vote_threshold=1, combo_confirm_days=10),
        'vote=1, combo=10,panic_cooldown=15,min_def=10 (stabilized)': replace(
            base, vote_threshold=1, combo_confirm_days=10, panic_cooldown_days=15, min_defensive_days=10),
    }

    print('Loading shared classifier data once...')
    clf_data = load_all()
    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)

    rows = []
    for name, params in variants.items():
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        strategy = FactorRotationStrategy(params, db_path=db_path, data=clf_data, max_positions=15)
        result = bt.backtest(lookback_years=20, end_year=2025, starting_fund=100000,
                              strategy=strategy, time_period='M')
        if result and result.snapshots:
            rows.append((name, result.cagr, result.max_drawdown, result.sharpe_ratio, result.total_return, len(result.trades)))

    print(f"\n{'='*100}\nSTABILIZED VOTE=1 VALIDATION (20yr, after 25% tax)\n{'='*100}")
    print(f"{'Variant':<62}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}{'Trades':>8}")
    print('-' * 94)
    for name, cagr, dd, sharpe, ret, ntrades in rows:
        print(f"{name:<62}{cagr*100:>7.1f}%{dd*100:>7.1f}%{sharpe:>8.2f}{ntrades:>8d}")

    target_hit = [r for r in rows if r[1] > 0.251 and r[2] < 0.30]
    print()
    if target_hit:
        print(f"MEETS TARGET (CAGR>25.1% AND MaxDD<30%): {[r[0] for r in target_hit]}")
    else:
        print("No variant meets the CAGR>25.1% AND MaxDD<30% target.")


if __name__ == '__main__':
    main()
