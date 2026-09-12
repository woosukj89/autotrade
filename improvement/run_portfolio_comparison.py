"""
Step 3 (final): real portfolio backtest for the best classifier found per
bear definition, plus the live RegimeAdaptiveStrategy (MacroMom) and SPY
as reference points. Whichever maximizes actual growth (CAGR, risk-
adjusted via Sharpe, and MaxDD as the risk check) wins - per the user's
explicit final criterion, this is the real arbiter, not classifier
coverage/FPR/defensive-return (those were Step 2's filter).

Uses the same tax-aware Backtest engine (25% realized-gains tax) and
methodology as every other backtest in this repo, for apples-to-apples
comparison against the archived Bond Rate Adaptive work and the live
MacroMom strategy.
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from backtest import Backtest
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from classifier import ClassifierParams
from portfolio_strategy import BinaryClassifierStrategy
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_summary():
    with open(os.path.join(RESULTS_DIR, 'multi_def_summary.json')) as f:
        return json.load(f)


def run_backtest(name, strategy, bt, years, end_year):
    print(f"\n{'='*60}\nRunning: {name} ({years}yr, ending {end_year})\n{'='*60}")
    return bt.backtest(
        lookback_years=years, end_year=end_year, starting_fund=100000,
        strategy=strategy, time_period='M',
    )


def _bench_maxdd(result):
    values = np.array(result.benchmark_values)
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    return float(((peak - values) / np.where(peak > 0, peak, 1)).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, default=20)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--tax-rate', type=float, default=0.25)
    parser.add_argument('--defs', type=str, default=None,
                         help='comma-separated definition names to run (default: all in summary)')
    args = parser.parse_args()

    summary = load_summary()
    def_names = args.defs.split(',') if args.defs else list(summary.keys())

    print('Loading shared classifier data (SPY/VIX/credit/breadth) once...')
    clf_data = load_all()

    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=args.tax_rate)

    results = {}

    for name in def_names:
        params = ClassifierParams(**summary[name]['best_params'])
        strategy = BinaryClassifierStrategy(params, db_path=db_path, data=clf_data)
        result = run_backtest(f'Binary[{name}]', strategy, bt, args.years, args.end_year)
        results[f'Binary[{name}]'] = result

    baseline = RegimeAdaptiveStrategy(db_path=db_path)
    results['RegimeAdaptive (MacroMom, live)'] = run_backtest(
        'RegimeAdaptive (MacroMom, live)', baseline, bt, args.years, args.end_year)

    print(f"\n{'='*100}\nPORTFOLIO COMPARISON ({args.years}yr, after {args.tax_rate*100:.0f}% tax, ending {args.end_year})\n{'='*100}")
    print(f"{'Strategy':<38}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}{'Return':>12}")
    print('-' * 100)
    rows = []
    for name, result in results.items():
        if result and result.snapshots:
            rows.append((name, result.cagr, result.max_drawdown, result.sharpe_ratio, result.total_return))
            print(f"{name:<38}{result.cagr*100:>7.1f}%{result.max_drawdown*100:>7.1f}%"
                  f"{result.sharpe_ratio:>8.2f}{result.total_return*100:>11.1f}%")

    first = next((r for r in results.values() if r), None)
    if first and first.benchmark_values:
        bdd = _bench_maxdd(first)
        print(f"{'SPY (no tax, benchmark)':<38}{first.benchmark_cagr*100:>7.1f}%{bdd*100:>7.1f}%"
              f"{'':>8}{first.benchmark_total_return*100:>11.1f}%")

    if rows:
        winner = max(rows, key=lambda r: r[1])  # by CAGR = actual capital growth
        print(f"\nWINNER by CAGR (capital growth): {winner[0]}  "
              f"({winner[1]*100:.1f}% CAGR, {winner[2]*100:.1f}% MaxDD, {winner[3]:.2f} Sharpe)")

    # Save results
    out = {}
    for name, result in results.items():
        if result and result.snapshots:
            out[name] = {'cagr': result.cagr, 'max_drawdown': result.max_drawdown,
                          'sharpe_ratio': result.sharpe_ratio, 'total_return': result.total_return}
            df = result.portfolio_summary_over_time()
            slug = name.lower().replace(' ', '_').replace('[', '_').replace(']', '').replace('(', '').replace(')', '').replace(',', '')
            df.to_csv(os.path.join(RESULTS_DIR, f'portfolio_{slug}_{args.years}yr.csv'), index=False)
    with open(os.path.join(RESULTS_DIR, f'portfolio_comparison_{args.years}yr.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/portfolio_comparison_{args.years}yr.json + per-strategy CSVs")


if __name__ == '__main__':
    main()
