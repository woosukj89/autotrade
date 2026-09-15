"""
Idea E test: factor rotation (always fully invested, switches which stock
sleeve it holds based on the calibrated regime classifier) run through the
real 20yr backtest, compared against every baseline established so far:
  - HighBetaOnly: the aggressive sleeve alone, always (26.9-27.0% CAGR /
    64-65% MaxDD across runs - see test_diversification.py)
  - QualityDefensiveOnly: the new defensive sleeve alone, always - not
    tried standalone yet, establishes its own baseline CAGR/MaxDD
  - RegimeAdaptive (MacroMom, live, unmodified)
  - FactorRotation[<definition>] for each of the 7 bear definitions'
    calibrated classifiers
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
from strategies.high_beta_strategy import HighBetaGrowthStrategy
from quality_strategy import QualityDefensiveStrategy
from classifier import ClassifierParams
from factor_rotation_strategy import FactorRotationStrategy
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_summary():
    with open(os.path.join(RESULTS_DIR, 'multi_def_summary.json')) as f:
        return json.load(f)


def run_backtest(name, strategy, bt, years, end_year):
    print(f"\n{'='*60}\nRunning: {name} ({years}yr, ending {end_year})\n{'='*60}")
    return bt.backtest(lookback_years=years, end_year=end_year, starting_fund=100000,
                        strategy=strategy, time_period='M')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, default=20)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--tax-rate', type=float, default=0.25)
    parser.add_argument('--defs', type=str, default=None)
    args = parser.parse_args()

    summary = load_summary()
    def_names = args.defs.split(',') if args.defs else list(summary.keys())

    print('Loading shared classifier data once...')
    clf_data = load_all()

    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=args.tax_rate)

    results = {}

    results['HighBetaOnly (no timing)'] = run_backtest(
        'HighBetaOnly (no timing)', HighBetaGrowthStrategy(db_path=db_path, max_positions=15),
        bt, args.years, args.end_year)

    results['QualityDefensiveOnly'] = run_backtest(
        'QualityDefensiveOnly', QualityDefensiveStrategy(db_path=db_path, max_positions=15),
        bt, args.years, args.end_year)

    results['RegimeAdaptive (MacroMom, live)'] = run_backtest(
        'RegimeAdaptive (MacroMom, live)', RegimeAdaptiveStrategy(db_path=db_path),
        bt, args.years, args.end_year)

    for name in def_names:
        params = ClassifierParams(**summary[name]['best_params'])
        strategy = FactorRotationStrategy(params, db_path=db_path, data=clf_data, max_positions=15)
        results[f'FactorRotation[{name}]'] = run_backtest(f'FactorRotation[{name}]', strategy, bt, args.years, args.end_year)

    print(f"\n{'='*100}\nFACTOR ROTATION COMPARISON ({args.years}yr, after {args.tax_rate*100:.0f}% tax, ending {args.end_year})\n{'='*100}")
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
        values = np.array(first.benchmark_values)
        peak = np.maximum.accumulate(values)
        bdd = float(((peak - values) / np.where(peak > 0, peak, 1)).max())
        print(f"{'SPY (no tax, benchmark)':<38}{first.benchmark_cagr*100:>7.1f}%{bdd*100:>7.1f}%"
              f"{'':>8}{first.benchmark_total_return*100:>11.1f}%")

    if rows:
        winner = max(rows, key=lambda r: r[1])
        print(f"\nWINNER by CAGR: {winner[0]}  ({winner[1]*100:.1f}% CAGR, {winner[2]*100:.1f}% MaxDD, {winner[3]:.2f} Sharpe)")
        target_hit = [r for r in rows if r[1] > 0.251 and r[2] < 0.30]
        if target_hit:
            print(f"MEETS TARGET (CAGR>25.1% AND MaxDD<30%): {[r[0] for r in target_hit]}")
        else:
            print("No strategy meets the CAGR>25.1% AND MaxDD<30% target yet.")

    out = {}
    for name, result in results.items():
        if result and result.snapshots:
            out[name] = {'cagr': result.cagr, 'max_drawdown': result.max_drawdown,
                          'sharpe_ratio': result.sharpe_ratio, 'total_return': result.total_return}
            df = result.portfolio_summary_over_time()
            slug = name.lower().replace(' ', '_').replace('[', '_').replace(']', '').replace('(', '').replace(')', '').replace(',', '')
            df.to_csv(os.path.join(RESULTS_DIR, f'factorrot_{slug}_{args.years}yr.csv'), index=False)
    with open(os.path.join(RESULTS_DIR, f'factor_rotation_comparison_{args.years}yr.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/factor_rotation_comparison_{args.years}yr.json + per-strategy CSVs")


if __name__ == '__main__':
    main()
