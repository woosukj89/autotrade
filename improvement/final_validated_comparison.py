"""
The redo requested after the honest session-21 checkpoint: point-in-time
SEC EDGAR fundamentals (no lookahead bias, no yfinance .info reliability
risk), daily execution cadence (matching the live cron, not the monthly
sampling used all prior session), real slippage + SEC/FINRA regulatory
fees, options/collar removed entirely (pure long-only), and an outage-
resilience check (simulates the bot being offline for multi-day windows,
per the observed Robinhood session-expiry issue).

Run modes (see --mode):
  smoke     - short window, quick correctness check before the full run
  full      - full 20yr daily-cadence backtest, no outages
  outage    - full 20yr daily-cadence backtest, WITH simulated outages,
              compared against the no-outage run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from backtest import Backtest
from classifier import ClassifierParams
from pit_factor_rotation_strategy import PitFactorRotationStrategy
from pit_strategies import PitFundamentalsStore, PitHighBetaGrowthStrategy, PitQualityDefensiveStrategy
from outage_strategy import OutageSimulatingStrategy, generate_outage_windows
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SLIPPAGE_BPS = 5.0  # 0.05% per trade side - moderate assumption for liquid S&P-ish names


def run_one(name, strategy, bt, years, end_year, time_period):
    print(f"\n{'='*60}\nRunning: {name} ({years}yr, {time_period}, ending {end_year})\n{'='*60}")
    return bt.backtest(lookback_years=years, end_year=end_year, starting_fund=100000,
                        strategy=strategy, time_period=time_period,
                        slippage_bps=SLIPPAGE_BPS, regulatory_fees=True)


def report(rows, target_cagr=0.251, target_dd=0.30):
    print(f"\n{'Strategy':<42}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}{'Slippage':>12}{'RegFees':>10}{'Tax':>12}")
    print('-' * 100)
    for name, r in rows:
        print(f"{name:<42}{r.cagr*100:>7.1f}%{r.max_drawdown*100:>7.1f}%{r.sharpe_ratio:>8.2f}"
              f"{r.slippage_cost:>12,.0f}{r.regulatory_fees:>10,.0f}{r.tax_paid:>12,.0f}")
    hits = [(n, r) for n, r in rows if r.cagr > target_cagr and r.max_drawdown < target_dd]
    if hits:
        print(f"\nMEETS TARGET: {[n for n, _ in hits]}")
    else:
        print("\nNo variant meets the CAGR>25.1% AND MaxDD<30% target.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full', 'outage'], default='smoke')
    parser.add_argument('--years', type=int, default=None)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--time-period', type=str, default='d')
    parser.add_argument('--defs', type=str, default='alltime_15')
    args = parser.parse_args()

    years = args.years or (3 if args.mode == 'smoke' else 20)

    print('Loading shared classifier data + point-in-time fundamentals store once...')
    clf_data = load_all()
    pit_store = PitFundamentalsStore()

    with open(os.path.join(RESULTS_DIR, 'multi_def_summary.json')) as f:
        summary = json.load(f)

    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)

    rows = []

    rows.append(('PitHighBetaOnly (no timing)', run_one(
        'PitHighBetaOnly (no timing)',
        PitHighBetaGrowthStrategy(pit_store=pit_store, db_path=db_path, max_positions=15),
        bt, years, args.end_year, args.time_period)))

    rows.append(('PitQualityDefensiveOnly', run_one(
        'PitQualityDefensiveOnly',
        PitQualityDefensiveStrategy(pit_store=pit_store, db_path=db_path, max_positions=15),
        bt, years, args.end_year, args.time_period)))

    for def_name in args.defs.split(','):
        params = ClassifierParams(**summary[def_name]['best_params'])
        base_strategy = PitFactorRotationStrategy(params, db_path=db_path, data=clf_data,
                                                    max_positions=15, pit_store=pit_store)
        name = f'PitFactorRotation[{def_name}]'

        if args.mode == 'outage':
            # Three severity levels: light/moderate/severe outage schedules.
            for label, mean_gap, mean_dur in [
                ('light (1 outage/quarter, ~3d)', 90, 3),
                ('moderate (1 outage/month, ~5d)', 30, 5),
                ('severe (2/month, ~4d each)', 15, 4),
            ]:
                from datetime import datetime
                start = datetime(args.end_year - years, 1, 1)
                end = datetime(args.end_year, 12, 31)
                windows = generate_outage_windows(start, end, mean_gap, mean_dur, seed=42)
                strat2 = PitFactorRotationStrategy(params, db_path=db_path, data=clf_data,
                                                    max_positions=15, pit_store=pit_store)
                outage_strat = OutageSimulatingStrategy(strat2, windows)
                oname = f'{name} + outages: {label}'
                result = run_one(oname, outage_strat, bt, years, args.end_year, args.time_period)
                print(f"  -> {len(windows)} outage windows, {outage_strat.outage_fraction*100:.1f}% of days offline")
                rows.append((oname, result))

        rows.append((name, run_one(name, base_strategy, bt, years, args.end_year, args.time_period)))

    report(rows)

    out = {n: {'cagr': r.cagr, 'max_drawdown': r.max_drawdown, 'sharpe_ratio': r.sharpe_ratio,
               'slippage_cost': r.slippage_cost, 'regulatory_fees': r.regulatory_fees, 'tax_paid': r.tax_paid}
           for n, r in rows}
    out_path = os.path.join(RESULTS_DIR, f'final_validated_{args.mode}_{years}yr.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
