"""
Root cause found for the run-to-run noise that sank Iteration 19's claim:
yfinance's `.info` endpoint (data/yahoo_data.py::get_fundamentals) silently
fails a variable subset of fetches each run (observed: "Fetching
fundamentals for 8 tickers... Got fundamentals for 0 tickers" - a complete
failure on that batch), a known reliability issue with that unofficial
API under concurrent load. This changes which stocks pass HighBetaGrowthStrategy's
score>=50 filter run to run, which is enough to move MaxDD by several
points on an identical config (confirmed: two runs of the same
FactorRotation[alltime_15] gave 33.5% and 35.1% MaxDD).

Rather than patch data/yahoo_data.py (shared with the live trader - out of
scope for this research branch), this runs the SAME strategy N times and
reports the distribution, so "does it meet target" is answered honestly
instead of off one draw - directly responding to the Iteration 19/20
lesson: don't trust a single run this close to a boundary.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from backtest import Backtest
from classifier import ClassifierParams
from factor_rotation_strategy import FactorRotationStrategy
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with open(os.path.join(RESULTS_DIR, 'multi_def_summary.json')) as f:
        summary = json.load(f)
    params = ClassifierParams(**summary['alltime_15']['best_params'])

    clf_data = load_all()
    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)

    rows = []
    for i in range(n_runs):
        print(f"\n{'='*60}\nRun {i+1}/{n_runs}\n{'='*60}")
        strategy = FactorRotationStrategy(params, db_path=db_path, data=clf_data, max_positions=15)
        result = bt.backtest(lookback_years=20, end_year=2025, starting_fund=100000,
                              strategy=strategy, time_period='M')
        if result and result.snapshots:
            rows.append((result.cagr, result.max_drawdown, result.sharpe_ratio))
            df = result.portfolio_summary_over_time()
            df.to_csv(os.path.join(RESULTS_DIR, f'stability_run{i}_20yr.csv'), index=False)

    print(f"\n{'='*60}\nSTABILITY CHECK: FactorRotation[alltime_15], {n_runs} independent runs\n{'='*60}")
    for i, (cagr, dd, sharpe) in enumerate(rows):
        print(f"  run {i}: CAGR={cagr*100:.1f}%  MaxDD={dd*100:.1f}%  Sharpe={sharpe:.2f}")
    cagrs = [r[0] for r in rows]
    dds = [r[1] for r in rows]
    sharpes = [r[2] for r in rows]
    n = len(rows)
    print(f"\nmean CAGR={sum(cagrs)/n*100:.1f}%  range=[{min(cagrs)*100:.1f}%, {max(cagrs)*100:.1f}%]")
    print(f"mean MaxDD={sum(dds)/n*100:.1f}%  range=[{min(dds)*100:.1f}%, {max(dds)*100:.1f}%]")
    print(f"mean Sharpe={sum(sharpes)/n:.2f}  range=[{min(sharpes):.2f}, {max(sharpes):.2f}]")


if __name__ == '__main__':
    main()
