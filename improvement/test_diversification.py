"""
Idea D probe: does the aggressive sleeve's concentration (defaults:
max_positions=15, max_sector_weight=0.50 - tech can be half the
portfolio, max_position_weight=0.15) explain a meaningful chunk of the
66% MaxDD seen in every backtest this session, independent of any
market-timing signal at all? Runs HighBetaGrowthStrategy ALONE (always
100% invested, no defensive switching) at a few diversification settings.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest import Backtest
from strategies.high_beta_strategy import HighBetaGrowthStrategy

VARIANTS = {
    'default (15pos/50%sector/15%pos)': dict(max_positions=15, max_sector_weight=0.50, max_position_weight=0.15),
    'diversified (25pos/30%sector/8%pos)': dict(max_positions=25, max_sector_weight=0.30, max_position_weight=0.08),
    'very_diversified (35pos/20%sector/5%pos)': dict(max_positions=35, max_sector_weight=0.20, max_position_weight=0.05),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, default=20)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--tax-rate', type=float, default=0.25)
    args = parser.parse_args()

    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=args.tax_rate)

    results = {}
    for name, kwargs in VARIANTS.items():
        strategy = HighBetaGrowthStrategy(db_path=db_path, **kwargs)
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        result = bt.backtest(lookback_years=args.years, end_year=args.end_year,
                              starting_fund=100000, strategy=strategy, time_period='M')
        results[name] = result

    print(f"\n{'='*90}\nDIVERSIFICATION COMPARISON ({args.years}yr, always 100% invested, no timing)\n{'='*90}")
    print(f"{'Variant':<42}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}{'Return':>12}")
    for name, result in results.items():
        if result and result.snapshots:
            print(f"{name:<42}{result.cagr*100:>7.1f}%{result.max_drawdown*100:>7.1f}%"
                  f"{result.sharpe_ratio:>8.2f}{result.total_return*100:>11.1f}%")


if __name__ == '__main__':
    main()
