"""Test the High Beta Growth Strategy."""

import warnings
import logging

# Suppress all warnings before any imports
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

import sys
import os

# Add current and parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from backtest import Backtest
from strategies.high_beta_strategy import HighBetaGrowthStrategy, HighBetaMomentumStrategy

print("=" * 80)
print("HIGH BETA GROWTH STRATEGY BACKTEST")
print("=" * 80)

# Create strategy
strategy = HighBetaGrowthStrategy(
    db_path="data/fundamentals.sqlite",
    max_positions=15,
    min_beta=1.0,
    min_score=50,
    rebalance_days=90,
    max_sector_weight=0.50,
)

# Run backtest
print("\nRunning 10-year backtest...")
bt = Backtest()

result = bt.backtest(
    lookback_years=10,
    end_year=2024,
    starting_fund=100000,
    strategy=strategy,
    time_period='d',
)

print("\n" + "=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)

# Print key metrics
print(f"\nStrategy: High Beta Growth")
print(f"Period: 10 years ending 2024")
print(f"Starting Capital: $100,000")

if result and result.snapshots:
    final_value = result.final_value
    total_return = result.total_return * 100
    cagr = result.cagr * 100

    print(f"\nFinal Portfolio Value: ${final_value:,.0f}")
    print(f"Total Return: {total_return:.1f}%")
    print(f"CAGR: {cagr:.1f}%")

    if result.benchmark_values:
        bench_final = result.benchmark_values[-1]
        bench_return = (bench_final / 100000 - 1) * 100
        bench_cagr = ((bench_final / 100000) ** (1/10) - 1) * 100
        print(f"\nS&P 500 (SPY) Final: ${bench_final:,.0f}")
        print(f"S&P 500 Return: {bench_return:.1f}%")
        print(f"S&P 500 CAGR: {bench_cagr:.1f}%")
        print(f"\nAlpha: {cagr - bench_cagr:.1f}%")

    print(f"Max Drawdown: {result.max_drawdown*100:.1f}%")

# Show final portfolio
print("\n" + "=" * 80)
print("FINAL PORTFOLIO HOLDINGS")
print("=" * 80)

if hasattr(strategy, '_holdings') and strategy._holdings:
    print(f"\n{'Ticker':<8} {'Shares':>10} {'Beta':>8} {'Score':>8} {'Sector':<20}")
    print("-" * 60)
    for ticker, data in strategy._holdings.items():
        print(f"{ticker:<8} {data['shares']:>10.0f} {data['beta']:>8.2f} {data['score']:>8.0f} {data['sector']:<20}")
else:
    print("No holdings to display")

# Plot results
print("\n" + "=" * 80)
print("Generating performance chart...")
print("=" * 80)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    if result and result.snapshots:
        df = result.to_dataframe()
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df['date'], df['total_value'], label='High Beta Growth', linewidth=2)
        if result.benchmark_values:
            ax.plot(df['date'], result.benchmark_values, label='S&P 500 (SPY)', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('High Beta Growth Strategy vs S&P 500')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('high_beta_strategy_results.png', dpi=150)
        print("Chart saved to: high_beta_strategy_results.png")
except Exception as e:
    print(f"Could not generate chart: {e}")
