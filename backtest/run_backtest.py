"""
Run backtest of FundamentalStrategy vs S&P 500 benchmark.

Usage:
    python run_backtest.py
"""

import warnings
import logging

# Suppress yfinance and other noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from backtest.backtest import Backtest
from fundamental_strategy import FundamentalStrategy


def main():
    print("=" * 60)
    print("Fundamental Value Strategy Backtest")
    print("=" * 60)

    # Initialize strategy with Buffett-style parameters.
    strategy = FundamentalStrategy(
        db_path="fundamentals.sqlite",
        max_positions=5,
        min_margin_of_safety=0.20,  # Require 20% discount to intrinsic value
        rebalance_days=90,          # Quarterly rebalancing
        max_position_weight=0.35,   # No single position > 35%
    )

    # Initialize backtest engine.
    engine = Backtest(db_path="fundamentals.sqlite")

    print("\nRunning backtest...")
    print("  - Lookback: 10 years")
    print("  - End year: 2024")
    print("  - Starting fund: $100,000")
    print("  - Buffer pricing: $2 (simulates slippage)")
    print("  - Time period: Daily")
    print("  - Benchmark: SPY (S&P 500 ETF)")
    print("\nThis may take a few minutes on first run (fetching shares outstanding)...\n")

    result = engine.backtest(
        lookback_years=10,
        end_year=2024,
        starting_fund=100_000,
        buffer_pricing=2,
        strategy=strategy,
        time_period="d",
        benchmark="SPY",
    )

    # Print summary statistics.
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    stats = result.full_stats()

    print("\nSTRATEGY PERFORMANCE")
    print("-" * 40)
    print(f"  Initial Value:     ${stats['initial_value']:>12,.2f}")
    print(f"  Final Value:       ${stats['final_value']:>12,.2f}")
    print(f"  Total Return:      {stats['total_return']*100:>12.2f}%")
    print(f"  CAGR:              {stats['cagr']*100:>12.2f}%")
    print(f"  Volatility:        {stats['volatility']*100:>12.2f}%")
    print(f"  Max Drawdown:      {stats['max_drawdown']*100:>12.2f}%")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:>12.2f}")
    print(f"  Sortino Ratio:     {stats['sortino_ratio']:>12.2f}")
    print(f"  Total Trades:      {stats['total_trades']:>12}")

    print(f"\nBENCHMARK ({stats['benchmark']})")
    print("-" * 40)
    print(f"  Total Return:      {stats['benchmark_total_return']*100:>12.2f}%")
    print(f"  CAGR:              {stats['benchmark_cagr']*100:>12.2f}%")
    print(f"  Volatility:        {stats['benchmark_volatility']*100:>12.2f}%")

    print("\nRELATIVE METRICS")
    print("-" * 40)
    print(f"  Alpha:             {stats['alpha']*100:>12.2f}%")
    print(f"  Beta:              {stats['beta']:>12.2f}")
    print(f"  Information Ratio: {stats['information_ratio']:>12.2f}")
    print(f"  Excess Return:     {stats['excess_return']*100:>12.2f}%")

    # Save data to CSV.
    result.to_dataframe().to_csv("backtest_equity_curve.csv", index=False)
    result.trades_to_dataframe().to_csv("backtest_trades.csv", index=False)
    result.portfolio_summary_over_time().to_csv("backtest_portfolio.csv", index=False)
    print("\nData saved to:")
    print("   - backtest_equity_curve.csv")
    print("   - backtest_trades.csv")
    print("   - backtest_portfolio.csv")

    # Plot the results.
    print("\nGenerating chart...")
    result.plot(
        title="Fundamental Value Strategy vs S&P 500 (10-Year Backtest)",
        save_path="backtest_chart.png"
    )

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
