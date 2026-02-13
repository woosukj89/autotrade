"""
Compare multiple value investing strategies against S&P 500 benchmark.

Usage:
    python compare_strategies.py
"""

import warnings
import logging

# Suppress yfinance and other noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from backtest.backtest import Backtest
from strategies.fundamental_strategy import FundamentalStrategy
from strategies.value_strategies import (
    BuyAndHoldValueStrategy,
    MomentumValueStrategy,
    ConcentratedValueStrategy,
    QualityGrowthStrategy,
    AdaptiveValueStrategy,
)
from improved_strategies import (
    CompoundersStrategy,
    TrueBuffettStrategy,
    MarketRegimeStrategy,
    EarningsGrowthStrategy,
)


def run_strategy_comparison(
    lookback_years: int = 10,
    end_year: int = 2024,
    starting_fund: float = 100_000,
    buffer_pricing: int = 2,
):
    """Run all strategies and compare results."""

    strategies = {
        # Original strategies (for comparison)
        "Original (Quarterly)": FundamentalStrategy(
            max_positions=5,
            min_margin_of_safety=0.20,
            rebalance_days=90,
        ),
        "Buy & Hold": BuyAndHoldValueStrategy(
            max_positions=5,
            min_margin_of_safety=0.25,
            rebalance_days=365,
            sell_threshold=-0.30,
        ),

        # Improved strategies
        "Compounders": CompoundersStrategy(
            max_positions=10,
            min_roic=0.15,
            min_moat=0.55,
            max_premium=0.20,  # Will pay up to 20% above fair value
            rebalance_days=180,
        ),
        "True Buffett": TrueBuffettStrategy(
            max_positions=6,
            min_roic=0.15,
            min_moat=0.60,
            min_margin_of_safety=0.10,
            rebalance_days=365,
        ),
        "Market Regime": MarketRegimeStrategy(
            max_positions=8,
            min_moat=0.50,
            min_margin_of_safety=0.15,
            rebalance_days=30,
        ),
        "Earnings Growth": EarningsGrowthStrategy(
            max_positions=10,
            min_earnings_growth=0.10,
            min_revenue_growth=0.08,
            min_roic=0.12,
            rebalance_days=90,
        ),
    }

    results = {}
    engine = Backtest(db_path="fundamentals.sqlite")

    print("=" * 70)
    print("STRATEGY COMPARISON BACKTEST")
    print("=" * 70)
    print(f"Period: {end_year - lookback_years} - {end_year} ({lookback_years} years)")
    print(f"Starting Fund: ${starting_fund:,.0f}")
    print(f"Buffer Pricing: ${buffer_pricing}")
    print("=" * 70)

    for name, strategy in strategies.items():
        print(f"\nRunning: {name}...")
        try:
            result = engine.backtest(
                lookback_years=lookback_years,
                end_year=end_year,
                starting_fund=starting_fund,
                buffer_pricing=buffer_pricing,
                strategy=strategy,
                time_period="d",
                benchmark="SPY",
            )
            results[name] = result
            print(f"  -> Final Value: ${result.final_value:,.0f} ({result.total_return*100:.1f}%)")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    return results


def print_comparison_table(results: dict):
    """Print a comparison table of all strategy results."""
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)

    headers = ["Strategy", "Final Value", "Return", "CAGR", "Volatility", "MaxDD", "Sharpe", "Alpha", "Beta"]
    print(f"{headers[0]:<20} {headers[1]:>12} {headers[2]:>8} {headers[3]:>8} {headers[4]:>10} {headers[5]:>8} {headers[6]:>8} {headers[7]:>8} {headers[8]:>6}")
    print("-" * 90)

    # Print benchmark first
    first_result = list(results.values())[0]
    print(f"{'Benchmark (SPY)':<20} ${first_result.benchmark_values[-1]:>10,.0f} {first_result.benchmark_total_return*100:>7.1f}% {first_result.benchmark_cagr*100:>7.1f}% {first_result.benchmark_volatility*100:>9.1f}% {'N/A':>8} {'N/A':>8} {'N/A':>8} {'1.00':>6}")
    print("-" * 90)

    # Sort by total return
    sorted_results = sorted(results.items(), key=lambda x: x[1].total_return, reverse=True)

    for name, result in sorted_results:
        stats = result.full_stats()
        print(
            f"{name:<20} "
            f"${stats['final_value']:>10,.0f} "
            f"{stats['total_return']*100:>7.1f}% "
            f"{stats['cagr']*100:>7.1f}% "
            f"{stats['volatility']*100:>9.1f}% "
            f"{stats['max_drawdown']*100:>7.1f}% "
            f"{stats['sharpe_ratio']:>8.2f} "
            f"{stats['alpha']*100:>7.2f}% "
            f"{stats['beta']:>6.2f}"
        )

    print("=" * 90)


def plot_comparison(results: dict, save_path: str = None):
    """Plot all strategy equity curves on one chart."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    fig.suptitle("Strategy Comparison: Value Investing Variations vs S&P 500", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(range(len(results) + 1))

    # Get dates from first result
    first_result = list(results.values())[0]
    dates = [s.date for s in first_result.snapshots]

    # Plot benchmark
    axes[0].plot(
        dates[:len(first_result.benchmark_values)],
        first_result.benchmark_values,
        label="SPY (Benchmark)",
        linewidth=2.5,
        color="black",
        linestyle="--",
    )

    # Plot each strategy
    for i, (name, result) in enumerate(results.items()):
        values = [s.total_value for s in result.snapshots]
        axes[0].plot(dates, values, label=name, linewidth=1.5, color=colors[i])

    axes[0].set_ylabel("Portfolio Value ($)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(dates[0], dates[-1])

    # Bottom chart: relative performance vs benchmark
    for i, (name, result) in enumerate(results.items()):
        values = [s.total_value for s in result.snapshots]
        bench = result.benchmark_values
        min_len = min(len(values), len(bench))
        relative = [(v / b - 1) * 100 for v, b in zip(values[:min_len], bench[:min_len])]
        axes[1].plot(dates[:min_len], relative, label=name, linewidth=1.5, color=colors[i])

    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Excess Return vs SPY (%)", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved to {save_path}")
    else:
        plt.show()


def save_detailed_results(results: dict):
    """Save detailed results to CSV files."""
    # Summary table
    summary_data = []
    for name, result in results.items():
        stats = result.full_stats()
        stats["strategy"] = name
        summary_data.append(stats)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("strategy_comparison_summary.csv", index=False)

    # Equity curves
    equity_data = {"date": [s.date for s in list(results.values())[0].snapshots]}
    for name, result in results.items():
        equity_data[name] = [s.total_value for s in result.snapshots]
    equity_data["SPY_Benchmark"] = list(results.values())[0].benchmark_values

    equity_df = pd.DataFrame(equity_data)
    equity_df.to_csv("strategy_equity_curves.csv", index=False)

    # Portfolio holdings over time for each strategy
    for name, result in results.items():
        portfolio_df = result.portfolio_summary_over_time()
        safe_name = name.replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
        portfolio_df.to_csv(f"portfolio_{safe_name}.csv", index=False)

    print("\nDetailed results saved to:")
    print("  - strategy_comparison_summary.csv")
    print("  - strategy_equity_curves.csv")
    print("  - portfolio_<strategy_name>.csv for each strategy")


def print_portfolio_summary(results: dict):
    """Print the final portfolio holdings for each strategy."""
    print("\n" + "=" * 90)
    print("FINAL PORTFOLIO HOLDINGS")
    print("=" * 90)

    for name, result in results.items():
        print(f"\n{name}:")
        print("-" * 50)

        if result.final_portfolio is None:
            print("  No portfolio data")
            continue

        portfolio = result.final_portfolio
        print(f"  Cash: ${portfolio.cash:,.2f}")

        if portfolio.positions:
            print("  Positions:")
            for ticker, pos in sorted(portfolio.positions.items()):
                print(f"    {ticker}: {pos.shares:,.0f} shares @ ${pos.avg_cost:.2f} avg cost")
        else:
            print("  Positions: None (all cash)")

        # Get last snapshot for position values
        if result.snapshots:
            last_snap = result.snapshots[-1]
            print(f"  Total Value: ${last_snap.total_value:,.2f}")


def print_portfolio_changes_over_time(results: dict, sample_dates: int = 5):
    """Print portfolio holdings at key dates for each strategy."""
    print("\n" + "=" * 90)
    print("PORTFOLIO EVOLUTION OVER TIME")
    print("=" * 90)

    for name, result in results.items():
        if not result.snapshots:
            continue

        print(f"\n{'='*60}")
        print(f"Strategy: {name}")
        print(f"{'='*60}")

        # Sample snapshots at regular intervals
        total_snapshots = len(result.snapshots)
        if total_snapshots <= sample_dates:
            indices = list(range(total_snapshots))
        else:
            step = total_snapshots // (sample_dates - 1)
            indices = [0] + [i * step for i in range(1, sample_dates - 1)] + [total_snapshots - 1]

        for idx in indices:
            snap = result.snapshots[idx]
            print(f"\n  Date: {snap.date.strftime('%Y-%m-%d')}")
            print(f"  Total Value: ${snap.total_value:,.0f} | Cash: ${snap.cash:,.0f} | Positions: ${snap.positions_value:,.0f}")
            if snap.positions:
                holdings = ", ".join([f"{t}({s:.0f})" for t, s in snap.positions.items()])
                print(f"  Holdings: {holdings}")
            else:
                print(f"  Holdings: None (all cash)")


def main():
    # Run comparison
    results = run_strategy_comparison(
        lookback_years=10,
        end_year=2024,
        starting_fund=100_000,
        buffer_pricing=2,
    )

    if not results:
        print("No results to display.")
        return

    # Print comparison table
    print_comparison_table(results)

    # Print portfolio holdings
    print_portfolio_summary(results)

    # Print portfolio evolution
    print_portfolio_changes_over_time(results, sample_dates=6)

    # Save detailed results
    save_detailed_results(results)

    # Plot comparison
    plot_comparison(results, save_path="strategy_comparison.png")

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
