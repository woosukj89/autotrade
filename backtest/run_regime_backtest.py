"""
Backtest the Regime-Adaptive Strategy.

This strategy dynamically allocates between:
- High Beta Growth (aggressive) - for bull markets
- Bear Beta Defensive - for corrections

Allocation is based on the MacroMom bear score predictor.
"""

import warnings
import logging
import sys
import os

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*YFPricesMissingError.*")
warnings.filterwarnings("ignore", message=".*possibly delisted.*")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from backtest.backtest import Backtest
    from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy, OptimizedRegimeStrategy
    from strategies.high_beta_strategy import HighBetaGrowthStrategy
    from strategies.bear_beta_strategy import BearBetaStrategy
except ImportError:
    from backtest import Backtest
    from regime_adaptive_strategy import RegimeAdaptiveStrategy, OptimizedRegimeStrategy
    from high_beta_strategy import HighBetaGrowthStrategy
    from bear_beta_strategy import BearBetaStrategy


def run_single_backtest(strategy, strategy_name, engine, **kwargs):
    """Run a single backtest and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {strategy_name}")
    print("="*60)

    result = engine.backtest(
        strategy=strategy,
        **kwargs
    )

    stats = result.full_stats()

    print(f"\n{strategy_name} Results:")
    print("-" * 40)
    print(f"  Total Return:      {stats['total_return']*100:>10.2f}%")
    print(f"  CAGR:              {stats['cagr']*100:>10.2f}%")
    print(f"  Max Drawdown:      {stats['max_drawdown']*100:>10.2f}%")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:     {stats['sortino_ratio']:>10.2f}")
    print(f"  Beta:              {stats['beta']:>10.2f}")
    print(f"  Alpha:             {stats['alpha']*100:>10.2f}%")
    print(f"  Total Trades:      {stats['total_trades']:>10}")

    return result, stats


def main():
    print("=" * 70)
    print("REGIME-ADAPTIVE STRATEGY BACKTEST")
    print("=" * 70)

    db_path = "data/fundamentals.sqlite"

    # Common backtest parameters
    backtest_params = {
        "lookback_years": 10,
        "end_year": 2025,
        "starting_fund": 100_000,
        "buffer_pricing": 1,  # $1 slippage
        "time_period": "M",   # Monthly for faster execution
        "benchmark": "SPY",
    }

    print(f"\nBacktest Parameters:")
    print(f"  - Lookback: {backtest_params['lookback_years']} years")
    print(f"  - End year: {backtest_params['end_year']}")
    print(f"  - Starting fund: ${backtest_params['starting_fund']:,}")
    print(f"  - Time period: Monthly")
    print(f"  - Benchmark: {backtest_params['benchmark']}")

    # Initialize strategies
    strategies = {
        "Regime-Adaptive": RegimeAdaptiveStrategy(
            db_path=db_path,
            max_positions=25,
            rebalance_days=30,
            regime_check_days=7,
        ),
        "High Beta Only": HighBetaGrowthStrategy(
            db_path=db_path,
            max_positions=15,
            rebalance_days=90,
        ),
        "Bear Beta Only": BearBetaStrategy(
            db_path=db_path,
            max_positions=20,
            rebalance_days=90,
        ),
    }

    # Run backtests
    engine = Backtest(db_path="data/fundamentals.sqlite")
    results = {}

    for name, strategy in strategies.items():
        try:
            result, stats = run_single_backtest(
                strategy=strategy,
                strategy_name=name,
                engine=engine,
                **backtest_params
            )
            results[name] = {"result": result, "stats": stats}
        except Exception as e:
            print(f"\n[ERROR] {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Comparison summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)

        print(f"\n{'Strategy':<20} {'Return':>10} {'CAGR':>10} {'MaxDD':>10} {'Sharpe':>10} {'Alpha':>10}")
        print("-" * 70)

        for name, data in results.items():
            s = data["stats"]
            print(f"{name:<20} "
                  f"{s['total_return']*100:>9.1f}% "
                  f"{s['cagr']*100:>9.1f}% "
                  f"{s['max_drawdown']*100:>9.1f}% "
                  f"{s['sharpe_ratio']:>10.2f} "
                  f"{s['alpha']*100:>9.1f}%")

        # Benchmark
        if "Regime-Adaptive" in results:
            s = results["Regime-Adaptive"]["stats"]
            print(f"{'SPY (Benchmark)':<20} "
                  f"{s['benchmark_total_return']*100:>9.1f}% "
                  f"{s['benchmark_cagr']*100:>9.1f}% "
                  f"{'N/A':>10} "
                  f"{'N/A':>10} "
                  f"{'0.0':>9}%")

        print("-" * 70)

    # Save results and generate charts
    if "Regime-Adaptive" in results:
        result = results["Regime-Adaptive"]["result"]

        # Save data
        result.to_dataframe().to_csv("regime_backtest_equity.csv", index=False)
        result.trades_to_dataframe().to_csv("regime_backtest_trades.csv", index=False)
        result.portfolio_summary_over_time().to_csv("regime_backtest_portfolio.csv", index=False)

        print("\nData saved to:")
        print("  - regime_backtest_equity.csv")
        print("  - regime_backtest_trades.csv")
        print("  - regime_backtest_portfolio.csv")

        # Generate chart
        print("\nGenerating chart...")
        result.plot(
            title="Regime-Adaptive Strategy vs S&P 500 (10-Year Backtest)",
            save_path="regime_backtest_chart.png"
        )

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
The Regime-Adaptive Strategy aims to:
1. Capture upside in bull markets (via High Beta stocks)
2. Protect capital during corrections (via low Bear Beta stocks)
3. Use MacroMom indicator to predict regime changes (~173 days lead time)

Expected characteristics:
- Lower max drawdown than pure High Beta
- Higher returns than pure Defensive
- Better risk-adjusted returns (Sharpe, Sortino)
- Positive alpha vs benchmark

Note: Monthly backtest frequency may miss some regime transitions.
For production, use weekly or daily frequency.
""")

    print("\n" + "=" * 70)
    print("Backtest complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
