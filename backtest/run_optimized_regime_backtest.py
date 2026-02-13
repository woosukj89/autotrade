"""
Optimized Regime-Adaptive Strategy Backtest

Compares:
1. Optimized Regime-Adaptive (aggressive thresholds + momentum confirmation)
2. Conservative Regime-Adaptive (original thresholds)
3. High Beta Only
4. Bear Beta Only (defensive)

Outputs:
- Console comparison table
- CSV files for each strategy
- Interactive chart with hover for portfolio positions
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import Backtest
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from strategies.high_beta_strategy import HighBetaGrowthStrategy
from strategies.bear_beta_strategy import BearBetaStrategy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def run_backtest(name, strategy, bt, lookback_years=10, end_year=2025, quiet=False):
    """Run a single backtest and return results."""
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)

    result = bt.backtest(
        lookback_years=lookback_years,
        end_year=end_year,
        starting_fund=100000,
        strategy=strategy,
        time_period='M',  # Monthly rebalancing
    )

    return result


def plot_comparison(results: dict, save_path: str = None, interactive: bool = False):
    """Plot all strategies on a single comparison chart with hover support."""

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True
    )
    fig.suptitle("Strategy Comparison: Regime-Adaptive Backtest", fontsize=14, fontweight="bold")

    colors = {
        'Regime-Adaptive (Optimized)': '#2E86AB',
        'Regime-Adaptive (Conservative)': '#F18F01',
        'High Beta Only': '#C73E1D',
        'Bear Beta Only': '#3A7D44',
    }

    all_dates = []
    all_values = {}
    all_snapshots = {}
    scatters = {}

    for name, result in results.items():
        if result and result.snapshots:
            dates = [s.date for s in result.snapshots]
            values = [s.total_value for s in result.snapshots]
            all_dates = dates  # They should all be the same
            all_values[name] = values
            all_snapshots[name] = result.snapshots

            color = colors.get(name, '#888888')
            ax1.plot(dates, values, label=name, linewidth=2, color=color)

            # Add scatter points for hover
            scatter = ax1.scatter(dates, values, s=15, color=color, alpha=0.5, zorder=5)
            scatters[name] = scatter

    # Add benchmark
    first_result = list(results.values())[0]
    if first_result and first_result.benchmark_values:
        ax1.plot(
            all_dates[:len(first_result.benchmark_values)],
            first_result.benchmark_values,
            label=f'Benchmark ({first_result.benchmark_ticker})',
            linewidth=2,
            color='#888888',
            linestyle='--',
            alpha=0.7,
        )

    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization of growth
    ax1.set_xlim(all_dates[0], all_dates[-1])

    # Drawdown for optimized strategy
    opt_result = results.get('Regime-Adaptive (Optimized)')
    if opt_result and opt_result.snapshots:
        values = [s.total_value for s in opt_result.snapshots]
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak * 100
        ax2.fill_between(all_dates, drawdown, 0, color='#E74C3C', alpha=0.4,
                         label='Optimized Drawdown')

    # Also show High Beta drawdown for comparison
    hb_result = results.get('High Beta Only')
    if hb_result and hb_result.snapshots:
        values = [s.total_value for s in hb_result.snapshots]
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak * 100
        ax2.plot(all_dates, drawdown, color='#C73E1D', alpha=0.7,
                 linewidth=1, linestyle='--', label='High Beta Drawdown')

    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Stats box
    stats_lines = ["Strategy Comparison", "─" * 35]
    for name, result in results.items():
        if result:
            stats = result.full_stats()
            short_name = name.replace('Regime-Adaptive ', 'RA-').replace('(', '').replace(')', '')
            stats_lines.append(f"{short_name[:18]:<18}")
            stats_lines.append(f"  Return: {stats['total_return']*100:>8.1f}%")
            stats_lines.append(f"  CAGR:   {stats['cagr']*100:>8.1f}%")
            stats_lines.append(f"  MaxDD:  {stats['max_drawdown']*100:>8.1f}%")
            stats_lines.append(f"  Sharpe: {stats['sharpe_ratio']:>8.2f}")
            stats_lines.append("")

    stats_text = "\n".join(stats_lines)
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray")
    ax1.text(
        1.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace", bbox=props
    )

    # Interactive hover
    if interactive and not save_path:
        annot = ax1.annotate(
            "", xy=(0, 0), xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.95),
            fontsize=8, fontfamily="monospace",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
            zorder=100
        )
        annot.set_visible(False)

        def on_hover(event):
            if event.inaxes != ax1:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                return

            for name, scatter in scatters.items():
                cont, ind = scatter.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    snap = all_snapshots[name][idx]
                    pos = scatter.get_offsets()[idx]
                    annot.xy = pos

                    positions = snap.positions
                    pos_text = "\n".join([f"  {t:<6} {s:>6.0f}" for t, s in list(positions.items())[:10]])
                    if len(positions) > 10:
                        pos_text += f"\n  ... +{len(positions) - 10} more"

                    text = (
                        f"{name}\n"
                        f"{'─' * 28}\n"
                        f"Date: {snap.date.strftime('%Y-%m-%d')}\n"
                        f"Total: ${snap.total_value:,.0f}\n"
                        f"Cash: ${snap.cash:,.0f}\n"
                        f"{'─' * 28}\n"
                        f"Holdings ({len(positions)}):\n{pos_text}"
                    )
                    annot.set_text(text)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return

            annot.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        print("\nHover over data points to see portfolio positions.")

    plt.tight_layout()
    plt.subplots_adjust(right=0.78)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved to {save_path}")
    else:
        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run regime-adaptive strategy backtest')
    parser.add_argument('--years', type=int, default=10, help='Lookback years (default: 10)')
    parser.add_argument('--end-year', type=int, default=2025, help='End year (default: 2025)')
    parser.add_argument('--save-chart', type=str, default=None, help='Save chart to file')
    parser.add_argument('--interactive', action='store_true', help='Show interactive chart')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    args = parser.parse_args()

    print("="*70)
    print("REGIME-ADAPTIVE STRATEGY COMPARISON")
    print("="*70)

    bt = Backtest()
    results = {}

    # 1. Optimized Regime-Adaptive (new aggressive thresholds)
    strategy_opt = RegimeAdaptiveStrategy(
        db_path='data/fundamentals.sqlite',
        max_positions=25,
        use_conservative=False,  # Use optimized thresholds
    )
    results['Regime-Adaptive (Optimized)'] = run_backtest(
        'Regime-Adaptive (Optimized)', strategy_opt, bt,
        lookback_years=args.years, end_year=args.end_year, quiet=args.quiet
    )

    # 2. Conservative Regime-Adaptive (original thresholds)
    strategy_cons = RegimeAdaptiveStrategy(
        db_path='data/fundamentals.sqlite',
        max_positions=25,
        use_conservative=True,  # Use original conservative thresholds
    )
    results['Regime-Adaptive (Conservative)'] = run_backtest(
        'Regime-Adaptive (Conservative)', strategy_cons, bt,
        lookback_years=args.years, end_year=args.end_year, quiet=args.quiet
    )

    # 3. High Beta Only
    strategy_hb = HighBetaGrowthStrategy(
        db_path='data/fundamentals.sqlite',
        max_positions=15,
    )
    results['High Beta Only'] = run_backtest(
        'High Beta Only', strategy_hb, bt,
        lookback_years=args.years, end_year=args.end_year, quiet=args.quiet
    )

    # 4. Bear Beta Only (Defensive)
    strategy_bb = BearBetaStrategy(
        db_path='data/fundamentals.sqlite',
        max_positions=15,
    )
    results['Bear Beta Only'] = run_backtest(
        'Bear Beta Only', strategy_bb, bt,
        lookback_years=args.years, end_year=args.end_year, quiet=args.quiet
    )

    # Print comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"\n{'Strategy':<35} {'Return':>10} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8}")
    print("-"*70)

    for name, result in results.items():
        if result and result.snapshots:
            ret = result.total_return * 100
            cagr = result.cagr * 100
            maxdd = result.max_drawdown * 100
            sharpe = result.sharpe_ratio
            print(f"{name:<35} {ret:>9.1f}% {cagr:>7.1f}% {maxdd:>7.1f}% {sharpe:>8.2f}")

    # Print benchmark
    first_result = list(results.values())[0]
    if first_result:
        bench_ret = first_result.benchmark_total_return * 100
        bench_cagr = first_result.benchmark_cagr * 100
        print(f"\n{'SPY (Benchmark)':<35} {bench_ret:>9.1f}% {bench_cagr:>7.1f}%")

    # Save results to CSV
    print("\n" + "="*70)
    print("Saving results...")

    for name, result in results.items():
        if result and result.snapshots:
            df = result.portfolio_summary_over_time()
            filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') + '_results.csv'
            df.to_csv(filename, index=False)
            print(f"  Saved: {filename}")

    # Generate chart
    print("\n" + "="*70)
    print("Generating comparison chart...")

    save_path = args.save_chart or 'regime_comparison_chart.png'

    if args.interactive:
        # Show interactive chart (don't save)
        plot_comparison(results, save_path=None, interactive=True)
    else:
        # Save static chart
        matplotlib.use('Agg')
        plot_comparison(results, save_path=save_path, interactive=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
