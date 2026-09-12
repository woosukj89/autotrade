"""
Plot Bond Rate Adaptive vs Regime Adaptive vs SPY from saved backtest CSVs.

Reads the portfolio_summary_over_time() CSVs already produced by
backtest/run_bond_rate_backtest.py --save-csv (bond_rate_*_results.csv at
repo root) and plots equity curves (log scale) + drawdown for the 10yr and
20yr windows.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def plot_window(years_label, br_csv, ra_csv, save_path, bear_periods=None):
    br = pd.read_csv(br_csv, parse_dates=['date'])
    ra = pd.read_csv(ra_csv, parse_dates=['date'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 1], sharex=True)
    fig.suptitle(f'Bond Rate Adaptive vs Regime Adaptive (MacroMom) vs SPY — {years_label}\n'
                 f'$100k start, monthly rebalance, 25% tax on realized gains modeled',
                 fontsize=13, fontweight='bold')

    ax1.plot(br['date'], br['total_value'], label='Bond Rate Adaptive', color='#2E86AB', linewidth=2.2)
    ax1.plot(ra['date'], ra['total_value'], label='Regime Adaptive (MacroMom, live strategy)', color='#F18F01', linewidth=2.0, linestyle='--')
    ax1.plot(br['date'], br['benchmark_value'], label='SPY (buy & hold, no tax)', color='#888888', linewidth=1.8, linestyle=':')

    if bear_periods:
        for start, end, label in bear_periods:
            ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='red', alpha=0.08)

    ax1.set_yscale('log')
    ax1.set_ylabel('Portfolio Value ($, log scale)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, which='both', alpha=0.25)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    def cagr(df):
        yrs = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        return (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) ** (1/yrs) - 1

    def maxdd(values):
        peak = np.maximum.accumulate(values)
        return ((peak - values) / peak).max()

    def bench_cagr(df):
        yrs = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        return (df['benchmark_value'].iloc[-1] / df['benchmark_value'].iloc[0]) ** (1/yrs) - 1

    stats_text = (
        f"{'Strategy':<20}{'CAGR':>8}{'MaxDD':>8}\n"
        f"{'-'*36}\n"
        f"{'Bond Rate Adapt.':<20}{cagr(br)*100:>7.1f}%{maxdd(br['total_value'].values)*100:>7.1f}%\n"
        f"{'Regime Adaptive':<20}{cagr(ra)*100:>7.1f}%{maxdd(ra['total_value'].values)*100:>7.1f}%\n"
        f"{'SPY (no tax)':<20}{bench_cagr(br)*100:>7.1f}%{maxdd(br['benchmark_value'].values)*100:>7.1f}%"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray')
    ax1.text(0.985, 0.03, stats_text, transform=ax1.transAxes, fontsize=9.5,
              verticalalignment='bottom', horizontalalignment='right', fontfamily='monospace', bbox=props)

    def dd_series(values):
        peak = np.maximum.accumulate(values)
        return (values - peak) / peak * 100

    ax2.fill_between(br['date'], dd_series(br['total_value'].values), 0, color='#2E86AB', alpha=0.35, label='Bond Rate Adaptive')
    ax2.plot(ra['date'], dd_series(ra['total_value'].values), color='#F18F01', linewidth=1.3, linestyle='--', label='Regime Adaptive')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved {save_path}')
    plt.close(fig)


BEAR_PERIODS = [
    ('2007-10-09', '2009-03-09', '2008 GFC'),
    ('2020-02-19', '2020-03-23', '2020 COVID'),
    ('2022-01-03', '2022-10-13', '2022 Inflation Bear'),
]

plot_window('10-Year (2015–2025)',
            'bond_rate_bond_rate_adaptive_10yr_results.csv',
            'bond_rate_regime_adaptive_macromom_10yr_results.csv',
            'bond_rate_comparison_10yr_final.png',
            bear_periods=BEAR_PERIODS)

plot_window('20-Year (2005–2025)',
            'bond_rate_bond_rate_adaptive_20yr_results.csv',
            'bond_rate_regime_adaptive_macromom_20yr_results.csv',
            'bond_rate_comparison_20yr_final.png',
            bear_periods=BEAR_PERIODS)
