"""
Plot the actual mechanism behind the Bond Rate Adaptive vs Regime Adaptive
comparison: not just the resulting equity curve, but WHEN each strategy's
signal crossed its threshold, WHAT allocation it moved to, and how that
allocation trajectory looked over time. Also includes the Regime Adaptive
(Improved Execution) control — same MacroMom signal, same execution
upgrade Bond Rate Adaptive got — to isolate signal quality from execution
quality.

Reads the CSVs saved by backtest/run_bond_rate_backtest.py --save-csv:
  bond_rate_<slug>_<years>yr_results.csv       (portfolio_summary_over_time)
  bond_rate_<slug>_<years>yr_scores.csv        (date, score)
  bond_rate_<slug>_<years>yr_repositions.csv   (date, from_hb, to_hb, score, trigger)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


# Each strategy's OWN allocation-table breakpoints (max_score -> hb_weight),
# used to draw threshold reference lines on the score panel. Must match
# strategies/bond_rate_strategy.py / regime_adaptive_strategy.py exactly.
THRESHOLDS_BOND_RATE = [50, 60, 70, 80]          # BondRateAdaptiveStrategy + Improved Exec
THRESHOLDS_MACROMOM_ORIGINAL = [55, 65, 75]       # RegimeAdaptiveStrategy (unmodified)

COLORS = {
    'Bond Rate Adaptive': '#2E86AB',
    'Regime Adaptive (orig)': '#F18F01',
    'Regime Adaptive (Improved Exec)': '#8E44AD',
    'SPY': '#888888',
}


def load(slug, years):
    results = pd.read_csv(f'bond_rate_{slug}_{years}yr_results.csv', parse_dates=['date'])
    try:
        scores = pd.read_csv(f'bond_rate_{slug}_{years}yr_scores.csv', parse_dates=['date'])
    except FileNotFoundError:
        scores = pd.DataFrame(columns=['date', 'score'])
    try:
        repos = pd.read_csv(f'bond_rate_{slug}_{years}yr_repositions.csv', parse_dates=['date'])
    except FileNotFoundError:
        repos = pd.DataFrame(columns=['date', 'from_hb', 'to_hb', 'score', 'trigger'])
    return results, scores, repos


def allocation_series(results: pd.DataFrame, repos: pd.DataFrame) -> pd.Series:
    """Reconstruct the HB-weight step function over the full date range from
    the reposition log (starts at 100% HB, the strategies' default)."""
    dates = results['date'].values
    hb = np.full(len(dates), 1.0)
    for _, row in repos.iterrows():
        hb[dates >= np.datetime64(row['date'])] = row['to_hb']
    return pd.Series(hb, index=results.index)


def plot(years, bear_periods, save_path):
    br_res, br_scores, br_repos = load('bond_rate_adaptive', years)
    ra_res, ra_scores, ra_repos = load('regime_adaptive_macromom', years)
    ie_res, ie_scores, ie_repos = load('regime_adaptive_improved_exec', years)

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 15),
        height_ratios=[3, 1.3, 1.1, 1.1], sharex=True
    )
    ax_eq, ax_score, ax_alloc, ax_dd = axes
    fig.suptitle(f'Bond Rate Adaptive vs Regime Adaptive — mechanism, {years}-Year\n'
                 f'$100k start, monthly rebalance, 25% tax on realized gains modeled',
                 fontsize=13, fontweight='bold')

    # ---- Panel 1: equity curves + reposition markers ----
    ax_eq.plot(br_res['date'], br_res['total_value'], label='Bond Rate Adaptive',
               color=COLORS['Bond Rate Adaptive'], linewidth=2.2)
    ax_eq.plot(ra_res['date'], ra_res['total_value'], label='Regime Adaptive (orig)',
               color=COLORS['Regime Adaptive (orig)'], linewidth=1.8, linestyle='--')
    ax_eq.plot(ie_res['date'], ie_res['total_value'], label='Regime Adaptive (Improved Exec)',
               color=COLORS['Regime Adaptive (Improved Exec)'], linewidth=1.6, linestyle='-.')
    ax_eq.plot(br_res['date'], br_res['benchmark_value'], label='SPY (buy & hold, no tax)',
               color=COLORS['SPY'], linewidth=1.5, linestyle=':')

    for start, end, _label in bear_periods:
        ax_eq.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='red', alpha=0.08)

    def add_markers(ax, res, repos, color, y_offset_frac):
        """Triangle markers at each reallocation: up=more aggressive, down=more defensive."""
        for _, row in repos.iterrows():
            idx = (res['date'] - row['date']).abs().idxmin()
            y = res['total_value'].iloc[idx] * y_offset_frac
            going_defensive = row['to_hb'] < row['from_hb']
            marker = 'v' if going_defensive else '^'
            ax.scatter(row['date'], y, marker=marker, s=55, color=color,
                       edgecolor='black', linewidth=0.6, zorder=6)

    add_markers(ax_eq, br_res, br_repos, COLORS['Bond Rate Adaptive'], 1.35)
    add_markers(ax_eq, ra_res, ra_repos, COLORS['Regime Adaptive (orig)'], 0.72)

    ax_eq.set_yscale('log')
    ax_eq.set_ylabel('Portfolio Value ($, log scale)', fontsize=10.5)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_eq.grid(True, which='both', alpha=0.25)

    def cagr(df, col='total_value'):
        yrs = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        return (df[col].iloc[-1] / df[col].iloc[0]) ** (1/yrs) - 1

    def maxdd(values):
        peak = np.maximum.accumulate(values)
        return ((peak - values) / peak).max()

    stats_text = (
        f"{'Strategy':<26}{'CAGR':>7}{'MaxDD':>7}\n" + '-'*40 + "\n"
        f"{'Bond Rate Adaptive':<26}{cagr(br_res)*100:>6.1f}%{maxdd(br_res['total_value'].values)*100:>6.1f}%\n"
        f"{'Regime Adaptive (orig)':<26}{cagr(ra_res)*100:>6.1f}%{maxdd(ra_res['total_value'].values)*100:>6.1f}%\n"
        f"{'Regime Adapt. (Impr.Exec)':<26}{cagr(ie_res)*100:>6.1f}%{maxdd(ie_res['total_value'].values)*100:>6.1f}%\n"
        f"{'SPY (no tax)':<26}{cagr(br_res, 'benchmark_value')*100:>6.1f}%{maxdd(br_res['benchmark_value'].values)*100:>6.1f}%"
    )
    props = dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.93, edgecolor='gray')
    ax_eq.text(0.985, 0.03, stats_text, transform=ax_eq.transAxes, fontsize=8.7,
               verticalalignment='bottom', horizontalalignment='right', fontfamily='monospace', bbox=props)

    marker_legend = [
        mlines.Line2D([], [], color='black', marker='v', linestyle='None', markersize=7, label='goes MORE defensive'),
        mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=7, label='goes MORE aggressive'),
    ]
    leg1 = ax_eq.legend(loc='upper left', fontsize=9.5)
    ax_eq.add_artist(leg1)
    ax_eq.legend(handles=marker_legend, loc='upper left', bbox_to_anchor=(0.0, 0.72), fontsize=8.5, framealpha=0.9)

    # ---- Panel 2: score history + each strategy's own thresholds ----
    ax_score.plot(br_scores['date'], br_scores['score'], color=COLORS['Bond Rate Adaptive'],
                  linewidth=1.6, label='Bond Rate score')
    ax_score.plot(ra_scores['date'], ra_scores['score'], color=COLORS['Regime Adaptive (orig)'],
                  linewidth=1.4, linestyle='--', label='MacroMom score (shared by both Regime variants)')

    # get_yaxis_transform keeps y in DATA units (0-100 here), x in axes-fraction —
    # so threshold labels are placed directly at their real data value, not a
    # 0-1 fraction (that mismatch was the earlier overlapping-text bug).
    for t in THRESHOLDS_BOND_RATE:
        ax_score.axhline(t, color=COLORS['Bond Rate Adaptive'], linestyle=':', linewidth=0.9, alpha=0.55)
        ax_score.text(1.005, t, f'BR {t}', transform=ax_score.get_yaxis_transform(),
                      fontsize=7.5, color=COLORS['Bond Rate Adaptive'], va='center', ha='left')
    for t in THRESHOLDS_MACROMOM_ORIGINAL:
        ax_score.axhline(t, color=COLORS['Regime Adaptive (orig)'], linestyle=':', linewidth=0.9, alpha=0.55)
        ax_score.text(1.075, t, f'MM {t}', transform=ax_score.get_yaxis_transform(),
                      fontsize=7.5, color=COLORS['Regime Adaptive (orig)'], va='center', ha='left')

    for start, end, _label in bear_periods:
        ax_score.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='red', alpha=0.08)
    ax_score.set_ylabel('Bear score (0-100)', fontsize=10.5)
    ax_score.set_ylim(0, 100)
    ax_score.legend(loc='upper left', fontsize=8.5)
    ax_score.grid(True, alpha=0.25)

    # ---- Panel 3: allocation % (high-beta weight) over time ----
    br_alloc = allocation_series(br_res, br_repos)
    ra_alloc = allocation_series(ra_res, ra_repos)
    ie_alloc = allocation_series(ie_res, ie_repos)

    ax_alloc.plot(br_res['date'], br_alloc.values * 100, color=COLORS['Bond Rate Adaptive'], linewidth=1.8, label='Bond Rate Adaptive')
    ax_alloc.plot(ra_res['date'], ra_alloc.values * 100, color=COLORS['Regime Adaptive (orig)'], linewidth=1.5, linestyle='--', label='Regime Adaptive (orig)')
    ax_alloc.plot(ie_res['date'], ie_alloc.values * 100, color=COLORS['Regime Adaptive (Improved Exec)'], linewidth=1.4, linestyle='-.', label='Regime Adaptive (Improved Exec)')

    for start, end, _label in bear_periods:
        ax_alloc.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='red', alpha=0.08)
    ax_alloc.set_ylabel('High-beta\nallocation (%)', fontsize=10)
    ax_alloc.set_ylim(-5, 105)
    ax_alloc.legend(loc='lower left', fontsize=8, ncol=3)
    ax_alloc.grid(True, alpha=0.25)

    # ---- Panel 4: drawdown ----
    def dd(values):
        peak = np.maximum.accumulate(values)
        return (values - peak) / peak * 100

    ax_dd.fill_between(br_res['date'], dd(br_res['total_value'].values), 0, color=COLORS['Bond Rate Adaptive'], alpha=0.30)
    ax_dd.plot(ra_res['date'], dd(ra_res['total_value'].values), color=COLORS['Regime Adaptive (orig)'], linewidth=1.2, linestyle='--')
    ax_dd.plot(ie_res['date'], dd(ie_res['total_value'].values), color=COLORS['Regime Adaptive (Improved Exec)'], linewidth=1.1, linestyle='-.')
    for start, end, label in bear_periods:
        ax_dd.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='red', alpha=0.08)
    ax_dd.set_ylabel('Drawdown (%)', fontsize=10.5)
    ax_dd.set_xlabel('Date', fontsize=11)
    ax_dd.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.subplots_adjust(right=0.90)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved {save_path}')
    plt.close(fig)


BEAR_PERIODS = [
    ('2007-10-09', '2009-03-09', '2008 GFC'),
    ('2020-02-19', '2020-03-23', '2020 COVID'),
    ('2022-01-03', '2022-10-13', '2022 Inflation Bear'),
]

plot(10, BEAR_PERIODS, 'bond_rate_mechanism_10yr.png')
plot(20, BEAR_PERIODS, 'bond_rate_mechanism_20yr.png')
