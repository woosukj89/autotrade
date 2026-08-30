"""
SPY Bear Market Analysis - 30-Year Study (1994-2026)
=====================================================
Compares three algorithmic bear-detection methods across 30+ years of SPY.

Methods:
  1. DRAWDOWN   Rolling peak-to-trough >= -20% sustained >= 2 weeks
  2. MA200      SPY below 200-day SMA (~40 weeks) for >= 4 consecutive weeks
  3. COMPOSITE  52-week return < 0 AND SPY below 52-week MA AND 13-week vol > 20%

Ground-truth bears:
  1998 LTCM / 2000-02 Dot-com / 2007-09 Financial Crisis /
  2018 Q4 Fed / 2020 COVID / 2022 Inflation

Output:
  spy_bear_analysis_chart.png   (20x22 fig, 150 dpi)
  Console: precision/recall table per method
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import yfinance as yf

warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_START      = '1993-01-01'   # SPY inception; warm-up period for MA200
ANALYSIS_START  = '1994-01-01'   # First full year; slice before eval/chart
DRAWDOWN_THRESH = -0.20          # -20% from rolling ATH
MIN_BEAR_WEEKS  = 2              # Min consecutive weeks for drawdown method
MA200_PERIOD_W  = 40             # 200-day SMA ≈ 40 trading weeks
VOL_LOOKBACK_W  = 13             # 13-week (~3 month) realized vol window
VOL_THRESHOLD   = 0.20           # Annualized vol > 20% = elevated regime

KNOWN_BEARS = [
    ('1998-07-17', '1998-10-08', '1998\nLTCM',        '#FFAAAA'),
    ('2000-03-24', '2002-10-09', '2000-02\nDot-com',  '#FF7777'),
    ('2007-10-09', '2009-03-09', '2007-09\nFin.Crs.', '#FF4444'),
    ('2018-09-20', '2018-12-24', '2018 Q4\nFed',      '#FFBBBB'),
    ('2020-02-19', '2020-03-23', '2020\nCOVID',       '#FF6666'),
    ('2022-01-03', '2022-10-13', '2022\nInflation',   '#FF3333'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_spy_weekly() -> pd.Series:
    print('  Fetching SPY from Yahoo Finance...', end='', flush=True)
    end_date = datetime.today().strftime('%Y-%m-%d')
    raw = yf.download('SPY', start=DATA_START, end=end_date,
                      auto_adjust=True, progress=False, timeout=30)
    if raw is None or raw.empty:
        raise RuntimeError('Failed to download SPY data from Yahoo Finance')

    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        spy = close.iloc[:, 0]
    else:
        spy = close

    weekly = spy.resample('W-FRI').last().ffill().dropna()
    print(f' {len(weekly)} weekly bars ({weekly.index[0].date()} to {weekly.index[-1].date()})')
    return weekly


# ─────────────────────────────────────────────────────────────────────────────
# Bear detection methods
# ─────────────────────────────────────────────────────────────────────────────
def compute_rolling_drawdown(spy_weekly: pd.Series) -> pd.Series:
    """No-look-ahead drawdown: (current - expanding peak) / expanding peak."""
    rolling_peak = spy_weekly.expanding().max()
    drawdown = (spy_weekly - rolling_peak) / rolling_peak
    return drawdown.rename('drawdown')


def _apply_min_duration(bool_series: pd.Series, min_weeks: int) -> pd.Series:
    """
    Causal minimum-duration filter. Uses cumcount within each run so the count
    only sees past weeks — avoids the look-ahead in transform('sum').
    A week becomes True only after min_weeks consecutive True values.
    """
    # Count consecutive True values up to each point (causal)
    not_true = (~bool_series).cumsum()
    consecutive = bool_series.groupby(not_true).cumcount() + bool_series.astype(int)
    return bool_series & (consecutive >= min_weeks)


def signal_drawdown(drawdown: pd.Series, thresh: float = DRAWDOWN_THRESH,
                    min_weeks: int = MIN_BEAR_WEEKS) -> pd.Series:
    raw = drawdown <= thresh
    return _apply_min_duration(raw, min_weeks).rename('DRAWDOWN')


def signal_ma200(spy_weekly: pd.Series, period: int = MA200_PERIOD_W,
                 min_weeks: int = 4) -> pd.Series:
    ma200 = spy_weekly.rolling(period, min_periods=period // 2).mean()
    raw = spy_weekly < ma200
    return _apply_min_duration(raw, min_weeks).rename('MA200')


def signal_composite(spy_weekly: pd.Series,
                     vol_lookback: int = VOL_LOOKBACK_W,
                     vol_thresh: float = VOL_THRESHOLD) -> pd.Series:
    ret_52w    = spy_weekly.pct_change(52)
    ma52       = spy_weekly.rolling(52, min_periods=26).mean()
    weekly_ret = spy_weekly.pct_change()
    realized_vol = weekly_ret.rolling(vol_lookback, min_periods=6).std() * np.sqrt(52)

    composite = (ret_52w < 0) & (spy_weekly < ma52) & (realized_vol > vol_thresh)
    return composite.rename('COMPOSITE')


def build_signal_grid(spy_weekly: pd.Series, sig_dd: pd.Series,
                      sig_ma: pd.Series, sig_comp: pd.Series) -> pd.DataFrame:
    idx = spy_weekly.index
    grid = pd.DataFrame({
        'DRAWDOWN':  sig_dd.reindex(idx).fillna(False),
        'MA200':     sig_ma.reindex(idx).fillna(False),
        'COMPOSITE': sig_comp.reindex(idx).fillna(False),
    })
    return grid[grid.index >= ANALYSIS_START]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _build_bear_mask(index: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(False, index=index)
    for s_str, e_str, *_ in KNOWN_BEARS:
        s_ts = pd.Timestamp(s_str)
        e_ts = pd.Timestamp(e_str)
        mask.loc[s_ts:e_ts] = True
    return mask


def evaluate_methods(signal_grid: pd.DataFrame):
    bear_mask = _build_bear_mask(signal_grid.index)
    rows = []
    per_bear_rows = []

    for method in signal_grid.columns:
        sig = signal_grid[method]
        tp  = int((sig & bear_mask).sum())
        fp  = int((sig & ~bear_mask).sum())
        fn  = int((~sig & bear_mask).sum())
        tn  = int((~sig & ~bear_mask).sum())
        prec  = tp / (tp + fp + 1e-9)
        rec   = tp / (tp + fn + 1e-9)
        f1    = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({'method': method, 'precision': prec, 'recall': rec,
                     'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})

    eval_df = pd.DataFrame(rows).set_index('method')

    for s_str, e_str, label, *_ in KNOWN_BEARS:
        s_ts, e_ts = pd.Timestamp(s_str), pd.Timestamp(e_str)
        window = signal_grid.loc[s_ts:e_ts]
        total = max(len(window), 1)
        row = {'bear': label.replace('\n', ' '), 'total_wks': total}
        for method in signal_grid.columns:
            row[method] = window[method].sum() / total
        per_bear_rows.append(row)

    per_bear_df = pd.DataFrame(per_bear_rows).set_index('bear')
    return eval_df, per_bear_df


def print_evaluation_table(eval_df: pd.DataFrame, per_bear_df: pd.DataFrame):
    print()
    print('=' * 68)
    print('  BEAR DETECTION METHOD COMPARISON')
    print('=' * 68)
    print(f"  {'Method':<12}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  "
          f"{'TP-wks':>7}  {'FP-wks':>7}  {'FN-wks':>7}")
    print('  ' + '-' * 64)
    for method, row in eval_df.iterrows():
        print(f"  {method:<12}  {row['precision']:>8.1%}  {row['recall']:>7.1%}  "
              f"{row['f1']:>7.1%}  {int(row['tp']):>7d}  {int(row['fp']):>7d}  "
              f"{int(row['fn']):>7d}")
    print()
    print('  NOTE: Ground truth = 6 known bears (1998-2022).')
    print('  1998 LTCM (-19%) is below the -20% threshold so DRAWDOWN will miss it.')

    print()
    print('=' * 68)
    print('  COVERAGE PER KNOWN BEAR (% of bear weeks flagged)')
    print('=' * 68)
    print(f"  {'Bear':<28}  {'Wks':>4}  {'DD':>6}  {'MA200':>6}  {'COMP':>6}")
    print('  ' + '-' * 56)
    for bear, row in per_bear_df.iterrows():
        print(f"  {bear:<28}  {int(row['total_wks']):>4d}  "
              f"{row['DRAWDOWN']:>5.0%}  {row['MA200']:>6.0%}  {row['COMPOSITE']:>6.0%}")
    print('=' * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Charting
# ─────────────────────────────────────────────────────────────────────────────
def _nearest(series: pd.Series, date: pd.Timestamp):
    try:
        idx = series.index.get_indexer([date], method='nearest')[0]
        if idx < 0:
            return None
        return series.iloc[idx]
    except Exception:
        return None


def _shade_known_bears(ax, trim_start, trim_end, alpha=0.18):
    for s_str, e_str, _label, color in KNOWN_BEARS:
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < trim_end and e > trim_start:
            ax.axvspan(max(s, trim_start), min(e, trim_end),
                       color=color, alpha=alpha, zorder=1)


def plot_bear_analysis(spy_weekly: pd.Series, drawdown: pd.Series,
                       signal_grid: pd.DataFrame, eval_df: pd.DataFrame,
                       save_path: str = 'spy_bear_analysis_chart.png'):

    spy = spy_weekly[spy_weekly.index >= ANALYSIS_START]
    dd  = drawdown[drawdown.index >= ANALYSIS_START]
    sig = signal_grid

    trim_start = pd.Timestamp(ANALYSIS_START)
    trim_end   = spy.index[-1]

    ma50  = spy_weekly.rolling(50,  min_periods=25).mean().reindex(spy.index)
    ma200 = spy_weekly.rolling(MA200_PERIOD_W, min_periods=20).mean().reindex(spy.index)

    ret_52w = spy_weekly.pct_change(52).reindex(spy.index)
    vol_13w = spy_weekly.pct_change().rolling(VOL_LOOKBACK_W, min_periods=6).std()
    vol_ann = (vol_13w * np.sqrt(52)).reindex(spy.index)

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle('SPY Bear Market Analysis  |  30-Year Study  |  1994–2026\n'
                 'Three Algorithmic Detection Methods vs. Six Known Bears',
                 fontsize=13, fontweight='bold', y=0.995)

    gs = GridSpec(5, 1, figure=fig,
                  height_ratios=[3.5, 1.5, 1.5, 1.5, 1.5],
                  hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4], sharex=ax1)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('#F8F8F8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')

    # ── Panel 1: SPY log-scale + known bear shading + MA overlays + trough markers
    _shade_known_bears(ax1, trim_start, trim_end, alpha=0.30)

    ax1.semilogy(spy.index, spy.values,
                 color='#222222', lw=1.8, zorder=5, label='SPY (weekly close)')
    ax1.semilogy(ma50.index,  ma50.values,
                 color='#F28E2B', lw=1.6, alpha=0.85, zorder=4, label='50-wk MA')
    ax1.semilogy(ma200.index, ma200.values,
                 color='#4E79A7', lw=1.8, alpha=0.90, zorder=4, label='200-d MA (~40wk)')

    # Trough markers and period labels
    for s_str, e_str, label, color in KNOWN_BEARS:
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < trim_end and e > trim_start:
            window = dd.loc[s:e]
            if not window.empty:
                trough_date = window.idxmin()
                trough_val  = _nearest(spy, trough_date)
                if trough_val:
                    ax1.scatter([trough_date], [trough_val],
                                marker='v', color='#880000', s=90, zorder=10)
            mid = s + (e - s) / 2
            if trim_start < mid < trim_end:
                spy_mid = _nearest(spy, mid)
                if spy_mid:
                    ax1.text(mid, spy_mid * 0.72, label,
                             fontsize=7.5, ha='center', color='#770000',
                             fontweight='bold', zorder=8)

    # Annotation: 1998 LTCM missed by threshold
    ax1.annotate('1998 LTCM\n~-19%\n(below -20% thresh)',
                 xy=(pd.Timestamp('1998-10-08'), _nearest(spy, pd.Timestamp('1998-10-08')) or 90),
                 xytext=(pd.Timestamp('2000-01-01'), 75),
                 fontsize=7, color='#AA4400',
                 arrowprops=dict(arrowstyle='->', color='#AA4400', lw=1.0),
                 zorder=9)

    legend_elements = [
        Line2D([0], [0], color='#222222', lw=1.8, label='SPY price'),
        Line2D([0], [0], color='#F28E2B', lw=1.6, label='50-wk MA'),
        Line2D([0], [0], color='#4E79A7', lw=1.8, label='200-d MA (~40wk)'),
        mpatches.Patch(color='#FF7777', alpha=0.30, label='Known bear periods'),
        Line2D([0], [0], marker='v', color='#880000', lw=0, ms=8, label='Drawdown trough'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8.5,
               framealpha=0.85, edgecolor='#CCCCCC')
    ax1.set_ylabel('SPY Price (log scale, $)', fontsize=9)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f'${x:.0f}'))
    ax1.grid(axis='y', which='both', lw=0.4, color='#DDDDDD', zorder=0)
    ax1.set_title('Panel 1 — SPY Price History (log scale) with Known Bear Periods',
                  fontsize=9, loc='left', pad=4)

    # ── Panel 2: Rolling drawdown from ATH
    _shade_known_bears(ax2, trim_start, trim_end, alpha=0.15)
    ax2.fill_between(dd.index, dd.values * 100, 0,
                     where=dd < 0,
                     color='#EE4444', alpha=0.55, interpolate=True,
                     label='Drawdown from ATH')
    ax2.axhline(-20, ls='--', color='#CC0000', lw=1.2, alpha=0.8, label='-20% threshold')
    ax2.axhline(-10, ls=':', color='#AA8888', lw=0.8, alpha=0.5)
    ax2.axhline(0, color='#888888', lw=0.6)
    ax2.set_ylabel('Drawdown (%)', fontsize=9)
    ax2.legend(loc='lower left', fontsize=8, framealpha=0.85, edgecolor='#CCCCCC')
    ax2.grid(axis='y', lw=0.4, color='#DDDDDD', zorder=0)
    ax2.set_title('Panel 2 — Rolling Drawdown from All-Time High',
                  fontsize=9, loc='left', pad=4)

    # ── Panel 3: Signal heatmap (pcolormesh)
    heatmap_data = sig[['DRAWDOWN', 'MA200', 'COMPOSITE']].astype(float).copy()
    bear_mask = _build_bear_mask(sig.index).astype(float)
    heatmap_data['TRUTH'] = bear_mask

    row_labels = list(heatmap_data.columns)
    n_rows = len(row_labels)

    # pcolormesh needs fence-post x-edges (n_cols+1)
    x_idx = heatmap_data.index
    td = pd.Timedelta(weeks=1)
    x_edges = pd.DatetimeIndex(list(x_idx) + [x_idx[-1] + td])
    y_edges = np.arange(n_rows + 1) - 0.5

    ax3.pcolormesh(x_edges, y_edges, heatmap_data.T.values,
                   cmap='RdYlGn', vmin=0, vmax=1, shading='flat')

    ax3.set_yticks(range(n_rows))
    ax3.set_yticklabels(row_labels, fontsize=9)
    ax3.set_ylim(-0.5, n_rows - 0.5)

    # Vertical lines for known bear boundaries
    for s_str, e_str, *_ in KNOWN_BEARS:
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if trim_start < e and s < trim_end:
            ax3.axvline(max(s, trim_start), color='#222222', lw=1.0, alpha=0.6)
            ax3.axvline(min(e, trim_end),   color='#222222', lw=0.7, alpha=0.4, ls='--')

    ax3.set_ylabel('Signal', fontsize=9)
    ax3.set_title('Panel 3 — Weekly Signal Heatmap  (green=bearish, red=bullish/neutral  |  TRUTH=known bears)',
                  fontsize=9, loc='left', pad=4)

    # ── Panel 4: SPY with MA200 shading
    _shade_known_bears(ax4, trim_start, trim_end, alpha=0.15)
    ax4.semilogy(spy.index, spy.values,
                 color='#444444', lw=1.2, alpha=0.6, zorder=5, label='SPY')
    ax4.semilogy(ma200.index, ma200.values,
                 color='#4E79A7', lw=2.0, zorder=6, label='200-d MA (~40wk)')

    # Shade where SPY < MA200
    below = spy < ma200
    ax4.fill_between(spy.index, spy.values, ma200.reindex(spy.index).values,
                     where=below.values,
                     color='#FF5555', alpha=0.25, interpolate=True,
                     label='SPY below 200d MA')

    ax4.legend(loc='upper left', fontsize=8, framealpha=0.85, edgecolor='#CCCCCC')
    ax4.set_ylabel('SPY Price (log)', fontsize=9)
    ax4.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f'${x:.0f}'))
    ax4.grid(axis='y', which='both', lw=0.4, color='#DDDDDD', zorder=0)
    ax4.set_title('Panel 4 — SPY vs 200-Day Moving Average (red shading = below MA200)',
                  fontsize=9, loc='left', pad=4)

    # ── Panel 5: 52-week return + realized vol
    _shade_known_bears(ax5, trim_start, trim_end, alpha=0.15)

    ret = ret_52w * 100
    ax5.fill_between(ret.index, ret.values, 0,
                     where=ret.values < 0,
                     color='#EE4444', alpha=0.50, interpolate=True, label='52-wk return < 0')
    ax5.fill_between(ret.index, ret.values, 0,
                     where=ret.values >= 0,
                     color='#44AA44', alpha=0.35, interpolate=True, label='52-wk return > 0')
    ax5.axhline(0, color='#555555', lw=0.8)
    ax5.set_ylabel('52-wk Return (%)', fontsize=9, color='#333333')
    ax5.tick_params(axis='y', labelcolor='#333333')

    ax5b = ax5.twinx()
    ax5b.plot(vol_ann.index, vol_ann.values * 100,
              color='#9C27B0', lw=1.4, alpha=0.75, label='13-wk vol (ann.)')
    ax5b.axhline(VOL_THRESHOLD * 100, color='#9C27B0', ls='--', lw=0.9, alpha=0.6)
    ax5b.set_ylabel('Realized Vol Ann. (%)', fontsize=9, color='#9C27B0')
    ax5b.tick_params(axis='y', labelcolor='#9C27B0')
    ax5b.set_ylim(0, 120)

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    vol_thresh_line = Line2D([0], [0], color='#9C27B0', ls='--', lw=0.9,
                             label=f'Vol threshold ({VOL_THRESHOLD:.0%})')
    ax5.legend(handles=lines1 + lines2 + [vol_thresh_line],
               loc='upper left', fontsize=8, framealpha=0.85, edgecolor='#CCCCCC')

    ax5.set_ylabel('52-wk Return (%)', fontsize=9)
    ax5.grid(axis='y', lw=0.4, color='#DDDDDD', zorder=0)
    ax5.set_title('Panel 5 — 52-Week Rolling Return + 13-Week Realized Volatility',
                  fontsize=9, loc='left', pad=4)

    # ── x-axis formatting (bottom panel only)
    ax5.xaxis.set_major_locator(mdates.YearLocator(2))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax5.xaxis.set_minor_locator(mdates.YearLocator(1))
    plt.setp(ax5.get_xticklabels(), rotation=0, ha='center', fontsize=8.5)
    ax5.set_xlim(trim_start, trim_end)

    # ── Method summary annotation box
    summary_lines = ['Detection Method Summary:']
    for method, row in eval_df.iterrows():
        summary_lines.append(
            f"  {method:<10} Prec={row['precision']:.0%}  Rec={row['recall']:.0%}  F1={row['f1']:.0%}"
        )
    summary_text = '\n'.join(summary_lines)
    ax1.text(0.995, 0.03, summary_text,
             transform=ax1.transAxes,
             fontsize=7.5, family='monospace',
             va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#AAAAAA', alpha=0.88))

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print('=' * 68)
    print('  SPY BEAR MARKET ANALYSIS — 30 YEARS (1994-2026)')
    print('=' * 68)

    spy_weekly = fetch_spy_weekly()

    print('  Computing signals...')
    drawdown  = compute_rolling_drawdown(spy_weekly)
    sig_dd    = signal_drawdown(drawdown)
    sig_ma    = signal_ma200(spy_weekly)
    sig_comp  = signal_composite(spy_weekly)

    signal_grid = build_signal_grid(spy_weekly, sig_dd, sig_ma, sig_comp)

    print('  Evaluating methods...')
    eval_df, per_bear_df = evaluate_methods(signal_grid)
    print_evaluation_table(eval_df, per_bear_df)

    chart_path = os.path.join(_ROOT, 'spy_bear_analysis_chart.png')
    print(f'\n  Generating chart...')
    plot_bear_analysis(
        spy_weekly=spy_weekly[spy_weekly.index >= ANALYSIS_START],
        drawdown=drawdown[drawdown.index >= ANALYSIS_START],
        signal_grid=signal_grid,
        eval_df=eval_df,
        save_path=chart_path,
    )
    print(f'  Chart saved: {chart_path}')
    print('=' * 68)
    print('  DONE')
    print('=' * 68)


if __name__ == '__main__':
    main()
