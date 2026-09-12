"""
Bear Profit Research — Signal-Driven Multi-Asset Study
=======================================================
Given a bear-hunt signal fires, what is the best instrument to hold
to actively profit from the bear?

Tests 35 instruments across bonds, gold, inverse ETFs, forex,
volatility, commodities, and defensive sectors.

Three analyses:
  1. Raw per-bear returns (signal-agnostic baseline)
  2. Signal-driven 25-year performance (hold instrument when ON, cash when OFF)
  3. Composite basket performance

Usage
-----
  python scripts/run_bear_profit_research.py

Output
------
  bear_profit_research_chart.png
  bear_profit_report.md
"""

import os
import sys
import time
import warnings
import importlib.util
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HUNT_PATH = os.path.join(_ROOT, 'scripts', 'spy_bear_hunt.py')
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_BEARS = [
    ('1998-07-17', '1998-10-08', '1998 LTCM'),
    ('2000-03-24', '2002-10-09', '2000-02 Dot-com'),
    ('2007-10-09', '2009-03-09', '2007-09 GFC'),
    ('2018-09-20', '2018-12-24', '2018 Q4 Fed'),
    ('2020-02-19', '2020-03-23', '2020 COVID'),
    ('2022-01-03', '2022-10-13', '2022 Inflation'),
]
BEAR_LABELS = [b[2] for b in KNOWN_BEARS]

INSTRUMENTS = {
    'Bonds': {
        'TLT':  '20-yr Treasury',
        'IEF':  '7-10yr Treasury',
        'SHY':  '1-3yr Treasury',
        'ZROZ': '25+yr Zero Coupon',
        'TMF':  '3x TLT (levered)',
        'TIP':  'TIPS (inflation-prot.)',
        'BND':  'Total Bond Market',
    },
    'Gold & Metals': {
        'GLD':  'Gold ETF',
        'IAU':  'Gold (iShares)',
        'GDX':  'Gold Miners',
        'GDXJ': 'Jr Gold Miners',
        'SLV':  'Silver ETF',
    },
    'Inverse Equity': {
        'SH':   '1x Short S&P500',
        'SDS':  '2x Short S&P500',
        'SPXS': '3x Short S&P500',
        'PSQ':  'Short QQQ',
        'RWM':  'Short Russell 2000',
        'DOG':  'Short Dow',
    },
    'Forex': {
        'UUP':  'US Dollar Bull',
        'FXY':  'Japanese Yen',
        'FXF':  'Swiss Franc',
        'FXE':  'Euro',
    },
    'Volatility': {
        'VXX':  'VIX Futures (1x)',
        'UVXY': '2x VIX Futures',
    },
    'Commodities': {
        'DBC':  'Broad Commodities',
        'DBA':  'Agriculture',
        'USO':  'Crude Oil',
        'UNG':  'Natural Gas',
    },
    'Defensive Sectors': {
        'XLU':  'Utilities',
        'XLP':  'Consumer Staples',
        'XLV':  'Healthcare',
        'BIL':  '1-3M T-Bills',
        'USMV': 'Min Volatility',
    },
}

ALL_TICKERS = list(t for cat in INSTRUMENTS.values() for t in cat.keys())
ALL_TICKERS_PLUS_SPY = ['SPY'] + ALL_TICKERS

CATEGORY_COLORS = {
    'Bonds':            '#2196F3',
    'Gold & Metals':    '#FFC107',
    'Inverse Equity':   '#E91E63',
    'Forex':            '#9C27B0',
    'Volatility':       '#FF5722',
    'Commodities':      '#795548',
    'Defensive Sectors':'#4CAF50',
    'SPY':              '#9E9E9E',
}

BASKETS = {
    'Bonds+Gold':      {'TLT': 0.60, 'GLD': 0.40},
    'Gold+Short':      {'GLD': 0.40, 'SH':  0.60},
    'Full Defense':    {'TLT': 0.30, 'GLD': 0.30, 'SH':  0.40},
    'Safe Haven':      {'TLT': 0.50, 'GLD': 0.30, 'UUP': 0.20},
    'Currency Safe':   {'FXY': 0.40, 'TLT': 0.30, 'GLD': 0.30},
    'Cash+Gold':       {'BIL': 0.50, 'GLD': 0.50},
    'Ultra Defense':   {'TLT': 0.20, 'GLD': 0.30, 'SH':  0.30, 'UUP': 0.20},
    'Levered Bonds':   {'TMF': 0.50, 'GLD': 0.50},
    'Inflation Bear':  {'GLD': 0.40, 'SHY': 0.40, 'SH':  0.20},
    'Recession Bear':  {'TLT': 0.50, 'GLD': 0.30, 'SHY': 0.20},
    'Sector Defense':  {'XLU': 0.33, 'XLP': 0.34, 'XLV': 0.33},
    'Short+Vol Mix':   {'VXX': 0.30, 'TLT': 0.40, 'GLD': 0.30},
    'Forex Hedge':     {'FXY': 0.30, 'FXF': 0.30, 'TLT': 0.40},
    'GLD+SHY':         {'GLD': 0.50, 'SHY': 0.50},
    'SPY Baseline':    {'SPY': 1.00},
}

SIGNAL_NAMES = ['WEIGHTED_SCORE', 'RSI_BELOW40', 'FAST_OR_MACRO']

# ─────────────────────────────────────────────────────────────────────────────
# Signal cache
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_CACHE: dict = {}


def _get_signals() -> dict:
    if _SIGNAL_CACHE:
        return _SIGNAL_CACHE
    print('\n  [Signals] Building bear-hunt signals...')
    t0 = time.time()
    spec = importlib.util.spec_from_file_location('spy_bear_hunt', _HUNT_PATH)
    hunt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hunt)

    fred, yahoo  = hunt.fetch_extended_data()
    weekly_index = yahoo['SPY'].index
    bear_mask    = hunt._build_bear_mask(weekly_index)

    signals_orig = hunt._C.build_all_signals(fred, yahoo, weekly_index)
    eval_orig    = hunt.evaluate_all_signals(signals_orig, bear_mask)
    signals_new  = hunt.build_new_signals(fred, yahoo, weekly_index)
    signals_all  = {**signals_orig, **signals_new}
    signal_df    = pd.DataFrame(signals_all)

    ws, _, _  = hunt.compute_weighted_composite(signal_df, eval_orig, bear_mask)
    fom        = hunt.compute_fast_or_macro(signal_df, weekly_index)

    _SIGNAL_CACHE.update({
        'WEIGHTED_SCORE': ws,
        'RSI_BELOW40':    signals_all['RSI_BELOW40'],
        'FAST_OR_MACRO':  fom,
        'bear_mask':      bear_mask,
        'weekly_index':   weekly_index,
    })
    print(f'  [Signals] Done in {time.time()-t0:.0f}s')
    return _SIGNAL_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Price fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(start: str = '1994-01-01') -> pd.DataFrame:
    print(f'\n  [Prices] Downloading {len(ALL_TICKERS_PLUS_SPY)} tickers ({start} -> today)...')
    t0 = time.time()
    raw = yf.download(
        ALL_TICKERS_PLUS_SPY,
        start=start, end=datetime.now().strftime('%Y-%m-%d'),
        progress=False, auto_adjust=True, timeout=60,
    )
    close = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw
    if isinstance(close.columns, pd.MultiIndex):
        close = close.droplevel(0, axis=1) if close.columns.nlevels > 1 else close
    # flatten in case yfinance returns (metric, ticker)
    if hasattr(close.columns, 'levels'):
        close.columns = close.columns.droplevel(0)
    close = close.resample('W-FRI').last().ffill()
    print(f'  [Prices] {close.shape[1]} tickers, {len(close)} weeks  ({time.time()-t0:.0f}s)')
    return close


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 — Per-bear returns
# ─────────────────────────────────────────────────────────────────────────────

def compute_bear_period_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return % during each known bear. Index = tickers, cols = bear labels."""
    records = {}
    for s_str, e_str, label in KNOWN_BEARS:
        bear_start = pd.Timestamp(s_str)
        bear_end   = pd.Timestamp(e_str)
        col = {}
        for ticker in prices.columns:
            s = prices[ticker].dropna()
            if len(s) == 0 or s.index[0] > bear_start:
                col[ticker] = np.nan
                continue
            p0 = float(s.loc[:bear_start].iloc[-1])
            p_end = s.loc[:bear_end]
            if len(p_end) == 0:
                col[ticker] = np.nan
                continue
            p1 = float(p_end.iloc[-1])
            col[ticker] = round((p1 / p0 - 1) * 100, 1)
        records[label] = col
    df = pd.DataFrame(records)
    df.index.name = 'Ticker'
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Signal-driven metrics
# ─────────────────────────────────────────────────────────────────────────────

def _annualised_cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0)


def _max_drawdown(equity_curve: pd.Series) -> float:
    peak = np.maximum.accumulate(equity_curve.values)
    dd   = (equity_curve.values - peak) / peak
    return float(dd.min()) if len(dd) > 0 else 0.0


def _sharpe(rets: pd.Series, rf_weekly: float = 0.04 / 52) -> float:
    active = rets[rets != 0]
    if len(active) < 10:
        return 0.0
    excess = active - rf_weekly
    return float(excess.mean() / excess.std() * np.sqrt(52)) if excess.std() > 0 else 0.0


def signal_driven_metrics(
    instr_ret : pd.Series,
    signal    : pd.Series,
    bear_mask : pd.Series,
) -> dict:
    # Align signal and bear mask to instrument return index
    sig   = signal.reindex(instr_ret.index, method='ffill').fillna(False).astype(bool)
    bmask = bear_mask.reindex(instr_ret.index, method='ffill').fillna(False).astype(bool)

    signal_ret   = instr_ret.where(sig, 0.0)
    equity_curve = (1 + signal_ret).cumprod() * 100_000

    true_pos  = instr_ret[sig & bmask]
    false_alm = instr_ret[sig & ~bmask]

    bear_weeks  = int(bmask.sum())
    cov_weeks   = int((sig & bmask).sum())

    return {
        'cagr':            _annualised_cagr(equity_curve),
        'max_dd':          _max_drawdown(equity_curve),
        'sharpe':          _sharpe(signal_ret),
        'pct_time_on':     float(sig.mean()),
        'true_pos_ret':    float(true_pos.mean() * 52) if len(true_pos) > 0 else 0.0,
        'false_alarm_ret': float(false_alm.mean() * 52) if len(false_alm) > 0 else 0.0,
        'bear_coverage':   cov_weeks / bear_weeks if bear_weeks > 0 else 0.0,
        'equity_curve':    equity_curve,
    }


def compute_all_signal_metrics(prices: pd.DataFrame, signals: dict) -> dict:
    """
    Returns nested dict: results[signal_name][ticker] = metrics_dict
    Also adds basket metrics: results[signal_name]['_basket_' + name]
    """
    bear_mask    = signals['bear_mask']
    weekly_index = signals['weekly_index']

    results = {}
    for sig_name in SIGNAL_NAMES:
        sig  = signals[sig_name]
        results[sig_name] = {}
        print(f'  Computing metrics for signal={sig_name}...')

        # Individual instruments
        for ticker in ALL_TICKERS_PLUS_SPY:
            if ticker not in prices.columns:
                continue
            s = prices[ticker].dropna()
            # Need at least 3yr of history
            if len(s) < 156:
                continue
            ret = s.pct_change().dropna()
            m = signal_driven_metrics(ret, sig, bear_mask)
            results[sig_name][ticker] = m

        # Baskets
        for basket_name, weights in BASKETS.items():
            avail = {t: w for t, w in weights.items() if t in prices.columns}
            if not avail:
                continue
            total_w = sum(avail.values())
            avail   = {t: w / total_w for t, w in avail.items()}

            # Compute weighted basket return
            ret_df = pd.DataFrame({
                t: prices[t].pct_change() for t in avail if t in prices.columns
            }).dropna()
            if len(ret_df) < 156:
                continue
            basket_ret = sum(ret_df[t] * w for t, w in avail.items() if t in ret_df.columns)
            basket_ret.name = basket_name

            m = signal_driven_metrics(basket_ret, sig, bear_mask)
            m['weights'] = avail
            results[sig_name]['_basket_' + basket_name] = m

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Category helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ticker_category(ticker: str) -> str:
    for cat, tickers in INSTRUMENTS.items():
        if ticker in tickers:
            return cat
    return 'SPY'


def _ticker_label(ticker: str) -> str:
    for cat, tickers in INSTRUMENTS.items():
        if ticker in tickers:
            return f"{ticker} — {tickers[ticker]}"
    return ticker


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(bear_returns: pd.DataFrame, all_metrics: dict, save_path: str):
    lines = []

    def p(*args, **kw):
        text = ' '.join(str(a) for a in args)
        print(text, **kw)
        lines.append(text)

    p('=' * 80)
    p('  BEAR PROFIT RESEARCH — SIGNAL-DRIVEN MULTI-ASSET STUDY')
    p('=' * 80)
    p()

    # ── Section 1: Per-bear returns ──────────────────────────────────────────
    p('SECTION 1 — RAW RETURNS DURING EACH BEAR (hold from bear start to end)')
    p('-' * 80)

    # Print by category
    for cat, cat_tickers in INSTRUMENTS.items():
        p(f'\n  [{cat}]')
        hdr  = f"  {'Ticker':<8}"
        for bl in BEAR_LABELS:
            hdr += f'{bl[:15]:>16}'
        p(hdr)
        p('  ' + '-' * (8 + 16 * len(BEAR_LABELS)))

        for ticker in cat_tickers:
            if ticker not in bear_returns.index:
                continue
            row_str = f'  {ticker:<8}'
            for bl in BEAR_LABELS:
                val = bear_returns.loc[ticker, bl] if bl in bear_returns.columns else np.nan
                if np.isnan(val):
                    row_str += f"{'N/A':>16}"
                else:
                    flag = '*' if val > 0 else ' '
                    row_str += f'{val:>14.1f}%{flag}'
            p(row_str)

    p()
    p('  SPY (benchmark):')
    if 'SPY' in bear_returns.index:
        spy_row = '  SPY     '
        for bl in BEAR_LABELS:
            val = bear_returns.loc['SPY', bl] if bl in bear_returns.columns else np.nan
            spy_row += f'{val:>14.1f}% ' if not np.isnan(val) else f"{'N/A':>16}"
        p(spy_row)

    p()
    p('=' * 80)
    p('SECTION 2 — SIGNAL-DRIVEN PERFORMANCE (hold instrument when signal ON, cash otherwise)')
    p('=' * 80)

    metric_cols = ['CAGR', 'MaxDD', 'Sharpe', '%TimeON', 'TruePosRet', 'FalseAlmRet', 'BearCov']

    for sig_name in SIGNAL_NAMES:
        p(f'\n  Signal: {sig_name}')
        p('  ' + '-' * 78)

        # Collect individual instruments
        indiv = {}
        for ticker, m in all_metrics[sig_name].items():
            if ticker.startswith('_basket_'):
                continue
            indiv[ticker] = m

        # Rank by CAGR
        ranked = sorted(indiv.items(), key=lambda x: x[1].get('cagr', -999), reverse=True)

        p(f"  {'Ticker':<8} {'Cat':<16} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'%TimeON':>8} {'TruePos':>8} {'FalseAlm':>9} {'BearCov':>8}")
        p('  ' + '-' * 78)
        for ticker, m in ranked[:15]:
            cat = _ticker_category(ticker)
            p(f"  {ticker:<8} {cat[:16]:<16} "
              f"{m['cagr']*100:>6.1f}% "
              f"{m['max_dd']*100:>6.1f}% "
              f"{m['sharpe']:>7.2f} "
              f"{m['pct_time_on']*100:>7.0f}% "
              f"{m['true_pos_ret']*100:>7.1f}% "
              f"{m['false_alarm_ret']*100:>8.1f}% "
              f"{m['bear_coverage']*100:>7.0f}%")

        # VXX / UVXY warning
        vxx_m  = indiv.get('VXX')
        uvxy_m = indiv.get('UVXY')
        if vxx_m or uvxy_m:
            p()
            p('  *** VOLATILITY WARNING: VXX/UVXY show strong bear-period gains but ')
            p('  *** catastrophic 25yr CAGR due to VIX futures contango decay (~80%/yr). ')
            p('  *** Not viable as long-term holdings. Best used for short-term crisis only.')

        # Basket ranking
        p(f'\n  Top Baskets (signal={sig_name}):')
        baskets = {k: v for k, v in all_metrics[sig_name].items() if k.startswith('_basket_')}
        basket_ranked = sorted(baskets.items(), key=lambda x: x[1].get('cagr', -999), reverse=True)
        p(f"  {'Basket':<20} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'%TimeON':>8} {'BearCov':>8}")
        p('  ' + '-' * 60)
        for name, m in basket_ranked[:10]:
            bn = name.replace('_basket_', '')
            p(f"  {bn[:20]:<20} "
              f"{m['cagr']*100:>6.1f}% "
              f"{m['max_dd']*100:>6.1f}% "
              f"{m['sharpe']:>7.2f} "
              f"{m['pct_time_on']*100:>7.0f}% "
              f"{m['bear_coverage']*100:>7.0f}%")

    p()
    p('=' * 80)
    p('SECTION 3 — CROSS-SIGNAL WINNERS')
    p('=' * 80)

    # Find instruments with top CAGR across all 3 signals
    all_tickers_union = set()
    for sig_name in SIGNAL_NAMES:
        all_tickers_union.update(
            k for k in all_metrics[sig_name] if not k.startswith('_basket_'))

    cross = {}
    for ticker in all_tickers_union:
        cagrs = []
        for sig_name in SIGNAL_NAMES:
            m = all_metrics[sig_name].get(ticker)
            if m:
                cagrs.append(m['cagr'])
        if len(cagrs) == 3:
            cross[ticker] = np.mean(cagrs)

    cross_ranked = sorted(cross.items(), key=lambda x: x[1], reverse=True)[:10]
    p('\n  Top instruments by avg CAGR across all 3 signals:')
    p(f"  {'Ticker':<8} {'Category':<20} {'AvgCAGR':>8} | {'WS':>7} {'RSI':>7} {'FOM':>7}")
    p('  ' + '-' * 60)
    for ticker, avg_c in cross_ranked:
        vals = [all_metrics[sig].get(ticker, {}).get('cagr', np.nan) * 100
                for sig in SIGNAL_NAMES]
        cat = _ticker_category(ticker)
        p(f"  {ticker:<8} {cat[:20]:<20} {avg_c*100:>7.1f}% | "
          f"{vals[0]:>6.1f}% {vals[1]:>6.1f}% {vals[2]:>6.1f}%")

    # Best baskets across signals
    all_basket_keys = set()
    for sig_name in SIGNAL_NAMES:
        all_basket_keys.update(k for k in all_metrics[sig_name] if k.startswith('_basket_'))

    cross_basket = {}
    for bk in all_basket_keys:
        cagrs = []
        for sig_name in SIGNAL_NAMES:
            m = all_metrics[sig_name].get(bk)
            if m:
                cagrs.append(m['cagr'])
        if len(cagrs) == 3:
            cross_basket[bk] = np.mean(cagrs)

    cb_ranked = sorted(cross_basket.items(), key=lambda x: x[1], reverse=True)[:8]
    p('\n  Top BASKETS by avg CAGR across all 3 signals:')
    p(f"  {'Basket':<22} {'AvgCAGR':>8} | {'WS':>7} {'RSI':>7} {'FOM':>7}")
    p('  ' + '-' * 55)
    for bk, avg_c in cb_ranked:
        bn   = bk.replace('_basket_', '')
        vals = [all_metrics[sig].get(bk, {}).get('cagr', np.nan) * 100
                for sig in SIGNAL_NAMES]
        p(f"  {bn[:22]:<22} {avg_c*100:>7.1f}% | "
          f"{vals[0]:>6.1f}% {vals[1]:>6.1f}% {vals[2]:>6.1f}%")

    p()
    p('=' * 80)

    report_text = '\n'.join(lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f'# Bear Profit Research\n\n```\n{report_text}\n```\n')
    print(f'\n  Report saved: {save_path}')
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _shade_bears(ax, x_min=None, x_max=None):
    for s_str, e_str, _ in KNOWN_BEARS:
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if x_min and s < x_min:
            s = x_min
        if x_max and e > x_max:
            e = x_max
        if s < e:
            ax.axvspan(s, e, color='#FFCCCC', alpha=0.55, zorder=1)


def plot_research_chart(
    bear_returns  : pd.DataFrame,
    all_metrics   : dict,
    prices        : pd.DataFrame,
    signals       : dict,
    save_path     : str,
):
    fig = plt.figure(figsize=(24, 22))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle(
        'Bear Profit Research — Signal-Driven Multi-Asset Study\n'
        'When each bear-hunt signal fires, which instrument should you hold to profit?',
        fontsize=11, fontweight='bold', y=0.999,
    )

    gs = GridSpec(4, 2, figure=fig,
                  height_ratios=[0.32, 0.27, 0.24, 0.17],
                  hspace=0.18, wspace=0.10)

    # ── Panel 1 (full width): Per-bear instrument heatmap ────────────────────
    ax_hm = fig.add_subplot(gs[0, :])
    ax_hm.set_facecolor('#F8F8F8')

    # Build ordered list: SPY first, then categories
    ordered_tickers = ['SPY']
    category_breaks = {}  # row index where new category starts
    for cat, cat_tickers in INSTRUMENTS.items():
        category_breaks[len(ordered_tickers)] = cat
        ordered_tickers.extend([t for t in cat_tickers if t in bear_returns.index])

    # Filter to available data
    ordered_tickers = [t for t in ordered_tickers if t in bear_returns.index]
    n_rows = len(ordered_tickers)
    n_cols = len(BEAR_LABELS)

    heat_data = np.full((n_rows, n_cols), np.nan)
    for i, ticker in enumerate(ordered_tickers):
        for j, label in enumerate(BEAR_LABELS):
            if label in bear_returns.columns and not np.isnan(bear_returns.loc[ticker, label]):
                heat_data[i, j] = bear_returns.loc[ticker, label]

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    im   = ax_hm.imshow(heat_data, cmap=cmap, norm=norm, aspect='auto')

    ax_hm.set_xticks(range(n_cols))
    ax_hm.set_xticklabels(BEAR_LABELS, fontsize=8)
    ax_hm.set_yticks(range(n_rows))
    ax_hm.set_yticklabels(ordered_tickers, fontsize=7)

    # Category labels on left margin
    for row_idx, cat in category_breaks.items():
        if row_idx < n_rows:
            ax_hm.axhline(row_idx - 0.5, color='white', lw=1.5)
            ax_hm.text(-0.8, row_idx, cat, fontsize=6.5, ha='right', va='top',
                       color=CATEGORY_COLORS.get(cat, 'black'), fontweight='bold',
                       transform=ax_hm.get_yaxis_transform())

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            val = heat_data[i, j]
            if not np.isnan(val):
                txt_color = 'white' if abs(val) > 35 else 'black'
                ax_hm.text(j, i, f'{val:.0f}%', ha='center', va='center',
                           fontsize=6, color=txt_color)
            else:
                ax_hm.text(j, i, 'N/A', ha='center', va='center',
                           fontsize=5.5, color='#888888')

    plt.colorbar(im, ax=ax_hm, label='Return (%)', fraction=0.015, pad=0.01)
    ax_hm.set_title(
        'Panel 1 — Raw Returns (%) During Each Bear Period  '
        '(green = positive / profited, red = lost, N/A = ETF not yet launched)',
        fontsize=8.5, loc='left', pad=3)

    # ── Panel 2 left: Signal-driven equity curves (WeightedScore) ────────────
    ax_eq = fig.add_subplot(gs[1, 0])
    ax_eq.set_facecolor('#F8F8F8')

    sig_name_eq = 'WEIGHTED_SCORE'
    instr_metrics = {k: v for k, v in all_metrics[sig_name_eq].items()
                     if not k.startswith('_basket_') and 'equity_curve' in v}
    ranked_instr  = sorted(instr_metrics.items(),
                           key=lambda x: x[1].get('cagr', -999), reverse=True)

    # Add SPY buy-hold equity curve (from prices)
    if 'SPY' in prices.columns:
        spy_s  = prices['SPY'].dropna()
        spy_eq = spy_s / spy_s.iloc[0] * 100_000
        spy_eq = spy_eq.reindex(ranked_instr[0][1]['equity_curve'].index, method='ffill')
        ax_eq.semilogy(spy_eq.index, spy_eq,
                       color='#9E9E9E', lw=1.5, ls='--', label='SPY buy-hold')

    _shade_bears(ax_eq)
    top5_colors = ['#E91E63', '#4CAF50', '#2196F3', '#FFC107', '#FF5722']
    for i, (ticker, m) in enumerate(ranked_instr[:5]):
        ec = m['equity_curve'].dropna()
        ax_eq.semilogy(ec.index, ec,
                       color=top5_colors[i], lw=1.8,
                       label=f'{ticker} ({m["cagr"]*100:.1f}%)')

    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    ax_eq.legend(fontsize=7, loc='upper left', framealpha=0.90)
    ax_eq.set_title(f'Panel 2 — Signal-Driven Equity Curves (signal={sig_name_eq})\n'
                    'hold instrument when ON, cash when OFF. Pink = bear.',
                    fontsize=7.5, loc='left', pad=2)
    ax_eq.grid(which='both', lw=0.3, color='#DDDDDD')
    ax_eq.set_xlim(ranked_instr[0][1]['equity_curve'].index[0],
                   ranked_instr[0][1]['equity_curve'].index[-1])

    # ── Panel 2 right: Basket CAGR bar chart ─────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_facecolor('#F8F8F8')

    basket_metrics_ws = {
        k.replace('_basket_', ''): v
        for k, v in all_metrics['WEIGHTED_SCORE'].items()
        if k.startswith('_basket_')
    }
    basket_names = list(basket_metrics_ws.keys())
    basket_cagrs = [basket_metrics_ws[b]['cagr'] * 100 for b in basket_names]
    bar_colors   = ['#4CAF50' if c > 0 else '#F44336' for c in basket_cagrs]

    bars = ax_bar.barh(range(len(basket_names)), basket_cagrs,
                       color=bar_colors, alpha=0.80)
    if 'SPY' in prices.columns:
        spy_ret = prices['SPY'].dropna()
        spy_years = (spy_ret.index[-1] - spy_ret.index[0]).days / 365.25
        spy_cagr = (spy_ret.iloc[-1] / spy_ret.iloc[0]) ** (1 / spy_years) - 1
        ax_bar.axvline(spy_cagr * 100, color='#9E9E9E', lw=1.5, ls='--',
                       label=f'SPY {spy_cagr*100:.1f}%')

    ax_bar.set_yticks(range(len(basket_names)))
    ax_bar.set_yticklabels(basket_names, fontsize=7.5)
    ax_bar.set_xlabel('CAGR (%)', fontsize=8)
    ax_bar.legend(fontsize=7)
    ax_bar.grid(axis='x', lw=0.3, color='#DDDDDD')
    ax_bar.set_title(f'Panel 3 — Basket CAGR (signal={sig_name_eq})\n'
                     'hold basket when signal ON, cash otherwise',
                     fontsize=7.5, loc='left', pad=2)

    # ── Panel 3 left: Cross-signal comparison table ───────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 0])
    ax_tbl.axis('off')
    ax_tbl.set_title('Panel 4 — Top Instruments: CAGR Across 3 Signals',
                     fontsize=7.5, loc='left', pad=2)

    # Build cross-signal ranking
    union_tickers = set()
    for sn in SIGNAL_NAMES:
        union_tickers.update(k for k in all_metrics[sn] if not k.startswith('_basket_'))

    cross_rows = []
    for ticker in union_tickers:
        cagrs = [all_metrics[sn].get(ticker, {}).get('cagr', np.nan) * 100
                 for sn in SIGNAL_NAMES]
        if not any(np.isnan(c) for c in cagrs):
            cross_rows.append((ticker, cagrs, np.mean(cagrs)))
    cross_rows.sort(key=lambda x: x[2], reverse=True)

    col_hdrs = ['Ticker', 'Category', 'WeightedScore', 'RSI_Below40', 'FastOrMacro', 'Avg']
    tbl_data = []
    tbl_colors = []
    for ticker, cagrs, avg_c in cross_rows[:12]:
        cat = _ticker_category(ticker)
        row = [ticker, cat[:14]] + [f'{c:.1f}%' for c in cagrs] + [f'{avg_c:.1f}%']
        tbl_data.append(row)
        row_colors = ['#EEEEEE', '#EEEEEE']
        for c in cagrs:
            if c > 5:
                row_colors.append('#C8E6C9')
            elif c < 0:
                row_colors.append('#FFCDD2')
            else:
                row_colors.append('#FFFFFF')
        row_colors.append('#E3F2FD' if avg_c > 3 else '#FFFFFF')
        tbl_colors.append(row_colors)

    tbl = ax_tbl.table(
        cellText=tbl_data, colLabels=col_hdrs,
        cellLoc='center', loc='center',
        cellColours=tbl_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.4)
    for j in range(len(col_hdrs)):
        tbl[(0, j)].set_facecolor('#37474F')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # ── Panel 3 right: Efficiency scatter ────────────────────────────────────
    ax_sc = fig.add_subplot(gs[2, 1])
    ax_sc.set_facecolor('#F8F8F8')

    sig_name_sc = 'FAST_OR_MACRO'
    for cat, cat_tickers in INSTRUMENTS.items():
        color = CATEGORY_COLORS.get(cat, 'grey')
        for ticker in cat_tickers:
            m = all_metrics[sig_name_sc].get(ticker)
            if m is None:
                continue
            cagr   = m['cagr'] * 100
            max_dd = abs(m['max_dd']) * 100
            cov    = m['bear_coverage'] * 100
            ax_sc.scatter([max_dd], [cagr],
                          s=max(20, cov * 4),
                          color=color, alpha=0.80, zorder=4)
            if cagr > 2.0 or max_dd < 5.0:
                ax_sc.annotate(ticker, (max_dd, cagr),
                               fontsize=6.5, ha='left', va='bottom',
                               xytext=(3, 3), textcoords='offset points')

    # Category legend
    handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c)
               for c in INSTRUMENTS.keys()]
    ax_sc.legend(handles=handles, fontsize=6, loc='upper right')
    ax_sc.set_xlabel('Max Drawdown (%)', fontsize=8)
    ax_sc.set_ylabel('CAGR (%)', fontsize=8)
    ax_sc.grid(lw=0.3, color='#DDDDDD')
    ax_sc.set_title(f'Panel 5 — Efficiency Scatter (signal={sig_name_sc})\n'
                    'x = max drawdown, y = CAGR, size = bear coverage',
                    fontsize=7.5, loc='left', pad=2)
    ax_sc.axhline(0, color='black', lw=0.8)

    # ── Panel 4 (full width): Signal timelines with bear shading ─────────────
    gs_bot = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3, :], hspace=0.05)

    for i, sig_name in enumerate(SIGNAL_NAMES):
        ax_sig = fig.add_subplot(gs_bot[i])
        sig    = signals[sig_name]

        # Fill signal ON periods
        x      = sig.index
        y_on   = sig.astype(float).values
        ax_sig.fill_between(x, 0, y_on, step='post',
                            color='#1976D2', alpha=0.45, label='Signal ON')
        _shade_bears(ax_sig)

        # Best basket weekly return line for this signal
        best_bk_key = max(
            (k for k in all_metrics[sig_name] if k.startswith('_basket_')),
            key=lambda k: all_metrics[sig_name][k].get('cagr', -999),
            default=None,
        )
        if best_bk_key:
            ec = all_metrics[sig_name][best_bk_key].get('equity_curve')
            if ec is not None and len(ec) > 2:
                ax_sig2 = ax_sig.twinx()
                ax_sig2.plot(ec.index, ec / ec.iloc[0] * 100,
                             color='#E91E63', lw=1.2, alpha=0.85,
                             label=best_bk_key.replace('_basket_', ''))
                ax_sig2.set_ylabel('Basket (norm 100)', fontsize=6, color='#E91E63')
                ax_sig2.tick_params(axis='y', labelsize=6, colors='#E91E63')
                ax_sig2.legend(loc='upper right', fontsize=6)

        ax_sig.set_ylabel(sig_name.replace('_', '\n'), fontsize=6.5)
        ax_sig.set_ylim(-0.05, 1.3)
        ax_sig.set_yticks([])
        ax_sig.set_xlim(x[0], x[-1])
        if i < 2:
            ax_sig.set_xticks([])
        else:
            ax_sig.xaxis.set_major_locator(mdates.YearLocator(2))
            ax_sig.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        if i == 0:
            ax_sig.set_title('Panel 6 — Signal Timelines (blue = signal ON, pink = bear, red = best basket equity)',
                             fontsize=7.5, loc='left', pad=2)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\n  Chart saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print('=' * 70)
    print('  BEAR PROFIT RESEARCH')
    print('=' * 70)
    t_total = time.time()

    # 1. Build signals
    signals = _get_signals()

    # 2. Fetch prices
    prices = fetch_prices(start='1993-01-01')

    # Fix yfinance multi-index if needed
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)

    # 3. Analysis 1: per-bear returns
    print('\n  [Analysis 1] Computing per-bear returns...')
    bear_returns = compute_bear_period_returns(prices)
    print(f'    {bear_returns.shape[0]} instruments × {bear_returns.shape[1]} bears')

    # 4. Analysis 2+3: signal-driven metrics + baskets
    print('\n  [Analysis 2+3] Computing signal-driven metrics...')
    all_metrics = compute_all_signal_metrics(prices, signals)

    # 5. Report
    report_path = os.path.join(_ROOT, 'bear_profit_report.md')
    print_report(bear_returns, all_metrics, report_path)

    # 6. Chart
    chart_path = os.path.join(_ROOT, 'bear_profit_research_chart.png')
    print('\n  [Chart] Generating...')
    plot_research_chart(bear_returns, all_metrics, prices, signals, chart_path)

    print(f'\n  Total elapsed: {(time.time()-t_total)/60:.1f} min')
    print('=' * 70)
    print('  DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()
