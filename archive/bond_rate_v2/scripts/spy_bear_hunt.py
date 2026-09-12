"""
SPY Bear Hunt — Catching All 6 Bears
=====================================
Extends the combo analysis with fast/velocity signals, cross-asset breadth,
and composite approaches designed to catch the 3 "fast bears":
  1998 LTCM, 2018 Q4 Fed, 2020 COVID

New signal categories:
  A) Fast/velocity (min_dur=1): VIX surge, drawdown velocity, credit dislocation
  B) Cross-asset breadth: IWM/SPY, sector rotation, emerging markets
  C) Macro velocity: yield spike, credit OAS acceleration, rate shock
  D) Composites: F1-weighted score, domain-diversity vote, FAST_OR_MACRO

Output:
  spy_bear_hunt_chart.png   (22x28, 150 dpi)
  Console: per-bear coverage matrix, all-bears ranking
"""

import os
import sys
import warnings
import importlib.util
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import yfinance as yf

warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from dotenv import load_dotenv; load_dotenv()

# ── Dynamic import of existing combo analysis ─────────────────────────────────
_COMBO_PATH = os.path.join(_ROOT, 'scripts', 'spy_bear_combo_analysis.py')
spec = importlib.util.spec_from_file_location('spy_combo', _COMBO_PATH)
_C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_C)

_apply_min_duration = _C._apply_min_duration
_build_bear_mask    = _C._build_bear_mask
_nearest            = _C._nearest
_shade_known_bears  = _C._shade_known_bears
_zscore_level       = _C._zscore_level
_zscore_momentum    = _C._zscore_momentum
evaluate_signal     = _C.evaluate_signal
evaluate_all_signals= _C.evaluate_all_signals
compute_lead_time   = _C.compute_lead_time

KNOWN_BEARS       = _C.KNOWN_BEARS
ANALYSIS_START    = _C.ANALYSIS_START
MONTHLY_LAG_WEEKS = _C.MONTHLY_LAG_WEEKS

# Extended domain map and colors
DOMAIN_MAP    = dict(_C.DOMAIN_MAP)
DOMAIN_COLORS = dict(_C.DOMAIN_COLORS)

NEW_DOMAIN_MAP = {
    'VIX_SURGE_FAST':    'Fast/Velocity',
    'DRAWDOWN_FAST':     'Fast/Velocity',
    'CREDIT_VEL_FAST':   'Fast/Velocity',
    'TLT_SURGE_FAST':    'Fast/Velocity',
    'ATR_SPIKE':         'Fast/Velocity',
    'SMALL_CAP_WEAK':    'Cross-Asset',
    'DEFENSIVE_ROT':     'Cross-Asset',
    'EEM_STRESS':        'Cross-Asset',
    'TECH_ROLLOVER':     'Cross-Asset',
    'YIELD_SPIKE':       'Macro Velocity',
    'CURVE_STEEPEN_INV': 'Macro Velocity',
    'CREDIT_OAS_ACCEL':  'Macro Velocity',
    'FED_RATE_SHOCK':    'Macro Velocity',
    'REAL_RATE_SPIKE':   'Macro Velocity',
    'WEIGHTED_SCORE':    'Composite',
    'DOMAIN_DIVERSE':    'Composite',
    'FAST_OR_MACRO':     'Composite',
}
DOMAIN_MAP.update(NEW_DOMAIN_MAP)
DOMAIN_COLORS.update({
    'Fast/Velocity': '#FF6B6B',
    'Cross-Asset':   '#6BCB77',
    'Macro Velocity':'#4D96FF',
    'Composite':     '#FFD93D',
})

EXTRA_YAHOO = ['IWM', 'XLU', 'XLP', 'XLY', 'XLK', 'EEM', 'IVV', 'XLV']


# ─────────────────────────────────────────────────────────────────────────────
# Extended data fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_extended_data():
    print('  Fetching base data (FRED + Yahoo)...')
    fred, yahoo = _C.fetch_all_data()

    print('  Fetching extended sector/breadth tickers...')
    end_str = datetime.today().strftime('%Y-%m-%d')
    try:
        raw = yf.download(EXTRA_YAHOO, start=_C.DATA_START_STR, end=end_str,
                          auto_adjust=True, progress=False, timeout=60)
        close = raw['Close']
        if isinstance(close, pd.DataFrame):
            for tk in close.columns:
                s = close[tk].dropna()
                if len(s) > 52:
                    yahoo[str(tk)] = s.resample('W-FRI').last().ffill().dropna()
    except Exception as e:
        print(f'    Batch failed: {e}')
        for tk in EXTRA_YAHOO:
            if tk not in yahoo:
                try:
                    df = yf.download(tk, start=_C.DATA_START_STR, end=end_str,
                                     auto_adjust=True, progress=False, timeout=20)
                    if df is not None and not df.empty:
                        s = df['Close']
                        if isinstance(s, pd.DataFrame):
                            s = s.iloc[:, 0]
                        if len(s) > 52:
                            yahoo[tk] = s.resample('W-FRI').last().ffill().dropna()
                except Exception:
                    pass

    print(f'    Total: {len(fred)} FRED + {len(yahoo)} Yahoo')
    return fred, yahoo


# ─────────────────────────────────────────────────────────────────────────────
# Category A: Fast/Velocity signals (min_duration=1)
# ─────────────────────────────────────────────────────────────────────────────
def sig_VIX_SURGE_FAST(vix):
    if vix is None or vix.empty:
        return pd.Series(dtype=bool, name='VIX_SURGE_FAST')
    z = _zscore_momentum(vix, lookback=2)
    return _apply_min_duration(z > 2.5, 1).rename('VIX_SURGE_FAST')


def sig_DRAWDOWN_FAST(spy_w):
    ret4 = spy_w.pct_change(4)
    return _apply_min_duration(ret4 < -0.07, 1).rename('DRAWDOWN_FAST')


def sig_CREDIT_VEL_FAST(hyg, lqd):
    if hyg is None or lqd is None or hyg.empty or lqd.empty:
        return pd.Series(dtype=bool, name='CREDIT_VEL_FAST')
    ratio = hyg / lqd.reindex(hyg.index).ffill()
    z = _zscore_momentum(ratio, lookback=2)
    return _apply_min_duration(z < -2.0, 1).rename('CREDIT_VEL_FAST')


def sig_TLT_SURGE_FAST(tlt):
    if tlt is None or tlt.empty:
        return pd.Series(dtype=bool, name='TLT_SURGE_FAST')
    z = _zscore_momentum(tlt, lookback=2)
    return _apply_min_duration(z > 2.0, 1).rename('TLT_SURGE_FAST')


def sig_ATR_SPIKE(spy_w):
    abs_ret = spy_w.pct_change().abs()
    z = _zscore_level(abs_ret)
    return _apply_min_duration(z > 2.0, 2).rename('ATR_SPIKE')


# ─────────────────────────────────────────────────────────────────────────────
# Category B: Cross-asset breadth (min_duration=2)
# ─────────────────────────────────────────────────────────────────────────────
def sig_SMALL_CAP_WEAK(iwm, spy_w):
    if iwm is None or iwm.empty:
        return pd.Series(dtype=bool, name='SMALL_CAP_WEAK')
    ratio = iwm.reindex(spy_w.index).ffill() / spy_w
    z = _zscore_momentum(ratio, lookback=8)
    return _apply_min_duration(z < -1.0, 2).rename('SMALL_CAP_WEAK')


def sig_DEFENSIVE_ROT(xlu, xlp, xly, xlk, spy_w):
    if any(x is None or x.empty for x in [xlu, xlp, xly, xlk]):
        return pd.Series(dtype=bool, name='DEFENSIVE_ROT')
    idx      = spy_w.index
    def_avg  = (xlu.reindex(idx).ffill() + xlp.reindex(idx).ffill()) / 2
    risk_avg = (xly.reindex(idx).ffill() + xlk.reindex(idx).ffill()) / 2
    ratio    = def_avg / (risk_avg + 1e-9)
    z        = _zscore_momentum(ratio, lookback=8)
    return _apply_min_duration(z > 1.0, 2).rename('DEFENSIVE_ROT')


def sig_EEM_STRESS(eem, spy_w):
    if eem is None or eem.empty:
        return pd.Series(dtype=bool, name='EEM_STRESS')
    rel = eem.reindex(spy_w.index).ffill() / spy_w
    z   = _zscore_momentum(rel, lookback=8)
    return _apply_min_duration(z < -1.2, 2).rename('EEM_STRESS')


def sig_TECH_ROLLOVER(xlk, spy_w):
    if xlk is None or xlk.empty:
        return pd.Series(dtype=bool, name='TECH_ROLLOVER')
    rel = xlk.reindex(spy_w.index).ffill() / spy_w
    z   = _zscore_momentum(rel, lookback=8)
    return _apply_min_duration(z < -1.2, 2).rename('TECH_ROLLOVER')


# ─────────────────────────────────────────────────────────────────────────────
# Category C: Macro velocity (min_duration=2)
# ─────────────────────────────────────────────────────────────────────────────
def sig_YIELD_SPIKE(dgs10):
    if dgs10 is None or dgs10.empty:
        return pd.Series(dtype=bool, name='YIELD_SPIKE')
    delta = dgs10.diff(4).abs()
    return _apply_min_duration(delta > 0.50, 2).rename('YIELD_SPIKE')


def sig_CURVE_STEEPEN_INV(t10y2y):
    """Re-steepening after deep inversion = recession arriving."""
    if t10y2y is None or t10y2y.empty:
        return pd.Series(dtype=bool, name='CURVE_STEEPEN_INV')
    inverted = t10y2y < -0.50
    rising   = t10y2y.diff(4) > 0
    return _apply_min_duration(inverted & rising, 2).rename('CURVE_STEEPEN_INV')


def sig_CREDIT_OAS_ACCEL(hy_oas):
    if hy_oas is None or hy_oas.empty:
        return pd.Series(dtype=bool, name='CREDIT_OAS_ACCEL')
    z = _zscore_momentum(hy_oas, lookback=2)
    return _apply_min_duration(z > 1.5, 2).rename('CREDIT_OAS_ACCEL')


def sig_FED_RATE_SHOCK(fedfunds):
    if fedfunds is None or fedfunds.empty:
        return pd.Series(dtype=bool, name='FED_RATE_SHOCK')
    lag     = fedfunds.shift(MONTHLY_LAG_WEEKS)
    delta13 = lag.diff(13)
    return _apply_min_duration(delta13 > 1.5, 2).rename('FED_RATE_SHOCK')


def sig_REAL_RATE_SPIKE(dfii5):
    if dfii5 is None or dfii5.empty:
        return pd.Series(dtype=bool, name='REAL_RATE_SPIKE')
    delta4 = dfii5.diff(4).abs()
    return _apply_min_duration(delta4 > 0.50, 2).rename('REAL_RATE_SPIKE')


# ─────────────────────────────────────────────────────────────────────────────
# Build new signals
# ─────────────────────────────────────────────────────────────────────────────
def build_new_signals(fred, yahoo, weekly_index):
    spy_w = yahoo.get('SPY')
    vix   = fred.get('VIXCLS')
    if vix is None or vix.empty:
        vix = yahoo.get('^VIX')

    raw = {
        'VIX_SURGE_FAST':    sig_VIX_SURGE_FAST(vix),
        'DRAWDOWN_FAST':     sig_DRAWDOWN_FAST(spy_w),
        'CREDIT_VEL_FAST':   sig_CREDIT_VEL_FAST(yahoo.get('HYG'), yahoo.get('LQD')),
        'TLT_SURGE_FAST':    sig_TLT_SURGE_FAST(yahoo.get('TLT')),
        'ATR_SPIKE':         sig_ATR_SPIKE(spy_w),
        'SMALL_CAP_WEAK':    sig_SMALL_CAP_WEAK(yahoo.get('IWM'), spy_w),
        'DEFENSIVE_ROT':     sig_DEFENSIVE_ROT(yahoo.get('XLU'), yahoo.get('XLP'),
                                               yahoo.get('XLY'), yahoo.get('XLK'), spy_w),
        'EEM_STRESS':        sig_EEM_STRESS(yahoo.get('EEM'), spy_w),
        'TECH_ROLLOVER':     sig_TECH_ROLLOVER(yahoo.get('XLK'), spy_w),
        'YIELD_SPIKE':       sig_YIELD_SPIKE(fred.get('DGS10')),
        'CURVE_STEEPEN_INV': sig_CURVE_STEEPEN_INV(fred.get('T10Y2Y')),
        'CREDIT_OAS_ACCEL':  sig_CREDIT_OAS_ACCEL(fred.get('BAMLH0A0HYM2')),
        'FED_RATE_SHOCK':    sig_FED_RATE_SHOCK(fred.get('FEDFUNDS')),
        'REAL_RATE_SPIKE':   sig_REAL_RATE_SPIKE(fred.get('DFII5')),
    }

    result = {}
    for name, s in raw.items():
        if s is None or (hasattr(s, 'empty') and s.empty):
            result[name] = pd.Series(False, index=weekly_index, name=name)
        else:
            result[name] = s.reindex(weekly_index).fillna(False).astype(bool)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Composite signal builders
# ─────────────────────────────────────────────────────────────────────────────
def compute_weighted_composite(signal_df, eval_df, bear_mask,
                                thresholds=None):
    """F1-weighted sum of top-15 signals across multiple thresholds."""
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    top15 = [n for n in eval_df.head(15).index if n in signal_df.columns]
    weights = eval_df.loc[top15, 'f1'].values

    grid  = signal_df[top15].astype(float)
    score = (grid * weights).sum(axis=1) / (weights.sum() + 1e-9)

    best_f1, best_thresh, best_sig = -1, 0.30, None

    for thresh in thresholds:
        raw = score > thresh
        sig = _apply_min_duration(raw, 1)
        r   = evaluate_signal(f'WEIGHTED_{thresh:.2f}', sig, bear_mask)
        if r['bears_caught'] == len(KNOWN_BEARS) and r['f1'] > best_f1:
            best_f1    = r['f1']
            best_thresh = thresh
            best_sig   = sig

    if best_sig is None:
        best_sig = _apply_min_duration(score > 0.30, 1)
        best_thresh = 0.30

    return best_sig.rename('WEIGHTED_SCORE'), score, best_thresh


def compute_domain_diversity_vote(signal_df, domain_map, min_domains=2, min_total=3):
    """Fire when ≥ min_domains distinct domains each have ≥1 active signal."""
    def _fires(row):
        active  = row.index[row.astype(bool)].tolist()
        if len(active) < min_total:
            return False
        domains = set(domain_map.get(n, 'Unknown') for n in active)
        return len(domains) >= min_domains

    raw = signal_df.apply(_fires, axis=1)
    return _apply_min_duration(raw, 2).rename('DOMAIN_DIVERSE')


def compute_fast_or_macro(signal_df, weekly_index):
    """Fast signal active  OR  (MA200 active AND any confirming signal)."""
    fast_names  = ['VIX_SURGE_FAST', 'DRAWDOWN_FAST', 'CREDIT_VEL_FAST', 'TLT_SURGE_FAST']
    confirmers  = ['RETURN_NEG52', 'VIX_HIGH20', 'STLFSI_HIGH', 'SENTIMENT_WEAK',
                   'CPI_HIGH3', 'REALIZED_VOL', 'ATR_SPIKE']

    false_s = pd.Series(False, index=weekly_index)

    fast_any = false_s.copy()
    for fn in fast_names:
        if fn in signal_df.columns:
            fast_any = fast_any | signal_df[fn]

    ma200 = signal_df.get('MA200', false_s)

    macro_any = false_s.copy()
    for mn in confirmers:
        if mn in signal_df.columns:
            macro_any = macro_any | signal_df[mn]

    raw = fast_any | (ma200 & macro_any)
    return _apply_min_duration(raw, 1).rename('FAST_OR_MACRO')


# ─────────────────────────────────────────────────────────────────────────────
# Coverage analysis
# ─────────────────────────────────────────────────────────────────────────────
def per_bear_coverage(signal, weekly_index):
    sig = signal.reindex(weekly_index).fillna(False).astype(bool)
    result = {}
    for s_str, e_str, label in KNOWN_BEARS:
        window = sig.loc[pd.Timestamp(s_str):pd.Timestamp(e_str)]
        result[label] = window.sum() / max(len(window), 1) * 100
    return result


def build_coverage_matrix(signals_dict, weekly_index):
    bear_labels = [b[2] for b in KNOWN_BEARS]
    rows = {name: per_bear_coverage(sig, weekly_index)
            for name, sig in signals_dict.items()}
    df = pd.DataFrame(rows, index=bear_labels).T  # (n_signals, 6)
    return df


def build_penalized_matrix(coverage_df, eval_df_all):
    """
    Penalized per-bear score = raw recall × overall precision.
    This cuts down always-on signals (e.g. DOMAIN_DIVERSE: 100% recall × 19% precision = 19%).
    A signal must be both present during a bear AND selective outside bears to score well.
    """
    bear_labels = [b[2] for b in KNOWN_BEARS]
    pen = coverage_df[bear_labels].copy()
    for name in pen.index:
        prec = eval_df_all.loc[name, 'prec'] if name in eval_df_all.index else 0.0
        pen.loc[name] = pen.loc[name] * prec
    return pen


def all_bears_composite_score(eval_row):
    # Pure F1 — already penalizes false signals via precision component
    return eval_row['f1']


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────
def print_coverage_matrix(coverage_df, penalized_df, eval_df_all):
    bear_labels = [b[2] for b in KNOWN_BEARS]

    # Sort by F1 (penalizes false signals) among those catching ≥5 bears, then mean penalized score
    pen = penalized_df[bear_labels].copy()
    pen['_bc']     = (coverage_df[bear_labels] > 10).sum(axis=1)
    pen['_f1']     = pen.index.map(lambda n: eval_df_all.loc[n, 'f1'] if n in eval_df_all.index else 0)
    pen['_pen_avg']= pen[bear_labels].mean(axis=1)
    pen = pen.sort_values(['_f1', '_pen_avg'], ascending=[False, False])

    print()
    print('=' * 100)
    print('  ALL-BEARS PENALIZED COVERAGE  (penalized = recall × precision per bear)')
    print('  Sorts by F1 — always-on signals are penalized by their low precision.')
    print('=' * 100)
    hdr = (f"  {'Signal':<22} {'1998':>6} {'2000':>6} {'2007':>6} {'2018':>6} "
           f"{'2020':>6} {'2022':>6}  {'All6':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(hdr)
    print('  ' + '-' * 96)
    for name in pen.index[:35]:
        pen_vals  = [f"{pen.loc[name, bl]:>5.0f}%" for bl in bear_labels]
        bc        = int(pen.loc[name, '_bc'])
        if name in eval_df_all.index:
            prec = eval_df_all.loc[name, 'prec'] * 100
            rec  = eval_df_all.loc[name, 'rec']  * 100
            f1   = eval_df_all.loc[name, 'f1']   * 100
        else:
            prec = rec = f1 = 0.0
        bc_str = f'{bc}/6'
        star   = ' *' if bc == 6 else ''
        print(f"  {name:<22} {'  '.join(pen_vals)}  {bc_str:>5}{star}  {prec:>5.1f}%  {rec:>5.1f}%  {f1:>5.1f}%")
    print('  * = catches all 6 bears  |  values shown = recall × precision (penalized score)')
    print('=' * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def _draw_signal_panel(fig, outer_cell, spy_w, signal, label, eval_result,
                        bear_mask, score_series=None, score_thresh=None):
    """Generic panel for one signal with entry/exit markers and optional score twinx."""
    has_score = score_series is not None

    if has_score:
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_cell,
                                           height_ratios=[0.55, 0.45], hspace=0.04)
        ax_p = fig.add_subplot(inner_gs[0])
        ax_s = fig.add_subplot(inner_gs[1], sharex=ax_p)
    else:
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_cell,
                                           height_ratios=[0.55, 0.45], hspace=0.04)
        ax_p = fig.add_subplot(inner_gs[0])
        ax_s = fig.add_subplot(inner_gs[1], sharex=ax_p)

    trim_start = pd.Timestamp(ANALYSIS_START)
    trim_end   = spy_w.index[-1]

    spy = spy_w.loc[trim_start:trim_end]
    sig = signal.reindex(spy.index).fillna(False)
    bm  = bear_mask.reindex(spy.index)

    for ax in [ax_p, ax_s]:
        ax.set_facecolor('#F8F8F8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')

    _shade_known_bears(ax_p, trim_start, trim_end, alpha=0.30)
    _shade_known_bears(ax_s, trim_start, trim_end, alpha=0.20)

    # Signal active: green axvspan
    in_sig = False; seg_start = None
    for dt, val in sig.items():
        if not in_sig and val:
            seg_start = dt; in_sig = True
        elif in_sig and not val:
            ax_p.axvspan(seg_start, dt, color='#AAFFAA', alpha=0.35, zorder=2)
            in_sig = False
    if in_sig and seg_start is not None:
        ax_p.axvspan(seg_start, sig.index[-1], color='#AAFFAA', alpha=0.35, zorder=2)

    ax_p.semilogy(spy.index, spy.values, color='#222222', lw=1.4, zorder=5)

    # Entry/exit arrows
    arr  = sig.values.astype(int)
    diff = np.diff(arr, prepend=0)
    for d in sig.index[diff == 1]:
        y = _nearest(spy, d)
        if y:
            ax_p.scatter([d], [y * 0.96], marker='v', color='#CC0000', s=40, zorder=10, linewidths=0)
    for d in sig.index[diff == -1]:
        y = _nearest(spy, d)
        if y:
            ax_p.scatter([d], [y * 1.04], marker='^', color='#006600', s=40, zorder=10, linewidths=0)

    # Lead time annotations
    for s_str, _e, bear_label in KNOWN_BEARS:
        lead  = compute_lead_time(sig, s_str)
        peak  = pd.Timestamp(s_str)
        if not (trim_start < peak < trim_end):
            continue
        y_val = _nearest(spy, peak)
        if y_val is None:
            continue
        if lead is not None:
            txt = f'+{lead}w' if lead > 0 else '0w'
            col = '#005500'
        else:
            txt = 'miss'
            col = '#CC0000'
        ax_p.text(peak, y_val * 1.20, txt, fontsize=5.5, ha='center',
                  color=col, fontweight='bold', zorder=11)

    ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}'))
    ax_p.grid(axis='y', which='both', lw=0.3, color='#DDDDDD')
    ax_p.set_ylabel('SPY', fontsize=8)
    plt.setp(ax_p.get_xticklabels(), visible=False)

    prec = eval_result['prec'] * 100
    rec  = eval_result['rec']  * 100
    f1   = eval_result['f1']   * 100
    bc   = eval_result['bears_caught']
    title = f"{label} | F1={f1:.1f}%  Prec={prec:.1f}%  Rec={rec:.1f}%  Bears={bc}/6"
    ax_p.set_title(title, fontsize=8, loc='left', pad=3,
                   bbox=dict(boxstyle='round,pad=0.3', fc='#FFFFF0', ec='#AAAAAA', alpha=0.92))

    legend_elements = [
        mpatches.Patch(color='#FFBBBB', alpha=0.60, label='Known bear'),
        mpatches.Patch(color='#AAFFAA', alpha=0.60, label='Signal active'),
        Line2D([0], [0], marker='v', color='#CC0000', lw=0, ms=6, label='Entry ▼'),
        Line2D([0], [0], marker='^', color='#006600', lw=0, ms=6, label='Exit ▲'),
    ]
    ax_p.legend(handles=legend_elements, loc='upper left', fontsize=6.5,
                framealpha=0.88, edgecolor='#CCCCCC', ncol=2)

    # Bottom panel: score (if provided) or binary bar
    if has_score and score_series is not None:
        sc = score_series.reindex(spy.index) * 100
        ax_s.plot(sc.index, sc.values, color='#9C27B0', lw=1.2, alpha=0.85)
        ax_s.fill_between(sc.index, 0, sc.values,
                          where=sc.values > (score_thresh or 30),
                          color='#9C27B0', alpha=0.18, step='post')
        if score_thresh is not None:
            ax_s.axhline(score_thresh * 100, color='#9C27B0', ls='--', lw=1.0,
                         label=f'Thresh {score_thresh:.0%}')
            ax_s.legend(fontsize=7, loc='upper left')
        bm_f = bm.reindex(sc.index).fillna(False).astype(float) * 50
        ax_s.fill_between(sc.index, 0, bm_f, step='post', color='#FF5555', alpha=0.20)
        ax_s.set_ylim(0, 100)
        ax_s.set_ylabel('Score %', fontsize=8)
    else:
        sig_f = sig.astype(float)
        ax_s.fill_between(sig.index, 0, sig_f, step='post', color='#44AA44', alpha=0.65)
        bm_f = bm.astype(float) * 0.45
        ax_s.fill_between(sig.index, 0, bm_f, step='post', color='#FF5555', alpha=0.25)
        ax_s.set_ylim(0, 1.2)
        ax_s.set_yticks([0, 1])
        ax_s.set_yticklabels(['off', 'on'], fontsize=7)
        ax_s.set_ylabel('Signal', fontsize=8)

    ax_s.grid(axis='y', lw=0.3, color='#DDDDDD')
    ax_s.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_s.get_xticklabels(), rotation=0, ha='center', fontsize=8)
    ax_s.set_xlim(trim_start, trim_end)


def plot_bear_heatmap(ax, raw_coverage_df, penalized_df, eval_df_all):
    """
    Heatmap shows penalized coverage = recall × precision per bear.
    Always-on signals (low precision) are automatically cut down.
    A side column shows overall Precision and F1 for each signal.
    """
    bear_labels = [b[2] for b in KNOWN_BEARS]
    short_cols  = ['1998\nLTCM\n(12w)', '2000-02\nDot-com\n(133w)',
                   '2007-09\nFin.Crs\n(74w)', '2018 Q4\nFed\n(14w)',
                   '2020\nCOVID\n(5w)', '2022\nInflation\n(40w)']

    # Sort by F1 (penalized) desc — this is the canonical ranking
    pen = penalized_df[bear_labels].copy()
    pen['_f1']  = pen.index.map(lambda n: eval_df_all.loc[n, 'f1'] if n in eval_df_all.index else 0)
    pen['_avg'] = pen[bear_labels].mean(axis=1)
    pen = pen.sort_values(['_f1', '_avg'], ascending=[False, False]).drop(columns=['_f1', '_avg'])

    data     = pen.values                    # penalized scores 0–100
    n_r, n_c = data.shape

    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100,
                   aspect='auto', interpolation='nearest')

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(short_cols, fontsize=7.5)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(pen.index, fontsize=7)
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

    for i in range(n_r):
        for j in range(n_c):
            val = data[i, j]
            tc  = 'white' if val < 22 or val > 65 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=6, color=tc, fontweight='bold')

    # Gold border for signals that catch all 6 bears (by raw recall > 10%)
    raw_bc = (raw_coverage_df[bear_labels] > 10).sum(axis=1)
    for i, name in enumerate(pen.index):
        if name in raw_bc.index and raw_bc[name] >= 6:
            ax.add_patch(plt.Rectangle(
                (-0.5, i - 0.5), n_c, 1, fill=False,
                edgecolor='gold', lw=2.5, zorder=5))

    # Side annotations: Prec% | F1%  (placed just outside right edge)
    for i, name in enumerate(pen.index):
        if name in eval_df_all.index:
            prec = eval_df_all.loc[name, 'prec'] * 100
            f1   = eval_df_all.loc[name, 'f1']   * 100
            ax.text(n_c + 0.05, i, f'P={prec:.0f}% F1={f1:.0f}%',
                    va='center', ha='left', fontsize=5.5, color='#333333')

    cbar = plt.colorbar(im, ax=ax, fraction=0.010, pad=0.10)
    cbar.set_label('Penalized Score\n(Recall × Precision)', fontsize=7.5)
    ax.set_title('Panel 1 — Penalized Per-Bear Coverage  =  Recall × Precision  '
                 '(gold border = all 6 bears caught by raw recall)',
                 fontsize=9, loc='left', pad=4)


def plot_all_bears_bar(ax, eval_df_all, raw_coverage_df):
    bear_labels  = [b[2] for b in KNOWN_BEARS]
    bears_caught = (raw_coverage_df[bear_labels] > 10).sum(axis=1)

    df = eval_df_all.copy()
    df['bears_caught'] = df.index.map(lambda n: int(bears_caught.get(n, 0)))
    # Sort by F1 — precision component already penalizes always-on signals
    df['composite']    = df['f1'] * 100
    df = df.sort_values('composite', ascending=True).tail(25)

    names  = list(df.index)
    scores = df['composite'].values
    bc_arr = df['bears_caught'].values
    colors = [DOMAIN_COLORS.get(DOMAIN_MAP.get(n, 'Fin.Stress'), '#BAB0AC') for n in names]

    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.82,
                   edgecolor='white', lw=0.5)

    for i, (bar, bc) in enumerate(zip(bars, bc_arr)):
        txt = f'{bc}/6'
        tc  = '#005500' if bc == 6 else '#AA4400' if bc >= 4 else '#CC0000'
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                txt, va='center', fontsize=7, color=tc, fontweight='bold')
        if bc == 6:
            ax.add_patch(plt.Rectangle(
                (0, bar.get_y()), bar.get_width(), bar.get_height(),
                fill=False, edgecolor='gold', lw=2.0, zorder=5))

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel('All-Bears Composite Score', fontsize=9)
    ax.set_xlim(0, 90)
    ax.axvline(50, color='#AAAAAA', lw=0.8, ls='--', label='F1=50%')
    ax.set_title('Panel 2 — F1 Score Ranking  (already penalizes false signals via precision  |  n/6 = bears caught  |  gold = 6/6)',
                 fontsize=9, loc='left', pad=4)
    ax.set_facecolor('#F8F8F8')
    ax.grid(axis='x', lw=0.3, color='#DDDDDD')

    # Domain legend
    shown_domains = set(DOMAIN_MAP.get(n, 'Fin.Stress') for n in names)
    patches = [mpatches.Patch(color=DOMAIN_COLORS.get(d, '#BAB0AC'), alpha=0.82, label=d)
               for d in shown_domains]
    ax.legend(handles=patches, loc='lower right', ncol=2, fontsize=7,
              framealpha=0.90, edgecolor='#CCCCCC')


# ─────────────────────────────────────────────────────────────────────────────
# Full chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_bear_hunt(spy_w, raw_coverage_df, penalized_df, eval_df_all,
                   weighted_sig, score_series, best_thresh, weighted_eval,
                   b_sig, b_eval, b_label,
                   fom_sig, fom_eval,
                   bear_mask, save_path):

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle(
        'SPY Bear Hunt — Can We Catch All 6 Bears?  |  Fast + Cross-Asset + Composites  |  1994–2026\n'
        'Heatmap = recall × precision per bear (penalizes always-on signals).  F1 bar sorts by combined precision+recall.',
        fontsize=12, fontweight='bold', y=0.998)

    gs = GridSpec(5, 1, figure=fig,
                  height_ratios=[0.30, 0.18, 0.175, 0.175, 0.17],
                  hspace=0.09)

    # Panel 1: heatmap (penalized)
    ax_heat = fig.add_subplot(gs[0])
    plot_bear_heatmap(ax_heat, raw_coverage_df, penalized_df, eval_df_all)

    # Panel 2: F1 bar (penalizes false signals)
    ax_bar = fig.add_subplot(gs[1])
    plot_all_bears_bar(ax_bar, eval_df_all, raw_coverage_df)

    # Panel 3: weighted composite (with score twinx)
    _draw_signal_panel(fig, gs[2], spy_w,
                       weighted_sig,
                       f'Composite A: F1-Weighted Score (threshold={best_thresh:.0%}, top-15 signals)',
                       weighted_eval, bear_mask,
                       score_series=score_series, score_thresh=best_thresh)

    # Panel 4: second-best all-6-bears signal
    _draw_signal_panel(fig, gs[3], spy_w, b_sig, b_label, b_eval, bear_mask)

    # Panel 5: FAST_OR_MACRO
    _draw_signal_panel(fig, gs[4], spy_w,
                       fom_sig,
                       'Composite C: FAST_OR_MACRO (any fast signal  OR  MA200 + confirming signal)',
                       fom_eval, bear_mask)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print('=' * 72)
    print('  SPY BEAR HUNT — CATCHING ALL 6 BEARS')
    print('=' * 72)

    # 1. Data
    print('\n  [1/6] Fetching data...')
    fred, yahoo = fetch_extended_data()
    spy_w = yahoo.get('SPY')
    if spy_w is None:
        raise RuntimeError('SPY data unavailable')
    weekly_index = spy_w[spy_w.index >= ANALYSIS_START].index
    bear_mask    = _build_bear_mask(weekly_index)

    # 2. Original 46 signals
    print('\n  [2/6] Building original 46 signals...')
    signals_orig = _C.build_all_signals(fred, yahoo, weekly_index)
    eval_orig    = evaluate_all_signals(signals_orig, bear_mask)

    # 3. New 14 signals
    print('\n  [3/6] Building 14 new signals (fast + cross-asset + velocity)...')
    signals_new = build_new_signals(fred, yahoo, weekly_index)
    signals_all = {**signals_orig, **signals_new}

    # 4. Composite signals
    print('\n  [4/6] Building composite signals...')
    signal_df = pd.DataFrame(signals_all)

    weighted_sig, score, best_thresh = compute_weighted_composite(
        signal_df, eval_orig, bear_mask)
    signals_all['WEIGHTED_SCORE'] = weighted_sig

    domain_sig = compute_domain_diversity_vote(signal_df, DOMAIN_MAP)
    signals_all['DOMAIN_DIVERSE'] = domain_sig

    fom_sig = compute_fast_or_macro(signal_df, weekly_index)
    signals_all['FAST_OR_MACRO'] = fom_sig

    # 5. Evaluate all
    print('\n  [5/6] Evaluating all signals...')
    eval_df_all   = evaluate_all_signals(signals_all, bear_mask)
    weighted_eval = evaluate_signal('WEIGHTED_SCORE', weighted_sig, bear_mask)
    domain_eval   = evaluate_signal('DOMAIN_DIVERSE', domain_sig,  bear_mask)
    fom_eval      = evaluate_signal('FAST_OR_MACRO',  fom_sig,     bear_mask)

    # Coverage matrix (top 15 orig + 14 new + 3 composites = 32 rows in heatmap)
    top15_orig   = list(eval_orig.head(15).index)
    heatmap_sigs = {k: signals_all[k] for k in
                    top15_orig + list(signals_new.keys()) +
                    ['WEIGHTED_SCORE', 'DOMAIN_DIVERSE', 'FAST_OR_MACRO']
                    if k in signals_all}
    raw_coverage_df = build_coverage_matrix(heatmap_sigs, weekly_index)
    for name, er in [('WEIGHTED_SCORE', weighted_eval),
                     ('DOMAIN_DIVERSE', domain_eval),
                     ('FAST_OR_MACRO',  fom_eval)]:
        if name not in eval_df_all.index:
            eval_df_all.loc[name] = er

    # Penalized matrix = recall × precision (penalizes always-on signals)
    penalized_df = build_penalized_matrix(raw_coverage_df, eval_df_all)

    print_coverage_matrix(raw_coverage_df, penalized_df, eval_df_all)

    # Pick the 3 chart panels by F1 among signals that catch all 6 bears (no duplicates)
    bear_labels = [b[2] for b in KNOWN_BEARS]
    raw_bc = (raw_coverage_df[bear_labels] > 10).sum(axis=1)
    all6   = [n for n in eval_df_all.index
              if raw_bc.get(n, 0) >= 6 and n in signals_all]
    all6_ranked = sorted(all6, key=lambda n: eval_df_all.loc[n, 'f1'], reverse=True)

    used = set()
    panel_a_name = all6_ranked[0]          # best F1 all-6-bears signal
    used.add(panel_a_name)

    # Panel C: prefer FAST_OR_MACRO as the rule-based composite
    panel_c_name = next(
        (n for n in ['FAST_OR_MACRO'] + all6_ranked if n in all6 and n not in used),
        all6_ranked[-1])
    used.add(panel_c_name)

    # Panel B: next-best with meaningful precision (>40%), not already used
    panel_b_name = next(
        (n for n in all6_ranked
         if n not in used and eval_df_all.loc[n, 'prec'] > 0.40),
        next((n for n in all6_ranked if n not in used), all6_ranked[1]))

    def _er(name):
        r = eval_df_all.loc[name]
        return {k: float(r[k]) for k in ('f1', 'prec', 'rec', 'bears_caught')}

    print(f'\n  Chart panels:')
    for label, name in [('A', panel_a_name), ('B', panel_b_name), ('C', panel_c_name)]:
        er = _er(name)
        print(f"    Composite {label}: {name:<20} F1={er['f1']*100:.1f}%  "
              f"Prec={er['prec']*100:.1f}%  Bears={er['bears_caught']:.0f}/6")

    # 6. Chart
    print('\n  [6/6] Generating chart...')
    chart_path = os.path.join(_ROOT, 'spy_bear_hunt_chart.png')
    plot_bear_hunt(
        spy_w           = spy_w,
        raw_coverage_df = raw_coverage_df,
        penalized_df    = penalized_df,
        eval_df_all     = eval_df_all,
        weighted_sig    = signals_all[panel_a_name],
        score_series    = score,
        best_thresh     = best_thresh,
        weighted_eval   = _er(panel_a_name),
        b_sig           = signals_all[panel_b_name],
        b_eval          = _er(panel_b_name),
        b_label         = f'Signal B: {panel_b_name}',
        fom_sig         = signals_all[panel_c_name],
        fom_eval        = _er(panel_c_name),
        bear_mask       = bear_mask,
        save_path       = chart_path,
    )
    print(f'  Chart saved: {chart_path}')
    print('=' * 72)
    print('  DONE')
    print('=' * 72)


if __name__ == '__main__':
    main()
