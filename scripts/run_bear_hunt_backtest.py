"""
Bear Hunt Regime Backtest — Full Strategy Comparison
=====================================================
Wires the three bear-hunt signals (WEIGHTED_SCORE, RSI_BELOW40, FAST_OR_MACRO)
into the RegimeAdaptiveStrategy portfolio framework and backtests them against
the current live MacroMom signal over 25 years (2001–2026).

Usage
-----
  python scripts/run_bear_hunt_backtest.py            # 25yr
  python scripts/run_bear_hunt_backtest.py --years 10 # 10yr quick test

Output
------
  bear_hunt_comparison_chart.png
  bear_hunt_macromom_Nyr_results.csv
  bear_hunt_weighted_Nyr_results.csv
  bear_hunt_rsi_Nyr_results.csv
  bear_hunt_fastormacro_Nyr_results.csv
"""

import os
import sys
import time
import argparse
import warnings
import importlib.util
from datetime import datetime

# Force UTF-8 output so box-drawing and other Unicode chars render on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'backtest'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from dotenv import load_dotenv
load_dotenv()

from backtest import Backtest, BacktestResult
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext

DB_PATH = os.path.join(_ROOT, 'data', 'fundamentals.sqlite')
if not os.path.exists(DB_PATH):
    DB_PATH = os.path.join(_ROOT, 'fundamentals.sqlite')

TAX_RATE = 0.0   # same baseline as MacroMom comparison (no tax for clean signal comparison)

STRATEGY_COLORS = {
    'MacroMom':     '#2196F3',   # blue
    'WeightedScore':'#4CAF50',   # green
    'RSI_Below40':  '#FF9800',   # orange
    'FastOrMacro':  '#E91E63',   # pink/red
    'SPY':          '#9E9E9E',   # grey
}

KNOWN_BEARS = [
    ('1998-07-17', '1998-10-08', '1998 LTCM'),
    ('2000-03-24', '2002-10-09', '2000–02 Dot-com'),
    ('2007-10-09', '2009-03-09', '2007–09 Fin.Crisis'),
    ('2018-09-20', '2018-12-24', '2018 Q4 Fed'),
    ('2020-02-19', '2020-03-23', '2020 COVID'),
    ('2022-01-03', '2022-10-13', '2022 Inflation'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Signal cache  (singleton: built once, shared across all 3 hunt strategies)
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_CACHE: dict = {}


def _get_or_build_signal_cache() -> dict:
    if _SIGNAL_CACHE:
        return _SIGNAL_CACHE

    print('\n  [Signal cache] Building all bear-hunt signals (one-time fetch)...')
    t0 = time.time()

    # Dynamic import so spy_bear_hunt doesn't need to be a package
    _hunt_path = os.path.join(_ROOT, 'scripts', 'spy_bear_hunt.py')
    spec = importlib.util.spec_from_file_location('spy_bear_hunt', _hunt_path)
    hunt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hunt)

    fred, yahoo = hunt.fetch_extended_data()
    spy_w = yahoo.get('SPY')
    if spy_w is None:
        raise RuntimeError('SPY data not available from extended fetch')

    # Use full history (no start trim — we need signals from 1994 for context)
    weekly_index = spy_w.index
    bear_mask    = hunt._build_bear_mask(weekly_index)

    print('    Building original 46 signals...')
    signals_orig = hunt._C.build_all_signals(fred, yahoo, weekly_index)
    eval_orig    = hunt.evaluate_all_signals(signals_orig, bear_mask)

    print('    Building 14 new signals...')
    signals_new = hunt.build_new_signals(fred, yahoo, weekly_index)
    signals_all = {**signals_orig, **signals_new}
    signal_df   = pd.DataFrame(signals_all)

    print('    Computing composites...')
    weighted_sig, _score, _thresh = hunt.compute_weighted_composite(
        signal_df, eval_orig, bear_mask)
    fom_sig = hunt.compute_fast_or_macro(signal_df, weekly_index)

    _SIGNAL_CACHE.update({
        'WEIGHTED_SCORE': weighted_sig,
        'RSI_BELOW40':    signals_all['RSI_BELOW40'],
        'FAST_OR_MACRO':  fom_sig,
    })
    print(f'    Done in {time.time()-t0:.0f}s. '
          f'Cache keys: {list(_SIGNAL_CACHE.keys())}')
    return _SIGNAL_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# BearHuntRegimeStrategy
# ─────────────────────────────────────────────────────────────────────────────

class BearHuntRegimeStrategy(RegimeAdaptiveStrategy):
    """
    RegimeAdaptiveStrategy with MacroMom replaced by a pre-computed bear-hunt
    signal (WEIGHTED_SCORE, RSI_BELOW40, or FAST_OR_MACRO).

    Signal ON  → bear_score=80 → 30% HB / 70% Bear-Beta  (EXTREME tier)
    Signal OFF → bear_score=20 → 100% HB                  (LOW tier)
    """

    def __init__(self, signal_name: str, db_path: str = DB_PATH,
                 max_positions: int = 25, **kwargs):
        super().__init__(db_path=db_path, max_positions=max_positions, **kwargs)
        self._signal_name = signal_name
        self._precomputed = _get_or_build_signal_cache()

    def _get_signal_at(self, date: datetime) -> bool:
        sig  = self._precomputed[self._signal_name]
        ts   = pd.Timestamp(date)
        prior = sig.index[sig.index <= ts]
        if len(prior) == 0:
            return False
        return bool(sig.loc[prior[-1]])

    def _update_regime(self, context: ExecutionContext) -> None:
        self._regime_inputs = self._fetch_regime_inputs_from_context(context)

        signal_active = self._get_signal_at(context.date)
        bear_score    = 80.0 if signal_active else 20.0
        self._current_bear_score = bear_score
        self._bear_score_history.append(bear_score)

        new_alloc = self._get_allocation_weights(bear_score)
        old_alloc = self._current_allocation

        # Lightweight momentum gate: only go defensive if price is already weak
        if new_alloc[0] < old_alloc[0]:
            sp500 = self._regime_inputs.get('sp500')
            if sp500 is not None and len(sp500) >= 4:
                momentum_ok = float(sp500.iloc[-1]) < float(sp500.iloc[-4:].mean())
            else:
                momentum_ok = True
            if not momentum_ok:
                new_alloc = old_alloc

        date_str = context.date.strftime('%Y-%m-%d')
        print(f"[{self._signal_name} {date_str}] "
              f"{'ON ' if signal_active else 'off'}  "
              f"Alloc={new_alloc[0]*100:.0f}%/{new_alloc[1]*100:.0f}% HB/Def")

        self._score_history_log.append({'date': context.date, 'score': bear_score})

        alloc_change = abs(new_alloc[0] - old_alloc[0])
        if alloc_change >= self.min_realloc_change:
            direction = '-> DEF' if new_alloc[0] < old_alloc[0] else '-> AGG'
            print(f"  REALLOC {direction}")
            self._current_allocation = new_alloc
            self._reposition_log.append({
                'date':    context.date,
                'trigger': 'bear-hunt',
                'from_hb': old_alloc[0],
                'to_hb':   new_alloc[0],
                'score':   bear_score,
            })

        self._last_regime_check = context.date


# ─────────────────────────────────────────────────────────────────────────────
# Backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(name, strategy, years, end_year, shared_cache=None):
    print(f"\n{'='*60}\nRunning: {name}  ({years}yr, end={end_year})\n{'='*60}")
    bt     = Backtest(db_path=DB_PATH, tax_rate=TAX_RATE)
    result = bt.backtest(
        lookback_years=years,
        end_year=end_year,
        starting_fund=100_000,
        strategy=strategy,
        time_period='M',
        preloaded_price_cache=shared_cache,
    )
    return result, bt._price_cache


def run_all(years: int = 25, end_year: int = None) -> dict:
    if end_year is None:
        end_year = datetime.now().year

    # Pre-build signal cache before creating strategy instances
    print('\nPre-building signal cache...')
    _get_or_build_signal_cache()

    strategies = {
        'MacroMom':     RegimeAdaptiveStrategy(db_path=DB_PATH, max_positions=25),
        'WeightedScore':BearHuntRegimeStrategy('WEIGHTED_SCORE', db_path=DB_PATH, max_positions=25),
        'RSI_Below40':  BearHuntRegimeStrategy('RSI_BELOW40',    db_path=DB_PATH, max_positions=25),
        'FastOrMacro':  BearHuntRegimeStrategy('FAST_OR_MACRO',  db_path=DB_PATH, max_positions=25),
    }

    results   = {}
    reposition_logs = {}
    score_logs      = {}
    shared_cache = None

    for name, strat in strategies.items():
        t0      = time.time()
        result, shared_cache = _run_one(name, strat, years, end_year, shared_cache)
        elapsed = time.time() - t0
        st      = result.full_stats()
        print(f"  -> CAGR={st['cagr']*100:.1f}%  MaxDD={st['max_drawdown']*100:.1f}%  "
              f"Sharpe={st['sharpe_ratio']:.2f}  ({elapsed:.0f}s)")

        # Save CSV
        csv_path = os.path.join(_ROOT, f'bear_hunt_{name.lower()}_{years}yr_results.csv')
        result.portfolio_summary_over_time().to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

        results[name] = result

        # Collect reposition + score logs from strategy
        if hasattr(strat, '_reposition_log'):
            reposition_logs[name] = strat._reposition_log
        if hasattr(strat, '_score_history_log'):
            score_logs[name] = strat._score_history_log

    return results, reposition_logs, score_logs


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct_time_defensive(score_log: list) -> float:
    """Fraction of regime checks where bear_score >= 65 (significant defense)."""
    if not score_log:
        return 0.0
    scores = [r['score'] for r in score_log]
    return sum(1 for s in scores if s >= 65) / len(scores)


def _bear_lead_times(reposition_log: list) -> dict:
    """
    For each known bear, find how many weeks before/after the bear START the
    signal triggered a defensive period that overlaps the bear.
    Positive = early (triggered before bear start), negative = late.
    """
    if not reposition_log:
        return {}

    events = sorted(reposition_log, key=lambda r: r['date'])
    defensive_periods = []
    in_def = False
    def_start = None
    for ev in events:
        going_def = ev['to_hb'] < ev['from_hb']
        going_agg = ev['to_hb'] > ev['from_hb']
        if going_def and not in_def:
            def_start = ev['date']
            in_def    = True
        elif going_agg and in_def and def_start is not None:
            defensive_periods.append((pd.Timestamp(def_start), pd.Timestamp(ev['date'])))
            in_def = False
    if in_def and def_start:
        defensive_periods.append((pd.Timestamp(def_start), pd.Timestamp('2099-01-01')))

    results = {}
    for s_str, e_str, label in KNOWN_BEARS:
        bear_start = pd.Timestamp(s_str)
        bear_end   = pd.Timestamp(e_str)
        # Find the defensive period that overlaps the bear and gives best lead
        best_lead = None
        for d_start, d_end in defensive_periods:
            if d_start <= bear_end and d_end >= bear_start:
                lead_weeks = (bear_start - d_start).days / 7
                if best_lead is None or abs(lead_weeks) < abs(best_lead):
                    best_lead = lead_weeks
        if best_lead is not None and abs(best_lead) <= 52:
            if best_lead >= 0:
                results[label] = f'+{best_lead:.0f}w early'
            else:
                results[label] = f'{abs(best_lead):.0f}w late'
        else:
            results[label] = 'MISS'
    return results


def print_report(results: dict, reposition_logs: dict, score_logs: dict, years: int):
    strat_names  = list(results.keys())
    bear_labels  = [b[2] for b in KNOWN_BEARS]

    # Compute SPY stats from benchmark values
    first_result = list(results.values())[0]

    print()
    print('=' * 78)
    print(f'  BEAR HUNT REGIME BACKTEST — {years}-YEAR COMPARISON')
    print('=' * 78)

    metrics = ['cagr', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
               'alpha', 'beta', 'information_ratio', 'volatility', 'total_trades']
    labels  = ['CAGR', 'Max Drawdown', 'Sharpe', 'Sortino', 'Alpha',
               'Beta', 'Info Ratio', 'Volatility', 'Total Trades']

    # Header
    col_w = 14
    print(f"  {'Metric':<18}", end='')
    for name in strat_names:
        print(f'{name:>{col_w}}', end='')
    print(f"{'SPY':>{col_w}}")
    print('  ' + '─' * (18 + col_w * (len(strat_names) + 1)))

    spy_cagr = first_result.benchmark_cagr

    for metric, label in zip(metrics, labels):
        print(f"  {label:<18}", end='')
        vals = []
        for name in strat_names:
            st  = results[name].full_stats()
            val = st[metric]
            vals.append(val)
            if metric in ('cagr', 'max_drawdown', 'alpha', 'volatility'):
                print(f'{val*100:>{col_w}.1f}%', end='')
            elif metric == 'total_trades':
                print(f'{int(val):>{col_w}}', end='')
            else:
                print(f'{val:>{col_w}.2f}', end='')
        # SPY column
        spy_result = results[strat_names[0]]
        spy_st = {'cagr': spy_result.benchmark_cagr,
                  'max_drawdown': ((min(spy_result.benchmark_values) /
                                    spy_result.benchmark_values[0]) - 1
                                   if spy_result.benchmark_values else 0),
                  'volatility': spy_result.benchmark_volatility}
        if metric in spy_st:
            val = spy_st[metric]
            if metric in ('cagr', 'max_drawdown', 'volatility'):
                print(f'{val*100:>{col_w}.1f}%', end='')
            else:
                print(f'{"--":>{col_w}}', end='')
        else:
            print(f'{"--":>{col_w}}', end='')
        print()

    # % time defensive
    print(f"  {'% Time Defensive':<18}", end='')
    for name in strat_names:
        pct = _pct_time_defensive(score_logs.get(name, []))
        print(f'{pct*100:>{col_w}.0f}%', end='')
    print(f'{"0%":>{col_w}}')

    # Per-bear lead times
    print()
    print('  Bear Response (+ = triggered X weeks before bear peak):')
    print(f"  {'Bear':<22}", end='')
    for name in strat_names:
        print(f'{name[:12]:>{col_w}}', end='')
    print()
    print('  ' + '─' * (22 + col_w * len(strat_names)))

    for _, _, bear_label in KNOWN_BEARS:
        print(f"  {bear_label:<22}", end='')
        for name in strat_names:
            lt = _bear_lead_times(reposition_logs.get(name, {}))
            val = lt.get(bear_label, 'MISS')
            print(f'{val:>{col_w}}', end='')
        print()

    print('=' * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _shade_bears(ax, x_min, x_max):
    for s_str, e_str, _ in KNOWN_BEARS:
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < x_max and e > x_min:
            ax.axvspan(max(s, x_min), min(e, x_max),
                       color='#FFCCCC', alpha=0.55, zorder=1)


def _reposition_markers(ax, reposition_log, spy_values, spy_dates, color,
                         offset_mult=1.0):
    """Draw ▼/▲ markers on the axis for each defensive/aggressive transition."""
    if not reposition_log or spy_values is None:
        return
    spy_s = pd.Series(spy_values, index=spy_dates)
    for ev in reposition_log:
        dt       = pd.Timestamp(ev['date'])
        going_def= ev['to_hb'] < ev['from_hb']
        if dt < spy_s.index[0] or dt > spy_s.index[-1]:
            continue
        idx  = spy_s.index.searchsorted(dt)
        idx  = min(max(idx, 0), len(spy_s) - 1)
        y    = spy_s.iloc[idx]
        if going_def:
            ax.scatter([dt], [y * 0.92 * offset_mult], marker='v',
                       color=color, s=45, zorder=10, linewidths=0)
        else:
            ax.scatter([dt], [y * 1.08 / offset_mult], marker='^',
                       color=color, s=45, zorder=10, linewidths=0)


def _equity_curve_from_result(result: BacktestResult):
    snapshots  = result.snapshots
    dates      = [s.date for s in snapshots]
    values     = [s.total_value for s in snapshots]
    bm_values  = list(result.benchmark_values)
    return dates, values, bm_values


def plot_comparison_chart(results: dict, reposition_logs: dict, years: int,
                           save_path: str):
    strat_names = list(results.keys())
    n_strats    = len(strat_names)

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle(
        f'Bear Hunt Regime Backtest — {years}-Year Comparison  |  '
        f'{datetime.now().year - years}–{datetime.now().year}\n'
        'MacroMom (current live)  vs  3 Bear-Hunt Signals as Regime Detectors  '
        '|  Same HighBeta+BearBeta portfolio, only the regime signal changes',
        fontsize=11, fontweight='bold', y=0.998)

    gs = GridSpec(4, 2, figure=fig,
                  height_ratios=[0.40, 0.22, 0.20, 0.18],
                  hspace=0.14, wspace=0.10)

    # ── Row 0: Equity curves (full width) ────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor('#F8F8F8')

    first_res  = results[strat_names[0]]
    first_dates, first_vals, bm_vals = _equity_curve_from_result(first_res)
    x_min = pd.Timestamp(first_dates[0])
    x_max = pd.Timestamp(first_dates[-1])

    _shade_bears(ax_eq, x_min, x_max)

    # SPY benchmark (use benchmark_values from first result, normalised to $100k)
    if bm_vals:
        bm_arr = np.array(bm_vals)
        bm_norm = bm_arr / bm_arr[0] * 100_000
        ax_eq.semilogy(first_dates[:len(bm_vals)], bm_norm,
                       color=STRATEGY_COLORS['SPY'], lw=1.5, ls='--',
                       label='SPY buy-&-hold', zorder=3)

    offsets = [1.0, 0.97, 0.94, 0.91]
    for i, name in enumerate(strat_names):
        dates, vals, _bm = _equity_curve_from_result(results[name])
        color = STRATEGY_COLORS.get(name, '#333333')
        ax_eq.semilogy(dates, vals, color=color, lw=2.0,
                       label=name, zorder=4 + i)
        # Reposition markers (offset slightly so they don't overlap)
        _reposition_markers(ax_eq, reposition_logs.get(name, []),
                            bm_vals, first_dates[:len(bm_vals)],
                            color=color, offset_mult=offsets[i])

    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1000:.0f}k'))
    ax_eq.set_ylabel('Portfolio Value', fontsize=9)
    ax_eq.legend(loc='upper left', fontsize=8.5, framealpha=0.90,
                 edgecolor='#CCCCCC')
    ax_eq.set_xlim(x_min, x_max)
    ax_eq.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_eq.grid(axis='y', which='both', lw=0.3, color='#DDDDDD')
    ax_eq.set_title('Panel 1 — Equity Curves (log scale)  |  pink = confirmed bear period  '
                    '|  ▼ defensive entry  ▲ aggressive re-entry',
                    fontsize=8.5, loc='left', pad=3)
    for spine in ax_eq.spines.values():
        spine.set_edgecolor('#CCCCCC')

    # ── Row 1 left: Drawdown ─────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1, 0])
    ax_dd.set_facecolor('#F8F8F8')
    _shade_bears(ax_dd, x_min, x_max)

    for name in strat_names:
        dates, vals, _ = _equity_curve_from_result(results[name])
        peak = np.maximum.accumulate(vals)
        dd   = (np.array(vals) - peak) / peak * 100
        ax_dd.plot(dates, dd, color=STRATEGY_COLORS.get(name), lw=1.5, label=name)

    if bm_vals:
        peak = np.maximum.accumulate(bm_vals)
        dd   = (np.array(bm_vals) - peak) / peak * 100
        ax_dd.plot(first_dates[:len(bm_vals)], dd,
                   color=STRATEGY_COLORS['SPY'], lw=1.2, ls='--', label='SPY')

    ax_dd.set_ylabel('Drawdown (%)', fontsize=8.5)
    ax_dd.legend(loc='lower left', fontsize=7.5, framealpha=0.90)
    ax_dd.set_xlim(x_min, x_max)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_dd.grid(lw=0.3, color='#DDDDDD')
    ax_dd.set_title('Panel 2 — Drawdown', fontsize=8.5, loc='left', pad=3)
    for spine in ax_dd.spines.values():
        spine.set_edgecolor('#CCCCCC')

    # ── Row 1 right: Regime signal activity (Gantt) ───────────────────────────
    ax_gantt = fig.add_subplot(gs[1, 1])
    ax_gantt.set_facecolor('#F8F8F8')
    _shade_bears(ax_gantt, x_min, x_max)

    y_labels = []
    for i, name in enumerate(strat_names):
        color = STRATEGY_COLORS.get(name)
        rlog  = reposition_logs.get(name, [])
        events = sorted(rlog, key=lambda r: r['date'])
        in_def = False
        d_start = None
        for ev in events:
            going_def = ev['to_hb'] < ev['from_hb']
            going_agg = ev['to_hb'] > ev['from_hb']
            if going_def and not in_def:
                d_start = pd.Timestamp(ev['date'])
                in_def  = True
            elif going_agg and in_def and d_start is not None:
                ax_gantt.barh(i, (pd.Timestamp(ev['date']) - d_start).days,
                              left=d_start, height=0.65,
                              color=color, alpha=0.80, zorder=4)
                in_def = False
        if in_def and d_start:
            ax_gantt.barh(i, (x_max - d_start).days,
                          left=d_start, height=0.65,
                          color=color, alpha=0.80, zorder=4)
        y_labels.append(name)

    ax_gantt.set_yticks(range(n_strats))
    ax_gantt.set_yticklabels(y_labels, fontsize=8)
    ax_gantt.set_xlim(x_min, x_max)
    ax_gantt.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_gantt.set_ylim(-0.5, n_strats - 0.5)
    ax_gantt.grid(axis='x', lw=0.3, color='#DDDDDD')
    ax_gantt.set_title('Panel 3 — Defensive Periods (bars = signal ON)',
                       fontsize=8.5, loc='left', pad=3)
    for spine in ax_gantt.spines.values():
        spine.set_edgecolor('#CCCCCC')

    # ── Row 2: Stats table (full width) ─────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('off')
    ax_tbl.set_title('Panel 4 — Performance Statistics', fontsize=8.5, loc='left', pad=3)

    col_headers  = ['Metric'] + strat_names + ['SPY']
    row_metrics  = [
        ('CAGR',         'cagr',              '{:.1f}%',  True,  True),
        ('Max Drawdown', 'max_drawdown',       '{:.1f}%',  True,  False),
        ('Sharpe',       'sharpe_ratio',       '{:.2f}',   True,  True),
        ('Sortino',      'sortino_ratio',      '{:.2f}',   True,  True),
        ('Alpha (ann.)', 'alpha',              '{:.1f}%',  True,  True),
        ('Beta',         'beta',               '{:.2f}',   False, False),
        ('Info Ratio',   'information_ratio',  '{:.2f}',   True,  True),
        ('Volatility',   'volatility',         '{:.1f}%',  False, False),
        ('Trades',       'total_trades',       '{:.0f}',   False, False),
    ]

    spy_stats = {
        'cagr':       results[strat_names[0]].benchmark_cagr,
        'max_drawdown': abs(min(
            (np.array(bm_vals) - np.maximum.accumulate(bm_vals)) / np.maximum.accumulate(bm_vals)
        )) if bm_vals else 0,
        'volatility': results[strat_names[0]].benchmark_volatility,
    }

    cell_data   = []
    cell_colors = []
    for label, key, fmt, higher_better, compare in row_metrics:
        row_vals   = []
        row_colors = []
        strat_vals = []
        for name in strat_names:
            st  = results[name].full_stats()
            val = st[key]
            strat_vals.append(val)
            if key in ('cagr', 'max_drawdown', 'alpha', 'volatility'):
                row_vals.append(fmt.format(val * 100))
            else:
                row_vals.append(fmt.format(val))

        # SPY value
        if key in spy_stats:
            spy_val = spy_stats[key]
            if key in ('cagr', 'max_drawdown', 'volatility'):
                spy_str = fmt.format(spy_val * 100)
            else:
                spy_str = fmt.format(spy_val)
        else:
            spy_str = '—'
            spy_val = None

        # Color coding: green=best, red=worst (only when compare=True)
        if compare and len(strat_vals) > 1:
            if higher_better:
                best_idx  = int(np.argmax(strat_vals))
                worst_idx = int(np.argmin(strat_vals))
            else:
                best_idx  = int(np.argmin(strat_vals))
                worst_idx = int(np.argmax(strat_vals))
            for j in range(len(strat_vals)):
                if j == best_idx:
                    row_colors.append('#C8E6C9')
                elif j == worst_idx:
                    row_colors.append('#FFCDD2')
                else:
                    row_colors.append('#FFFFFF')
        else:
            row_colors = ['#FFFFFF'] * len(strat_vals)

        row_colors.append('#F5F5F5')  # SPY column
        cell_data.append([label] + row_vals + [spy_str])
        cell_colors.append(['#EEEEEE'] + row_colors)

    tbl = ax_tbl.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.55)

    # Style header
    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor('#37474F')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # ── Row 3: Bear response heatmap ─────────────────────────────────────────
    ax_bear = fig.add_subplot(gs[3, :])
    ax_bear.set_facecolor('#F8F8F8')
    ax_bear.set_title('Panel 5 — Bear Response  '
                       '(+ = signal triggered X weeks BEFORE bear peak  |  '
                       '- = triggered late  |  MISS = never triggered)',
                       fontsize=8.5, loc='left', pad=3)

    bear_labels = [b[2] for b in KNOWN_BEARS]
    hm_data     = []
    hm_colors   = []
    hm_text     = []

    for name in strat_names:
        lt   = _bear_lead_times(reposition_logs.get(name, {}))
        row_c = []
        row_t = []
        row_v = []
        for bl in bear_labels:
            val = lt.get(bl, 'MISS')
            row_t.append(val)
            if val == 'MISS':
                row_c.append('#FFCDD2')
                row_v.append(0)
            elif 'early' in val:
                weeks = float(val.replace('+', '').replace('w early', '').strip())
                row_c.append('#C8E6C9' if weeks >= 4 else '#DCEDC8')
                row_v.append(weeks)
            else:
                weeks = float(val.replace('w late', '').strip())
                row_c.append('#FFF9C4')
                row_v.append(weeks)
        hm_colors.append(row_c)
        hm_text.append(row_t)
        hm_data.append(row_v)

    tbl2 = ax_bear.table(
        cellText=hm_text,
        rowLabels=strat_names,
        colLabels=bear_labels,
        cellLoc='center',
        loc='center',
        cellColours=hm_colors,
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8.0)
    tbl2.scale(1.0, 1.8)

    # Style header row
    for j in range(len(bear_labels)):
        tbl2[(0, j)].set_facecolor('#546E7A')
        tbl2[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=7.5)

    # Style row headers
    for i in range(len(strat_names)):
        tbl2[(i + 1, -1)].set_facecolor('#ECEFF1')
        tbl2[(i + 1, -1)].set_text_props(fontweight='bold', fontsize=8)

    ax_bear.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\n  Chart saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Bear Hunt Regime Backtest')
    parser.add_argument('--years',    type=int, default=25,
                        help='Number of backtest years (default: 25)')
    parser.add_argument('--end-year', type=int, default=None,
                        help='End year (default: current year)')
    args = parser.parse_args()

    end_year = args.end_year or datetime.now().year

    print()
    print('=' * 70)
    print('  BEAR HUNT REGIME BACKTEST')
    print(f'  {end_year - args.years}–{end_year}  ({args.years} years)')
    print('=' * 70)

    t_total = time.time()
    results, reposition_logs, score_logs = run_all(args.years, end_year)

    print_report(results, reposition_logs, score_logs, args.years)

    chart_path = os.path.join(_ROOT, 'bear_hunt_comparison_chart.png')
    plot_comparison_chart(results, reposition_logs, args.years, chart_path)

    print(f'\n  Total elapsed: {(time.time()-t_total)/60:.1f} min')
    print('=' * 70)
    print('  DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()
