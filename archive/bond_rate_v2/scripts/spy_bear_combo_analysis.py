"""
SPY Bear Market Combo Analysis — ~50 Signals x 460 Combinations
================================================================
Evaluates every macro/price indicator domain across 6 known SPY bears,
ranks all combinations, and charts the top 3 with entry/exit signals.

Domains: Price, Volatility, Yield Curve, Credit, Inflation, Real Rates,
         Monetary, Consumer/Activity, Gold/Commodities, Financial Stress

Output:
  spy_bear_combo_analysis_chart.png   (22x26, 150 dpi)
  Console: individual signal ranking + top-10 combo table
"""

import os
import sys
import warnings
import itertools
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import yfinance as yf

warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from dotenv import load_dotenv; load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_START_STR     = '1991-01-01'   # extra warm-up before analysis window
ANALYSIS_START     = '1994-01-01'
MONTHLY_LAG_WEEKS  = 5              # release lag for monthly FRED series
ZSCORE_WINDOW      = 104            # 2-year rolling baseline
MIN_DURATION_WKS   = 2              # min consecutive weeks for all signals
TOP_N_INDIVIDUAL   = 15             # top signals fed into pair/triple search
TOP_N_TRIPLES      = 10             # top signals for triples
TOP_N_QUADS        = 5              # top signals for quads
DISPLAY_TOP        = 25             # signals shown in bar chart + console

KNOWN_BEARS = [
    ('1998-07-17', '1998-10-08', '1998 LTCM'),
    ('2000-03-24', '2002-10-09', '2000-02 Dot-com'),
    ('2007-10-09', '2009-03-09', '2007-09 Fin.Crisis'),
    ('2018-09-20', '2018-12-24', '2018 Q4 Fed'),
    ('2020-02-19', '2020-03-23', '2020 COVID'),
    ('2022-01-03', '2022-10-13', '2022 Inflation'),
]

DOMAIN_MAP = {
    'MA200': 'Price/Tech', 'MA50': 'Price/Tech',
    'RSI_BELOW40': 'Price/Tech', 'RETURN_NEG52': 'Price/Tech', 'BREADTH_WEAK': 'Price/Tech',
    'VIX_HIGH20': 'Volatility', 'VIX_HIGH25': 'Volatility',
    'VIX_SPIKE': 'Volatility', 'VIX_ZSCORE': 'Volatility', 'REALIZED_VOL': 'Volatility',
    'CURVE_2Y_INV': 'Yield Curve', 'CURVE_3M_INV': 'Yield Curve',
    'CURVE_2Y_FLAT': 'Yield Curve', 'CURVE_3M_FLAT': 'Yield Curve', 'CURVE_BUTTERFLY': 'Yield Curve',
    'HYG_LQD_STRESS': 'Credit', 'HY_OAS_HIGH': 'Credit',
    'HY_OAS_SPIKE': 'Credit', 'IG_OAS_HIGH': 'Credit', 'HY_IG_RATIO': 'Credit',
    'CPI_HIGH3': 'Inflation', 'CPI_HIGH4': 'Inflation',
    'CORECPI_ZSCORE': 'Inflation', 'FED_BEHIND': 'Inflation', 'PPI_CPI_SPREAD': 'Inflation',
    'REAL_Y10_HIGH': 'Real Rates', 'REAL_Y5_ZSCORE': 'Real Rates',
    'REAL_Y_MOMENTUM': 'Real Rates', 'TIPS_SPREAD': 'Real Rates',
    'M2_CONTRACT': 'Monetary', 'M2_ZSCORE': 'Monetary',
    'USD_SURGE': 'Monetary', 'FF_HIKE': 'Monetary',
    'SENTIMENT_WEAK': 'Consumer', 'CLAIMS_SPIKE': 'Consumer',
    'RETAIL_WEAK': 'Consumer', 'INDPRO_WEAK': 'Consumer', 'LEI_WEAK': 'Consumer',
    'GOLD_SURGE': 'Gold/Commod', 'GOLD_VS_SPY': 'Gold/Commod',
    'OIL_CRASH': 'Gold/Commod', 'COPPER_WEAK': 'Gold/Commod',
    'NFCI_TIGHT': 'Fin.Stress', 'NFCI_SPIKE': 'Fin.Stress',
    'STLFSI_HIGH': 'Fin.Stress', 'TLT_FLIGHT': 'Fin.Stress',
}

DOMAIN_COLORS = {
    'Price/Tech':  '#4E79A7',
    'Volatility':  '#F28E2B',
    'Yield Curve': '#59A14F',
    'Credit':      '#E15759',
    'Inflation':   '#76B7B2',
    'Real Rates':  '#EDC948',
    'Monetary':    '#B07AA1',
    'Consumer':    '#FF9DA7',
    'Gold/Commod': '#9C755F',
    'Fin.Stress':  '#BAB0AC',
}

FRED_SERIES_LIST = [
    'DGS10', 'DGS2', 'DGS3MO', 'DGS5', 'DGS30',
    'T10Y2Y', 'T10Y3M',
    'DFII10', 'DFII5',
    'T10YIE', 'T5YIE', 'T5YIFR',
    'BAMLH0A0HYM2', 'BAMLC0A0CM',
    'CPIAUCSL', 'CPILFESL', 'PPIACO', 'PCEPILFE',
    'FEDFUNDS', 'M2SL',
    'VIXCLS', 'NFCI', 'STLFSI4',
    'UMCSENT', 'ICSA', 'RSXFS', 'HOUST', 'MORTGAGE30US',
    'INDPRO', 'UNRATE', 'PSAVERT', 'USSLIND',
]

YAHOO_TICKERS = ['SPY', 'HYG', 'LQD', 'TLT', 'GLD', 'UUP', '^VIX', 'GC=F', 'CL=F', 'HG=F']


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_all_data():
    end_str = datetime.today().strftime('%Y-%m-%d')

    # FRED via fredapi
    print('  Fetching FRED series...')
    fred = {}
    try:
        from fredapi import Fred
        fred_client = Fred(api_key=os.getenv('FRED_API_KEY', ''))
        for sid in FRED_SERIES_LIST:
            try:
                s = fred_client.get_series(sid, observation_start=DATA_START_STR,
                                           observation_end=end_str)
                if s is not None and not s.empty:
                    s = s.resample('W-FRI').last().ffill().dropna()
                    if len(s) > 52:
                        fred[sid] = s
            except Exception:
                pass
        print(f'    FRED: {len(fred)} series fetched')
    except Exception as e:
        print(f'    FRED fetch error: {e}')

    # Fallback for RSXFS -> RSAFS if RSXFS missing
    if 'RSXFS' not in fred:
        try:
            from fredapi import Fred
            fred_client = Fred(api_key=os.getenv('FRED_API_KEY', ''))
            s = fred_client.get_series('RSAFS', observation_start=DATA_START_STR,
                                       observation_end=end_str)
            if s is not None and not s.empty:
                s = s.resample('W-FRI').last().ffill().dropna()
                fred['RSXFS'] = s
        except Exception:
            pass

    # Yahoo Finance
    print('  Fetching Yahoo tickers...')
    yahoo = {}
    try:
        raw = yf.download(YAHOO_TICKERS, start=DATA_START_STR, end=end_str,
                          auto_adjust=True, progress=False, timeout=60)
        close = raw['Close']
        if isinstance(close, pd.DataFrame):
            for tk in close.columns:
                s = close[tk].dropna()
                if len(s) > 52:
                    yahoo[str(tk)] = s.resample('W-FRI').last().ffill().dropna()
    except Exception as e:
        print(f'    Yahoo batch failed: {e}')

    for tk in YAHOO_TICKERS:
        if tk not in yahoo:
            try:
                df = yf.download(tk, start=DATA_START_STR, end=end_str,
                                 auto_adjust=True, progress=False, timeout=20)
                if df is not None and not df.empty:
                    s = df['Close']
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    s = s.dropna()
                    if len(s) > 52:
                        yahoo[tk] = s.resample('W-FRI').last().ffill().dropna()
            except Exception:
                pass

    print(f'    Yahoo: {len(yahoo)} tickers fetched')
    return fred, yahoo


# ─────────────────────────────────────────────────────────────────────────────
# Utility primitives
# ─────────────────────────────────────────────────────────────────────────────
def _apply_min_duration(bool_series: pd.Series, min_weeks: int = 2) -> pd.Series:
    """Causal minimum-duration filter — no look-ahead."""
    bs = bool_series.fillna(False).astype(bool)
    not_true = (~bs).cumsum()
    consecutive = bs.groupby(not_true).cumcount() + bs.astype(int)
    return bs & (consecutive >= min_weeks)


def _build_bear_mask(index: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(False, index=index)
    for s_str, e_str, *_ in KNOWN_BEARS:
        mask.loc[pd.Timestamp(s_str):pd.Timestamp(e_str)] = True
    return mask


def _nearest(series: pd.Series, date: pd.Timestamp):
    try:
        idx = series.index.get_indexer([date], method='nearest')[0]
        return series.iloc[idx] if idx >= 0 else None
    except Exception:
        return None


def _shade_known_bears(ax, trim_start, trim_end, alpha=0.20, color='#FFBBBB'):
    for s_str, e_str, *_ in KNOWN_BEARS:
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        if s < trim_end and e > trim_start:
            ax.axvspan(max(s, trim_start), min(e, trim_end),
                       color=color, alpha=alpha, zorder=1)


def _zscore_level(series: pd.Series, window: int = 104) -> pd.Series:
    s = series.ffill()
    mu  = s.rolling(window, min_periods=window // 4).mean()
    std = s.rolling(window, min_periods=window // 4).std()
    return (s - mu) / (std + 1e-9)


def _zscore_momentum(series: pd.Series, lookback: int, window: int = 104) -> pd.Series:
    delta = series.diff(lookback)
    std   = series.rolling(window, min_periods=window // 4).std()
    return delta / (std + 1e-9)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period, min_periods=period // 2).mean()
    avg_loss = loss.rolling(period, min_periods=period // 2).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return (100 - (100 / (1 + rs))).fillna(50)


def _safe_bool(s, name):
    if s is None or (hasattr(s, 'empty') and s.empty):
        return pd.Series(dtype=bool, name=name)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Signal builders — Domain 1: Price/Technical
# ─────────────────────────────────────────────────────────────────────────────
def sig_MA200(spy_w):
    ma  = spy_w.rolling(40, min_periods=20).mean()
    raw = spy_w < ma
    return _apply_min_duration(raw, 4).rename('MA200')


def sig_MA50(spy_w):
    ma  = spy_w.rolling(10, min_periods=5).mean()
    raw = spy_w < ma
    return _apply_min_duration(raw, 2).rename('MA50')


def sig_RSI_BELOW40(spy_w):
    rsi = _compute_rsi(spy_w, 14)
    raw = rsi < 40
    return _apply_min_duration(raw, 2).rename('RSI_BELOW40')


def sig_RETURN_NEG52(spy_w):
    ret = spy_w.pct_change(52)
    raw = ret < -0.05
    return _apply_min_duration(raw, 2).rename('RETURN_NEG52')


def sig_BREADTH_WEAK(spy_w, tlt_w):
    if tlt_w is None or tlt_w.empty:
        return _safe_bool(None, 'BREADTH_WEAK')
    ratio = spy_w / tlt_w.reindex(spy_w.index).ffill()
    ma52  = ratio.rolling(52, min_periods=26).mean()
    raw   = ratio < ma52
    return _apply_min_duration(raw, 2).rename('BREADTH_WEAK')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 2: Volatility
# ─────────────────────────────────────────────────────────────────────────────
def sig_VIX_HIGH20(vix):
    if vix is None or vix.empty: return _safe_bool(None, 'VIX_HIGH20')
    return _apply_min_duration(vix > 20, 2).rename('VIX_HIGH20')


def sig_VIX_HIGH25(vix):
    if vix is None or vix.empty: return _safe_bool(None, 'VIX_HIGH25')
    return _apply_min_duration(vix > 25, 2).rename('VIX_HIGH25')


def sig_VIX_SPIKE(vix):
    if vix is None or vix.empty: return _safe_bool(None, 'VIX_SPIKE')
    z = _zscore_momentum(vix, lookback=4)
    return _apply_min_duration(z > 1.5, 2).rename('VIX_SPIKE')


def sig_VIX_ZSCORE(vix):
    if vix is None or vix.empty: return _safe_bool(None, 'VIX_ZSCORE')
    z = _zscore_level(vix)
    return _apply_min_duration(z > 1.2, 2).rename('VIX_ZSCORE')


def sig_REALIZED_VOL(spy_w):
    rv  = spy_w.pct_change().rolling(13, min_periods=6).std() * np.sqrt(52)
    raw = rv > 0.20
    return _apply_min_duration(raw, 2).rename('REALIZED_VOL')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 3: Yield Curve
# ─────────────────────────────────────────────────────────────────────────────
def sig_CURVE_2Y_INV(t10y2y):
    if t10y2y is None or t10y2y.empty: return _safe_bool(None, 'CURVE_2Y_INV')
    return _apply_min_duration(t10y2y < 0, 2).rename('CURVE_2Y_INV')


def sig_CURVE_3M_INV(t10y3m):
    if t10y3m is None or t10y3m.empty: return _safe_bool(None, 'CURVE_3M_INV')
    return _apply_min_duration(t10y3m < 0, 2).rename('CURVE_3M_INV')


def sig_CURVE_2Y_FLAT(t10y2y):
    if t10y2y is None or t10y2y.empty: return _safe_bool(None, 'CURVE_2Y_FLAT')
    z = _zscore_momentum(t10y2y, lookback=8)
    return _apply_min_duration(z < -1.0, 2).rename('CURVE_2Y_FLAT')


def sig_CURVE_3M_FLAT(t10y3m):
    if t10y3m is None or t10y3m.empty: return _safe_bool(None, 'CURVE_3M_FLAT')
    z = _zscore_momentum(t10y3m, lookback=8)
    return _apply_min_duration(z < -1.0, 2).rename('CURVE_3M_FLAT')


def sig_CURVE_BUTTERFLY(dgs5, dgs2, dgs10):
    if any(x is None or x.empty for x in [dgs5, dgs2, dgs10]):
        return _safe_bool(None, 'CURVE_BUTTERFLY')
    idx      = dgs5.index.intersection(dgs2.index).intersection(dgs10.index)
    btfly    = dgs5.reindex(idx) - 0.5 * (dgs2.reindex(idx) + dgs10.reindex(idx))
    z        = _zscore_level(btfly)
    return _apply_min_duration(z < -1.0, 2).rename('CURVE_BUTTERFLY')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 4: Credit
# ─────────────────────────────────────────────────────────────────────────────
def sig_HYG_LQD_STRESS(hyg, lqd):
    if hyg is None or lqd is None or hyg.empty or lqd.empty:
        return _safe_bool(None, 'HYG_LQD_STRESS')
    ratio = hyg / lqd.reindex(hyg.index).ffill()
    z     = _zscore_momentum(ratio, lookback=4)
    return _apply_min_duration(z < -1.0, 2).rename('HYG_LQD_STRESS')


def sig_HY_OAS_HIGH(hy_oas):
    if hy_oas is None or hy_oas.empty: return _safe_bool(None, 'HY_OAS_HIGH')
    z = _zscore_level(hy_oas)
    return _apply_min_duration(z > 1.2, 2).rename('HY_OAS_HIGH')


def sig_HY_OAS_SPIKE(hy_oas):
    if hy_oas is None or hy_oas.empty: return _safe_bool(None, 'HY_OAS_SPIKE')
    z = _zscore_momentum(hy_oas, lookback=4)
    return _apply_min_duration(z > 1.2, 2).rename('HY_OAS_SPIKE')


def sig_IG_OAS_HIGH(ig_oas):
    if ig_oas is None or ig_oas.empty: return _safe_bool(None, 'IG_OAS_HIGH')
    z = _zscore_level(ig_oas)
    return _apply_min_duration(z > 1.0, 2).rename('IG_OAS_HIGH')


def sig_HY_IG_RATIO(hy_oas, ig_oas):
    if hy_oas is None or ig_oas is None or hy_oas.empty or ig_oas.empty:
        return _safe_bool(None, 'HY_IG_RATIO')
    idx   = hy_oas.index.intersection(ig_oas.index)
    ratio = hy_oas.reindex(idx) / (ig_oas.reindex(idx) + 1e-9)
    z     = _zscore_level(ratio)
    return _apply_min_duration(z > 1.0, 2).rename('HY_IG_RATIO')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 5: Inflation  (monthly FRED — 5-week release lag)
# ─────────────────────────────────────────────────────────────────────────────
def sig_CPI_HIGH3(cpi):
    if cpi is None or cpi.empty: return _safe_bool(None, 'CPI_HIGH3')
    yoy = cpi.pct_change(52) * 100
    lag = yoy.shift(MONTHLY_LAG_WEEKS)
    return _apply_min_duration(lag > 3.0, 2).rename('CPI_HIGH3')


def sig_CPI_HIGH4(cpi):
    if cpi is None or cpi.empty: return _safe_bool(None, 'CPI_HIGH4')
    yoy = cpi.pct_change(52) * 100
    lag = yoy.shift(MONTHLY_LAG_WEEKS)
    return _apply_min_duration(lag > 4.0, 2).rename('CPI_HIGH4')


def sig_CORECPI_ZSCORE(core_cpi):
    if core_cpi is None or core_cpi.empty: return _safe_bool(None, 'CORECPI_ZSCORE')
    yoy = core_cpi.pct_change(52) * 100
    lag = yoy.shift(MONTHLY_LAG_WEEKS)
    z   = _zscore_level(lag)
    return _apply_min_duration(z > 1.0, 2).rename('CORECPI_ZSCORE')


def sig_FED_BEHIND(cpi, fedfunds):
    if cpi is None or fedfunds is None or cpi.empty or fedfunds.empty:
        return _safe_bool(None, 'FED_BEHIND')
    yoy    = cpi.pct_change(52) * 100
    cpi_l  = yoy.shift(MONTHLY_LAG_WEEKS)
    ff_l   = fedfunds.shift(MONTHLY_LAG_WEEKS)
    idx    = cpi_l.index.intersection(ff_l.index)
    gap    = cpi_l.reindex(idx) - ff_l.reindex(idx)
    return _apply_min_duration(gap > 2.0, 2).rename('FED_BEHIND')


def sig_PPI_CPI_SPREAD(ppi, cpi):
    if ppi is None or cpi is None or ppi.empty or cpi.empty:
        return _safe_bool(None, 'PPI_CPI_SPREAD')
    ppi_yoy = ppi.pct_change(52) * 100
    cpi_yoy = cpi.pct_change(52) * 100
    idx     = ppi_yoy.index.intersection(cpi_yoy.index)
    spread  = ppi_yoy.reindex(idx).shift(MONTHLY_LAG_WEEKS) - cpi_yoy.reindex(idx).shift(MONTHLY_LAG_WEEKS)
    z       = _zscore_level(spread)
    return _apply_min_duration(z > 1.0, 2).rename('PPI_CPI_SPREAD')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 6: Real Rates  (daily FRED — no lag)
# ─────────────────────────────────────────────────────────────────────────────
def sig_REAL_Y10_HIGH(dfii10):
    if dfii10 is None or dfii10.empty: return _safe_bool(None, 'REAL_Y10_HIGH')
    return _apply_min_duration(dfii10 > 1.5, 2).rename('REAL_Y10_HIGH')


def sig_REAL_Y5_ZSCORE(dfii5):
    if dfii5 is None or dfii5.empty: return _safe_bool(None, 'REAL_Y5_ZSCORE')
    z = _zscore_level(dfii5)
    return _apply_min_duration(z > 1.0, 2).rename('REAL_Y5_ZSCORE')


def sig_REAL_Y_MOMENTUM(dfii5):
    if dfii5 is None or dfii5.empty: return _safe_bool(None, 'REAL_Y_MOMENTUM')
    z = _zscore_momentum(dfii5, lookback=8)
    return _apply_min_duration(z > 1.0, 2).rename('REAL_Y_MOMENTUM')


def sig_TIPS_SPREAD(dfii10, dfii5):
    if dfii10 is None or dfii5 is None or dfii10.empty or dfii5.empty:
        return _safe_bool(None, 'TIPS_SPREAD')
    idx    = dfii10.index.intersection(dfii5.index)
    spread = dfii10.reindex(idx) - dfii5.reindex(idx)
    z      = _zscore_momentum(spread, lookback=8)
    return _apply_min_duration(z > 1.0, 2).rename('TIPS_SPREAD')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 7: Monetary/Liquidity  (M2 monthly — 5-week lag; FF monthly — lag)
# ─────────────────────────────────────────────────────────────────────────────
def sig_M2_CONTRACT(m2):
    if m2 is None or m2.empty: return _safe_bool(None, 'M2_CONTRACT')
    yoy = m2.pct_change(52) * 100
    lag = yoy.shift(MONTHLY_LAG_WEEKS)
    return _apply_min_duration(lag < 0, 2).rename('M2_CONTRACT')


def sig_M2_ZSCORE(m2):
    if m2 is None or m2.empty: return _safe_bool(None, 'M2_ZSCORE')
    yoy = m2.pct_change(52) * 100
    lag = yoy.shift(MONTHLY_LAG_WEEKS)
    z   = _zscore_level(lag)
    return _apply_min_duration(z < -1.0, 2).rename('M2_ZSCORE')


def sig_USD_SURGE(uup):
    if uup is None or uup.empty: return _safe_bool(None, 'USD_SURGE')
    z = _zscore_momentum(uup, lookback=8)
    return _apply_min_duration(z > 1.0, 2).rename('USD_SURGE')


def sig_FF_HIKE(fedfunds):
    if fedfunds is None or fedfunds.empty: return _safe_bool(None, 'FF_HIKE')
    lag = fedfunds.shift(MONTHLY_LAG_WEEKS)
    z   = _zscore_momentum(lag, lookback=13)
    return _apply_min_duration(z > 1.0, 2).rename('FF_HIKE')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 8: Consumer/Activity  (mostly monthly — 5-week lag; ICSA weekly)
# ─────────────────────────────────────────────────────────────────────────────
def sig_SENTIMENT_WEAK(umcsent):
    if umcsent is None or umcsent.empty: return _safe_bool(None, 'SENTIMENT_WEAK')
    lag = umcsent.shift(MONTHLY_LAG_WEEKS)
    z   = _zscore_level(lag)
    return _apply_min_duration(z < -1.0, 2).rename('SENTIMENT_WEAK')


def sig_CLAIMS_SPIKE(icsa):
    if icsa is None or icsa.empty: return _safe_bool(None, 'CLAIMS_SPIKE')
    z = _zscore_momentum(icsa, lookback=4)
    return _apply_min_duration(z > 1.2, 2).rename('CLAIMS_SPIKE')


def sig_RETAIL_WEAK(rsxfs):
    if rsxfs is None or rsxfs.empty: return _safe_bool(None, 'RETAIL_WEAK')
    lag = rsxfs.shift(MONTHLY_LAG_WEEKS)
    z   = _zscore_momentum(lag, lookback=8)
    return _apply_min_duration(z < -1.0, 2).rename('RETAIL_WEAK')


def sig_INDPRO_WEAK(indpro):
    if indpro is None or indpro.empty: return _safe_bool(None, 'INDPRO_WEAK')
    lag = indpro.shift(MONTHLY_LAG_WEEKS)
    z   = _zscore_momentum(lag, lookback=8)
    return _apply_min_duration(z < -1.0, 2).rename('INDPRO_WEAK')


def sig_LEI_WEAK(usslind):
    if usslind is None or usslind.empty: return _safe_bool(None, 'LEI_WEAK')
    lag = usslind.shift(MONTHLY_LAG_WEEKS)
    z   = _zscore_momentum(lag, lookback=8)
    return _apply_min_duration(z < -1.0, 2).rename('LEI_WEAK')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 9: Gold/Commodities
# ─────────────────────────────────────────────────────────────────────────────
def sig_GOLD_SURGE(gld):
    if gld is None or gld.empty: return _safe_bool(None, 'GOLD_SURGE')
    z = _zscore_momentum(gld, lookback=4)
    return _apply_min_duration(z > 1.2, 2).rename('GOLD_SURGE')


def sig_GOLD_VS_SPY(gld, spy_w):
    if gld is None or gld.empty: return _safe_bool(None, 'GOLD_VS_SPY')
    ratio = gld.reindex(spy_w.index).ffill() / spy_w
    z     = _zscore_momentum(ratio, lookback=8)
    return _apply_min_duration(z > 1.0, 2).rename('GOLD_VS_SPY')


def sig_OIL_CRASH(oil):
    if oil is None or oil.empty: return _safe_bool(None, 'OIL_CRASH')
    ret4w = oil.pct_change(4)
    return _apply_min_duration(ret4w < -0.15, 2).rename('OIL_CRASH')


def sig_COPPER_WEAK(copper):
    if copper is None or copper.empty: return _safe_bool(None, 'COPPER_WEAK')
    z = _zscore_momentum(copper, lookback=13)
    return _apply_min_duration(z < -1.0, 2).rename('COPPER_WEAK')


# ─────────────────────────────────────────────────────────────────────────────
# Domain 10: Financial Stress
# ─────────────────────────────────────────────────────────────────────────────
def sig_NFCI_TIGHT(nfci):
    if nfci is None or nfci.empty: return _safe_bool(None, 'NFCI_TIGHT')
    return _apply_min_duration(nfci > 0, 2).rename('NFCI_TIGHT')


def sig_NFCI_SPIKE(nfci):
    if nfci is None or nfci.empty: return _safe_bool(None, 'NFCI_SPIKE')
    z = _zscore_momentum(nfci, lookback=4)
    return _apply_min_duration(z > 1.0, 2).rename('NFCI_SPIKE')


def sig_STLFSI_HIGH(stlfsi):
    if stlfsi is None or stlfsi.empty: return _safe_bool(None, 'STLFSI_HIGH')
    z = _zscore_level(stlfsi)
    return _apply_min_duration(z > 1.0, 2).rename('STLFSI_HIGH')


def sig_TLT_FLIGHT(tlt):
    if tlt is None or tlt.empty: return _safe_bool(None, 'TLT_FLIGHT')
    z = _zscore_momentum(tlt, lookback=8)
    return _apply_min_duration(z > 1.0, 2).rename('TLT_FLIGHT')


# ─────────────────────────────────────────────────────────────────────────────
# Master signal builder
# ─────────────────────────────────────────────────────────────────────────────
def build_all_signals(fred: dict, yahoo: dict, weekly_index: pd.DatetimeIndex) -> dict:
    spy_w = yahoo.get('SPY')
    if spy_w is None:
        raise RuntimeError('SPY data is required')

    vix = fred.get('VIXCLS')
    if vix is None or vix.empty:
        vix = yahoo.get('^VIX')
    gld = yahoo.get('GLD')
    if gld is None or gld.empty:
        gld = yahoo.get('GC=F')

    raw = {
        # Price/Technical
        'MA200':         sig_MA200(spy_w),
        'MA50':          sig_MA50(spy_w),
        'RSI_BELOW40':   sig_RSI_BELOW40(spy_w),
        'RETURN_NEG52':  sig_RETURN_NEG52(spy_w),
        'BREADTH_WEAK':  sig_BREADTH_WEAK(spy_w, yahoo.get('TLT')),
        # Volatility
        'VIX_HIGH20':    sig_VIX_HIGH20(vix),
        'VIX_HIGH25':    sig_VIX_HIGH25(vix),
        'VIX_SPIKE':     sig_VIX_SPIKE(vix),
        'VIX_ZSCORE':    sig_VIX_ZSCORE(vix),
        'REALIZED_VOL':  sig_REALIZED_VOL(spy_w),
        # Yield Curve
        'CURVE_2Y_INV':   sig_CURVE_2Y_INV(fred.get('T10Y2Y')),
        'CURVE_3M_INV':   sig_CURVE_3M_INV(fred.get('T10Y3M')),
        'CURVE_2Y_FLAT':  sig_CURVE_2Y_FLAT(fred.get('T10Y2Y')),
        'CURVE_3M_FLAT':  sig_CURVE_3M_FLAT(fred.get('T10Y3M')),
        'CURVE_BUTTERFLY':sig_CURVE_BUTTERFLY(fred.get('DGS5'), fred.get('DGS2'), fred.get('DGS10')),
        # Credit
        'HYG_LQD_STRESS': sig_HYG_LQD_STRESS(yahoo.get('HYG'), yahoo.get('LQD')),
        'HY_OAS_HIGH':    sig_HY_OAS_HIGH(fred.get('BAMLH0A0HYM2')),
        'HY_OAS_SPIKE':   sig_HY_OAS_SPIKE(fred.get('BAMLH0A0HYM2')),
        'IG_OAS_HIGH':    sig_IG_OAS_HIGH(fred.get('BAMLC0A0CM')),
        'HY_IG_RATIO':    sig_HY_IG_RATIO(fred.get('BAMLH0A0HYM2'), fred.get('BAMLC0A0CM')),
        # Inflation
        'CPI_HIGH3':      sig_CPI_HIGH3(fred.get('CPIAUCSL')),
        'CPI_HIGH4':      sig_CPI_HIGH4(fred.get('CPIAUCSL')),
        'CORECPI_ZSCORE': sig_CORECPI_ZSCORE(fred.get('CPILFESL')),
        'FED_BEHIND':     sig_FED_BEHIND(fred.get('CPIAUCSL'), fred.get('FEDFUNDS')),
        'PPI_CPI_SPREAD': sig_PPI_CPI_SPREAD(fred.get('PPIACO'), fred.get('CPIAUCSL')),
        # Real Rates
        'REAL_Y10_HIGH':   sig_REAL_Y10_HIGH(fred.get('DFII10')),
        'REAL_Y5_ZSCORE':  sig_REAL_Y5_ZSCORE(fred.get('DFII5')),
        'REAL_Y_MOMENTUM': sig_REAL_Y_MOMENTUM(fred.get('DFII5')),
        'TIPS_SPREAD':     sig_TIPS_SPREAD(fred.get('DFII10'), fred.get('DFII5')),
        # Monetary
        'M2_CONTRACT':  sig_M2_CONTRACT(fred.get('M2SL')),
        'M2_ZSCORE':    sig_M2_ZSCORE(fred.get('M2SL')),
        'USD_SURGE':    sig_USD_SURGE(yahoo.get('UUP')),
        'FF_HIKE':      sig_FF_HIKE(fred.get('FEDFUNDS')),
        # Consumer
        'SENTIMENT_WEAK': sig_SENTIMENT_WEAK(fred.get('UMCSENT')),
        'CLAIMS_SPIKE':   sig_CLAIMS_SPIKE(fred.get('ICSA')),
        'RETAIL_WEAK':    sig_RETAIL_WEAK(fred.get('RSXFS')),
        'INDPRO_WEAK':    sig_INDPRO_WEAK(fred.get('INDPRO')),
        'LEI_WEAK':       sig_LEI_WEAK(fred.get('USSLIND')),
        # Gold/Commodities
        'GOLD_SURGE':   sig_GOLD_SURGE(gld),
        'GOLD_VS_SPY':  sig_GOLD_VS_SPY(gld, spy_w),
        'OIL_CRASH':    sig_OIL_CRASH(yahoo.get('CL=F')),
        'COPPER_WEAK':  sig_COPPER_WEAK(yahoo.get('HG=F')),
        # Financial Stress
        'NFCI_TIGHT':  sig_NFCI_TIGHT(fred.get('NFCI')),
        'NFCI_SPIKE':  sig_NFCI_SPIKE(fred.get('NFCI')),
        'STLFSI_HIGH': sig_STLFSI_HIGH(fred.get('STLFSI4')),
        'TLT_FLIGHT':  sig_TLT_FLIGHT(yahoo.get('TLT')),
    }

    # Reindex all to common analysis window, fill NaN as False
    signals = {}
    for name, s in raw.items():
        if s is None or (hasattr(s, 'empty') and s.empty):
            signals[name] = pd.Series(False, index=weekly_index, name=name)
        else:
            signals[name] = s.reindex(weekly_index).fillna(False).astype(bool)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_signal(name: str, signal: pd.Series, bear_mask: pd.Series) -> dict:
    sig = signal.reindex(bear_mask.index).fillna(False).astype(bool)
    bm  = bear_mask.astype(bool)

    tp = int((sig & bm).sum())
    fp = int((sig & ~bm).sum())
    fn = int((~sig & bm).sum())
    tn = int((~sig & ~bm).sum())

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    bears_caught = sum(
        1 for s_str, e_str, *_ in KNOWN_BEARS
        if sig.loc[pd.Timestamp(s_str):pd.Timestamp(e_str)].any()
    )

    return {
        'name': name, 'prec': prec, 'rec': rec, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'bears_caught': bears_caught, 'total_bears': len(KNOWN_BEARS),
    }


def evaluate_all_signals(signals: dict, bear_mask: pd.Series) -> pd.DataFrame:
    rows = [evaluate_signal(name, sig, bear_mask) for name, sig in signals.items()]
    df   = pd.DataFrame(rows).set_index('name')
    return df.sort_values('f1', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Combination search
# ─────────────────────────────────────────────────────────────────────────────
def compute_lead_time(signal: pd.Series, bear_start_str: str):
    peak_ts = pd.Timestamp(bear_start_str)
    window  = signal.loc[peak_ts - pd.Timedelta(weeks=52): peak_ts]
    active  = window[window]
    if active.empty:
        return None
    return max(0, int((peak_ts - active.index[0]).days / 7))


def _combine_signals(components: list, logic: str, signals: dict,
                     weekly_index: pd.DatetimeIndex) -> pd.Series:
    stack = pd.DataFrame({
        name: signals[name].reindex(weekly_index).fillna(False)
        for name in components
    })
    if logic == 'AND':
        raw = stack.all(axis=1)
    else:  # MAJORITY
        n   = len(components)
        raw = stack.sum(axis=1) > (n / 2)
    return _apply_min_duration(raw.astype(bool), MIN_DURATION_WKS)


def search_combinations(signals: dict, eval_df: pd.DataFrame,
                         bear_mask: pd.Series,
                         weekly_index: pd.DatetimeIndex) -> pd.DataFrame:
    top15 = list(eval_df.head(TOP_N_INDIVIDUAL).index)
    top10 = list(eval_df.head(TOP_N_TRIPLES).index)
    top5  = list(eval_df.head(TOP_N_QUADS).index)

    combos = []
    for pair   in itertools.combinations(top15, 2):
        combos.extend([(list(pair), 'AND'), (list(pair), 'MAJORITY')])
    for triple in itertools.combinations(top10, 3):
        combos.extend([(list(triple), 'AND'), (list(triple), 'MAJORITY')])
    for quad   in itertools.combinations(top5, 4):
        combos.extend([(list(quad), 'AND'), (list(quad), 'MAJORITY')])

    rows = []
    for components, logic in combos:
        combo_sig = _combine_signals(components, logic, signals, weekly_index)
        r = evaluate_signal('+'.join(components), combo_sig, bear_mask)
        r['components'] = components
        r['logic']      = logic
        rows.append(r)

    df = pd.DataFrame(rows).sort_values('f1', ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────
def print_individual_ranking(eval_df: pd.DataFrame):
    print()
    print('=' * 74)
    print('  INDIVIDUAL SIGNAL RANKING  (Top 25 by F1)')
    print('=' * 74)
    print(f"  {'Rank':<5} {'Signal':<20} {'Domain':<14} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Bears':>8}")
    print('  ' + '-' * 70)
    for rank, (name, row) in enumerate(eval_df.head(DISPLAY_TOP).iterrows(), 1):
        dom   = DOMAIN_MAP.get(name, 'Unknown')[:13]
        bears = f"{int(row['bears_caught'])}/{int(row['total_bears'])}"
        print(f"  {rank:<5} {name:<20} {dom:<14} "
              f"{row['prec']:>6.1%} {row['rec']:>7.1%} {row['f1']:>7.1%} {bears:>8}")


def print_combo_ranking(combo_df: pd.DataFrame):
    print()
    print('=' * 74)
    print('  TOP 10 COMBINATIONS  (by F1 score)')
    print('=' * 74)
    print(f"  {'#':<4} {'Combination':<42} {'Logic':<9} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print('  ' + '-' * 70)
    for rank, (_, row) in enumerate(combo_df.head(10).iterrows(), 1):
        combo = ' + '.join(row['components'])
        if len(combo) > 40:
            combo = combo[:37] + '...'
        print(f"  {rank:<4} {combo:<42} {row['logic']:<9} "
              f"{row['prec']:>6.1%} {row['rec']:>7.1%} {row['f1']:>7.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Charting
# ─────────────────────────────────────────────────────────────────────────────
def _draw_combo_panel(fig, outer_cell, spy_w: pd.Series, combo_sig: pd.Series,
                      combo_row: dict, panel_num: int, bear_mask: pd.Series):

    inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_cell,
                                       height_ratios=[0.55, 0.45], hspace=0.04)
    ax_p = fig.add_subplot(inner_gs[0])
    ax_s = fig.add_subplot(inner_gs[1], sharex=ax_p)

    trim_start = pd.Timestamp(ANALYSIS_START)
    trim_end   = spy_w.index[-1]

    spy = spy_w.loc[trim_start:trim_end]
    sig = combo_sig.reindex(spy.index).fillna(False)
    bm  = bear_mask.reindex(spy.index)

    for ax in [ax_p, ax_s]:
        ax.set_facecolor('#F8F8F8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')

    # Known bear shading on both panels
    _shade_known_bears(ax_p, trim_start, trim_end, alpha=0.30)
    _shade_known_bears(ax_s, trim_start, trim_end, alpha=0.20)

    # Combo signal active: green axvspan loop (avoids semilogy incompatibility)
    in_sig = False; seg_start = None
    for dt, val in sig.items():
        if not in_sig and val:
            seg_start = dt; in_sig = True
        elif in_sig and not val:
            ax_p.axvspan(seg_start, dt, color='#AAFFAA', alpha=0.35, zorder=2)
            in_sig = False
    if in_sig and seg_start is not None:
        ax_p.axvspan(seg_start, sig.index[-1], color='#AAFFAA', alpha=0.35, zorder=2)

    # SPY log price
    ax_p.semilogy(spy.index, spy.values, color='#222222', lw=1.4, zorder=5)

    # Entry (▼) and exit (▲) markers on price
    arr  = sig.values.astype(int)
    diff = np.diff(arr, prepend=0)
    for d in sig.index[diff == 1]:
        y = _nearest(spy, d)
        if y:
            ax_p.scatter([d], [y * 0.96], marker='v', color='#CC0000',
                         s=45, zorder=10, linewidths=0)
    for d in sig.index[diff == -1]:
        y = _nearest(spy, d)
        if y:
            ax_p.scatter([d], [y * 1.04], marker='^', color='#006600',
                         s=45, zorder=10, linewidths=0)

    # Lead time annotations at each bear
    for s_str, _e, bear_label in KNOWN_BEARS:
        lead = compute_lead_time(sig, s_str)
        peak = pd.Timestamp(s_str)
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
            col = '#AA0000'
        ax_p.text(peak, y_val * 1.18, txt,
                  fontsize=5.5, ha='center', color=col,
                  fontweight='bold', zorder=11)

    ax_p.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'${x:.0f}'))
    ax_p.grid(axis='y', which='both', lw=0.3, color='#DDDDDD', zorder=0)
    ax_p.set_ylabel('SPY', fontsize=8)
    plt.setp(ax_p.get_xticklabels(), visible=False)

    # Signal indicator bar (bottom panel)
    sig_f = sig.astype(float)
    ax_s.fill_between(sig.index, 0, sig_f, step='post',
                      color='#44AA44', alpha=0.65, zorder=5)
    # Bear mask overlay at 0.4 height
    bm_f = bm.reindex(sig.index).fillna(False).astype(float) * 0.4
    ax_s.fill_between(sig.index, 0, bm_f, step='post',
                      color='#FF5555', alpha=0.30, zorder=3)
    ax_s.axhline(0.5, color='#999999', lw=0.5, ls='--')
    ax_s.set_ylim(0, 1.2)
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels(['off', 'on'], fontsize=7)
    ax_s.set_ylabel('Signal', fontsize=8)
    ax_s.grid(axis='y', lw=0.3, color='#DDDDDD')
    ax_s.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_s.get_xticklabels(), rotation=0, ha='center', fontsize=8)
    ax_s.set_xlim(trim_start, trim_end)

    # Title with metrics
    prec = combo_row['prec'] * 100
    rec  = combo_row['rec']  * 100
    f1   = combo_row['f1']   * 100
    logic= combo_row['logic']
    comp = ' + '.join(combo_row['components'])
    if len(comp) > 62:
        comp = comp[:59] + '...'
    title = (f"Combo #{panel_num}: {comp}  [{logic}] "
             f"| F1={f1:.1f}%  Prec={prec:.1f}%  Rec={rec:.1f}%")
    ax_p.set_title(title, fontsize=8, loc='left', pad=3,
                   bbox=dict(boxstyle='round,pad=0.3', fc='#FFFFF0',
                             ec='#AAAAAA', alpha=0.92))

    # Legend
    legend_elements = [
        mpatches.Patch(color='#FFBBBB', alpha=0.60, label='Known bear (ground truth)'),
        mpatches.Patch(color='#AAFFAA', alpha=0.60, label='Signal active (predicted bearish)'),
        Line2D([0], [0], marker='v', color='#CC0000', lw=0, ms=6, label='Signal entry ▼'),
        Line2D([0], [0], marker='^', color='#006600', lw=0, ms=6, label='Signal exit ▲'),
    ]
    ax_p.legend(handles=legend_elements, loc='upper left', fontsize=6.5,
                framealpha=0.88, edgecolor='#CCCCCC', ncol=2)


def plot_combo_analysis(spy_w: pd.Series, eval_df: pd.DataFrame,
                         combo_df: pd.DataFrame, signals: dict,
                         bear_mask: pd.Series, top3_combos: list,
                         save_path: str):

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle(
        'SPY Bear Market Analysis — Comprehensive Signal Sweep  |  ~46 Signals × 460 Combinations  |  1994–2026\n'
        'Green shading = signal predicting bear.  Pink shading = confirmed bear period.',
        fontsize=12, fontweight='bold', y=0.998)

    gs = GridSpec(4, 1, figure=fig,
                  height_ratios=[0.29, 0.235, 0.235, 0.24],
                  hspace=0.08)

    # ── Panel 1: Horizontal bar chart ──────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.set_facecolor('#F8F8F8')

    top25 = eval_df.head(DISPLAY_TOP)
    names  = list(reversed(top25.index.tolist()))
    f1vals = [top25.loc[n, 'f1'] * 100 for n in names]
    colors = [DOMAIN_COLORS.get(DOMAIN_MAP.get(n, 'Fin.Stress'), '#BAB0AC') for n in names]
    prec_v = [top25.loc[n, 'prec'] * 100 for n in names]
    rec_v  = [top25.loc[n, 'rec']  * 100 for n in names]

    y_pos = range(len(names))
    bars = ax_bar.barh(y_pos, f1vals, color=colors, alpha=0.82,
                       edgecolor='white', lw=0.5, zorder=3)

    # Precision dots and recall dots
    ax_bar.scatter(prec_v, y_pos, marker='|', color='#333333', s=60,
                   zorder=5, linewidths=1.5, label='Precision')
    ax_bar.scatter(rec_v, y_pos, marker='D', color='#666666', s=12,
                   zorder=5, label='Recall')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=7.5)
    ax_bar.set_xlabel('Score (%)', fontsize=9)
    ax_bar.set_xlim(0, 102)
    ax_bar.axvline(50, color='#AAAAAA', lw=0.7, ls=':', zorder=2)

    # MA200 baseline
    if 'MA200' in eval_df.index:
        ma200_f1 = eval_df.loc['MA200', 'f1'] * 100
        ax_bar.axvline(ma200_f1, color='#4E79A7', lw=1.5, ls='--', alpha=0.8,
                       label=f'MA200 baseline F1={ma200_f1:.1f}%')

    # Value labels
    for bar, val, rec in zip(bars, f1vals, rec_v):
        ax_bar.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}%', va='center', fontsize=6.5, color='#333333')

    # Domain legend
    domain_patches = [mpatches.Patch(color=c, alpha=0.82, label=d)
                      for d, c in DOMAIN_COLORS.items()]
    extra_patches = [
        Line2D([0], [0], marker='|', color='#333333', lw=0, ms=8,
               markeredgewidth=1.5, label='Precision'),
        Line2D([0], [0], marker='D', color='#666666', lw=0, ms=5, label='Recall'),
    ]
    if 'MA200' in eval_df.index:
        extra_patches.append(Line2D([0], [0], color='#4E79A7', lw=1.5, ls='--',
                                    label=f'MA200 F1={eval_df.loc["MA200","f1"]*100:.1f}%'))
    ax_bar.legend(handles=domain_patches + extra_patches,
                  loc='lower right', ncol=3, fontsize=7, framealpha=0.90,
                  edgecolor='#CCCCCC')

    ax_bar.set_title(
        'Panel 1 — Individual Signal F1 Scores  (|  = Precision,  ◆  = Recall)',
        fontsize=9, loc='left', pad=4)
    ax_bar.grid(axis='x', lw=0.4, color='#DDDDDD', zorder=0)

    # ── Panels 2-4: Top 3 combinations ─────────────────────────────────────────
    for i, (combo_sig, combo_row) in enumerate(top3_combos):
        _draw_combo_panel(fig, gs[i + 1], spy_w, combo_sig,
                          combo_row, i + 1, bear_mask)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print('=' * 68)
    print('  SPY BEAR COMBO ANALYSIS — ~46 SIGNALS x 460 COMBINATIONS')
    print('=' * 68)

    # 1. Fetch
    print('\n  [1/5] Fetching data...')
    fred, yahoo = fetch_all_data()
    spy_w = yahoo.get('SPY')
    if spy_w is None:
        raise RuntimeError('SPY data unavailable — check internet/Yahoo Finance')

    weekly_index = spy_w[spy_w.index >= ANALYSIS_START].index
    bear_mask    = _build_bear_mask(weekly_index)

    # 2. Signals
    print('\n  [2/5] Building signals...')
    signals = build_all_signals(fred, yahoo, weekly_index)
    active  = sum(1 for s in signals.values() if s.any())
    print(f'    {len(signals)} signals built; {active} have at least 1 active week')

    # 3. Evaluate individuals
    print('\n  [3/5] Evaluating individual signals...')
    eval_df = evaluate_all_signals(signals, bear_mask)
    print_individual_ranking(eval_df)

    # 4. Combination search
    print(f'\n  [4/5] Searching combinations...')
    combo_df = search_combinations(signals, eval_df, bear_mask, weekly_index)
    total = len(combo_df)
    print(f'    Tested {total} combinations')
    print_combo_ranking(combo_df)

    # 5. Chart
    print('\n  [5/5] Generating chart...')
    top3_combos = []
    for _, row in combo_df.head(3).iterrows():
        sig = _combine_signals(row['components'], row['logic'], signals, weekly_index)
        top3_combos.append((sig, row.to_dict()))

    chart_path = os.path.join(_ROOT, 'spy_bear_combo_analysis_chart.png')
    plot_combo_analysis(
        spy_w       = spy_w,
        eval_df     = eval_df,
        combo_df    = combo_df,
        signals     = signals,
        bear_mask   = bear_mask,
        top3_combos = top3_combos,
        save_path   = chart_path,
    )
    print(f'  Chart saved: {chart_path}')
    print('=' * 68)
    print('  DONE')
    print('=' * 68)


if __name__ == '__main__':
    main()
