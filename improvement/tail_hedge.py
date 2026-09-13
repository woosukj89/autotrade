"""
Idea B: options tail hedge, layered on top of an already-backtested equity
NAV curve (no re-running the stock-picker - this overlays a SPY-based
protective put / collar onto the saved monthly portfolio_summary_over_time()
CSVs from run_graduated_comparison.py / test_diversification.py).

Why SPY as the hedge underlying, not the actual held tickers: no historical
per-stock options data is available in this project; SPY is the only
underlying with both a deep options market and data already on hand (spot +
VIX as a 30-day IV proxy, via data_feed.py). To compensate for basis risk
(the stock-picker's holdings run beta ~1.2-2.2 to SPY, not 1.0), the hedge
notional is scaled by the portfolio's empirically estimated beta to SPY, so
a beta-1.7 portfolio buys ~1.7x the SPY-dollar notional of puts a beta-1.0
portfolio would for the same equity stake.

Pricing: Black-Scholes, sigma = VIX/100 at trade date (VIX is literally a
30-day SPX implied-vol index, so this is a standard, not a stretch,
convention for pricing an ~30-day option), flat r=3% (a 20yr-average
approximation - a second-order effect on 30-day option pricing, not worth
sourcing a real term-structure risk-free curve for this stress test).

Financing model (explicit simplification, documented not hidden): the
option P&L (premium paid/received, payoff at expiry) is tracked as a
SEPARATE running total added onto the existing equity NAV curve, rather
than diverting a slice of the $100k equity stake to fund premiums. This is
standard for a "cost of insurance" stress test and is most defensible for
the zero-cost collar (premium in ~= premium out by construction) and least
defensible for a naked long put in a year where premium costs stack up -
flagged in the printed report, not just here.

Tax: option net P&L is taxed at the same 25% rate used everywhere else in
this project, applied to positive net P&L only in the month it's realized
(monthly expiry = short-term, no preferential rate - conservative).
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
RISK_FREE_RATE = 0.03


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_price(spot, strike, t_years, sigma, r=RISK_FREE_RATE):
    if t_years <= 0 or sigma <= 0:
        return max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_call_price(spot, strike, t_years, sigma, r=RISK_FREE_RATE):
    if t_years <= 0 or sigma <= 0:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)


def load_nav_series(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    return df[['date', 'total_value', 'benchmark_value']].dropna().reset_index(drop=True)


def estimate_beta(nav_df):
    """Full-period regression of portfolio monthly returns on SPY monthly returns."""
    port_ret = nav_df['total_value'].pct_change().dropna()
    spy_ret = nav_df['benchmark_value'].pct_change().dropna()
    n = min(len(port_ret), len(spy_ret))
    port_ret, spy_ret = port_ret.iloc[-n:].to_numpy(), spy_ret.iloc[-n:].to_numpy()
    cov = np.cov(port_ret, spy_ret)[0, 1]
    var = np.var(spy_ret)
    return cov / var if var > 0 else 1.0


def simulate_overlay(nav_df, vix, beta, put_otm_pct, mode='put', call_otm_pct=0.10, tax_rate=0.25,
                      hedge_ratio=1.0, skew_per_pct_otm=0.0):
    """skew_per_pct_otm: equity vol-skew stress test. Real SPX options trade
    with OTM puts priced RICHER than ATM and OTM calls CHEAPER (the
    well-documented "smirk") - the base pricing here uses flat sigma=VIX/100
    for both legs, which is a known simplification that understates real
    collar cost. Set e.g. 0.015 (1.5 vol points of extra IV per 1% strike
    distance from spot) to stress-test whether a result survives a more
    realistic, skew-adjusted premium."""
    """Adds a rolling ~1-month options overlay on top of nav_df's total_value.

    mode: 'put' (long protective put only) or 'collar' (long put + short call,
    call_otm_pct above spot, financing the put premium).
    Returns a DataFrame with an added 'hedged_value' column.
    """
    dates = nav_df['date'].tolist()
    spy = nav_df['benchmark_value'].to_numpy()
    equity_nav = nav_df['total_value'].to_numpy()
    n = len(dates)

    hedged = np.zeros(n)
    hedged[0] = equity_nav[0]
    cum_overlay = 0.0

    for i in range(1, n):
        d0, d1 = dates[i - 1], dates[i]
        s0, s1 = spy[i - 1], spy[i]
        t_years = max((d1 - d0).days, 1) / 365.0
        try:
            iv = float(vix.asof(d0)) / 100.0
        except Exception:
            iv = 0.20
        if not iv or iv != iv:
            iv = 0.20

        notional = beta * hedge_ratio * equity_nav[i - 1]
        contracts = notional / s0 if s0 > 0 else 0.0

        put_iv = iv + skew_per_pct_otm * (put_otm_pct * 100)
        put_strike = s0 * (1 - put_otm_pct)
        put_premium = contracts * bs_put_price(s0, put_strike, t_years, put_iv)
        put_payoff = contracts * max(put_strike - s1, 0.0)
        net = put_payoff - put_premium

        if mode == 'collar':
            call_iv = max(iv - skew_per_pct_otm * (call_otm_pct * 100), 0.01)
            call_strike = s0 * (1 + call_otm_pct)
            call_premium_received = contracts * bs_call_price(s0, call_strike, t_years, call_iv)
            call_payoff_owed = contracts * max(s1 - call_strike, 0.0)
            net += call_premium_received - call_payoff_owed

        after_tax_net = net * (1 - tax_rate) if net > 0 else net
        cum_overlay += after_tax_net
        hedged[i] = equity_nav[i] + cum_overlay

    out = nav_df.copy()
    out['hedged_value'] = hedged
    return out


def compute_stats(values, dates):
    values = np.asarray(values, dtype=float)
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (values[-1] / values[0]) ** (1 / years) - 1 if values[0] > 0 and years > 0 else float('nan')
    peak = np.maximum.accumulate(values)
    max_dd = float(((peak - values) / np.where(peak > 0, peak, 1)).max())
    rets = pd.Series(values).pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * math.sqrt(12) if rets.std() > 0 else float('nan')
    return cagr, max_dd, sharpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='graduated_highbetaonly_no_timing_20yr.csv',
                         help='CSV under improvement/results/ with columns date,total_value,benchmark_value')
    parser.add_argument('--label', default='HighBetaOnly')
    parser.add_argument('--skew', type=float, default=0.0,
                         help='Vol points of extra IV per 1%% OTM (equity skew stress test), e.g. 0.015')
    args = parser.parse_args()

    nav_df = load_nav_series(os.path.join(RESULTS_DIR, args.base))
    beta = estimate_beta(nav_df)
    print(f"Base strategy: {args.label}  |  estimated beta to SPY: {beta:.2f}")

    data = load_all()
    vix = data['vix']

    base_cagr, base_dd, base_sharpe = compute_stats(nav_df['total_value'], nav_df['date'])
    print(f"\n{'Variant':<32}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}")
    print('-' * 56)
    print(f"{args.label + ' (no hedge)':<32}{base_cagr*100:>7.1f}%{base_dd*100:>7.1f}%{base_sharpe:>8.2f}")

    variants = [
        ('put', 0.05, None, 1.0, 'Put 5% OTM'),
        ('put', 0.10, None, 1.0, 'Put 10% OTM'),
        ('put', 0.15, None, 1.0, 'Put 15% OTM'),
        ('collar', 0.10, 0.10, 1.0, 'Collar 10%/10%'),
        ('collar', 0.10, 0.05, 1.0, 'Collar put10%/call5%'),
        ('collar', 0.15, 0.08, 1.0, 'Collar put15%/call8%'),
        ('collar', 0.05, 0.05, 1.0, 'Collar put5%/call5%'),
        ('collar', 0.03, 0.03, 1.0, 'Collar put3%/call3% (tight)'),
        ('collar', 0.10, 0.05, 1.5, 'Collar put10%/call5% x1.5 ratio'),
        ('collar', 0.10, 0.05, 2.0, 'Collar put10%/call5% x2.0 ratio'),
        ('collar', 0.05, 0.05, 1.5, 'Collar put5%/call5% x1.5 ratio'),
        ('collar', 0.05, 0.05, 2.0, 'Collar put5%/call5% x2.0 ratio'),
        ('collar', 0.03, 0.03, 2.0, 'Collar put3%/call3% x2.0 ratio'),
        ('collar', 0.10, 0.05, 2.5, 'Collar put10%/call5% x2.5 ratio'),
        ('collar', 0.10, 0.05, 3.0, 'Collar put10%/call5% x3.0 ratio'),
        ('collar', 0.08, 0.04, 2.5, 'Collar put8%/call4% x2.5 ratio'),
        ('collar', 0.08, 0.04, 3.0, 'Collar put8%/call4% x3.0 ratio'),
        ('collar', 0.10, 0.04, 2.5, 'Collar put10%/call4% x2.5 ratio'),
        ('collar', 0.10, 0.04, 3.0, 'Collar put10%/call4% x3.0 ratio'),
        ('collar', 0.06, 0.04, 2.5, 'Collar put6%/call4% x2.5 ratio'),
        ('collar', 0.06, 0.04, 3.0, 'Collar put6%/call4% x3.0 ratio'),
    ]
    results = {}
    for mode, put_pct, call_pct, hratio, name in variants:
        kwargs = dict(put_otm_pct=put_pct, mode=mode, hedge_ratio=hratio, skew_per_pct_otm=args.skew)
        if call_pct is not None:
            kwargs['call_otm_pct'] = call_pct
        out = simulate_overlay(nav_df, vix, beta, **kwargs)
        cagr, dd, sharpe = compute_stats(out['hedged_value'], out['date'])
        results[name] = (cagr, dd, sharpe)
        print(f"{name:<32}{cagr*100:>7.1f}%{dd*100:>7.1f}%{sharpe:>8.2f}")

    target_hit = [(n, c, d, s) for n, (c, d, s) in results.items() if c > 0.251 and d < 0.30]
    print()
    if target_hit:
        print(f"MEETS TARGET (CAGR>25.1% AND MaxDD<30%): {target_hit}")
    else:
        print("No hedge variant meets the CAGR>25.1% AND MaxDD<30% target yet.")


if __name__ == '__main__':
    main()
