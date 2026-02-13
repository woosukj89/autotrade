"""Analyze why our model missed Apple."""

import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('fundamentals.sqlite')

print("=" * 70)
print("APPLE (AAPL) FUNDAMENTAL ANALYSIS")
print("=" * 70)

# Check if AAPL is in the database
df = pd.read_sql_query("SELECT DISTINCT ticker FROM fundamentals WHERE ticker = 'AAPL'", conn)
print(f"\nAAPL in database: {len(df) > 0}")

# Get AAPL fundamentals for key years
print("\n" + "=" * 70)
print("AAPL FUNDAMENTALS BY YEAR (Raw Data)")
print("=" * 70)

df = pd.read_sql_query('''
    SELECT fy, field, value
    FROM fundamentals
    WHERE ticker = 'AAPL' AND fy BETWEEN 2012 AND 2018
    ORDER BY fy, field
''', conn)

# Pivot to see by year
pivot = df.pivot_table(index='fy', columns='field', values='value', aggfunc='first')
print(pivot.to_string())

# Calculate key metrics that our model uses
print("\n" + "=" * 70)
print("CALCULATED METRICS (What Our Model Sees)")
print("=" * 70)

metrics_by_year = []
for fy in sorted(df['fy'].unique()):
    year_data = df[df['fy'] == fy].set_index('field')['value'].to_dict()

    cfo = year_data.get('CashFromOperations', 0)
    capex = abs(year_data.get('CapitalExpenditures', 0))
    dep = year_data.get('Depreciation', 0)
    op_inc = year_data.get('OperatingIncome', 0)
    net_inc = year_data.get('NetIncome', 0)
    equity = year_data.get('Equity', 0)
    debt = year_data.get('TotalDebt', 0)
    cash = year_data.get('CashAndEquivalents', 0)
    revenue = year_data.get('Revenue', 0)

    # Owner earnings
    maint_capex = min(capex, dep * 0.8) if dep else capex * 0.5
    owner_earnings = cfo - maint_capex

    # ROIC
    invested_capital = equity + debt - cash if (equity + debt - cash) > 0 else equity
    roic = (op_inc * 0.75) / invested_capital if invested_capital > 0 else 0

    # ROE
    roe = net_inc / equity if equity > 0 else 0

    # Operating margin
    op_margin = op_inc / revenue if revenue > 0 else 0

    metrics_by_year.append({
        'FY': fy,
        'Revenue ($B)': revenue / 1e9,
        'Net Income ($B)': net_inc / 1e9,
        'Owner Earnings ($B)': owner_earnings / 1e9,
        'ROIC': roic,
        'ROE': roe,
        'Op Margin': op_margin,
        'Equity ($B)': equity / 1e9,
    })

metrics_df = pd.DataFrame(metrics_by_year)
print(metrics_df.to_string(index=False))

# Calculate growth rates
print("\n" + "=" * 70)
print("GROWTH METRICS")
print("=" * 70)

if len(metrics_df) >= 3:
    revenues = metrics_df['Revenue ($B)'].values
    earnings = metrics_df['Net Income ($B)'].values

    # CAGR calculation
    years = len(revenues) - 1
    if revenues[0] > 0 and revenues[-1] > 0:
        rev_cagr = (revenues[-1] / revenues[0]) ** (1/years) - 1
        print(f"Revenue CAGR ({int(metrics_df['FY'].iloc[0])}-{int(metrics_df['FY'].iloc[-1])}): {rev_cagr*100:.1f}%")

    if earnings[0] > 0 and earnings[-1] > 0:
        earn_cagr = (earnings[-1] / earnings[0]) ** (1/years) - 1
        print(f"Earnings CAGR ({int(metrics_df['FY'].iloc[0])}-{int(metrics_df['FY'].iloc[-1])}): {earn_cagr*100:.1f}%")

    avg_roic = metrics_df['ROIC'].mean()
    print(f"Average ROIC: {avg_roic*100:.1f}%")

    roic_above_15 = (metrics_df['ROIC'] > 0.15).mean()
    print(f"% Years ROIC > 15%: {roic_above_15*100:.0f}%")

# Calculate moat score as our model does
print("\n" + "=" * 70)
print("MOAT SCORE CALCULATION (Our Model)")
print("=" * 70)

roic_persistence = (metrics_df['ROIC'] > 0.15).mean()
margin_stability = 1.0 / (1.0 + metrics_df['Op Margin'].std())
capex_data = df[df['field'] == 'CapitalExpenditures']['value'].abs()
revenue_data = df[df['field'] == 'Revenue']['value']
if len(capex_data) > 0 and len(revenue_data) > 0:
    capex_intensity = (capex_data.values / revenue_data.values).mean()
    capex_light = max(0, 1 - capex_intensity)
else:
    capex_light = 0.5

moat_score = 0.5 * roic_persistence + 0.3 * margin_stability + 0.2 * capex_light

print(f"ROIC Persistence (>15%): {roic_persistence:.2f}")
print(f"Margin Stability: {margin_stability:.2f}")
print(f"Capex Light: {capex_light:.2f}")
print(f"MOAT SCORE: {moat_score:.2f}")

# Get historical prices for 2016
print("\n" + "=" * 70)
print("AAPL PRICE DATA (2016 - When Buffett Bought)")
print("=" * 70)

import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

aapl = yf.download('AAPL', start='2016-01-01', end='2016-12-31', progress=False)
if not aapl.empty:
    print(f"Price Range 2016: ${aapl['Close'].min():.2f} - ${aapl['Close'].max():.2f}")
    avg_price_2016 = aapl['Close'].mean()
    print(f"Average Price 2016: ${avg_price_2016:.2f}")

    # Get shares outstanding (current, as approximation)
    info = yf.Ticker('AAPL').info
    shares = info.get('sharesOutstanding', 16e9)  # ~16B shares

    # Calculate what our model would have computed
    # Using 2015 data for 2016 valuation
    fy2015 = metrics_df[metrics_df['FY'] == 2015].iloc[0] if 2015 in metrics_df['FY'].values else None
    if fy2015 is not None:
        owner_earnings_2015 = fy2015['Owner Earnings ($B)'] * 1e9

        # Our DCF calculation
        growth = min(avg_roic * 0.5, 0.05)  # Our old cap at 5%
        discount_rate = 0.09
        cap_years = 10 if moat_score > 0.4 else 5

        pv = 0
        for t in range(1, cap_years + 1):
            pv += owner_earnings_2015 * ((1 + growth) ** t) / ((1 + discount_rate) ** t)
        terminal = owner_earnings_2015 * (1 + growth) ** cap_years * 1.02 / (discount_rate - 0.02)
        pv += terminal / ((1 + discount_rate) ** cap_years)

        intrinsic_per_share_old = pv / shares

        print(f"\n--- Our OLD Model (5% growth cap) ---")
        print(f"Growth rate used: {growth*100:.1f}%")
        print(f"Intrinsic Value: ${intrinsic_per_share_old:.2f}")
        print(f"Market Price: ${avg_price_2016:.2f}")
        margin_of_safety_old = (intrinsic_per_share_old - avg_price_2016) / intrinsic_per_share_old
        print(f"Margin of Safety: {margin_of_safety_old*100:.1f}%")
        print(f"Would we buy (MoS > 20%)? {'YES' if margin_of_safety_old > 0.20 else 'NO'}")

        # New calculation with higher growth
        growth_new = min(avg_roic * 0.6, 0.12)  # Allow up to 12%
        discount_rate_new = 0.08  # Lower for quality
        cap_years_new = 15  # Longer for high moat

        pv_new = 0
        for t in range(1, cap_years_new + 1):
            pv_new += owner_earnings_2015 * ((1 + growth_new) ** t) / ((1 + discount_rate_new) ** t)
        terminal_new = owner_earnings_2015 * (1 + growth_new) ** cap_years_new * 1.025 / (discount_rate_new - 0.025)
        pv_new += terminal_new / ((1 + discount_rate_new) ** cap_years_new)

        intrinsic_per_share_new = pv_new / shares

        print(f"\n--- IMPROVED Model (12% growth cap) ---")
        print(f"Growth rate used: {growth_new*100:.1f}%")
        print(f"Intrinsic Value: ${intrinsic_per_share_new:.2f}")
        print(f"Market Price: ${avg_price_2016:.2f}")
        margin_of_safety_new = (intrinsic_per_share_new - avg_price_2016) / intrinsic_per_share_new
        print(f"Margin of Safety: {margin_of_safety_new*100:.1f}%")
        print(f"Would we buy (MoS > 10%)? {'YES' if margin_of_safety_new > 0.10 else 'NO'}")

conn.close()

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
