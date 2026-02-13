"""
Test script for Bear Beta and Regime-Adaptive strategies.

This script:
1. Analyzes bear betas for various tickers
2. Shows current regime assessment
3. Demonstrates the allocation logic
"""

import sys
import os

# Add current and parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def test_bear_beta_analysis():
    """Test bear beta calculation."""
    print("=" * 70)
    print("TEST 1: BEAR BETA ANALYSIS")
    print("=" * 70)

    try:
        from strategies.bear_beta_strategy import analyze_bear_betas
    except ImportError:
        from bear_beta_strategy import analyze_bear_betas

    # Analyze a diverse set of tickers
    tickers = [
        # Benchmarks
        'SPY', 'QQQ', 'IWM',
        # High beta tech
        'NVDA', 'AMD', 'TSLA', 'MRVL',
        # Defensive staples
        'PG', 'KO', 'PEP', 'WMT', 'JNJ',
        # Utilities
        'NEE', 'DUK', 'SO',
        # Healthcare
        'UNH', 'LLY', 'PFE',
        # Gold (potential negative bear beta)
        'GLD', 'NEM', 'GOLD',
        # Bonds
        'TLT', 'SHY', 'AGG',
    ]

    print(f"\nAnalyzing {len(tickers)} tickers for bear beta...")
    print("(Bear beta = correlation with market during down days)\n")

    df = analyze_bear_betas(tickers)

    if len(df) == 0:
        print("No data returned - check network connection")
        return

    print("\nBEAR BETA RANKINGS (sorted by bear beta, low = defensive):")
    print("-" * 90)
    print(f"{'Ticker':8s} {'Bear Beta':>10s} {'Bull Beta':>10s} {'Total Beta':>10s} "
          f"{'Down Cap':>10s} {'Up Cap':>10s} {'Return':>10s}")
    print("-" * 90)

    for _, row in df.iterrows():
        print(f"{row['ticker']:8s} "
              f"{row['bear_beta']:10.2f} "
              f"{row['bull_beta']:10.2f} " if pd.notna(row['bull_beta']) else f"{'N/A':>10s} "
              f"{row['total_beta']:10.2f} "
              f"{row['down_capture']*100:9.0f}% "
              f"{row['up_capture']*100:9.0f}% " if pd.notna(row['up_capture']) else f"{'N/A':>10s} "
              f"{row['total_return']*100:9.1f}%")

    print("-" * 90)

    # Categorize
    print("\n\nCATEGORIES:")
    print("-" * 50)

    negative = df[df['bear_beta'] < 0]
    if len(negative) > 0:
        print(f"\nNegative Bear Beta (true hedges, move UP when market DOWN):")
        for _, row in negative.iterrows():
            print(f"  {row['ticker']:8s} beta={row['bear_beta']:.2f}")

    defensive = df[(df['bear_beta'] >= 0) & (df['bear_beta'] < 0.5)]
    if len(defensive) > 0:
        print(f"\nLow Bear Beta < 0.5 (defensive, limited downside):")
        for _, row in defensive.iterrows():
            print(f"  {row['ticker']:8s} beta={row['bear_beta']:.2f}")

    moderate = df[(df['bear_beta'] >= 0.5) & (df['bear_beta'] < 1.0)]
    if len(moderate) > 0:
        print(f"\nModerate Bear Beta 0.5-1.0 (inline with market):")
        for _, row in moderate.iterrows():
            print(f"  {row['ticker']:8s} beta={row['bear_beta']:.2f}")

    aggressive = df[df['bear_beta'] >= 1.0]
    if len(aggressive) > 0:
        print(f"\nHigh Bear Beta >= 1.0 (amplifies losses):")
        for _, row in aggressive.iterrows():
            print(f"  {row['ticker']:8s} beta={row['bear_beta']:.2f}")

    return df


def test_regime_assessment():
    """Test regime indicator."""
    print("\n\n" + "=" * 70)
    print("TEST 2: REGIME ASSESSMENT")
    print("=" * 70)

    try:
        from strategies.regime_adaptive_strategy import demo_regime_allocation
    except ImportError:
        from regime_adaptive_strategy import demo_regime_allocation

    demo_regime_allocation()


def test_allocation_scenarios():
    """Test allocation logic for different bear scores."""
    print("\n\n" + "=" * 70)
    print("TEST 3: ALLOCATION SCENARIOS")
    print("=" * 70)

    try:
        from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
    except ImportError:
        from regime_adaptive_strategy import RegimeAdaptiveStrategy

    strategy = RegimeAdaptiveStrategy()

    print("\nAllocation by Bear Score:")
    print("-" * 60)
    print(f"{'Bear Score':^15s} {'High Beta':^15s} {'Bear Beta':^15s} {'Regime':^15s}")
    print("-" * 60)

    test_scores = [20, 35, 45, 52, 58, 67, 75, 85]
    regimes = ['VERY LOW', 'LOW', 'WATCH', 'CAUTION', 'ELEVATED', 'HIGH', 'VERY HIGH', 'EXTREME']

    for i, score in enumerate(test_scores):
        hb, bb = strategy._get_allocation_weights(score)
        regime = regimes[i] if i < len(regimes) else 'EXTREME'
        print(f"{score:^15.0f} {hb*100:^14.0f}% {bb*100:^14.0f}% {regime:^15s}")

    print("-" * 60)

    print("""
Interpretation:
- Bear Score < 40:  Bull market - maximum aggression
- Bear Score 40-55: Warning signs - start adding defense
- Bear Score 55-70: High risk - shift to mostly defensive
- Bear Score > 70:  Correction likely - maximum defense
""")


def test_combined_portfolio_example():
    """Show an example combined portfolio."""
    print("\n\n" + "=" * 70)
    print("TEST 4: EXAMPLE COMBINED PORTFOLIO")
    print("=" * 70)

    print("""
Example portfolio at Bear Score = 55 (CAUTION level):
Allocation: 60% High Beta + 40% Defensive

HIGH BETA COMPONENT (60% of portfolio):
----------------------------------------
  NVDA    12%   Beta=1.8   Score=92   Tech leader, high growth
  MSFT    10%   Beta=1.1   Score=85   Quality + growth
  AMD     10%   Beta=1.9   Score=80   Semiconductor momentum
  CRWD     8%   Beta=1.4   Score=78   Cybersecurity growth
  ANET     8%   Beta=1.3   Score=75   Networking infrastructure
  AVGO     6%   Beta=1.2   Score=72   Semiconductor quality
  META     6%   Beta=1.3   Score=70   Digital advertising

DEFENSIVE COMPONENT (40% of portfolio):
----------------------------------------
  GLD      8%   Bear Beta=-0.15   Gold hedge
  JNJ      6%   Bear Beta=0.35    Healthcare staple
  PG       6%   Bear Beta=0.40    Consumer defensive
  KO       5%   Bear Beta=0.42    Consumer staple
  NEE      5%   Bear Beta=0.48    Utility leader
  PEP      5%   Bear Beta=0.45    Consumer staple
  WMT      5%   Bear Beta=0.52    Retail defensive

PORTFOLIO CHARACTERISTICS:
--------------------------
  Weighted Beta:           0.95 (reduced from 1.4 pure high-beta)
  Weighted Bear Beta:      0.62 (protected in crashes)
  Expected Down Capture:   75% (vs 110%+ for pure high-beta)
  Expected Up Capture:     92% (still captures most upside)
""")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BEAR BETA & REGIME-ADAPTIVE STRATEGY TESTS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    try:
        # Test 1: Bear Beta Analysis
        df = test_bear_beta_analysis()

        # Test 2: Regime Assessment
        test_regime_assessment()

        # Test 3: Allocation Scenarios
        test_allocation_scenarios()

        # Test 4: Example Portfolio
        test_combined_portfolio_example()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
