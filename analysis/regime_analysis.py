"""
Historical Bear Score Analysis
==============================

Visualizes S&P 500 with bear score overlay for the past 20 years.
Shows how the bear score indicator would have performed historically.
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Import regime functions
from regime import (
    yield_curve_score,
    credit_spread_score,
    liquidity_score,
    breadth_score,
    volatility_score,
    valuation_score,
    estimate_bear_magnitude,
    WEIGHTS
)


class HistoricalRegimeAnalyzer:
    """
    Analyzes historical market regimes and bear scores.
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key or os.environ.get('FRED_API_KEY')
        self._cache = {}

    def fetch_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch all required historical data.

        Returns DataFrame with all indicators aligned to weekly frequency.
        """
        import yfinance as yf

        print("Fetching historical data...")

        # S&P 500
        print("  - S&P 500...")
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        sp500_close = sp500['Close'].squeeze()

        # VIX
        print("  - VIX...")
        vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        vix = vix_data['Close'].squeeze()

        # VIX 3-month (may not have full history)
        print("  - VIX 3M...")
        try:
            vix3m_data = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)
            vix_3m = vix3m_data['Close'].squeeze()
            if len(vix_3m) < 100:
                raise ValueError("Insufficient VIX3M data")
        except:
            # Approximate VIX3M as VIX with dampening
            vix_3m = vix.rolling(20).mean() * 0.95
            print("    (using VIX approximation)")

        # Treasury yields from Yahoo
        print("  - Treasury yields...")
        try:
            tnx = yf.download('^TNX', start=start_date, end=end_date, progress=False)
            y10 = tnx['Close'].squeeze()
        except:
            y10 = pd.Series(dtype=float)

        try:
            irx = yf.download('^IRX', start=start_date, end=end_date, progress=False)
            y3m = irx['Close'].squeeze()
        except:
            y3m = pd.Series(dtype=float)

        # FRED data if available
        credit_spread = None
        m2_yoy = None
        real_rate = None

        if self.fred_api_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=self.fred_api_key)

                print("  - Credit spreads (FRED)...")
                hy_oas = fred.get_series('BAMLH0A0HYM2', start_date, end_date)
                ig_oas = fred.get_series('BAMLC0A0CM', start_date, end_date)
                credit_spread = hy_oas - ig_oas

                print("  - M2 money supply (FRED)...")
                m2 = fred.get_series('M2SL', start_date, end_date)
                m2_yoy = m2.pct_change(periods=12) * 100

                print("  - Inflation & Fed Funds (FRED)...")
                fed_funds = fred.get_series('FEDFUNDS', start_date, end_date)
                cpi = fred.get_series('CPIAUCSL', start_date, end_date)
                cpi_yoy = cpi.pct_change(periods=12) * 100
                real_rate = fed_funds - cpi_yoy

            except Exception as e:
                print(f"    Warning: FRED data fetch failed: {e}")

        # Calculate breadth (% above 200 DMA)
        print("  - Market breadth...")
        ma200 = sp500_close.rolling(200).mean()
        pct_above_200dma = (sp500_close > ma200).astype(float) * 100

        # Calculate CAPE percentile approximation
        print("  - Valuation percentile...")
        # Use rolling 10-year percentile of P/E proxy (price/earnings approximation)
        cape_percentile = sp500_close.rolling(252 * 10, min_periods=252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        # Resample all to weekly
        print("  - Aligning data to weekly frequency...")

        df = pd.DataFrame({
            'sp500': sp500_close,
            'vix': vix,
            'vix_3m': vix_3m,
            'y10': y10 if len(y10) > 0 else np.nan,
            'y3m': y3m if len(y3m) > 0 else np.nan,
            'pct_above_200dma': pct_above_200dma,
            'cape_percentile': cape_percentile,
        })

        # Add FRED data if available
        if credit_spread is not None:
            df['credit_spread'] = credit_spread
        if m2_yoy is not None:
            df['m2_yoy'] = m2_yoy
        if real_rate is not None:
            df['real_rate'] = real_rate

        # Resample to weekly and forward fill
        df = df.resample('W-FRI').last()
        df = df.ffill().bfill()

        print(f"  [OK] Loaded {len(df)} weekly data points")

        return df

    def calculate_rolling_bear_score(
        self,
        df: pd.DataFrame,
        lookback: int = 252  # 1 year of daily data for z-scores
    ) -> pd.DataFrame:
        """
        Calculate bear score for each point in time.

        Returns DataFrame with bear_score and component scores.
        """
        print("\nCalculating historical bear scores...")

        results = []
        dates = df.index.tolist()

        for i, date in enumerate(dates):
            if i < 52:  # Need at least 1 year of data
                continue

            if i % 52 == 0:
                print(f"  Processing {date.strftime('%Y-%m-%d')}...")

            # Get historical window
            window = df.iloc[max(0, i - lookback):i + 1]

            try:
                scores = {}

                # Yield curve score
                if 'y10' in window.columns and 'y3m' in window.columns:
                    y10 = window['y10'].dropna()
                    y3m = window['y3m'].dropna()
                    if len(y10) > 30 and len(y3m) > 30:
                        scores['yield_curve'] = yield_curve_score(y10, y3m)
                    else:
                        scores['yield_curve'] = 0.5
                else:
                    scores['yield_curve'] = 0.5

                # Credit spread score
                if 'credit_spread' in window.columns:
                    cs = window['credit_spread'].dropna()
                    if len(cs) > 252:
                        scores['credit'] = credit_spread_score(cs)
                    else:
                        scores['credit'] = 0.5
                else:
                    scores['credit'] = 0.5

                # Liquidity score
                if 'm2_yoy' in window.columns and 'real_rate' in window.columns:
                    m2 = window['m2_yoy'].dropna()
                    rr = window['real_rate'].dropna()
                    if len(m2) > 0 and len(rr) > 0:
                        scores['liquidity'] = liquidity_score(m2, rr)
                    else:
                        scores['liquidity'] = 0.5
                else:
                    scores['liquidity'] = 0.5

                # Breadth score
                pct = window['pct_above_200dma'].dropna()
                if len(pct) > 0:
                    scores['breadth'] = breadth_score(pct)
                else:
                    scores['breadth'] = 0.5

                # Volatility score
                vix = window['vix'].dropna()
                vix_3m = window['vix_3m'].dropna()
                if len(vix) > 0 and len(vix_3m) > 0:
                    scores['volatility'] = volatility_score(vix, vix_3m)
                else:
                    scores['volatility'] = 0.5

                # Valuation score
                cape = window['cape_percentile'].dropna()
                if len(cape) > 0:
                    scores['valuation'] = valuation_score(cape)
                else:
                    scores['valuation'] = 0.5

                # Calculate weighted bear score
                bear_score = sum(scores[k] * WEIGHTS[k] for k in scores) * 100
                magnitude = estimate_bear_magnitude(scores)

                results.append({
                    'date': date,
                    'bear_score': bear_score,
                    'expected_drawdown': magnitude,
                    **{f'score_{k}': v for k, v in scores.items()}
                })

            except Exception as e:
                # Skip problematic dates
                continue

        result_df = pd.DataFrame(results).set_index('date')
        print(f"  [OK] Calculated {len(result_df)} bear scores")

        return result_df

    def identify_corrections(
        self,
        sp500: pd.Series,
        threshold: float = 0.10
    ) -> pd.DataFrame:
        """
        Identify market corrections (drawdowns > threshold).
        """
        # Calculate drawdown
        rolling_max = sp500.expanding().max()
        drawdown = (sp500 - rolling_max) / rolling_max

        # Find correction periods
        corrections = []
        in_correction = False
        start_date = None
        peak_value = None

        for date, dd in drawdown.items():
            if not in_correction and dd < -threshold:
                in_correction = True
                start_date = date
                peak_value = rolling_max[date]
            elif in_correction and dd >= -0.05:  # Recovery
                trough_idx = drawdown[start_date:date].idxmin()
                corrections.append({
                    'start': start_date,
                    'trough': trough_idx,
                    'end': date,
                    'max_drawdown': drawdown[trough_idx] * 100,
                    'peak_value': peak_value,
                    'trough_value': sp500[trough_idx],
                })
                in_correction = False

        return pd.DataFrame(corrections)


def create_regime_visualization(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    corrections: pd.DataFrame,
    save_path: str = 'regime_analysis.png'
):
    """
    Create comprehensive visualization with bear score overlay.
    """
    print("\nCreating visualization...")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Define grid
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)

    # =========================================================================
    # Main chart: S&P 500 with bear score gradient background
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])

    # Align data
    common_dates = scores_df.index.intersection(df.index)
    sp500 = df.loc[common_dates, 'sp500']
    bear_scores = scores_df.loc[common_dates, 'bear_score']

    # Create gradient background based on bear score
    # Green (low risk) to Yellow (moderate) to Red (high risk)
    cmap = LinearSegmentedColormap.from_list(
        'bear_cmap',
        [(0, '#2ecc71'),     # Green - low risk
         (0.3, '#f1c40f'),   # Yellow - caution
         (0.5, '#e67e22'),   # Orange - elevated
         (0.7, '#e74c3c'),   # Red - high risk
         (1.0, '#8e44ad')]   # Purple - extreme
    )

    # Plot gradient background as vertical bars
    dates_num = mdates.date2num(common_dates.to_pydatetime())

    for i in range(len(dates_num) - 1):
        score = bear_scores.iloc[i] / 100  # Normalize to 0-1
        color = cmap(score)
        ax1.axvspan(dates_num[i], dates_num[i + 1], alpha=0.3, color=color, linewidth=0)

    # Plot S&P 500
    ax1.plot(common_dates, sp500, 'k-', linewidth=1.5, label='S&P 500')
    ax1.set_ylabel('S&P 500', fontsize=12)
    ax1.set_yscale('log')

    # Mark major corrections
    for _, corr in corrections.iterrows():
        if corr['max_drawdown'] < -15:  # Mark significant corrections
            ax1.axvspan(corr['start'], corr['end'], alpha=0.2, color='red')
            ax1.annotate(
                f"{corr['max_drawdown']:.0f}%",
                xy=(corr['trough'], corr['trough_value']),
                xytext=(10, -20),
                textcoords='offset points',
                fontsize=8,
                color='red',
                fontweight='bold'
            )

    ax1.set_xlim(common_dates[0], common_dates[-1])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title('S&P 500 with Bear Score Overlay (20-Year Analysis)', fontsize=14, fontweight='bold')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', pad=0.01, aspect=30)
    cbar.set_label('Bear Score', fontsize=10)

    # =========================================================================
    # Bear Score panel
    # =========================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax2.fill_between(common_dates, bear_scores, alpha=0.7, color='navy')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Low Risk (<30)')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Elevated (50)')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='High Risk (>70)')

    ax2.set_ylabel('Bear Score', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.tick_params(labelbottom=False)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # =========================================================================
    # Expected Drawdown panel
    # =========================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    expected_dd = scores_df.loc[common_dates, 'expected_drawdown']
    ax3.fill_between(common_dates, expected_dd, alpha=0.7, color='crimson')
    ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.7)
    ax3.axhline(y=30, color='red', linestyle='--', alpha=0.7)

    ax3.set_ylabel('Expected\nDrawdown %', fontsize=10)
    ax3.set_ylim(10, 50)
    ax3.tick_params(labelbottom=False)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Component scores panel
    # =========================================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # Plot key component scores
    components = ['score_yield_curve', 'score_credit', 'score_volatility', 'score_breadth']
    colors = ['purple', 'blue', 'orange', 'green']
    labels = ['Yield Curve', 'Credit', 'Volatility', 'Breadth']

    for comp, color, label in zip(components, colors, labels):
        if comp in scores_df.columns:
            data = scores_df.loc[common_dates, comp].rolling(4).mean()  # Smooth
            ax4.plot(common_dates, data, color=color, alpha=0.7, linewidth=1, label=label)

    ax4.set_ylabel('Component\nScores', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax4.xaxis.set_major_locator(mdates.YearLocator(2))
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=8, ncol=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved to {save_path}")

    return fig


def create_correction_analysis(
    scores_df: pd.DataFrame,
    corrections: pd.DataFrame,
    save_path: str = 'correction_prediction_analysis.png'
):
    """
    Analyze how well bear score predicted corrections.
    """
    print("\nAnalyzing prediction accuracy...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # =========================================================================
    # 1. Bear score before corrections
    # =========================================================================
    ax1 = axes[0, 0]

    pre_correction_scores = []
    for _, corr in corrections.iterrows():
        if corr['max_drawdown'] < -10:  # Significant corrections only
            # Get bear score 4 weeks before correction started
            pre_date = corr['start'] - timedelta(weeks=4)
            mask = (scores_df.index >= pre_date) & (scores_df.index < corr['start'])
            if mask.any():
                avg_score = scores_df.loc[mask, 'bear_score'].mean()
                pre_correction_scores.append({
                    'date': corr['start'],
                    'pre_score': avg_score,
                    'drawdown': abs(corr['max_drawdown'])
                })

    if pre_correction_scores:
        pre_df = pd.DataFrame(pre_correction_scores)
        ax1.scatter(pre_df['pre_score'], pre_df['drawdown'], s=100, alpha=0.7, c='red')

        # Add trend line
        z = np.polyfit(pre_df['pre_score'], pre_df['drawdown'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(pre_df['pre_score'].min(), pre_df['pre_score'].max(), 100)
        ax1.plot(x_line, p(x_line), 'k--', alpha=0.5)

        ax1.set_xlabel('Bear Score (4 weeks before)', fontsize=11)
        ax1.set_ylabel('Actual Drawdown %', fontsize=11)
        ax1.set_title('Bear Score vs Actual Drawdown', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # =========================================================================
    # 2. Distribution of bear scores
    # =========================================================================
    ax2 = axes[0, 1]

    ax2.hist(scores_df['bear_score'], bins=30, alpha=0.7, color='navy', edgecolor='white')
    ax2.axvline(x=30, color='green', linestyle='--', linewidth=2, label='Low Risk')
    ax2.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='Elevated')
    ax2.axvline(x=70, color='red', linestyle='--', linewidth=2, label='High Risk')

    ax2.set_xlabel('Bear Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Bear Scores', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # 3. Component score correlations
    # =========================================================================
    ax3 = axes[1, 0]

    components = ['score_yield_curve', 'score_credit', 'score_liquidity',
                  'score_breadth', 'score_volatility', 'score_valuation']
    available = [c for c in components if c in scores_df.columns]

    if len(available) > 1:
        corr_matrix = scores_df[available].corr()
        im = ax3.imshow(corr_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(available)))
        ax3.set_yticks(range(len(available)))
        labels = [c.replace('score_', '').title() for c in available]
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_yticklabels(labels)
        ax3.set_title('Component Score Correlations', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax3)

    # =========================================================================
    # 4. Time spent in each regime
    # =========================================================================
    ax4 = axes[1, 1]

    regimes = [
        ('Low Risk (0-30)', (scores_df['bear_score'] < 30).mean() * 100, 'green'),
        ('Caution (30-50)', ((scores_df['bear_score'] >= 30) & (scores_df['bear_score'] < 50)).mean() * 100, 'yellow'),
        ('Elevated (50-70)', ((scores_df['bear_score'] >= 50) & (scores_df['bear_score'] < 70)).mean() * 100, 'orange'),
        ('High Risk (>70)', (scores_df['bear_score'] >= 70).mean() * 100, 'red'),
    ]

    labels, values, colors = zip(*regimes)
    bars = ax4.bar(labels, values, color=colors, edgecolor='black', alpha=0.7)
    ax4.set_ylabel('% of Time', fontsize=11)
    ax4.set_title('Time Spent in Each Regime', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved to {save_path}")

    return fig


def main():
    """Run the full historical regime analysis."""
    print("=" * 70)
    print("HISTORICAL BEAR SCORE ANALYSIS")
    print("=" * 70)

    # Check for FRED API key
    fred_key = os.environ.get('FRED_API_KEY')
    if not fred_key:
        print("\n[!] FRED_API_KEY not set. Credit spreads and liquidity will use defaults.")
        print("   For full analysis, register free at:")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html\n")

    # Initialize analyzer
    analyzer = HistoricalRegimeAnalyzer(fred_api_key=fred_key)

    # Date range: 20 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 20)

    print(f"\nAnalysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Fetch data
    df = analyzer.fetch_historical_data(start_date, end_date)

    # Calculate bear scores
    scores_df = analyzer.calculate_rolling_bear_score(df)

    # Identify corrections
    corrections = analyzer.identify_corrections(df['sp500'])
    print(f"\nIdentified {len(corrections)} corrections > 10%")

    # Create visualizations
    create_regime_visualization(df, scores_df, corrections, 'regime_analysis.png')
    create_correction_analysis(scores_df, corrections, 'correction_prediction_analysis.png')

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nBear Score Statistics:")
    print(f"  Mean:   {scores_df['bear_score'].mean():.1f}")
    print(f"  Median: {scores_df['bear_score'].median():.1f}")
    print(f"  Std:    {scores_df['bear_score'].std():.1f}")
    print(f"  Min:    {scores_df['bear_score'].min():.1f}")
    print(f"  Max:    {scores_df['bear_score'].max():.1f}")

    print(f"\nCurrent Bear Score: {scores_df['bear_score'].iloc[-1]:.1f}")
    print(f"Current Expected Drawdown: {scores_df['expected_drawdown'].iloc[-1]:.1f}%")

    # Show major corrections and pre-correction scores
    print(f"\nMajor Corrections (>15% drawdown):")
    for _, corr in corrections.iterrows():
        if corr['max_drawdown'] < -15:
            pre_date = corr['start'] - timedelta(weeks=4)
            mask = (scores_df.index >= pre_date) & (scores_df.index < corr['start'])
            if mask.any():
                pre_score = scores_df.loc[mask, 'bear_score'].mean()
                print(f"  {corr['start'].strftime('%Y-%m')}: {corr['max_drawdown']:.1f}% drawdown, "
                      f"pre-correction bear score: {pre_score:.1f}")

    print("\n[OK] Analysis complete. Check regime_analysis.png and correction_prediction_analysis.png")


if __name__ == "__main__":
    main()
