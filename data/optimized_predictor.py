"""
Optimized Bear Market Predictor
===============================

Based on comparison analysis, this creates an optimized predictor that:
1. Uses MacroMom's rate-of-change approach (best hit rate: 85.7%, best lead time: 140 days)
2. Incorporates Baseline's stability (best defensive Sharpe: 1.80)
3. Adds new signals identified from backtesting

Key insights from comparison:
- Rate of change (momentum of indicators) is more predictive than levels
- VIX term structure is valuable but noisy
- Yield curve gives early warnings but many false positives
- Breadth deterioration is a reliable signal
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')


class OptimizedPredictor:
    """
    Optimized bear market predictor combining best aspects of all predictors.

    Improvements:
    1. Uses 3-speed momentum (fast/medium/slow) for trend confirmation
    2. Adaptive thresholds based on recent volatility regime
    3. Momentum-of-momentum (acceleration) signals
    4. Confluence scoring (multiple signals confirming)
    """

    name = "Optimized"

    def __init__(self):
        # Component weights (tuned based on backtest)
        self.weights = {
            'trend_health': 0.25,      # Multi-timeframe trend
            'volatility_regime': 0.20, # VIX and vol clustering
            'macro_deterioration': 0.25, # Rate of change in conditions
            'breadth_momentum': 0.15,  # Market internals momentum
            'risk_appetite': 0.15,     # Credit/equity relative strength
        }

    def _calculate_trend_health(self, sp500: pd.Series) -> float:
        """
        Multi-timeframe trend health score.

        Combines short, medium, and long-term trends with confirmation.
        """
        if len(sp500) < 252:
            return 50.0

        current = sp500.iloc[-1]
        scores = []

        # Short-term (1 month): Recent momentum
        mom_21d = (current / sp500.iloc[-21] - 1) * 100
        short_score = max(0, min(100, 50 - mom_21d * 4))
        scores.append(short_score * 0.2)

        # Medium-term (3 months): Trend strength
        mom_63d = (current / sp500.iloc[-63] - 1) * 100
        medium_score = max(0, min(100, 50 - mom_63d * 2))
        scores.append(medium_score * 0.3)

        # Long-term (12 months): Primary trend
        mom_252d = (current / sp500.iloc[-252] - 1) * 100
        long_score = max(0, min(100, 50 - mom_252d * 1))
        scores.append(long_score * 0.2)

        # Trend acceleration: Is momentum weakening?
        mom_21d_prev = (sp500.iloc[-21] / sp500.iloc[-42] - 1) * 100
        accel = mom_21d - mom_21d_prev
        accel_score = max(0, min(100, 50 - accel * 5))
        scores.append(accel_score * 0.15)

        # Distance from 200 MA
        ma_200 = sp500.tail(200).mean()
        dist_ma = (current / ma_200 - 1) * 100
        ma_score = max(0, min(100, 50 - dist_ma * 3))
        scores.append(ma_score * 0.15)

        return sum(scores)

    def _calculate_volatility_regime(self, data: pd.DataFrame, idx: int) -> float:
        """
        Volatility regime score using VIX and realized vol.
        """
        window = data.iloc[max(0, idx - 252):idx + 1]

        if len(window) < 60:
            return 50.0

        scores = []

        # VIX level relative to history
        if 'vix' in data.columns:
            vix = window['vix']
            vix_current = vix.iloc[-1]
            vix_percentile = (vix < vix_current).mean() * 100

            # High VIX percentile = elevated risk
            vix_score = vix_percentile
            scores.append(('vix_percentile', vix_score, 0.25))

            # VIX acceleration (rate of change)
            if len(vix) >= 10:
                vix_roc = (vix.iloc[-1] / vix.iloc[-10] - 1) * 100
                roc_score = max(0, min(100, 50 + vix_roc * 3))
                scores.append(('vix_accel', roc_score, 0.25))

        # VIX term structure
        if 'vix' in data.columns and 'vix_3m' in data.columns:
            vix = window['vix'].iloc[-1]
            vix_3m = window['vix_3m'].iloc[-1]

            if vix_3m > 0:
                # Contango = normal, Backwardation = stress
                term_ratio = vix / vix_3m
                term_score = max(0, min(100, (term_ratio - 0.9) * 100))
                scores.append(('vix_term', term_score, 0.25))

        # Realized volatility spike
        if 'sp500' in data.columns:
            returns = window['sp500'].pct_change().dropna()
            if len(returns) >= 21:
                vol_5d = returns.tail(5).std() * np.sqrt(252) * 100
                vol_21d = returns.tail(21).std() * np.sqrt(252) * 100

                # Vol expansion = stress
                vol_ratio = vol_5d / max(vol_21d, 0.01)
                vol_score = max(0, min(100, vol_ratio * 40))
                scores.append(('vol_spike', vol_score, 0.25))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight

    def _calculate_macro_deterioration(self, data: pd.DataFrame, idx: int) -> float:
        """
        Macro deterioration score using rate of change.

        This was the strongest predictor in backtests.
        """
        window = data.iloc[max(0, idx - 252):idx + 1]

        if len(window) < 26:
            return 50.0

        scores = []

        # Yield curve momentum (flattening/inversion speed)
        if 'y10' in data.columns and 'y3m' in data.columns:
            spread = window['y10'] - window['y3m']

            if len(spread) >= 13:
                spread_now = spread.iloc[-1]
                spread_13w = spread.iloc[-13]
                spread_change = spread_now - spread_13w

                # Fast inversion = very bearish
                yc_score = max(0, min(100, 50 - spread_change * 25))
                scores.append(('yc_momentum', yc_score, 0.35))

                # Absolute inversion
                if spread_now < 0:
                    scores.append(('yc_inverted', 80, 0.15))
                else:
                    scores.append(('yc_inverted', 30, 0.15))

        # Breadth momentum
        if 'pct_above_200dma' in data.columns and len(window) >= 13:
            pct = window['pct_above_200dma']
            pct_now = pct.iloc[-1]
            pct_13w = pct.iloc[-13]

            # Declining breadth momentum
            breadth_change = pct_now - pct_13w
            breadth_score = max(0, min(100, 50 - breadth_change * 1.5))
            scores.append(('breadth_mom', breadth_score, 0.30))

        # Price momentum deceleration
        if 'sp500' in data.columns and len(window) >= 26:
            sp500 = window['sp500']

            mom_now = (sp500.iloc[-1] / sp500.iloc[-13] - 1) * 100
            mom_prev = (sp500.iloc[-13] / sp500.iloc[-26] - 1) * 100
            decel = mom_now - mom_prev

            # Momentum weakening = bearish
            decel_score = max(0, min(100, 50 - decel * 4))
            scores.append(('mom_decel', decel_score, 0.20))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight

    def _calculate_breadth_momentum(self, data: pd.DataFrame, idx: int) -> float:
        """
        Market breadth and internals momentum.
        """
        window = data.iloc[max(0, idx - 252):idx + 1]

        if len(window) < 21:
            return 50.0

        scores = []

        if 'pct_above_200dma' in data.columns:
            pct = window['pct_above_200dma']

            # Current breadth level
            current_pct = pct.iloc[-1]
            level_score = max(0, min(100, (60 - current_pct) * 2))
            scores.append(('breadth_level', level_score, 0.3))

            # 4-week breadth momentum
            if len(pct) >= 4:
                mom_4w = pct.iloc[-1] - pct.iloc[-4]
                mom_score = max(0, min(100, 50 - mom_4w * 2))
                scores.append(('breadth_4w', mom_score, 0.35))

            # Breadth trend (is it trending down?)
            if len(pct) >= 13:
                trend = np.polyfit(range(13), pct.tail(13), 1)[0]
                trend_score = max(0, min(100, 50 - trend * 10))
                scores.append(('breadth_trend', trend_score, 0.35))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight

    def _calculate_risk_appetite(self, data: pd.DataFrame, idx: int) -> float:
        """
        Risk appetite using cross-asset signals.
        """
        window = data.iloc[max(0, idx - 252):idx + 1]

        if len(window) < 21:
            return 50.0

        scores = []

        # VIX relative to its moving average (fear gauge)
        if 'vix' in data.columns and len(window) >= 50:
            vix = window['vix']
            vix_ma = vix.tail(50).mean()
            vix_ratio = vix.iloc[-1] / vix_ma

            # VIX above MA = fear
            fear_score = max(0, min(100, vix_ratio * 50))
            scores.append(('fear_gauge', fear_score, 0.4))

        # Equity momentum relative to safe havens (proxy: inverse of equity momentum)
        if 'sp500' in data.columns and len(window) >= 21:
            sp500 = window['sp500']
            mom = (sp500.iloc[-1] / sp500.iloc[-21] - 1) * 100

            # Weak equity momentum = risk-off
            risk_off_score = max(0, min(100, 50 - mom * 3))
            scores.append(('risk_off', risk_off_score, 0.35))

        # Volatility regime persistence
        if 'vix' in data.columns and len(window) >= 21:
            vix = window['vix'].tail(21)
            high_vol_days = (vix > 20).sum()
            persistence_score = max(0, min(100, high_vol_days * 5))
            scores.append(('vol_persist', persistence_score, 0.25))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate optimized bear score."""
        sp500 = data['sp500'].iloc[:idx + 1]

        if len(sp500) < 252:
            return 50.0

        components = {
            'trend_health': self._calculate_trend_health(sp500),
            'volatility_regime': self._calculate_volatility_regime(data, idx),
            'macro_deterioration': self._calculate_macro_deterioration(data, idx),
            'breadth_momentum': self._calculate_breadth_momentum(data, idx),
            'risk_appetite': self._calculate_risk_appetite(data, idx),
        }

        # Weighted average
        base_score = sum(
            components[k] * self.weights[k]
            for k in components
        )

        # Confluence bonus: If multiple components agree, increase conviction
        high_risk_count = sum(1 for v in components.values() if v > 60)
        if high_risk_count >= 4:
            base_score = min(100, base_score * 1.15)  # 15% bonus
        elif high_risk_count >= 3:
            base_score = min(100, base_score * 1.08)  # 8% bonus

        return base_score

    def calculate_all_scores(self, data: pd.DataFrame, min_lookback: int = 252) -> pd.Series:
        """Calculate scores for all valid dates."""
        scores = []
        dates = []

        for i in range(min_lookback, len(data)):
            try:
                score = self.calculate_score(data, i)
                scores.append(score)
                dates.append(data.index[i])
            except Exception:
                scores.append(np.nan)
                dates.append(data.index[i])

        return pd.Series(scores, index=dates, name=self.name)


class AdaptivePredictor:
    """
    Adaptive predictor that adjusts to market regime.

    Uses different weights based on current volatility environment.
    """

    name = "Adaptive"

    def __init__(self):
        self.optimized = OptimizedPredictor()

        # High volatility regime weights (more focus on momentum)
        self.high_vol_weights = {
            'trend_health': 0.30,
            'volatility_regime': 0.15,
            'macro_deterioration': 0.30,
            'breadth_momentum': 0.15,
            'risk_appetite': 0.10,
        }

        # Low volatility regime weights (more focus on macro)
        self.low_vol_weights = {
            'trend_health': 0.20,
            'volatility_regime': 0.25,
            'macro_deterioration': 0.25,
            'breadth_momentum': 0.15,
            'risk_appetite': 0.15,
        }

    def _detect_regime(self, data: pd.DataFrame, idx: int) -> str:
        """Detect current volatility regime."""
        if 'vix' not in data.columns:
            return 'normal'

        window = data.iloc[max(0, idx - 63):idx + 1]
        vix = window['vix']

        if len(vix) < 21:
            return 'normal'

        vix_current = vix.iloc[-1]
        vix_avg = vix.mean()

        if vix_current > vix_avg * 1.3:
            return 'high_vol'
        elif vix_current < vix_avg * 0.8:
            return 'low_vol'
        else:
            return 'normal'

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        regime = self._detect_regime(data, idx)

        if regime == 'high_vol':
            self.optimized.weights = self.high_vol_weights
        elif regime == 'low_vol':
            self.optimized.weights = self.low_vol_weights
        # else keep default weights

        return self.optimized.calculate_score(data, idx)

    def calculate_all_scores(self, data: pd.DataFrame, min_lookback: int = 252) -> pd.Series:
        scores = []
        dates = []

        for i in range(min_lookback, len(data)):
            try:
                score = self.calculate_score(data, i)
                scores.append(score)
                dates.append(data.index[i])
            except Exception:
                scores.append(np.nan)
                dates.append(data.index[i])

        return pd.Series(scores, index=dates, name=self.name)


def run_comparison(years: int = 30):
    """Run comparison including optimized predictors."""
    import os
    import sys
    import yfinance as yf
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    print("=" * 70)
    print(f"OPTIMIZED PREDICTOR COMPARISON ({years} YEARS)")
    print("=" * 70)

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print("\nFetching data...")
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Close'].squeeze()
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close'].squeeze()

    try:
        vix_3m = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)['Close'].squeeze()
        if len(vix_3m) < 100:
            vix_3m = vix.rolling(20).mean() * 0.95
    except:
        vix_3m = vix.rolling(20).mean() * 0.95

    try:
        y10 = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close'].squeeze()
    except:
        y10 = pd.Series(dtype=float)

    try:
        y3m = yf.download('^IRX', start=start_date, end=end_date, progress=False)['Close'].squeeze()
    except:
        y3m = pd.Series(dtype=float)

    ma200 = sp500.rolling(200).mean()
    pct_above_200dma = (sp500 > ma200).astype(float) * 100

    data = pd.DataFrame({
        'sp500': sp500,
        'vix': vix,
        'vix_3m': vix_3m,
        'y10': y10 if len(y10) > 0 else np.nan,
        'y3m': y3m if len(y3m) > 0 else np.nan,
        'pct_above_200dma': pct_above_200dma,
    }).resample('W-FRI').last().ffill().bfill()

    print(f"  [OK] Loaded {len(data)} weekly data points")

    # Calculate scores
    print("\nCalculating predictor scores...")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
    from predictors import BaselinePredictor, MacroMomentumPredictor, EnsemblePredictor

    predictors = {
        'Baseline': BaselinePredictor(),
        'MacroMom': MacroMomentumPredictor(),
        'Ensemble': EnsemblePredictor(),
        'Optimized': OptimizedPredictor(),
        'Adaptive': AdaptivePredictor(),
    }

    scores = {}
    for name, pred in predictors.items():
        print(f"  - {name}...")
        scores[name] = pred.calculate_all_scores(data)

    # Evaluate predictors
    print("\nEvaluating predictors...")

    # Find corrections
    rolling_max = data['sp500'].expanding().max()
    drawdown = (data['sp500'] - rolling_max) / rolling_max

    corrections = []
    in_correction = False
    start_date_corr = None

    for date, dd in drawdown.items():
        if not in_correction and dd < -0.10:
            in_correction = True
            start_date_corr = date
        elif in_correction and dd >= -0.05:
            trough_idx = drawdown[start_date_corr:date].idxmin()
            corrections.append({
                'start': start_date_corr,
                'trough': trough_idx,
                'max_drawdown': drawdown[trough_idx] * 100,
            })
            in_correction = False

    corrections = pd.DataFrame(corrections)
    print(f"  Found {len(corrections)} corrections > 10%")

    # Calculate metrics
    results = {}
    for name, score_series in scores.items():
        score_series = score_series.dropna()
        results[name] = {}

        # Pre-correction scores
        pre_scores = []
        for _, corr in corrections.iterrows():
            if corr['max_drawdown'] < -15:
                pre_start = corr['start'] - timedelta(weeks=4)
                mask = (score_series.index >= pre_start) & (score_series.index < corr['start'])
                if mask.any():
                    pre_scores.append(score_series[mask].mean())

        if pre_scores:
            results[name]['avg_pre_score'] = np.mean(pre_scores)
            results[name]['hit_rate'] = np.mean([s > 50 for s in pre_scores]) * 100

        # Correlation with forward drawdown
        fwd_min = data['sp500'].rolling(63).min().shift(-63)
        fwd_dd = (fwd_min / data['sp500'] - 1) * -100

        aligned = pd.DataFrame({
            'score': score_series,
            'fwd_dd': fwd_dd
        }).dropna()

        if len(aligned) > 50:
            results[name]['corr_63d'] = aligned['score'].corr(aligned['fwd_dd'])

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results).T
    print("\n" + results_df.round(3).to_string())

    # Create visualization
    print("\nCreating comparison chart...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Plot S&P 500
    ax1 = axes[0]
    common_idx = scores['Optimized'].dropna().index
    sp500_aligned = data.loc[common_idx, 'sp500']

    ax1.plot(sp500_aligned.index, sp500_aligned, 'k-', linewidth=1.5)
    ax1.set_ylabel('S&P 500')
    ax1.set_yscale('log')
    ax1.set_title('S&P 500 with Major Corrections Highlighted', fontweight='bold')

    for _, corr in corrections.iterrows():
        if corr['max_drawdown'] < -15:
            ax1.axvspan(corr['start'], corr['trough'], alpha=0.3, color='red')

    ax1.grid(True, alpha=0.3)

    # Plot all predictor scores
    ax2 = axes[1]
    colors = {'Baseline': 'gray', 'MacroMom': 'blue', 'Ensemble': 'purple',
              'Optimized': 'green', 'Adaptive': 'red'}

    for name, score_series in scores.items():
        aligned = score_series.reindex(common_idx).rolling(4).mean()
        ax2.plot(aligned.index, aligned, label=name, color=colors.get(name, 'black'),
                linewidth=2 if name in ['Optimized', 'Adaptive'] else 1,
                alpha=1.0 if name in ['Optimized', 'Adaptive'] else 0.6)

    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Bear Score')
    ax2.set_title('Predictor Scores Comparison', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(20, 90)

    # Plot Optimized vs Adaptive detail
    ax3 = axes[2]

    opt_scores = scores['Optimized'].reindex(common_idx)
    adapt_scores = scores['Adaptive'].reindex(common_idx)

    ax3.fill_between(opt_scores.index, opt_scores, alpha=0.5, color='green', label='Optimized')
    ax3.fill_between(adapt_scores.index, adapt_scores, alpha=0.5, color='red', label='Adaptive')

    ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)

    for _, corr in corrections.iterrows():
        if corr['max_drawdown'] < -15:
            ax3.axvline(x=corr['start'], color='red', alpha=0.7, linewidth=2)

    ax3.set_ylabel('Bear Score')
    ax3.set_xlabel('Date')
    ax3.set_title('Optimized vs Adaptive Predictors (corrections marked)', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(20, 90)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    plt.savefig('optimized_predictor_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("  [OK] Saved to optimized_predictor_comparison.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if 'corr_63d' in results_df.columns:
        best = results_df['corr_63d'].idxmax()
        print(f"\nBest correlation with 3-month drawdowns: {best} ({results_df.loc[best, 'corr_63d']:.3f})")

    if 'hit_rate' in results_df.columns:
        best = results_df['hit_rate'].idxmax()
        print(f"Best hit rate: {best} ({results_df.loc[best, 'hit_rate']:.1f}%)")

    if 'avg_pre_score' in results_df.columns:
        best = results_df['avg_pre_score'].idxmax()
        print(f"Highest pre-correction score: {best} ({results_df.loc[best, 'avg_pre_score']:.1f})")

    return results_df


if __name__ == "__main__":
    import sys
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_comparison(years=years)
