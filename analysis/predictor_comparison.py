"""
Predictor Comparison Framework
==============================

Compares all bear market predictors against historical corrections.

Metrics:
1. Correlation with forward drawdowns
2. Warning lead time before corrections
3. False positive rate
4. Sharpe ratio of "defensive" strategy
5. Hit rate on major corrections
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

from predictors import get_all_predictors, MLPredictor


class PredictorEvaluator:
    """Evaluates and compares multiple bear market predictors."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with historical market data.

        Args:
            data: DataFrame with sp500, vix, y10, y3m, etc.
        """
        self.data = data
        self.predictors = get_all_predictors(include_ml=True)
        self.scores: Dict[str, pd.Series] = {}
        self.metrics: Dict[str, Dict] = {}

    def calculate_all_scores(self, min_lookback: int = 252) -> None:
        """Calculate scores for all predictors."""
        print("Calculating scores for all predictors...")

        for name, predictor in self.predictors.items():
            print(f"  - {name}...")

            # Special handling for ML predictor
            if isinstance(predictor, MLPredictor):
                # Train on first 70% of data
                train_end = int(len(self.data) * 0.7)
                predictor.train(self.data.iloc[:train_end])

            self.scores[name] = predictor.calculate_all_scores(self.data, min_lookback)

        print(f"  [OK] Calculated scores for {len(self.scores)} predictors")

    def identify_corrections(
        self,
        threshold: float = 0.10,
        recovery_threshold: float = 0.05
    ) -> pd.DataFrame:
        """Identify market corrections."""
        sp500 = self.data['sp500']
        rolling_max = sp500.expanding().max()
        drawdown = (sp500 - rolling_max) / rolling_max

        corrections = []
        in_correction = False
        start_date = None
        peak_value = None

        for date, dd in drawdown.items():
            if not in_correction and dd < -threshold:
                in_correction = True
                start_date = date
                peak_value = rolling_max[date]
            elif in_correction and dd >= -recovery_threshold:
                trough_idx = drawdown[start_date:date].idxmin()
                corrections.append({
                    'start': start_date,
                    'trough': trough_idx,
                    'end': date,
                    'max_drawdown': drawdown[trough_idx] * 100,
                    'peak_value': peak_value,
                    'trough_value': sp500[trough_idx],
                    'duration_days': (trough_idx - start_date).days,
                })
                in_correction = False

        return pd.DataFrame(corrections)

    def calculate_forward_drawdowns(self, windows: List[int] = [21, 63, 126]) -> Dict[int, pd.Series]:
        """Calculate forward maximum drawdowns for each window."""
        sp500 = self.data['sp500']
        forward_dds = {}

        for window in windows:
            forward_min = sp500.rolling(window).min().shift(-window)
            forward_dd = (forward_min / sp500 - 1) * -100  # Positive = drawdown %
            forward_dds[window] = forward_dd

        return forward_dds

    def evaluate_predictor(
        self,
        name: str,
        corrections: pd.DataFrame,
        forward_dds: Dict[int, pd.Series]
    ) -> Dict:
        """Evaluate a single predictor."""
        scores = self.scores[name].dropna()

        if len(scores) < 100:
            return {'error': 'Insufficient data'}

        metrics = {'name': name}

        # 1. Correlation with forward drawdowns
        for window, fwd_dd in forward_dds.items():
            aligned = pd.DataFrame({
                'score': scores,
                'fwd_dd': fwd_dd
            }).dropna()

            if len(aligned) > 50:
                corr = aligned['score'].corr(aligned['fwd_dd'])
                metrics[f'corr_{window}d'] = corr

        # 2. Pre-correction scores
        pre_scores = []
        for _, corr in corrections.iterrows():
            if corr['max_drawdown'] < -10:  # Significant corrections
                # Get average score 4 weeks before
                pre_start = corr['start'] - timedelta(weeks=4)
                mask = (scores.index >= pre_start) & (scores.index < corr['start'])
                if mask.any():
                    pre_scores.append({
                        'drawdown': abs(corr['max_drawdown']),
                        'pre_score': scores[mask].mean()
                    })

        if pre_scores:
            pre_df = pd.DataFrame(pre_scores)
            metrics['avg_pre_correction_score'] = pre_df['pre_score'].mean()
            metrics['pre_score_drawdown_corr'] = pre_df['pre_score'].corr(pre_df['drawdown'])

            # Hit rate: % of corrections where score was elevated (>50)
            metrics['hit_rate'] = (pre_df['pre_score'] > 50).mean() * 100

        # 3. False positive rate
        # High score (>60) without subsequent 10% correction in 3 months
        high_score_dates = scores[scores > 60].index
        false_positives = 0
        true_positives = 0

        for date in high_score_dates:
            # Check if correction started within 3 months
            found_correction = False
            for _, corr in corrections.iterrows():
                if corr['start'] >= date and corr['start'] <= date + timedelta(days=90):
                    if corr['max_drawdown'] < -10:
                        found_correction = True
                        break

            if found_correction:
                true_positives += 1
            else:
                false_positives += 1

        total_warnings = true_positives + false_positives
        if total_warnings > 0:
            metrics['false_positive_rate'] = false_positives / total_warnings * 100
            metrics['precision'] = true_positives / total_warnings * 100

        # 4. Average lead time for detected corrections
        lead_times = []
        for _, corr in corrections.iterrows():
            if corr['max_drawdown'] < -15:  # Major corrections
                # Find first date with score > 55 before correction
                pre_mask = (scores.index < corr['start']) & (scores.index > corr['start'] - timedelta(days=180))
                high_scores = scores[pre_mask & (scores > 55)]

                if len(high_scores) > 0:
                    first_warning = high_scores.index[0]
                    lead_time = (corr['start'] - first_warning).days
                    lead_times.append(lead_time)

        if lead_times:
            metrics['avg_lead_time_days'] = np.mean(lead_times)

        # 5. Defensive strategy performance
        # When score > 60, go to cash; otherwise stay invested
        aligned_data = pd.DataFrame({
            'score': scores,
            'sp500': self.data['sp500']
        }).dropna()

        if len(aligned_data) > 100:
            returns = aligned_data['sp500'].pct_change()
            defensive = (aligned_data['score'] <= 60).shift(1).fillna(True)
            strategy_returns = returns * defensive

            # Calculate Sharpe ratio
            excess_returns = strategy_returns - 0.04 / 252  # Risk-free rate
            if strategy_returns.std() > 0:
                metrics['defensive_sharpe'] = (
                    excess_returns.mean() / strategy_returns.std() * np.sqrt(252)
                )

            # Calculate defensive strategy total return
            metrics['defensive_total_return'] = (1 + strategy_returns).prod() - 1
            metrics['buy_hold_return'] = (1 + returns.dropna()).prod() - 1

        # 6. Score statistics
        metrics['score_mean'] = scores.mean()
        metrics['score_std'] = scores.std()
        metrics['score_median'] = scores.median()
        metrics['pct_time_elevated'] = (scores > 50).mean() * 100
        metrics['pct_time_high_risk'] = (scores > 70).mean() * 100

        return metrics

    def evaluate_all(self) -> pd.DataFrame:
        """Evaluate all predictors and return comparison DataFrame."""
        corrections = self.identify_corrections()
        forward_dds = self.calculate_forward_drawdowns()

        print(f"\nEvaluating {len(self.predictors)} predictors...")
        print(f"Found {len(corrections)} corrections (>10% drawdown)")

        for name in self.scores:
            print(f"  - {name}...")
            self.metrics[name] = self.evaluate_predictor(name, corrections, forward_dds)

        # Convert to DataFrame
        results = pd.DataFrame(self.metrics).T

        return results

    def create_comparison_charts(self, save_path: str = 'predictor_comparison.png'):
        """Create comprehensive comparison visualization."""
        print("\nCreating comparison charts...")

        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Align all scores to common dates
        all_scores = pd.DataFrame(self.scores)
        all_scores = all_scores.dropna(how='all')

        sp500 = self.data.loc[all_scores.index, 'sp500']
        corrections = self.identify_corrections()

        # =====================================================================
        # 1. All predictors over time (top row, full width)
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, :])

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.scores)))

        for (name, scores), color in zip(self.scores.items(), colors):
            scores_aligned = scores.reindex(all_scores.index)
            smoothed = scores_aligned.rolling(4).mean()
            ax1.plot(smoothed.index, smoothed, label=name, alpha=0.8, linewidth=1.5, color=color)

        # Mark corrections
        for _, corr in corrections.iterrows():
            if corr['max_drawdown'] < -15:
                ax1.axvspan(corr['start'], corr['trough'], alpha=0.2, color='red')

        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Bear Score')
        ax1.set_title('All Predictors Over Time (with major corrections shaded)', fontweight='bold')
        ax1.legend(loc='upper right', ncol=3, fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(20, 90)

        # =====================================================================
        # 2. Correlation with forward drawdowns
        # =====================================================================
        ax2 = fig.add_subplot(gs[1, 0])

        metrics_df = pd.DataFrame(self.metrics).T
        corr_cols = [c for c in metrics_df.columns if c.startswith('corr_')]

        if corr_cols:
            corr_data = metrics_df[corr_cols].astype(float)
            corr_data.columns = [c.replace('corr_', '').replace('d', '-day') for c in corr_cols]

            x = np.arange(len(corr_data.index))
            width = 0.25
            multiplier = 0

            for col in corr_data.columns:
                offset = width * multiplier
                bars = ax2.bar(x + offset, corr_data[col], width, label=col, alpha=0.8)
                multiplier += 1

            ax2.set_xticks(x + width)
            ax2.set_xticklabels(corr_data.index, rotation=45, ha='right')
            ax2.set_ylabel('Correlation')
            ax2.set_title('Correlation with Forward Drawdowns', fontweight='bold')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

        # =====================================================================
        # 3. Hit rate and precision
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 1])

        if 'hit_rate' in metrics_df.columns and 'precision' in metrics_df.columns:
            hit_rate = metrics_df['hit_rate'].astype(float)
            precision = metrics_df['precision'].astype(float).fillna(0)

            x = np.arange(len(hit_rate))
            width = 0.35

            ax3.bar(x - width/2, hit_rate, width, label='Hit Rate', color='green', alpha=0.7)
            ax3.bar(x + width/2, precision, width, label='Precision', color='blue', alpha=0.7)

            ax3.set_xticks(x)
            ax3.set_xticklabels(hit_rate.index, rotation=45, ha='right')
            ax3.set_ylabel('Percentage')
            ax3.set_title('Hit Rate & Precision', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

        # =====================================================================
        # 4. Lead time before corrections
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 2])

        if 'avg_lead_time_days' in metrics_df.columns:
            lead_times = metrics_df['avg_lead_time_days'].dropna().astype(float)

            bars = ax4.bar(lead_times.index, lead_times, color='purple', alpha=0.7)
            ax4.set_ylabel('Days')
            ax4.set_title('Average Lead Time Before Major Corrections', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, lead_times):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}d', ha='center', fontsize=8)

        # =====================================================================
        # 5. Defensive strategy returns
        # =====================================================================
        ax5 = fig.add_subplot(gs[2, 0])

        if 'defensive_total_return' in metrics_df.columns:
            def_ret = metrics_df['defensive_total_return'].astype(float) * 100
            bh_ret = metrics_df['buy_hold_return'].iloc[0] * 100 if 'buy_hold_return' in metrics_df.columns else 0

            colors = ['green' if r > bh_ret else 'red' for r in def_ret]
            bars = ax5.bar(def_ret.index, def_ret, color=colors, alpha=0.7)
            ax5.axhline(y=bh_ret, color='blue', linestyle='--', linewidth=2,
                       label=f'Buy & Hold: {bh_ret:.1f}%')

            ax5.set_ylabel('Total Return %')
            ax5.set_title('Defensive Strategy Returns vs Buy & Hold', fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        # =====================================================================
        # 6. Pre-correction score analysis
        # =====================================================================
        ax6 = fig.add_subplot(gs[2, 1])

        if 'avg_pre_correction_score' in metrics_df.columns:
            pre_scores = metrics_df['avg_pre_correction_score'].astype(float)

            bars = ax6.bar(pre_scores.index, pre_scores, alpha=0.7,
                          color=['green' if s > 50 else 'gray' for s in pre_scores])
            ax6.axhline(y=50, color='red', linestyle='--', label='Warning Threshold (50)')

            ax6.set_ylabel('Average Score')
            ax6.set_title('Avg Score 4 Weeks Before Corrections', fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')

        # =====================================================================
        # 7. Summary ranking
        # =====================================================================
        ax7 = fig.add_subplot(gs[2, 2])

        # Create composite score
        ranking_metrics = ['corr_63d', 'hit_rate', 'precision', 'avg_lead_time_days']
        available_metrics = [m for m in ranking_metrics if m in metrics_df.columns]

        if available_metrics:
            # Normalize each metric to 0-100
            normalized = pd.DataFrame()
            for col in available_metrics:
                vals = metrics_df[col].astype(float)
                if col == 'avg_lead_time_days':
                    # Higher lead time is better
                    normalized[col] = (vals - vals.min()) / (vals.max() - vals.min() + 0.001) * 100
                else:
                    normalized[col] = (vals - vals.min()) / (vals.max() - vals.min() + 0.001) * 100

            # Composite score
            composite = normalized.mean(axis=1).sort_values(ascending=False)

            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(composite)))[::-1]
            bars = ax7.barh(composite.index, composite, color=colors)

            ax7.set_xlabel('Composite Score')
            ax7.set_title('Overall Predictor Ranking', fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for bar, val in zip(bars, composite):
                ax7.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=9)

        plt.suptitle('Bear Market Predictor Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  [OK] Saved to {save_path}")

        return fig

    def create_detailed_score_chart(self, save_path: str = 'predictor_scores_detail.png'):
        """Create detailed chart showing each predictor vs S&P 500."""
        print("\nCreating detailed score charts...")

        n_predictors = len(self.scores)
        fig, axes = plt.subplots(n_predictors + 1, 1, figsize=(16, 3 * (n_predictors + 1)), sharex=True)

        # Align data
        common_idx = self.scores[list(self.scores.keys())[0]].dropna().index
        sp500 = self.data.loc[common_idx, 'sp500']
        corrections = self.identify_corrections()

        # S&P 500 chart at top
        ax = axes[0]
        ax.plot(sp500.index, sp500, 'k-', linewidth=1)
        ax.set_ylabel('S&P 500')
        ax.set_yscale('log')
        ax.set_title('S&P 500 Price', fontweight='bold')

        for _, corr in corrections.iterrows():
            if corr['max_drawdown'] < -15:
                ax.axvspan(corr['start'], corr['trough'], alpha=0.3, color='red')
                ax.annotate(f"{corr['max_drawdown']:.0f}%",
                           xy=(corr['trough'], corr['trough_value']),
                           fontsize=8, color='red')

        ax.grid(True, alpha=0.3)

        # Individual predictor charts
        for i, (name, scores) in enumerate(self.scores.items()):
            ax = axes[i + 1]
            scores_aligned = scores.reindex(common_idx)

            # Color based on score level
            ax.fill_between(scores_aligned.index, scores_aligned,
                           where=scores_aligned <= 40, color='green', alpha=0.5, label='Low Risk')
            ax.fill_between(scores_aligned.index, scores_aligned,
                           where=(scores_aligned > 40) & (scores_aligned <= 60),
                           color='yellow', alpha=0.5, label='Caution')
            ax.fill_between(scores_aligned.index, scores_aligned,
                           where=scores_aligned > 60, color='red', alpha=0.5, label='High Risk')

            ax.plot(scores_aligned.index, scores_aligned, 'k-', linewidth=0.5)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)

            ax.set_ylabel(name)
            ax.set_ylim(20, 90)
            ax.grid(True, alpha=0.3)

            # Mark corrections
            for _, corr in corrections.iterrows():
                if corr['max_drawdown'] < -15:
                    ax.axvline(x=corr['start'], color='red', alpha=0.5, linewidth=1)

        axes[-1].set_xlabel('Date')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  [OK] Saved to {save_path}")

        return fig


def fetch_data(years: int = 20) -> pd.DataFrame:
    """Fetch historical data for analysis."""
    import yfinance as yf

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"Fetching {years} years of historical data...")

    # S&P 500
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Close'].squeeze()

    # VIX
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close'].squeeze()

    # VIX 3M
    try:
        vix_3m = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)['Close'].squeeze()
        if len(vix_3m) < 100:
            vix_3m = vix.rolling(20).mean() * 0.95
    except:
        vix_3m = vix.rolling(20).mean() * 0.95

    # Treasury yields
    try:
        y10 = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close'].squeeze()
    except:
        y10 = pd.Series(dtype=float)

    try:
        y3m = yf.download('^IRX', start=start_date, end=end_date, progress=False)['Close'].squeeze()
    except:
        y3m = pd.Series(dtype=float)

    # Calculate breadth
    ma200 = sp500.rolling(200).mean()
    pct_above_200dma = (sp500 > ma200).astype(float) * 100

    # Calculate CAPE percentile approximation
    cape_percentile = sp500.rolling(252 * 10, min_periods=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )

    df = pd.DataFrame({
        'sp500': sp500,
        'vix': vix,
        'vix_3m': vix_3m,
        'y10': y10 if len(y10) > 0 else np.nan,
        'y3m': y3m if len(y3m) > 0 else np.nan,
        'pct_above_200dma': pct_above_200dma,
        'cape_percentile': cape_percentile,
    })

    # Resample to weekly
    df = df.resample('W-FRI').last().ffill().bfill()

    print(f"  [OK] Loaded {len(df)} weekly data points")

    return df


def main():
    """Run the full predictor comparison."""
    print("=" * 70)
    print("BEAR MARKET PREDICTOR COMPARISON")
    print("=" * 70)

    # Fetch data
    data = fetch_data(years=20)

    # Initialize evaluator
    evaluator = PredictorEvaluator(data)

    # Calculate all scores
    evaluator.calculate_all_scores()

    # Evaluate all predictors
    results = evaluator.evaluate_all()

    # Create visualizations
    evaluator.create_comparison_charts('predictor_comparison.png')
    evaluator.create_detailed_score_chart('predictor_scores_detail.png')

    # Print summary table
    print("\n" + "=" * 70)
    print("PREDICTOR COMPARISON RESULTS")
    print("=" * 70)

    # Select key metrics for display
    display_cols = [
        'corr_63d', 'hit_rate', 'precision', 'avg_lead_time_days',
        'avg_pre_correction_score', 'defensive_sharpe'
    ]
    available_cols = [c for c in display_cols if c in results.columns]

    print("\n" + results[available_cols].round(2).to_string())

    # Determine winner
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    if 'corr_63d' in results.columns:
        best_corr = results['corr_63d'].astype(float).idxmax()
        print(f"\nBest correlation with 3-month drawdowns: {best_corr} ({results.loc[best_corr, 'corr_63d']:.3f})")

    if 'hit_rate' in results.columns:
        best_hit = results['hit_rate'].astype(float).idxmax()
        print(f"Best hit rate on corrections: {best_hit} ({results.loc[best_hit, 'hit_rate']:.1f}%)")

    if 'avg_lead_time_days' in results.columns:
        lead_times = results['avg_lead_time_days'].dropna().astype(float)
        if len(lead_times) > 0:
            best_lead = lead_times.idxmax()
            print(f"Best lead time: {best_lead} ({lead_times[best_lead]:.0f} days)")

    if 'defensive_sharpe' in results.columns:
        best_sharpe = results['defensive_sharpe'].astype(float).idxmax()
        print(f"Best defensive Sharpe ratio: {best_sharpe} ({results.loc[best_sharpe, 'defensive_sharpe']:.2f})")

    print("\n[OK] Analysis complete. Check predictor_comparison.png and predictor_scores_detail.png")


if __name__ == "__main__":
    main()
