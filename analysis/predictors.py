"""
Alternative Bear Market Predictors
==================================

Multiple approaches to predicting market corrections, compared against baseline.

Predictors:
1. BaselinePredictor - Original regime.py approach
2. MomentumPredictor - Price momentum and trend following
3. VolatilityRegimePredictor - VIX term structure and volatility clustering
4. TechnicalPredictor - RSI, MACD, Bollinger Bands composite
5. MacroMomentumPredictor - Rate of change in macro indicators
6. EnsemblePredictor - Weighted combination of all predictors
7. MLPredictor - Machine learning approach (if sklearn available)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')


class BasePredictor(ABC):
    """Abstract base class for all predictors."""

    name: str = "Base"

    @abstractmethod
    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        """
        Calculate bear score at a specific index.

        Args:
            data: DataFrame with all market data
            idx: Current index position

        Returns:
            Score from 0-100 (higher = more bearish)
        """
        pass

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


# =============================================================================
# Predictor 1: Baseline (Original regime.py approach)
# =============================================================================

class BaselinePredictor(BasePredictor):
    """Original bear score from regime.py."""

    name = "Baseline"

    def __init__(self):
        self.weights = {
            "yield_curve": 0.25,
            "credit": 0.25,
            "liquidity": 0.20,
            "breadth": 0.15,
            "volatility": 0.10,
            "valuation": 0.05
        }

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _clamp(self, x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        window = data.iloc[max(0, idx - 252):idx + 1]
        scores = {}

        # Yield curve
        if 'y10' in data.columns and 'y3m' in data.columns:
            y10 = window['y10'].dropna()
            y3m = window['y3m'].dropna()
            if len(y10) > 30 and len(y3m) > 30:
                spread = y10 - y3m
                inv_depth = -spread.clip(upper=0)
                score = self._sigmoid(inv_depth.rolling(30).mean().iloc[-1] * 10)
                scores['yield_curve'] = score
            else:
                scores['yield_curve'] = 0.5
        else:
            scores['yield_curve'] = 0.5

        # Breadth
        if 'pct_above_200dma' in data.columns:
            pct = window['pct_above_200dma'].iloc[-1]
            scores['breadth'] = self._clamp((50 - pct) / 50)
        else:
            scores['breadth'] = 0.5

        # Volatility
        if 'vix' in data.columns and 'vix_3m' in data.columns:
            vix = window['vix'].iloc[-1]
            vix_3m = window['vix_3m'].iloc[-1]
            if vix_3m > 0:
                term = (vix - vix_3m) / vix_3m
                scores['volatility'] = self._clamp(self._sigmoid(term * 5))
            else:
                scores['volatility'] = 0.5
        else:
            scores['volatility'] = 0.5

        # Valuation
        if 'cape_percentile' in data.columns:
            cape = window['cape_percentile'].iloc[-1]
            scores['valuation'] = self._clamp((cape - 80) / 20)
        else:
            scores['valuation'] = 0.5

        # Credit and liquidity - use defaults if not available
        scores['credit'] = 0.5
        scores['liquidity'] = 0.5

        if 'credit_spread' in data.columns:
            cs = window['credit_spread'].dropna()
            if len(cs) > 252:
                z = zscore(cs[-252:])
                scores['credit'] = self._clamp(self._sigmoid(z[-1]))

        bear_score = sum(scores[k] * self.weights[k] for k in scores) * 100
        return bear_score


# =============================================================================
# Predictor 2: Momentum-Based
# =============================================================================

class MomentumPredictor(BasePredictor):
    """
    Uses price momentum across multiple timeframes.

    Logic: Deteriorating momentum often precedes corrections.
    """

    name = "Momentum"

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        sp500 = data['sp500'].iloc[:idx + 1]

        if len(sp500) < 252:
            return 50.0

        current = sp500.iloc[-1]
        scores = []

        # 1-month momentum (weight: 0.15)
        if len(sp500) >= 21:
            mom_1m = (current / sp500.iloc[-21] - 1) * 100
            # Negative momentum = bearish
            score_1m = max(0, min(100, 50 - mom_1m * 5))
            scores.append(('1m', score_1m, 0.15))

        # 3-month momentum (weight: 0.25)
        if len(sp500) >= 63:
            mom_3m = (current / sp500.iloc[-63] - 1) * 100
            score_3m = max(0, min(100, 50 - mom_3m * 3))
            scores.append(('3m', score_3m, 0.25))

        # 6-month momentum (weight: 0.30)
        if len(sp500) >= 126:
            mom_6m = (current / sp500.iloc[-126] - 1) * 100
            score_6m = max(0, min(100, 50 - mom_6m * 2))
            scores.append(('6m', score_6m, 0.30))

        # 12-month momentum (weight: 0.20)
        if len(sp500) >= 252:
            mom_12m = (current / sp500.iloc[-252] - 1) * 100
            score_12m = max(0, min(100, 50 - mom_12m * 1.5))
            scores.append(('12m', score_12m, 0.20))

        # Distance from 200 DMA (weight: 0.10)
        ma_200 = sp500.tail(200).mean()
        dist_ma = (current / ma_200 - 1) * 100
        score_ma = max(0, min(100, 50 - dist_ma * 3))
        scores.append(('ma200', score_ma, 0.10))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight

        return weighted_score


# =============================================================================
# Predictor 3: Volatility Regime
# =============================================================================

class VolatilityRegimePredictor(BasePredictor):
    """
    Focuses on volatility patterns and VIX term structure.

    Logic: Volatility clustering and term structure inversion signal stress.
    """

    name = "VolRegime"

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        window = data.iloc[max(0, idx - 252):idx + 1]

        if len(window) < 60:
            return 50.0

        scores = []

        # VIX level (weight: 0.25)
        if 'vix' in data.columns:
            vix = window['vix'].iloc[-1]
            # VIX > 25 is elevated, > 35 is high stress
            vix_score = max(0, min(100, (vix - 12) * 4))
            scores.append(('vix_level', vix_score, 0.25))

        # VIX term structure (weight: 0.25)
        if 'vix' in data.columns and 'vix_3m' in data.columns:
            vix = window['vix'].iloc[-1]
            vix_3m = window['vix_3m'].iloc[-1]
            if vix_3m > 0:
                # Inverted term structure (VIX > VIX3M) is bearish
                inversion = (vix / vix_3m - 1) * 100
                term_score = max(0, min(100, 50 + inversion * 5))
                scores.append(('term_structure', term_score, 0.25))

        # VIX rate of change (weight: 0.20)
        if 'vix' in data.columns and len(window) >= 21:
            vix_now = window['vix'].iloc[-1]
            vix_21d = window['vix'].iloc[-21]
            vix_roc = (vix_now / vix_21d - 1) * 100
            roc_score = max(0, min(100, 50 + vix_roc * 2))
            scores.append(('vix_roc', roc_score, 0.20))

        # Realized volatility vs VIX (weight: 0.15)
        if 'sp500' in data.columns and 'vix' in data.columns:
            returns = window['sp500'].pct_change().dropna()
            if len(returns) >= 21:
                realized_vol = returns.tail(21).std() * np.sqrt(252) * 100
                vix = window['vix'].iloc[-1]
                # High realized vol relative to VIX = stress
                vol_ratio = realized_vol / max(vix, 1)
                vol_score = max(0, min(100, vol_ratio * 50))
                scores.append(('vol_ratio', vol_score, 0.15))

        # Volatility of volatility (weight: 0.15)
        if 'vix' in data.columns and len(window) >= 21:
            vix_vol = window['vix'].tail(21).std()
            vvix_score = max(0, min(100, vix_vol * 10))
            scores.append(('vvix', vvix_score, 0.15))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight

        return weighted_score


# =============================================================================
# Predictor 4: Technical Indicators
# =============================================================================

class TechnicalPredictor(BasePredictor):
    """
    Combines classic technical indicators.

    Uses RSI, MACD, Bollinger Bands, and moving average crossovers.
    """

    name = "Technical"

    def _rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        if loss.iloc[-1] == 0:
            return 100 if gain.iloc[-1] > 0 else 50

        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def _macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and signal line."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd.iloc[-1], signal.iloc[-1]

    def _bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Return position within Bollinger Bands (0-1)."""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std

        current = prices.iloc[-1]
        band_width = upper.iloc[-1] - lower.iloc[-1]

        if band_width == 0:
            return 0.5

        position = (current - lower.iloc[-1]) / band_width
        return max(0, min(1, position))

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        sp500 = data['sp500'].iloc[:idx + 1]

        if len(sp500) < 50:
            return 50.0

        scores = []

        # RSI (weight: 0.25) - Overbought is bearish signal
        rsi = self._rsi(sp500)
        # RSI > 70 = overbought (bearish), RSI < 30 = oversold (bullish)
        rsi_score = max(0, min(100, (rsi - 30) * 1.25))
        scores.append(('rsi', rsi_score, 0.25))

        # MACD (weight: 0.25) - Negative histogram is bearish
        macd, signal = self._macd(sp500)
        histogram = macd - signal
        # Normalize histogram
        macd_score = max(0, min(100, 50 - histogram * 0.5))
        scores.append(('macd', macd_score, 0.25))

        # Bollinger Bands (weight: 0.20) - Near upper band is bearish
        bb_pos = self._bollinger_position(sp500)
        bb_score = bb_pos * 100  # Higher position = more bearish
        scores.append(('bollinger', bb_score, 0.20))

        # Moving average crossovers (weight: 0.30)
        if len(sp500) >= 200:
            ma_50 = sp500.tail(50).mean()
            ma_200 = sp500.tail(200).mean()
            current = sp500.iloc[-1]

            # Below both MAs = very bearish
            # Between MAs = neutral
            # Above both = bullish (low bear score)
            if current < ma_200:
                ma_score = 80  # Below 200 MA
            elif current < ma_50:
                ma_score = 60  # Below 50 MA but above 200
            elif ma_50 < ma_200:
                ma_score = 55  # Death cross
            else:
                # Distance above MAs
                dist = (current / ma_200 - 1) * 100
                ma_score = max(0, min(50, 50 - dist))

            scores.append(('ma_cross', ma_score, 0.30))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight

        return weighted_score


# =============================================================================
# Predictor 5: Macro Momentum (Rate of Change)
# =============================================================================

class MacroMomentumPredictor(BasePredictor):
    """
    Focuses on rate of change in macro indicators.

    Logic: Rapid deterioration in conditions often precedes corrections.
    """

    name = "MacroMom"

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        window = data.iloc[max(0, idx - 252):idx + 1]

        if len(window) < 60:
            return 50.0

        scores = []

        # Yield curve momentum (weight: 0.30)
        if 'y10' in data.columns and 'y3m' in data.columns:
            spread = window['y10'] - window['y3m']
            if len(spread) >= 13:
                spread_now = spread.iloc[-1]
                spread_13w = spread.iloc[-13]
                spread_change = spread_now - spread_13w

                # Flattening/inverting curve is bearish
                yc_score = max(0, min(100, 50 - spread_change * 20))
                scores.append(('yc_mom', yc_score, 0.30))

        # VIX momentum (weight: 0.25)
        if 'vix' in data.columns and len(window) >= 13:
            vix_now = window['vix'].iloc[-1]
            vix_13w = window['vix'].iloc[-13]
            vix_change = (vix_now / vix_13w - 1) * 100

            # Rising VIX is bearish
            vix_score = max(0, min(100, 50 + vix_change * 2))
            scores.append(('vix_mom', vix_score, 0.25))

        # Breadth momentum (weight: 0.25)
        if 'pct_above_200dma' in data.columns and len(window) >= 13:
            pct_now = window['pct_above_200dma'].iloc[-1]
            pct_13w = window['pct_above_200dma'].iloc[-13]
            pct_change = pct_now - pct_13w

            # Declining breadth is bearish
            breadth_score = max(0, min(100, 50 - pct_change))
            scores.append(('breadth_mom', breadth_score, 0.25))

        # Price momentum acceleration (weight: 0.20)
        if 'sp500' in data.columns and len(window) >= 26:
            sp500 = window['sp500']
            mom_now = (sp500.iloc[-1] / sp500.iloc[-13] - 1) * 100
            mom_prev = (sp500.iloc[-13] / sp500.iloc[-26] - 1) * 100
            mom_accel = mom_now - mom_prev

            # Decelerating momentum is bearish
            accel_score = max(0, min(100, 50 - mom_accel * 3))
            scores.append(('mom_accel', accel_score, 0.20))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight

        return weighted_score


# =============================================================================
# Predictor 6: Ensemble (Weighted Average)
# =============================================================================

class EnsemblePredictor(BasePredictor):
    """
    Combines all predictors with optimized weights.
    """

    name = "Ensemble"

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.predictors = {
            'baseline': BaselinePredictor(),
            'momentum': MomentumPredictor(),
            'vol_regime': VolatilityRegimePredictor(),
            'technical': TechnicalPredictor(),
            'macro_mom': MacroMomentumPredictor(),
        }

        # Default weights (can be optimized)
        self.weights = weights or {
            'baseline': 0.20,
            'momentum': 0.25,
            'vol_regime': 0.20,
            'technical': 0.15,
            'macro_mom': 0.20,
        }

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        scores = {}

        for name, predictor in self.predictors.items():
            try:
                scores[name] = predictor.calculate_score(data, idx)
            except Exception:
                scores[name] = 50.0  # Default to neutral

        weighted_score = sum(
            scores[name] * self.weights[name]
            for name in self.predictors
        )

        return weighted_score


# =============================================================================
# Predictor 7: ML-Based (if sklearn available)
# =============================================================================

class MLPredictor(BasePredictor):
    """
    Machine learning predictor using Random Forest.

    Trains on historical data to predict future drawdowns.
    """

    name = "ML_RF"

    def __init__(self):
        self.model = None
        self.is_trained = False
        self._feature_names = []

    def _extract_features(self, data: pd.DataFrame, idx: int) -> Optional[np.ndarray]:
        """Extract features for ML model."""
        if idx < 252:
            return None

        window = data.iloc[max(0, idx - 252):idx + 1]
        sp500 = window['sp500']

        features = []
        self._feature_names = []

        # Price momentum features
        for period in [5, 10, 21, 63, 126, 252]:
            if len(sp500) >= period:
                mom = (sp500.iloc[-1] / sp500.iloc[-period] - 1)
                features.append(mom)
                self._feature_names.append(f'mom_{period}d')

        # Volatility features
        returns = sp500.pct_change().dropna()
        if len(returns) >= 21:
            vol_21 = returns.tail(21).std() * np.sqrt(252)
            vol_63 = returns.tail(min(63, len(returns))).std() * np.sqrt(252)
            features.extend([vol_21, vol_63, vol_21 / max(vol_63, 0.01)])
            self._feature_names.extend(['vol_21d', 'vol_63d', 'vol_ratio'])

        # VIX features
        if 'vix' in data.columns:
            vix = window['vix']
            features.append(vix.iloc[-1])
            features.append(vix.iloc[-1] / max(vix.mean(), 1))
            self._feature_names.extend(['vix', 'vix_zscore'])

            if 'vix_3m' in data.columns:
                vix_3m = window['vix_3m'].iloc[-1]
                features.append(vix.iloc[-1] / max(vix_3m, 1))
                self._feature_names.append('vix_term')

        # Breadth features
        if 'pct_above_200dma' in data.columns:
            pct = window['pct_above_200dma']
            features.append(pct.iloc[-1])
            if len(pct) >= 21:
                features.append(pct.iloc[-1] - pct.iloc[-21])
            self._feature_names.extend(['breadth', 'breadth_change'])

        # Yield curve features
        if 'y10' in data.columns and 'y3m' in data.columns:
            spread = (window['y10'] - window['y3m']).iloc[-1]
            features.append(spread)
            self._feature_names.append('yield_spread')

        # Distance from moving averages
        for ma_period in [50, 200]:
            if len(sp500) >= ma_period:
                ma = sp500.tail(ma_period).mean()
                dist = sp500.iloc[-1] / ma - 1
                features.append(dist)
                self._feature_names.append(f'dist_ma{ma_period}')

        if len(features) < 5:
            return None

        return np.array(features)

    def train(self, data: pd.DataFrame, forward_window: int = 63):
        """
        Train the model on historical data.

        Args:
            data: Full historical data
            forward_window: Days ahead to predict drawdown
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("Warning: sklearn not available, ML predictor disabled")
            return

        print(f"Training ML model on {len(data)} data points...")

        X = []
        y = []

        # Calculate forward drawdowns for labels
        sp500 = data['sp500']
        forward_min = sp500.rolling(forward_window).min().shift(-forward_window)
        forward_drawdown = (forward_min / sp500 - 1) * -100  # Positive = drawdown %

        for i in range(252, len(data) - forward_window):
            features = self._extract_features(data, i)
            if features is not None and not np.isnan(forward_drawdown.iloc[i]):
                X.append(features)
                y.append(forward_drawdown.iloc[i])

        if len(X) < 100:
            print("Insufficient training data")
            return

        X = np.array(X)
        y = np.array(y)

        # Handle NaN/Inf
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

        print(f"  Trained on {len(X)} samples")
        print(f"  Feature importance (top 5):")
        importances = list(zip(self._feature_names[:len(self.model.feature_importances_)],
                               self.model.feature_importances_))
        importances.sort(key=lambda x: -x[1])
        for name, imp in importances[:5]:
            print(f"    {name}: {imp:.3f}")

    def calculate_score(self, data: pd.DataFrame, idx: int) -> float:
        if not self.is_trained or self.model is None:
            return 50.0

        features = self._extract_features(data, idx)
        if features is None:
            return 50.0

        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            predicted_drawdown = self.model.predict(features_scaled)[0]

            # Convert predicted drawdown to score (0-100)
            # 0% drawdown = 0 score, 30%+ drawdown = 100 score
            score = max(0, min(100, predicted_drawdown * 3.33))
            return score
        except Exception:
            return 50.0


# =============================================================================
# Utility: Get all predictors
# =============================================================================

def get_all_predictors(include_ml: bool = True) -> Dict[str, BasePredictor]:
    """Return dictionary of all available predictors."""
    predictors = {
        'Baseline': BaselinePredictor(),
        'Momentum': MomentumPredictor(),
        'VolRegime': VolatilityRegimePredictor(),
        'Technical': TechnicalPredictor(),
        'MacroMom': MacroMomentumPredictor(),
        'Ensemble': EnsemblePredictor(),
    }

    if include_ml:
        try:
            import sklearn
            predictors['ML_RF'] = MLPredictor()
        except ImportError:
            pass

    return predictors
