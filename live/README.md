# Live Regime-Adaptive Trading System

This directory contains the live trading implementation of the Regime-Adaptive Strategy.

## Overview

The live trader uses the **exact same `RegimeAdaptiveStrategy`** from `strategies/regime_adaptive_strategy.py`
that is used in backtesting. This ensures consistency between backtesting and live trading.

The system:
1. Connects to Robinhood via the `robin_stocks` library
2. Runs the RegimeAdaptiveStrategy which:
   - Fetches market regime data (VIX, yield curve, breadth, etc.)
   - Calculates bear score using MacroMom predictor
   - Determines allocation between HighBetaGrowthStrategy and BearBetaStrategy
   - Selects specific stocks based on fundamentals and beta characteristics
3. Calculates the difference between current and target portfolio
4. Executes trades to rebalance
5. Sends email reports with actions taken and portfolio status

## Files

- `live_regime_trader.py` - Main trading runner (uses RegimeAdaptiveStrategy)
- `scheduler.py` - Scheduled execution for daily trading
- `__init__.py` - Package exports

## Installation

```bash
# Install required packages
pip install robin_stocks pyotp yfinance pandas numpy schedule pytz
```

## Configuration

Set the following environment variables:

### Robinhood Credentials
```bash
# Required
export ROBINHOOD_USERNAME="your_email@example.com"
export ROBINHOOD_PASSWORD="your_password"

# Optional (for 2FA auto-generation)
export ROBINHOOD_TOTP_SECRET="your_2fa_secret"
```

### Email Notifications (Gmail)
```bash
export SMTP_USERNAME="your_gmail@gmail.com"
export SMTP_PASSWORD="your_app_password"
```

**Note:** For Gmail, you must:
1. Enable 2-Step Verification on your Google account
2. Create an App Password at https://myaccount.google.com/apppasswords
3. Use the App Password (not your regular password) for SMTP_PASSWORD

## Usage

### Dry Run (No Real Trades)
```bash
# Run once immediately
python live/live_regime_trader.py --dry-run

# Or via scheduler
python live/scheduler.py --once
```

### Live Trading (Real Money!)
```bash
# Run once with real trades
python live/live_regime_trader.py --live

# Or scheduled daily at 9:45 AM ET
python live/scheduler.py --live
```

### Custom Schedule
```bash
# Run daily at 10:00 AM Eastern
python live/scheduler.py --time 10:00 --timezone US/Eastern

# Send reports to different email
python live/scheduler.py --email different@email.com
```

## Trading Strategy

### Stock Selection

The live trader uses the same strategy as backtesting:

**Aggressive Component (HighBetaGrowthStrategy):**
- Selects stocks with Beta > 1.0 (captures market upside)
- High ROE > 15% (profitability)
- Operating Margin > 15%
- Gross Margin > 40%
- Revenue Growth > 10%
- Prefers Technology, Healthcare, Consumer Discretionary sectors
- Universe: S&P 500 stocks from Yahoo Finance

**Defensive Component (BearBetaStrategy):**
- Selects stocks with low Bear Beta < 0.8 (defensive during crashes)
- Bear Beta = Covariance with market on down days / Variance of down days
- Prefers Consumer Staples, Healthcare, Utilities, Gold miners
- Prioritizes dividend-paying large-cap stocks
- Universe: S&P 500 stocks from Yahoo Finance

### Allocation Thresholds (Optimized)

| Bear Score | Risk Level | Aggressive | Defensive |
|------------|------------|------------|-----------|
| 0-55       | Low-Moderate | 100% | 0% |
| 55-65      | Elevated | 85% | 15% |
| 65-75      | High | 60% | 40% |
| 75-100     | Extreme | 30% | 70% |

### Regime Detection

The MacroMom bear predictor uses:
- **Yield Curve Momentum**: Tracks slope changes in yield curve
- **VIX Term Structure**: Monitors VIX vs VIX3M spread
- **Market Breadth**: Percent of stocks above 200 DMA
- **Price Momentum**: SPY relative to moving averages

These factors are combined into a bear score (0-100) with ~173 days lead time on corrections.

## Safety Features

1. **Dry Run Mode** - Default mode simulates trades without executing
2. **Live Mode Confirmation** - Requires typing 'CONFIRM' to enable live trading
3. **Order Validation** - Validates buying power and position sizes before trading
4. **Market Hours Check** - Won't execute trades when market is closed
5. **Min Trade Value** - Ignores trades below $50 to avoid tiny orders
6. **Max Position Weight** - Limits single positions to 25% of portfolio
7. **Rebalance Threshold** - Only rebalances when allocation change > 10%

## Email Reports

When rebalancing occurs, an email report is sent containing:
- Current regime assessment (bear score, risk level)
- Actions taken (buys/sells with prices and values)
- Portfolio summary (positions, values, weights)
- Reason for rebalancing

## Troubleshooting

### "Robinhood credentials required"
Set the ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD environment variables.

### "2FA required"
Either:
- Provide the TOTP secret for automatic 2FA code generation
- Or manually enter the MFA code when prompted

### "Email not configured"
Set SMTP_USERNAME and SMTP_PASSWORD for email notifications.

### "Cannot execute trades while market is closed"
The strategy only executes trades during market hours (9:30 AM - 4:00 PM ET).

## Disclaimer

This is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider consulting a financial advisor.
