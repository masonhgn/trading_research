# ETF Statistical Arbitrage Strategy

A statistical arbitrage strategy for SPY vs VOO using z-score mean reversion.

## Files

- **`etf_arb.py`** - Main strategy implementation with optimized parameters
- **`backtest.py`** - Simple backtesting utility
- **`tools.py`** - Data fetching utilities
- **`requirements.txt`** - Dependencies

## Quick Start

```bash
# Run the main strategy
python etf_arb.py

# Run a custom backtest
python backtest.py
```

## Strategy Parameters (Optimized)

- **Symbols**: SPY vs VOO
- **Window**: 90 periods
- **Entry Threshold**: 1.5 z-score
- **Exit Threshold**: 0.2 z-score
- **Position Size**: 200 shares per leg
- **Trading Hours**: 9:30 AM - 4:00 PM ET only

## Performance

- **Expected PnL**: ~$800 per 3-day period
- **Win Rate**: ~29%
- **Investment**: ~$240k
- **Return**: ~0.33%
quant trading research
