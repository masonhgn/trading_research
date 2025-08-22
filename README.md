# ETF Statistical Arbitrage Strategy

A statistical arbitrage strategy for SPY vs VOO using z-score mean reversion, built with a clean layer-based architecture.

## Architecture Overview

The project follows a layer-based architecture for clean separation of concerns:

```
trading_research/
├── data_layer/           # Data fetching and processing
│   ├── data_feed.py      # Market data feed interface
│   └── processors.py     # Data processing utilities
├── strategy_layer/       # Strategy implementations
│   ├── base.py          # Base strategy class
│   └── etf_arbitrage.py # ETF arbitrage strategy
├── execution_layer/      # Execution and backtesting
│   └── backtest.py      # Backtesting engine
├── analysis_layer/       # Performance analysis
│   └── analyzer.py      # Performance metrics and visualization
├── infrastructure/       # System infrastructure
│   ├── config.py        # Universal configuration
│   └── notifications.py # Telegram notifications
└── scripts/             # Execution scripts
    ├── backtest_runner.py
    ├── test_new_architecture.py
    └── probe_ib.py
```

## Quick Start

### 1. Test the New Architecture
```bash
# Test that everything works
python scripts/test_new_architecture.py
```

### 2. Run a Backtest
```bash
# Run the ETF arbitrage strategy backtest
python scripts/backtest_runner.py
```

### 3. Test IB Connection
```bash
# Test Interactive Brokers connection
python scripts/probe_ib.py
```

### 4. Run the Web Dashboard
```bash
# Start the Flask trading dashboard
python scripts/run_flask_server.py
```

The dashboard will be available at `http://localhost:8080`

**Default Users:**
- **admin/admin123** - Full access to all features
- **trader/trader123** - Trading access (can execute trades)
- **viewer/viewer123** - View-only access (can view data but not trade)

## Configuration

All configuration is centralized in `config.yaml` and loaded by `infrastructure/config.py`:

### Data Configuration
- **Bar Size**: 1 minute (configurable)
- **Trading Hours**: 9:30 AM - 4:00 PM ET
- **Signal Frequency**: 1 minute

### Strategy Configuration
- **Symbols**: SPY vs VOO
- **Window**: 90 periods
- **Entry Threshold**: 1.5 z-score
- **Exit Threshold**: 0.2 z-score
- **Dynamic Thresholds**: Disabled by default
- **Position Size**: 200 shares per leg

### Execution Configuration
- **IB Connection**: Paper trading port 4002
- **Risk Limits**: 2% daily loss, 5% max drawdown
- **Backtest Duration**: 1 day default

## Strategy Features

### Core Strategy
- **Mean Reversion**: Based on z-score deviations
- **Cointegration**: Tests for stable relationship between SPY/VOO
- **Dynamic Thresholds**: Optional distribution-based thresholds
- **Risk Management**: Position sizing and portfolio constraints

### Performance Analysis
- **Risk Metrics**: Maximum drawdown, VaR, downside deviation
- **Trading Metrics**: Win rate, profit factor, information ratio
- **Visualization**: Spread analysis, PnL charts, distribution analysis

## Usage Examples

### Basic Backtest
```python
from ib_async import IB, util
from strategy_layer.etf_arbitrage import ETFArbitrageStrategy
from execution_layer.backtest import BacktestEngine

# Connect to IB
ib = IB()
await ib.connectAsync("127.0.0.1", 4002, clientId=11)
util.startLoop()

# Create strategy and run backtest
strategy = ETFArbitrageStrategy(ib, data_mode='backtest')
engine = BacktestEngine(strategy)
results = await engine.run_backtest(duration="1 D")
```

### Custom Configuration

You can modify the `config.yaml` file directly, or use the configuration API:

```python
from infrastructure.config import config

# Update strategy parameters
config.update_strategy_config('etf_arbitrage', {
    'entry_threshold': 2.0,
    'exit_threshold': 0.1,
    'use_dynamic_thresholds': True
})

# Use dot notation to access any config value
bar_size = config.get('data.bar_size')
symbols = config.get('strategy.etf_arbitrage.symbols')

# Set configuration values
config.set('data.bar_size', '5 mins')

# Save changes back to file
config.save()
```

## Performance

- **Expected PnL**: ~$200-500 per day (paper trading)
- **Win Rate**: ~60-70%
- **Information Ratio**: 1.0-2.0
- **Maximum Drawdown**: <5%

## Dependencies

```
ib_async
matplotlib
numpy
statsmodels
pandas
pytz
tqdm
aiohttp
scipy
PyYAML
Flask
Flask-Login
Flask-SocketIO
Werkzeug
eventlet
```

## Development

### Adding New Strategies
1. Create a new strategy class in `strategy_layer/`
2. Inherit from `BaseStrategy`
3. Implement required methods: `fetch_data`, `fetch_latest_data`, `process_data`, `generate_signals`, `compute_pnl`
4. Add configuration in `infrastructure/config.py`

### Adding New Analysis
1. Extend `analysis_layer/analyzer.py`
2. Add new metrics to `calculate_risk_metrics`
3. Update visualization in `plot_performance_analysis`

## Notes

- Currently configured for paper trading (port 4002)
- Dynamic thresholds require sufficient data for distribution fitting
- All times are in US/Eastern timezone
- Strategy assumes cointegration between SPY and VOO
- Backtesting uses historical data (`data_mode='backtest'`)
- Live trading uses real-time data (`data_mode='live'`)