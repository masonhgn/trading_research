"""
Script to run backtests using the new architecture.
"""

import asyncio
import datetime as dt
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ib_async import IB, util
from infrastructure.config import config
from strategy_layer.etf_arbitrage import ETFArbitrageStrategy
from execution_layer.backtest import BacktestEngine


async def main():
    """
    Main function to run ETF arbitrage backtest.
    """
    # Validate configuration
    config.validate_config()
    print()
    
    # Connect to Interactive Brokers
    ib = IB()
    await ib.connectAsync(
        config.EXECUTION_CONFIG['ib_connection']['host'],
        config.EXECUTION_CONFIG['ib_connection']['paper_port'],
        clientId=config.EXECUTION_CONFIG['ib_connection']['client_id']
    )
    util.startLoop()
    
    try:
        # Create strategy instance
        strategy = ETFArbitrageStrategy(ib, data_mode='backtest')
        
        # Create backtest engine
        engine = BacktestEngine(strategy)
        
        # Run backtest
        results = await engine.run_backtest(
            duration="1 D",
            end_datetime=dt.datetime(2025, 8, 14, 16),
            show_plots=True,
            show_analysis=True
        )
        
        print("\nBacktest completed successfully!")
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        raise
    finally:
        ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
