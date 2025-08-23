"""
Script to run backtests using the new architecture.
"""

import asyncio
import datetime as dt
import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ib_async import IB, util
from infrastructure.config import config
from strategy_layer.etf_arbitrage import ETFArbitrageStrategy
from execution_layer.backtest import BacktestEngine


async def run_backtest(duration: str = "1 D", end_datetime: dt.datetime = None):
    """
    Run backtest using live IB connection.
    """
    print("Running ETF Arbitrage backtest...")
    
    # Validate configuration
    config.validate_config()
    print()
    
    # Connect to Interactive Brokers
    ib = IB()
    
    try:
        await ib.connectAsync(
            config.EXECUTION_CONFIG['ib_connection']['host'],
            config.EXECUTION_CONFIG['ib_connection']['paper_port'],
            clientId=config.EXECUTION_CONFIG['ib_connection']['client_id']
        )
        util.startLoop()
        
        print(f"Connected to IB TWS/Gateway successfully")
        
        # Create strategy instance
        strategy = ETFArbitrageStrategy(ib, data_mode='backtest')
        
        # Create backtest engine
        engine = BacktestEngine(strategy)
        
        # Run backtest
        results = await engine.run_backtest(
            duration=duration,
            end_datetime=end_datetime,
            show_plots=True,
            show_analysis=True
        )
        
        print("\nBacktest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        raise
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("Disconnected from IB")


async def main():
    """
    Main function to run ETF arbitrage backtest.
    """
    parser = argparse.ArgumentParser(description='Run ETF Arbitrage Backtest')
    parser.add_argument('--duration', type=str, default='1 D',
                       help='Duration for backtest (e.g., "1 D", "5 D", "1 M")')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    # Parse end datetime if provided
    end_datetime = None
    if args.end_date:
        try:
            end_datetime = dt.datetime.strptime(args.end_date, '%Y-%m-%d')
            # Set to end of trading day
            end_datetime = end_datetime.replace(hour=16, minute=0, second=0, microsecond=0)
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD")
            return
    
    try:
        # Run backtest with live IB connection
        results = await run_backtest(args.duration, end_datetime)
        
        # Print summary
        if results and 'analysis_summary' in results:
            print("\n" + "="*60)
            print("BACKTEST SUMMARY")
            print("="*60)
            summary = results['analysis_summary']
            print(f"Total Return: {summary.get('total_return', 0):.2%}")
            print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
            print(f"Total Trades: {summary.get('total_trades', 0)}")
            print(f"Win Rate: {summary.get('win_rate', 0):.2%}")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
    except Exception as e:
        print(f"Fatal error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
