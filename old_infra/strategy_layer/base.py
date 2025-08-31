"""
Base strategy class for the trading system.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
from ib_async import IB
from infrastructure.config import config


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, ib: IB, strategy_name: str = 'etf_arbitrage', data_mode: str = 'backtest'):
        """
        Initialize the base strategy.
        
        Args:
            ib: Interactive Brokers connection
            strategy_name: Name of the strategy configuration to use
            data_mode: Data mode - 'backtest' for historical data, 'live' for real-time data
        """
        self.ib = ib
        self.strategy_name = strategy_name
        self.strategy_config = config.get_strategy_config(strategy_name)
        self.data_mode = data_mode
        
        if data_mode not in ['backtest', 'live']:
            raise ValueError(f"Invalid data_mode: {data_mode}. Must be 'backtest' or 'live'")
        
    @abstractmethod
    async def fetch_data(self, duration: str, end_datetime: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data required for the strategy.
        
        Args:
            duration: Duration string for data fetching
            end_datetime: End datetime for data fetching
            
        Returns:
            DataFrame with strategy data
        """
        pass
    
    @abstractmethod
    async def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch the latest data for live trading (real-time bars).
        Only used in 'live' mode.
        
        Returns:
            DataFrame with latest real-time data
        """
        pass
    
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data into strategy signals.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with processed signals
        """
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from processed data.
        
        Args:
            df: Processed data DataFrame
            
        Returns:
            DataFrame with trading signals
        """
        pass
    
    @abstractmethod
    def compute_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PnL and performance metrics.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with PnL calculations
        """
        pass
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current strategy configuration.
        """
        return {
            'strategy_name': self.strategy_name,
            'data_mode': self.data_mode,
            'bar_size': config.DATA_CONFIG['bar_size'],
            'signal_frequency': config.DATA_CONFIG['signal_frequency'],
            'trading_hours': config.DATA_CONFIG['trading_hours'],
            **self.strategy_config
        }
    
    async def run_backtest(self, duration: str, end_datetime: Optional[str] = None) -> pd.DataFrame:
        """
        Run a complete backtest for the strategy.
        
        Args:
            duration: Duration string for backtest
            end_datetime: End datetime for backtest
            
        Returns:
            DataFrame with complete backtest results
        """
        # Ensure we're in backtest mode
        if self.data_mode != 'backtest':
            raise ValueError("run_backtest() can only be called in 'backtest' mode")
        
        # Fetch data
        df = await self.fetch_data(duration, end_datetime)
        
        # Process data
        df = self.process_data(df)
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Compute PnL
        df = self.compute_pnl(df)
        
        return df
    
    async def get_latest_signals(self) -> pd.DataFrame:
        """
        Get the latest signals for live trading.
        Only used in 'live' mode.
        
        Returns:
            DataFrame with latest signals
        """
        # Ensure we're in live mode
        if self.data_mode != 'live':
            raise ValueError("get_latest_signals() can only be called in 'live' mode")
        
        # Fetch latest data
        df = await self.fetch_latest_data()
        
        if df.empty:
            return pd.DataFrame()
        
        # Process data
        df = self.process_data(df)
        
        # Generate signals
        df = self.generate_signals(df)
        
        return df
