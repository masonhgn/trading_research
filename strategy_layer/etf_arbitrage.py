"""
ETF Statistical Arbitrage Strategy implementation.
"""

import datetime as dt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

import sys
import os

# Add the project root to the Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategy_layer.base import BaseStrategy
    from data_layer.data_feed import (
        fetch_pair_data, fetch_historical_bars, start_real_time_bars, cancel_real_time_bars,
        fetch_market_data, filter_trading_hours, detect_trading_hours_automatically,
        create_stock_contract, qualify_contract
    )
    from data_layer.processors import (
        compute_spread, compute_rolling_stats, compute_dynamic_zscore,
        generate_signals, compute_pnl
    )
    from infrastructure.config import config
except ImportError:
    # Fallback for direct execution
    from base import BaseStrategy
    from data_feed import (
        fetch_pair_data, fetch_historical_bars, start_real_time_bars, cancel_real_time_bars,
        fetch_market_data, filter_trading_hours, detect_trading_hours_automatically,
        create_stock_contract, qualify_contract
    )
    from processors import (
        compute_spread, compute_rolling_stats, compute_dynamic_zscore,
        generate_signals, compute_pnl
    )
    from config import config


class ETFArbitrageStrategy(BaseStrategy):
    """
    ETF Statistical Arbitrage Strategy using z-score mean reversion.
    """
    
    def __init__(self, ib, strategy_name: str = 'etf_arbitrage', data_mode: str = 'backtest'):
        super().__init__(ib, strategy_name, data_mode)
        self.sym1, self.sym2 = self.strategy_config['symbols']
        
        # Create contracts for both symbols
        self.contract1 = create_stock_contract(self.sym1)
        self.contract2 = create_stock_contract(self.sym2)
        
        # For live mode: store historical data for rolling calculations
        self.historical_data = pd.DataFrame()
        self.last_update_time = None
        
        # For live mode: real-time bars containers
        self.real_time_bars1 = None
        self.real_time_bars2 = None
        self.last_bar1 = {}
        self.last_bar2 = {}
        
        # For live mode: track latest processed data
        self.latest_processed_data = None
        self.last_signal_time = None
    
    async def fetch_data(self, duration: str, end_datetime: Optional[dt.datetime] = None) -> pd.DataFrame:
        """
        Fetch historical data for backtesting.
        """
        if self.data_mode != 'backtest':
            raise ValueError("fetch_data() is for backtesting only. Use fetch_latest_data() for live mode.")
        
        if end_datetime is None:
            end_datetime = dt.datetime.now()
        
        # Fetch historical data for backtesting using the new data_feed functions
        df = await fetch_pair_data(
            self.ib, 
            self.contract1, 
            self.contract2, 
            duration=duration,
            end_datetime=end_datetime
        )
        
        # Filter trading hours if enabled
        if self.strategy_config['filter_trading_hours']:
            if self.strategy_config['auto_detect_hours']:
                # First compute spread to enable automatic detection
                df = self.process_data(df)
                trading_mask = detect_trading_hours_automatically(
                    df, 
                    self.strategy_config['volatility_threshold']
                )
                df = df[trading_mask].copy()
                # Recompute spread and zscore with filtered data
                df = self.process_data(df)
            else:
                df = filter_trading_hours(df)
        
        return df
    
    async def start_live_data_stream(self):
        """
        Start real-time data streams for live trading.
        """
        if self.data_mode != 'live':
            raise ValueError("start_live_data_stream() is for live mode only.")
        
        try:
            # Use config bar_size for consistency
            bar_size = int(config.DATA_CONFIG['bar_size'].split()[0])  # Extract number from "10 secs"
            
            # Start real-time bars for both symbols
            self.real_time_bars1 = start_real_time_bars(
                self.ib,
                self.contract1,
                self.last_bar1,
                bar_size=bar_size,  # Use config bar_size
                what_to_show="TRADES",
                use_rth=True
            )
            
            self.real_time_bars2 = start_real_time_bars(
                self.ib,
                self.contract2,
                self.last_bar2,
                bar_size=bar_size,  # Use config bar_size
                what_to_show="TRADES",
                use_rth=True
            )
            
            print(f"Started real-time data streams for {self.sym1} and {self.sym2} with {bar_size}-second bars")
            
        except Exception as e:
            print(f"Error starting live data streams: {e}")
            raise
    
    def stop_live_data_stream(self):
        """
        Stop real-time data streams.
        """
        if self.data_mode != 'live':
            return
        
        try:
            if self.real_time_bars1:
                cancel_real_time_bars(self.ib, self.real_time_bars1, self.contract1)
                self.real_time_bars1 = None
            
            if self.real_time_bars2:
                cancel_real_time_bars(self.ib, self.real_time_bars2, self.contract2)
                self.real_time_bars2 = None
                
            print("Stopped real-time data streams")
            
        except Exception as e:
            print(f"Error stopping live data streams: {e}")
    
    async def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest real-time data for live trading.
        """
        if self.data_mode != 'live':
            raise ValueError("fetch_latest_data() is for live mode only. Use fetch_data() for backtesting.")
        
        try:
            # Check if we have recent data from real-time bars
            current_time = dt.datetime.now()
            
            # If we don't have real-time bars set up, try to get market data
            if not self.real_time_bars1 or not self.real_time_bars2:
                # Fallback to market data if real-time bars not available
                data1 = await fetch_market_data(self.ib, self.contract1)
                data2 = await fetch_market_data(self.ib, self.contract2)
                
                if not data1 or not data2:
                    return pd.DataFrame()
                
                # Create a single row DataFrame with current market data
                df = pd.DataFrame([{
                    'datetime': current_time,
                    f'close_{self.sym1}': data1.get('last', data1.get('market_price', 0)),
                    f'close_{self.sym2}': data2.get('last', data2.get('market_price', 0))
                }])
            else:
                # Use real-time bars data
                if not self.last_bar1 or not self.last_bar2:
                    return pd.DataFrame()
                
                # Check if data is recent (within last 15 seconds to account for bar size)
                time_diff1 = (current_time - self.last_bar1.get('timestamp', dt.datetime.min)).total_seconds()
                time_diff2 = (current_time - self.last_bar2.get('timestamp', dt.datetime.min)).total_seconds()
                
                if time_diff1 > 15 or time_diff2 > 15:
                    print("Warning: Real-time data may be stale")
                
                df = pd.DataFrame([{
                    'datetime': current_time,
                    f'close_{self.sym1}': self.last_bar1.get('close', 0),
                    f'close_{self.sym2}': self.last_bar2.get('close', 0)
                }])
            
            # Add to historical data for rolling calculations
            self.historical_data = pd.concat([self.historical_data, df], ignore_index=True)
            
            # Keep only recent data for rolling window calculations (window + buffer)
            window_size = self.strategy_config['window'] + 50  # Keep extra data for safety
            if len(self.historical_data) > window_size:
                self.historical_data = self.historical_data.tail(window_size).copy()
            
            return df
            
        except Exception as e:
            print(f"Error fetching latest data: {e}")
            return pd.DataFrame()
    
    async def fetch_historical_data_for_live(self, duration: str = "5 D") -> pd.DataFrame:
        """
        Fetch historical data for live trading initialization.
        """
        if self.data_mode != 'live':
            raise ValueError("fetch_historical_data_for_live() is for live mode only.")
        
        try:
            # Fetch historical data to initialize rolling calculations
            df = await fetch_pair_data(
                self.ib,
                self.contract1,
                self.contract2,
                duration=duration,
                end_datetime=dt.datetime.now()
            )
            
            # Filter trading hours if enabled
            if self.strategy_config['filter_trading_hours']:
                if self.strategy_config['auto_detect_hours']:
                    # First compute spread to enable automatic detection
                    df = self.process_data(df)
                    trading_mask = detect_trading_hours_automatically(
                        df, 
                        self.strategy_config['volatility_threshold']
                    )
                    df = df[trading_mask].copy()
                    # Recompute spread and zscore with filtered data
                    df = self.process_data(df)
                else:
                    df = filter_trading_hours(df)
            
            # Store as historical data for live trading
            self.historical_data = df.copy()
            print(f"Initialized historical data with {len(df)} bars for live trading")
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for live trading: {e}")
            return pd.DataFrame()
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data into strategy signals.
        """
        if df.empty:
            return df
        
        # For live mode with single-row data, use historical data for rolling calculations
        if self.data_mode == 'live' and len(df) == 1 and not self.historical_data.empty:
            # Compute spread on current data
            df = compute_spread(df, self.sym1, self.sym2)
            
            # Use historical data for rolling statistics
            historical_df = self.historical_data.copy()
            historical_df = compute_spread(historical_df, self.sym1, self.sym2)
            historical_df = compute_rolling_stats(historical_df, self.strategy_config['window'])
            
            # Get the latest rolling stats from historical data
            if len(historical_df) > 0:
                latest_stats = historical_df.iloc[-1]  # Get last row, not tail(1)
                
                # Apply latest stats to current data
                for col in ['rolling_mean', 'rolling_std', 'upper', 'lower']:
                    if col in latest_stats.index:
                        df[col] = latest_stats[col]
                
                # Compute z-score for current data
                if 'rolling_std' in df.columns and df['rolling_std'].iloc[0] > 0:
                    df['zscore'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']
                else:
                    df['zscore'] = 0.0
            else:
                # No historical data available, set default values
                df['rolling_mean'] = 0.0
                df['rolling_std'] = 1.0
                df['upper'] = 1.0
                df['lower'] = -1.0
                df['zscore'] = 0.0
        else:
            # For backtest mode or multi-row data
            # Compute spread
            df = compute_spread(df, self.sym1, self.sym2)
            
            # Compute rolling statistics
            df = compute_rolling_stats(df, self.strategy_config['window'])
        
        # Add dynamic z-score calculation if enabled
        if self.strategy_config['use_dynamic_thresholds']:
            df = compute_dynamic_zscore(
                df, 
                self.strategy_config['window'],
                self.strategy_config['distribution_type']
            )
            print(f"Using dynamic thresholds (distribution: {self.strategy_config['distribution_type']}, confidence: {self.strategy_config['confidence_level']})")
        else:
            print(f"Using fixed thresholds (entry: {self.strategy_config['entry_threshold']}, exit: {self.strategy_config['exit_threshold']})")
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from processed data.
        """
        return generate_signals(
            df,
            self.strategy_config['use_dynamic_thresholds'],
            self.strategy_config['entry_threshold'],
            self.strategy_config['exit_threshold']
        )
    
    def compute_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PnL and performance metrics.
        """
        return compute_pnl(df)
    
    async def get_latest_signals(self) -> pd.DataFrame:
        """
        Get the latest signals for live trading with proper timing control.
        """
        if self.data_mode != 'live':
            raise ValueError("get_latest_signals() is for live mode only.")
        
        current_time = dt.datetime.now()
        
        # Check if enough time has passed since last signal (respect bar size)
        bar_size_seconds = int(config.DATA_CONFIG['bar_size'].split()[0])
        if (self.last_signal_time and 
            (current_time - self.last_signal_time).total_seconds() < bar_size_seconds):
            # Return cached result if not enough time has passed
            return self.latest_processed_data if self.latest_processed_data is not None else pd.DataFrame()
        
        # Fetch latest data
        df = await self.fetch_latest_data()
        
        if df.empty:
            return pd.DataFrame()
        
        # Process data
        df = self.process_data(df)
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Cache the result and update timing
        self.latest_processed_data = df
        self.last_signal_time = current_time
        
        return df
    
    def run_cointegration_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run cointegration tests on the pair.
        """
        score, pvalue, _ = coint(df[f'close_{self.sym1}'], df[f'close_{self.sym2}'])
        adf_stat, adf_pval, _, _, crit_vals, _ = adfuller(df['spread'])
        
        return {
            "coint_score": score,
            "coint_pval": pvalue,
            "adf_stat": adf_stat,
            "adf_pval": adf_pval,
            "crit_vals": crit_vals
        }
    
    def plot_spread(self, df: pd.DataFrame):
        """
        Plot the spread with rolling statistics.
        """
        # import matplotlib.pyplot as plt  # Temporarily disabled for web interface
        
        plt.figure(figsize=(15, 6))
        
        # Create a continuous time index for better visualization
        df_plot = df.copy()
        df_plot['time_index'] = range(len(df_plot))
        
        # Plot using the numeric index explicitly
        x_values = df_plot['time_index'].values
        
        plt.plot(x_values, df_plot['spread'].values, 'b-', linewidth=2, label='Spread')
        plt.plot(x_values, df_plot['rolling_mean'].values, 'r--', linewidth=1.5, label='Rolling Mean')
        plt.plot(x_values, df_plot['upper'].values, 'g--', linewidth=1, label='+1σ')
        plt.plot(x_values, df_plot['lower'].values, 'orange', linestyle='--', linewidth=1, label='-1σ')
        
        # Add time labels at regular intervals
        label_interval = max(1, len(df_plot) // 10)  # Show ~10 labels
        label_indices = range(0, len(df_plot), label_interval)
        label_times = [df_plot.iloc[i]['datetime'].strftime('%m-%d %H:%M') for i in label_indices]
        
        plt.xticks(label_indices, label_times, rotation=45)
        
        # Ensure x-axis is treated as numeric, not datetime
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: ''))
        
        plt.title(f"Spread: {self.sym1} - {self.sym2}")
        plt.xlabel("Time")
        plt.ylabel("Spread ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_pnl(self, df: pd.DataFrame):
        """
        Plot the cumulative PnL.
        """
        # import matplotlib.pyplot as plt  # Temporarily disabled for web interface
        
        plt.figure(figsize=(15, 6))
        
        # Create a continuous time index for better visualization
        df_plot = df.copy()
        df_plot['time_index'] = range(len(df_plot))
        
        # Plot using the numeric index explicitly
        x_values = df_plot['time_index'].values
        y_values = df_plot['cumulative_pnl'].values
        
        plt.plot(x_values, y_values, 'b-', linewidth=2, label='Cumulative PnL')
        
        # Add time labels at regular intervals
        label_interval = max(1, len(df_plot) // 10)  # Show ~10 labels
        label_indices = range(0, len(df_plot), label_interval)
        label_times = [df_plot.iloc[i]['datetime'].strftime('%m-%d %H:%M') for i in label_indices]
        
        plt.xticks(label_indices, label_times, rotation=45)
        
        # Ensure x-axis is treated as numeric, not datetime
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: ''))
        
        plt.title(f"Cumulative PnL: {self.sym1} - {self.sym2}")
        plt.xlabel("Time")
        plt.ylabel("PnL ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_distribution_analysis(self, df: pd.DataFrame):
        """
        Plot the fitted distribution vs normal assumption for the spread.
        """
        # import matplotlib.pyplot as plt  # Temporarily disabled for web interface
        from scipy import stats
        
        if not self.strategy_config['use_dynamic_thresholds'] or 'dynamic_zscore' not in df.columns:
            print("Dynamic thresholds not enabled or not computed yet.")
            return
        
        # Get recent spread data for analysis
        recent_spread = df['spread'].tail(self.strategy_config['window']).dropna()
        
        if len(recent_spread) < 50:
            print("Not enough data for distribution analysis.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Histogram with fitted distributions
        ax1.hist(recent_spread, bins=30, density=True, alpha=0.7, color='lightblue', label='Actual Data')
        
        # Fit and plot normal distribution
        mu, sigma = stats.norm.fit(recent_spread)
        x = np.linspace(recent_spread.min(), recent_spread.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal (μ={mu:.4f}, σ={sigma:.4f})')
        
        # Fit and plot t-distribution
        try:
            params = stats.t.fit(recent_spread)
            ax1.plot(x, stats.t.pdf(x, *params), 'g-', linewidth=2, label=f't-dist (df={params[0]:.2f})')
        except:
            pass
        
        ax1.set_title('Spread Distribution Analysis')
        ax1.set_xlabel('Spread Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot to compare with normal distribution
        stats.probplot(recent_spread, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot vs Normal Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print distribution statistics
        print(f"\nDistribution Analysis for {self.sym1} - {self.sym2}:")
        print(f"Sample size: {len(recent_spread)}")
        print(f"Mean: {recent_spread.mean():.6f}")
        print(f"Std: {recent_spread.std():.6f}")
        print(f"Skewness: {stats.skew(recent_spread):.4f}")
        print(f"Kurtosis: {stats.kurtosis(recent_spread):.4f}")
        
        # Test for normality
        _, p_value = stats.normaltest(recent_spread)
        print(f"Normality test p-value: {p_value:.4f}")
        print(f"Data is {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)")
    
    def plot_dynamic_thresholds(self, df: pd.DataFrame):
        """
        Plot the dynamic thresholds over time.
        """
        # import matplotlib.pyplot as plt  # Temporarily disabled for web interface
        
        if not self.strategy_config['use_dynamic_thresholds'] or 'dynamic_entry_threshold' not in df.columns:
            print("Dynamic thresholds not enabled or not computed yet.")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Spread and z-scores
        df_plot = df.dropna(subset=['dynamic_zscore']).copy()
        df_plot['time_index'] = range(len(df_plot))
        
        ax1.plot(df_plot['time_index'], df_plot['spread'], 'b-', linewidth=1, label='Spread', alpha=0.7)
        ax1.set_ylabel('Spread ($)')
        ax1.set_title(f'Spread: {self.sym1} - {self.sym2}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Z-scores and dynamic thresholds
        ax2.plot(df_plot['time_index'], df_plot['dynamic_zscore'], 'b-', linewidth=1, label='Dynamic Z-Score', alpha=0.7)
        ax2.plot(df_plot['time_index'], df_plot['dynamic_entry_threshold'], 'r--', linewidth=1, label='Entry Threshold', alpha=0.8)
        ax2.plot(df_plot['time_index'], -df_plot['dynamic_entry_threshold'], 'r--', linewidth=1, alpha=0.8)
        ax2.plot(df_plot['time_index'], df_plot['dynamic_exit_threshold'], 'g--', linewidth=1, label='Exit Threshold', alpha=0.8)
        ax2.plot(df_plot['time_index'], -df_plot['dynamic_exit_threshold'], 'g--', linewidth=1, alpha=0.8)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Z-Score')
        ax2.set_xlabel('Time')
        ax2.set_title('Dynamic Z-Scores and Thresholds')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
