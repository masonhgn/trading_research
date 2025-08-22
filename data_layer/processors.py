"""
Data processing utilities for the trading system.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict
from infrastructure.config import config


def compute_spread(df: pd.DataFrame, sym1: str, sym2: str) -> pd.DataFrame:
    """
    Compute the spread between two symbols.
    """
    df = df.copy()
    df['spread'] = df[f'close_{sym1}'] - df[f'close_{sym2}']
    return df


def compute_rolling_stats(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """
    Compute rolling statistics for the spread.
    """
    if window is None:
        window = config.get_strategy_config()['window']
    
    df = df.copy()
    df['rolling_mean'] = df['spread'].rolling(window=window, min_periods=window).mean()
    df['rolling_std'] = df['spread'].rolling(window=window, min_periods=window).std()
    df['zscore'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']
    df['upper'] = df['rolling_mean'] + df['rolling_std']
    df['lower'] = df['rolling_mean'] - df['rolling_std']
    return df


def fit_spread_distribution(spread_series: pd.Series, distribution_type: str = 't') -> Tuple:
    """
    Fit the spread data to a specified distribution.
    """
    if distribution_type == 't':
        # Fit t-distribution (more robust to fat tails)
        params = stats.t.fit(spread_series)
        fitted_dist = stats.t(*params)
    elif distribution_type == 'normal':
        # Fit normal distribution
        params = stats.norm.fit(spread_series)
        fitted_dist = stats.norm(*params)
    elif distribution_type == 'skewnorm':
        # Fit skewed normal distribution
        params = stats.skewnorm.fit(spread_series)
        fitted_dist = stats.skewnorm(*params)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    return params, fitted_dist


def calculate_dynamic_thresholds(
    spread_series: pd.Series,
    confidence_level: float = 0.95,
    distribution_type: str = 't'
) -> Tuple[float, float]:
    """
    Calculate dynamic entry and exit thresholds based on fitted distribution.
    """
    if len(spread_series) < 60:  # Minimum window
        return 1.5, 0.2  # Default thresholds
    
    try:
        params, fitted_dist = fit_spread_distribution(spread_series, distribution_type)
        
        # Calculate percentiles for entry and exit
        entry_percentile = 1 - confidence_level  # e.g., 0.05 for 95% confidence
        exit_percentile = 0.5  # Median (mean reversion target)
        
        # Get threshold values from fitted distribution
        entry_value = fitted_dist.ppf(entry_percentile)
        exit_value = fitted_dist.ppf(exit_percentile)
        
        # Convert to z-scores relative to the fitted distribution
        entry_zscore = (entry_value - fitted_dist.mean()) / fitted_dist.std()
        exit_zscore = (exit_value - fitted_dist.mean()) / fitted_dist.std()
        
        # Ensure reasonable bounds
        entry_zscore = max(1.0, min(3.0, abs(entry_zscore)))
        exit_zscore = max(0.1, min(0.5, abs(exit_zscore)))
        
        return entry_zscore, exit_zscore
        
    except Exception as e:
        print(f"Error calculating dynamic thresholds: {e}")
        # Fall back to fixed thresholds
        return 1.5, 0.2


def compute_dynamic_zscore(df: pd.DataFrame, window: int = None, distribution_type: str = 't') -> pd.DataFrame:
    """
    Compute z-scores using fitted distribution instead of normal assumption.
    """
    if window is None:
        window = config.get_strategy_config()['window']
    
    df = df.copy()
    dynamic_zscores = []
    entry_thresholds = []
    exit_thresholds = []
    
    for i in range(window, len(df)):
        window_spread = df['spread'].iloc[i-window:i]
        
        try:
            # Fit distribution to rolling window
            params, fitted_dist = fit_spread_distribution(window_spread, distribution_type)
            
            # Calculate current spread's percentile in fitted distribution
            current_spread = df['spread'].iloc[i]
            percentile = fitted_dist.cdf(current_spread)
            
            # Convert to z-score (inverse normal CDF of percentile)
            dynamic_zscore = stats.norm.ppf(percentile)
            
            # Calculate dynamic thresholds for this window
            entry_thresh, exit_thresh = calculate_dynamic_thresholds(window_spread, distribution_type=distribution_type)
            
        except Exception as e:
            # Fall back to traditional z-score calculation
            mean = window_spread.mean()
            std = window_spread.std()
            dynamic_zscore = (current_spread - mean) / std if std > 0 else 0
            entry_thresh, exit_thresh = 1.5, 0.2
        
        dynamic_zscores.append(dynamic_zscore)
        entry_thresholds.append(entry_thresh)
        exit_thresholds.append(exit_thresh)
    
    # Pad the beginning with NaN values
    df['dynamic_zscore'] = [np.nan] * window + dynamic_zscores
    df['dynamic_entry_threshold'] = [np.nan] * window + entry_thresholds
    df['dynamic_exit_threshold'] = [np.nan] * window + exit_thresholds
    
    return df


def generate_signals(
    df: pd.DataFrame,
    use_dynamic_thresholds: bool = False,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.2
) -> pd.DataFrame:
    """
    Generate trading signals based on z-scores.
    """
    df = df.copy()
    
    if use_dynamic_thresholds and 'dynamic_zscore' in df.columns:
        # Use dynamic thresholds
        df['long_entry'] = df['dynamic_zscore'] < -df['dynamic_entry_threshold']
        df['short_entry'] = df['dynamic_zscore'] > df['dynamic_entry_threshold']
        df['exit'] = df['dynamic_zscore'].abs() < df['dynamic_exit_threshold']
    else:
        # Use fixed thresholds
        df['long_entry'] = df['zscore'] < -entry_threshold
        df['short_entry'] = df['zscore'] > entry_threshold
        df['exit'] = df['zscore'].abs() < exit_threshold

    # Generate position signals
    position = 0
    positions = []
    shares = []
    
    # First pass: generate basic signals without portfolio constraints
    for is_long, is_short, is_exit in zip(df['long_entry'], df['short_entry'], df['exit']):
        if is_long and position == 0:
            position = 1
        elif is_short and position == 0:
            position = -1
        elif is_exit:
            position = 0
        positions.append(position)
    
    # Second pass: apply portfolio constraints
    strategy_config = config.get_strategy_config()
    initial_capital = strategy_config['initial_capital']
    shares_per_leg = strategy_config['shares_per_leg']
    
    constrained_shares = []
    portfolio_value = initial_capital
    
    for i, (pos, price1, price2) in enumerate(zip(positions, df[f'close_{strategy_config["symbols"][0]}'], df[f'close_{strategy_config["symbols"][1]}'])):
        current_shares = 0
        
        if pos == 1:  # Long position
            # For long positions: buy SPY, sell VOO
            max_affordable_shares = int(portfolio_value / price1)
            current_shares = min(shares_per_leg, max_affordable_shares)
            
        elif pos == -1:  # Short position
            # For short positions: sell SPY, buy VOO  
            max_affordable_shares = int(portfolio_value / price2)
            current_shares = min(shares_per_leg, max_affordable_shares)
            
        # If we can't afford the position, set position to 0
        if current_shares == 0 and pos != 0:
            positions[i] = 0
            
        constrained_shares.append(current_shares)
    
    df['position'] = positions
    df['shares'] = constrained_shares
    return df


def compute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PnL and related metrics.
    """
    strategy_config = config.get_strategy_config()
    shares_per_leg = strategy_config['shares_per_leg']
    slippage = strategy_config['slippage']
    fee = strategy_config['fee']
    
    df = df.copy()
    df['spread_change'] = df['spread'].diff()
    df['shifted_position'] = df['position'].shift()
    df['shifted_shares'] = df['shares'].shift()
    
    # Calculate PnL using dynamic shares
    df['raw_pnl'] = df['shifted_position'] * df['spread_change'] * df['shifted_shares']
    df['trade'] = df['position'].diff().fillna(0).abs() > 0
    df['cost'] = df['trade'] * 2 * (slippage + fee) * df['shares']
    df['net_pnl'] = df['raw_pnl'] - df['cost']
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    
    # Calculate portfolio value over time
    df['portfolio_value'] = strategy_config['initial_capital'] + df['cumulative_pnl']
    
    return df
