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
    
    # First pass: generate basic signals without portfolio constraints
    for is_long, is_short, is_exit in zip(df['long_entry'], df['short_entry'], df['exit']):
        if is_long and position == 0:
            position = 1
        elif is_short and position == 0:
            position = -1
        elif is_exit:
            position = 0
        positions.append(position)
    
    df['position'] = positions
    return df


def compute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PnL and related metrics with proper spread trading logic.
    """
    strategy_config = config.get_strategy_config()
    slippage = strategy_config.get('slippage', 0.001)  # Default 0.1%
    fee = strategy_config.get('fee', 0.0005)  # Default 0.05%
    initial_capital = strategy_config.get('initial_capital', 100000)
    symbols = strategy_config['symbols']
    margin_requirement = strategy_config.get('margin_requirement', 1.5)  # 150% for short positions
    
    df = df.copy()
    
    # Initialize tracking variables
    shares = []
    portfolio_values = []
    raw_pnl_list = []
    cost_list = []
    net_pnl_list = []
    current_portfolio = initial_capital
    
    for i, (pos, price1, price2) in enumerate(zip(df['position'], df[f'close_{symbols[0]}'], df[f'close_{symbols[1]}'])):
        # Calculate position size based on current position type
        if pos == 1:  # Long spread: Buy sym1, Sell sym2 (profit when spread widens)
            # For long spread, we need capital to buy sym1 and margin for short sym2
            # Conservative approach: use total exposure
            total_exposure = price1 + (price2 * margin_requirement)
            current_shares = int(current_portfolio / total_exposure) if total_exposure > 0 else 0
            
        elif pos == -1:  # Short spread: Sell sym1, Buy sym2 (profit when spread narrows)
            # For short spread, we need capital to buy sym2 and margin for short sym1
            total_exposure = price2 + (price1 * margin_requirement)
            current_shares = int(current_portfolio / total_exposure) if total_exposure > 0 else 0
            
        else:  # No position
            current_shares = 0
        
        shares.append(current_shares)
        
        # Calculate PnL for this period
        if i == 0:
            raw_pnl_list.append(0)
            cost_list.append(0)
            net_pnl_list.append(0)
        else:
            # Get previous period data
            prev_pos = df['position'].iloc[i-1]
            prev_shares = shares[i-1]
            spread_change = df['spread'].iloc[i] - df['spread'].iloc[i-1]
            
            # Calculate raw PnL from spread change
            if prev_pos == 1:  # Was long spread
                # Long spread profits when spread increases
                raw_pnl = spread_change * prev_shares
            elif prev_pos == -1:  # Was short spread
                # Short spread profits when spread decreases
                raw_pnl = -spread_change * prev_shares
            else:  # Was flat
                raw_pnl = 0
            
            raw_pnl_list.append(raw_pnl)
            
            # Calculate trading costs when position changes
            position_changed = pos != prev_pos
            costs = 0
            
            if position_changed:
                # Calculate the number of shares traded
                if prev_pos == 0:  # Entering new position
                    shares_traded = current_shares
                elif pos == 0:  # Exiting position
                    shares_traded = prev_shares
                else:  # Changing position (rare but possible)
                    shares_traded = max(current_shares, prev_shares)
                
                if shares_traded > 0:
                    # Cost for both legs of the spread trade
                    leg1_cost = shares_traded * price1 * (slippage + fee)
                    leg2_cost = shares_traded * price2 * (slippage + fee)
                    costs = leg1_cost + leg2_cost
            
            cost_list.append(costs)
            
            # Net PnL
            net_pnl = raw_pnl - costs
            net_pnl_list.append(net_pnl)
            
            # Update portfolio value
            current_portfolio += net_pnl
        
        # Ensure portfolio doesn't go negative
        current_portfolio = max(current_portfolio, initial_capital * 0.01)  # Minimum 1% of initial capital
        portfolio_values.append(current_portfolio)
    
    # Add all calculated columns to dataframe
    df['shares'] = shares
    df['portfolio_value'] = portfolio_values
    df['raw_pnl'] = raw_pnl_list
    df['cost'] = cost_list
    df['net_pnl'] = net_pnl_list
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    
    # Add additional analysis columns
    df['spread_change'] = df['spread'].diff()
    df['position_prev'] = df['position'].shift(1).fillna(0)
    df['shares_prev'] = df['shares'].shift(1).fillna(0)
    df['trade'] = (df['position'] != df['position_prev']).astype(int)
    df['returns'] = df['net_pnl'] / df['portfolio_value'].shift(1)
    df['cumulative_returns'] = (df['portfolio_value'] / initial_capital) - 1
    
    return df


def validate_pnl_calculation(df: pd.DataFrame, debug_periods: int = 5) -> None:
    """
    Validate PnL calculations by printing detailed breakdown for first few periods.
    """
    print("=== PnL Calculation Validation ===")
    print(f"Initial Capital: ${df['portfolio_value'].iloc[0]:,.2f}")
    
    for i in range(min(debug_periods, len(df))):
        if i == 0:
            continue
            
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        print(f"\nPeriod {i}:")
        print(f"  Position: {prev_row['position']:.0f} -> {row['position']:.0f}")
        print(f"  Shares: {prev_row['shares']:.0f} -> {row['shares']:.0f}")
        print(f"  Spread: {prev_row['spread']:.4f} -> {row['spread']:.4f} (change: {row['spread_change']:.4f})")
        print(f"  Raw PnL: ${row['raw_pnl']:.2f}")
        print(f"  Costs: ${row['cost']:.2f}")
        print(f"  Net PnL: ${row['net_pnl']:.2f}")
        print(f"  Portfolio: ${prev_row['portfolio_value']:.2f} -> ${row['portfolio_value']:.2f}")
        
        # Manual verification
        if prev_row['position'] != 0:
            expected_raw_pnl = prev_row['position'] * row['spread_change'] * prev_row['shares']
            print(f"  Expected Raw PnL: ${expected_raw_pnl:.2f} {'✓' if abs(expected_raw_pnl - row['raw_pnl']) < 0.01 else '✗'}")


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for the strategy.
    """
    # Basic metrics
    total_return = df['cumulative_returns'].iloc[-1]
    total_trades = df['trade'].sum()
    
    # Risk metrics
    returns = df['returns'].dropna()
    if len(returns) > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = (df['cumulative_returns'].cummax() - df['cumulative_returns']).max()
        volatility = returns.std() * np.sqrt(252)
        win_rate = (returns > 0).mean()
    else:
        sharpe_ratio = max_drawdown = volatility = win_rate = 0
    
    # Trading metrics
    avg_trade_pnl = df[df['trade'] == 1]['net_pnl'].mean() if total_trades > 0 else 0
    total_costs = df['cost'].sum()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_trade_pnl': avg_trade_pnl,
        'total_costs': total_costs,
        'final_portfolio_value': df['portfolio_value'].iloc[-1]
    }