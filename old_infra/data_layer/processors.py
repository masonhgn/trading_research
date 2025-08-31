"""
Data processing utilities for the trading system - FIXED VERSION
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict
from infrastructure.config import config
import sys



from statsmodels.api import OLS, add_constant

def compute_hedge_ratio(Y: pd.Series, X: pd.Series) -> float:
    X = add_constant(X)
    model = OLS(Y, X).fit()
    return model.params.iloc[1]  # Return the slope coefficient (hedge ratio) - use iloc to avoid deprecation warning


def compute_holdings_percentage(y_shares: int, x_shares: int, y_price: float, x_price: float) -> tuple:
    """
    Compute the percentage holdings for a pairs trade.
    Similar to the reference code's computeHoldingsPct function.
    
    Args:
        y_shares: Number of shares for stock Y (first stock)
        x_shares: Number of shares for stock X (second stock)
        y_price: Price of stock Y
        x_price: Price of stock X
    
    Returns:
        tuple: (y_percentage, x_percentage) of total notional
    """
    y_dollars = y_shares * y_price
    x_dollars = x_shares * x_price
    total_notional = abs(y_dollars) + abs(x_dollars)
    
    if total_notional == 0:
        return (0.0, 0.0)
    
    y_percentage = y_dollars / total_notional
    x_percentage = x_dollars / total_notional
    
    return (y_percentage, x_percentage)


def compute_spread(df: pd.DataFrame, sym1: str, sym2: str, use_rolling_hedge: bool = False, window: int = 60) -> pd.DataFrame:
    df = df.copy()
    
    if use_rolling_hedge and len(df) >= window:
        # Compute rolling hedge ratio
        hedge_ratios = []
        for i in range(window, len(df)):
            Y = df[f'close_{sym1}'].iloc[i-window:i]
            X = df[f'close_{sym2}'].iloc[i-window:i]
            try:
                hedge_ratios.append(compute_hedge_ratio(Y, X))
            except:
                # Fallback to previous hedge ratio or 1.0
                prev_ratio = hedge_ratios[-1] if hedge_ratios else 1.0
                hedge_ratios.append(prev_ratio)
        
        # Pad the beginning with the first computed hedge ratio
        df['hedge_ratio'] = [hedge_ratios[0]] * window + hedge_ratios
        df['spread'] = df[f'close_{sym1}'] - df['hedge_ratio'] * df[f'close_{sym2}']
    else:
        # Use static hedge ratio computed on full dataset
        hedge_ratio = compute_hedge_ratio(df[f'close_{sym1}'], df[f'close_{sym2}'])
        df['spread'] = df[f'close_{sym1}'] - hedge_ratio * df[f'close_{sym2}']
        df['hedge_ratio'] = hedge_ratio
    
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
    Compute PnL and related metrics with proper pairs trading logic.
    Only trade when position changes to avoid unnecessary transaction costs.
    Uses IBKR-style per-share commissions plus realistic slippage.
    """
    strategy_config = config.get_strategy_config()
    initial_capital = strategy_config.get('initial_capital', 100000)
    symbols = strategy_config['symbols']
    max_reasonable_shares = strategy_config.get('max_shares_per_trade', 10000)
    
    # IBKR-style realistic costs
    commission_per_share = 0.005  # $0.005 per share (IBKR Fixed rate)
    slippage_rate = 0.0001       # 0.01% for liquid stocks
    
    df = df.copy()
    
    # Initialize tracking variables
    portfolio_values = []
    raw_pnl_list = []
    cost_list = []
    net_pnl_list = []
    shares_stock1_list = []
    shares_stock2_list = []
    
    current_portfolio = initial_capital
    prev_shares_stock1 = 0
    prev_shares_stock2 = 0
    
    for i, (pos, price1, price2) in enumerate(zip(df['position'], df[f'close_{symbols[0]}'], df[f'close_{symbols[1]}'])):
        
        # Validate prices
        if price1 <= 0 or price2 <= 0:
            # Skip this period with invalid prices
            raw_pnl = 0
            costs = 0
            new_shares_stock1 = prev_shares_stock1
            new_shares_stock2 = prev_shares_stock2
        else:
            if i == 0:
                # First period - no PnL to calculate
                raw_pnl = 0
                prev_pos = 0
            else:
                # Step 1: Calculate PnL from positions held since last bar
                prev_price1 = df[f'close_{symbols[0]}'].iloc[i-1]
                prev_price2 = df[f'close_{symbols[1]}'].iloc[i-1]
                prev_pos = df['position'].iloc[i-1]
                
                stock1_pnl = (price1 - prev_price1) * prev_shares_stock1
                stock2_pnl = (price2 - prev_price2) * prev_shares_stock2
                raw_pnl = stock1_pnl + stock2_pnl
            
            # Check if position changed
            position_changed = (i == 0) or (pos != prev_pos)
            
            if position_changed:
                # Step 2: Use previous period's portfolio value for position sizing
                if i == 0:
                    sizing_capital = initial_capital
                else:
                    sizing_capital = portfolio_values[i-1]  # Use previous portfolio value
                
                capital_per_leg = sizing_capital / 2
                
                # Get hedge ratio for this period (use the most recent available)
                hedge_ratio = df['hedge_ratio'].iloc[i] if 'hedge_ratio' in df.columns else 1.0
                
                # Convert position signal to individual stock positions using hedge ratio
                # Following the reference code pattern: y_target_shares = 1, x_target_shares = -hedge for long
                if pos == 1:  # Long the spread (long stock1, short stock2)
                    y_target_shares = 1
                    x_target_shares = -hedge_ratio
                elif pos == -1:  # Short the spread (short stock1, long stock2)  
                    y_target_shares = -1
                    x_target_shares = hedge_ratio
                else:  # Flat
                    y_target_shares = 0
                    x_target_shares = 0
                
                # Calculate share amounts based on target shares and available capital
                # Use a base position size and scale by the hedge ratio
                base_position_size = int(capital_per_leg / price1)
                base_position_size = min(base_position_size, max_reasonable_shares)
                
                if y_target_shares != 0:
                    new_shares_stock1 = y_target_shares * base_position_size
                else:
                    new_shares_stock1 = 0
                    
                if x_target_shares != 0:
                    # For stock2, use the hedge ratio to determine the number of shares
                    # This ensures proper hedging based on the regression relationship
                    new_shares_stock2 = x_target_shares * base_position_size
                    # Ensure we don't exceed reasonable limits
                    if abs(new_shares_stock2) > max_reasonable_shares:
                        new_shares_stock2 = max_reasonable_shares if new_shares_stock2 > 0 else -max_reasonable_shares
                else:
                    new_shares_stock2 = 0
                
                # Calculate realistic transaction costs only when position changes
                if i == 0:
                    # First period - only opening costs
                    opening_commission = (abs(new_shares_stock1) + abs(new_shares_stock2)) * commission_per_share
                    opening_slippage = (abs(new_shares_stock1) * price1 + abs(new_shares_stock2) * price2) * slippage_rate
                    costs = opening_commission + opening_slippage
                else:
                    # Calculate closing costs for previous position
                    closing_commission = (abs(prev_shares_stock1) + abs(prev_shares_stock2)) * commission_per_share
                    closing_slippage = (abs(prev_shares_stock1) * price1 + abs(prev_shares_stock2) * price2) * slippage_rate
                    closing_costs = closing_commission + closing_slippage
                    
                    # Calculate opening costs for new position
                    opening_commission = (abs(new_shares_stock1) + abs(new_shares_stock2)) * commission_per_share
                    opening_slippage = (abs(new_shares_stock1) * price1 + abs(new_shares_stock2) * price2) * slippage_rate
                    opening_costs = opening_commission + opening_slippage
                    
                    costs = closing_costs + opening_costs
                print(f'costs: {costs}')
            else:
                # No position change - keep same positions, no trading costs
                new_shares_stock1 = prev_shares_stock1
                new_shares_stock2 = prev_shares_stock2
                costs = 0
        
        # Update tracking
        shares_stock1_list.append(new_shares_stock1)
        shares_stock2_list.append(new_shares_stock2)
        raw_pnl_list.append(raw_pnl)
        print(f'raw pnl: {raw_pnl}')
        cost_list.append(costs)
        
        # Net PnL and portfolio update
        net_pnl = raw_pnl - costs
        net_pnl_list.append(net_pnl)
        current_portfolio += net_pnl
        
        # Ensure portfolio doesn't go negative
        current_portfolio = max(current_portfolio, initial_capital * 0.05)
        portfolio_values.append(current_portfolio)
        
        # Update previous shares for next iteration
        prev_shares_stock1 = new_shares_stock1
        prev_shares_stock2 = new_shares_stock2
    
    # Add all calculated columns to dataframe
    df[f'shares_{symbols[0]}'] = shares_stock1_list
    df[f'shares_{symbols[1]}'] = shares_stock2_list
    df['portfolio_value'] = portfolio_values
    df['raw_pnl'] = raw_pnl_list
    df['cost'] = cost_list
    df['net_pnl'] = net_pnl_list
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    
    # Add additional analysis columns
    df['position_prev'] = df['position'].shift(1).fillna(0)
    df['trade'] = (df['position'] != df['position_prev']).astype(int)
    df['returns'] = df['net_pnl'] / df['portfolio_value'].shift(1)
    df['cumulative_returns'] = (df['portfolio_value'] / initial_capital) - 1
    
    return df

















def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for the strategy.
    """
    # Basic metrics
    initial_capital = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1
    total_trades = df['trade'].sum()
    
    # Risk metrics
    returns = df['returns'].dropna()
    if len(returns) > 0:
        mean_return = returns.mean()
        return_std = returns.std()
        sharpe_ratio = mean_return / return_std * np.sqrt(252) if return_std > 0 else 0
        max_drawdown = (df['cumulative_returns'].cummax() - df['cumulative_returns']).max()
        volatility = return_std * np.sqrt(252)
        win_rate = (returns > 0).mean()
    else:
        mean_return = sharpe_ratio = max_drawdown = volatility = win_rate = 0
    
    # Trading metrics
    winning_trades = df[(df['trade'] == 1) & (df['net_pnl'] > 0)]
    losing_trades = df[(df['trade'] == 1) & (df['net_pnl'] < 0)]
    
    avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
    
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
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_costs': total_costs,
        'final_portfolio_value': final_value,
        'mean_return_per_period': mean_return
    }