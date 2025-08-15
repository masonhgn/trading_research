import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats


class BacktestAnalyzer:
    """
    Comprehensive backtest analysis for trading strategies.
    Calculates key performance metrics including returns, drawdowns, Sharpe ratio, etc.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the backtest analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, df: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
        """
        Calculate various return metrics from the backtest data.
        
        Args:
            df: DataFrame with 'cumulative_pnl' column
            initial_capital: Initial capital used for the strategy
            
        Returns:
            DataFrame with additional return columns
        """
        df = df.copy()
        
        # Calculate portfolio value over time
        df['portfolio_value'] = initial_capital + df['cumulative_pnl']
        
        # For intraday strategies, calculate returns based on PnL changes
        # This is more accurate than portfolio value changes
        df['pct_return'] = df['net_pnl'] / initial_capital  # Use net PnL for returns
        df['cumulative_pct_return'] = (df['portfolio_value'] / initial_capital) - 1
        
        # Calculate log returns for better statistical properties
        # Handle zero returns properly
        df['log_return'] = np.where(
            df['portfolio_value'] > 0,
            np.log(df['portfolio_value'] / df['portfolio_value'].shift(1)),
            0
        )
        
        return df
    
    def calculate_drawdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown metrics.
        
        Args:
            df: DataFrame with 'portfolio_value' column
            
        Returns:
            DataFrame with drawdown columns
        """
        df = df.copy()
        
        # Calculate running maximum
        df['running_max'] = df['portfolio_value'].expanding().max()
        
        # Calculate drawdown
        df['drawdown'] = (df['portfolio_value'] - df['running_max']) / df['running_max']
        df['drawdown_pct'] = df['drawdown'] * 100
        
        # Calculate absolute drawdown
        df['absolute_drawdown'] = df['portfolio_value'] - df['running_max']
        
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 252, periods_per_year: int = 252) -> pd.DataFrame:
        """
        Calculate rolling volatility metrics.
        
        Args:
            df: DataFrame with 'pct_return' column
            window: Rolling window for volatility calculation
            periods_per_year: Number of periods per year for annualization
            
        Returns:
            DataFrame with volatility columns
        """
        df = df.copy()
        
        # For intraday strategies, use daily annualization
        # This is the standard approach in finance
        if periods_per_year > 252:  # Intraday data
            annualization_factor = np.sqrt(252)
            rolling_window = min(window, 60)  # Use smaller window for intraday
        else:
            annualization_factor = np.sqrt(periods_per_year)
            rolling_window = window
        
        # Calculate rolling volatility (annualized)
        df['rolling_volatility'] = df['pct_return'].rolling(window=rolling_window).std() * annualization_factor
        
        # Calculate realized volatility
        df['realized_volatility'] = df['pct_return'].expanding().std() * annualization_factor
        
        return df
    
    def calculate_rolling_metrics(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Calculate rolling return and volatility metrics (no risk-free rate required).
        
        Args:
            df: DataFrame with 'pct_return' column
            window: Rolling window for calculations
            
        Returns:
            DataFrame with rolling metrics
        """
        df = df.copy()
        
        # Rolling return (annualized)
        df['rolling_return'] = df['pct_return'].rolling(window=window).mean() * 252
        
        # Rolling volatility (annualized)
        df['rolling_volatility'] = df['pct_return'].rolling(window=window).std() * np.sqrt(252)
        
        # Rolling information ratio (return / volatility)
        df['rolling_info_ratio'] = df['rolling_return'] / df['rolling_volatility']
        
        return df
    
    def calculate_trade_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate trade-specific statistics.
        
        Args:
            df: DataFrame with 'position' and 'trade' columns
            
        Returns:
            Dictionary with trade statistics
        """
        # Count trades
        total_trades = df['trade'].sum()
        
        # Calculate trade durations
        trade_durations = []
        current_duration = 0
        
        for position in df['position']:
            if position != 0:
                current_duration += 1
            elif current_duration > 0:
                trade_durations.append(current_duration)
                current_duration = 0
        
        # Add final trade if still open
        if current_duration > 0:
            trade_durations.append(current_duration)
        
        # Calculate position statistics
        long_positions = (df['position'] > 0).sum()
        short_positions = (df['position'] < 0).sum()
        neutral_positions = (df['position'] == 0).sum()
        
        return {
            'total_trades': total_trades,
            'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
            'max_trade_duration': max(trade_durations) if trade_durations else 0,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'neutral_positions': neutral_positions,
            'position_utilization': (long_positions + short_positions) / len(df)
        }
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics that don't require risk-free rates.
        
        Args:
            df: DataFrame with return and drawdown columns
            
        Returns:
            Dictionary with risk metrics
        """
        # Maximum drawdown
        max_drawdown = df['drawdown'].min()
        max_drawdown_pct = df['drawdown_pct'].min()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(df['pct_return'].dropna(), 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = df['pct_return'][df['pct_return'] <= var_95].mean()
        
        # Downside deviation
        downside_returns = df['pct_return'][df['pct_return'] < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Annualized volatility (using daily annualization)
        annual_volatility = df['pct_return'].std() * np.sqrt(252)
        
        # Information ratio (return / volatility)
        mean_return = df['pct_return'].mean()
        information_ratio = mean_return / df['pct_return'].std() if df['pct_return'].std() > 0 else 0
        
        # Win rate
        winning_periods = (df['pct_return'] > 0).sum()
        total_periods = len(df['pct_return'].dropna())
        win_rate = winning_periods / total_periods if total_periods > 0 else 0
        
        # Average win and loss
        wins = df['pct_return'][df['pct_return'] > 0]
        losses = df['pct_return'][df['pct_return'] < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,
            'annual_volatility': annual_volatility,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def analyze_backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform comprehensive backtest analysis (no risk-free rates required).
        
        Args:
            df: DataFrame with backtest results
            initial_capital: Initial capital used
            
        Returns:
            Tuple of (enhanced DataFrame, analysis summary)
        """
        # Calculate all metrics
        df = self.calculate_returns(df, initial_capital)
        df = self.calculate_drawdown(df)
        df = self.calculate_volatility(df)
        df = self.calculate_rolling_metrics(df)
        
        # Calculate trade statistics
        trade_stats = self.calculate_trade_statistics(df)
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(df)
        
        # Combine all statistics
        analysis_summary = {
            'performance': {
                'total_return': df['cumulative_pct_return'].iloc[-1],
                'total_pnl': df['cumulative_pnl'].iloc[-1],
                'final_portfolio_value': df['portfolio_value'].iloc[-1],
                'annual_volatility': risk_metrics['annual_volatility']
            },
            'risk': risk_metrics,
            'trades': trade_stats,
            'data': {
                'total_periods': len(df),
                'trading_days': len(df),
                'start_date': df['datetime'].min() if 'datetime' in df.columns else None,
                'end_date': df['datetime'].max() if 'datetime' in df.columns else None
            }
        }
        
        return df, analysis_summary
    
    def print_summary(self, analysis_summary: Dict):
        """
        Print a formatted summary of the backtest results.
        
        Args:
            analysis_summary: Dictionary with analysis results
        """
        print("\n" + "="*80)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*80)
        
        # Performance metrics
        perf = analysis_summary['performance']
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Total Return: {perf['total_return']:.2%}")
        print(f"  Total PnL: ${perf['total_pnl']:,.2f}")
        print(f"  Final Portfolio Value: ${perf['final_portfolio_value']:,.2f}")
        print(f"  Annual Volatility: {perf['annual_volatility']:.2%}")
        
        # Risk metrics
        risk = analysis_summary['risk']
        print(f"\nRISK METRICS:")
        print(f"  Maximum Drawdown: {risk['max_drawdown_pct']:.2f}%")
        print(f"  Value at Risk (95%): {risk['var_95']:.2%}")
        print(f"  Conditional VaR (95%): {risk['cvar_95']:.2%}")
        print(f"  Downside Deviation: {risk['downside_deviation']:.2%}")
        
        # Trading metrics
        print(f"\nTRADING METRICS:")
        print(f"  Information Ratio: {risk['information_ratio']:.3f}")
        print(f"  Win Rate: {risk['win_rate']:.2%}")
        print(f"  Average Win: {risk['avg_win']:.4f}")
        print(f"  Average Loss: {risk['avg_loss']:.4f}")
        print(f"  Profit Factor: {risk['profit_factor']:.3f}")
        
        # Trade statistics
        trades = analysis_summary['trades']
        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades: {trades['total_trades']}")
        print(f"  Average Trade Duration: {trades['avg_trade_duration']:.1f} periods")
        print(f"  Maximum Trade Duration: {trades['max_trade_duration']} periods")
        print(f"  Position Utilization: {trades['position_utilization']:.2%}")
        
        # Data summary
        data = analysis_summary['data']
        print(f"\nDATA SUMMARY:")
        print(f"  Total Periods: {data['total_periods']}")
        print(f"  Start Date: {data['start_date']}")
        print(f"  End Date: {data['end_date']}")
        
        print("="*80)
    
    def debug_returns(self, df: pd.DataFrame):
        """
        Debug function to understand return calculation issues.
        
        Args:
            df: DataFrame with return data
        """
        print("\n" + "="*60)
        print("RETURN CALCULATION DEBUG")
        print("="*60)
        
        # Check for zero returns
        zero_returns = (df['pct_return'] == 0).sum()
        total_returns = len(df['pct_return'].dropna())
        
        print(f"Total return periods: {total_returns}")
        print(f"Zero returns: {zero_returns} ({zero_returns/total_returns:.1%})")
        print(f"Non-zero returns: {total_returns - zero_returns}")
        
        # Check return statistics
        non_zero_returns = df['pct_return'][df['pct_return'] != 0]
        if len(non_zero_returns) > 0:
            print(f"\nNon-zero return statistics:")
            print(f"  Mean: {non_zero_returns.mean():.6f}")
            print(f"  Std: {non_zero_returns.std():.6f}")
            print(f"  Min: {non_zero_returns.min():.6f}")
            print(f"  Max: {non_zero_returns.max():.6f}")
            print(f"  Range: {non_zero_returns.max() - non_zero_returns.min():.6f}")
        
        # Check PnL statistics
        print(f"\nPnL statistics:")
        print(f"  Total PnL: ${df['cumulative_pnl'].iloc[-1]:.2f}")
        print(f"  Net PnL mean: {df['net_pnl'].mean():.4f}")
        print(f"  Net PnL std: {df['net_pnl'].std():.4f}")
        print(f"  Non-zero PnL periods: {(df['net_pnl'] != 0).sum()}")
        
        print("="*60)
    
    def plot_performance_analysis(self, df: pd.DataFrame, analysis_summary: Dict):
        """
        Create comprehensive performance analysis plots.
        
        Args:
            df: DataFrame with analysis results
            analysis_summary: Dictionary with analysis summary
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Backtest Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Portfolio Value and Drawdown
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        # Portfolio value
        ax1.plot(df.index, df['portfolio_value'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Drawdown
        ax1_twin.fill_between(df.index, df['drawdown_pct'], 0, alpha=0.3, color='red', label='Drawdown')
        ax1_twin.set_ylabel('Drawdown (%)', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Portfolio Value and Drawdown')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling Information Ratio
        ax2 = axes[0, 1]
        ax2.plot(df.index, df['rolling_info_ratio'], 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Rolling Information Ratio')
        ax2.set_ylabel('Information Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Volatility
        ax3 = axes[1, 0]
        ax3.plot(df.index, df['rolling_volatility'], 'orange', linewidth=2)
        ax3.set_title('Rolling Volatility (Annualized)')
        ax3.set_ylabel('Volatility')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Return Distribution
        ax4 = axes[1, 1]
        returns = df['pct_return'].dropna()
        ax4.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax4.axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.4f}')
        ax4.set_title('Return Distribution')
        ax4.set_xlabel('Return')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Drawdown analysis
        plt.figure(figsize=(12, 6))
        plt.fill_between(df.index, df['drawdown_pct'], 0, alpha=0.7, color='red')
        plt.title('Drawdown Analysis')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()


def quick_analysis(df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """
    Quick analysis function for simple backtest results.
    
    Args:
        df: DataFrame with 'cumulative_pnl' column
        initial_capital: Initial capital used
        
    Returns:
        Dictionary with key performance metrics
    """
    analyzer = BacktestAnalyzer()
    _, summary = analyzer.analyze_backtest(df, initial_capital)
    return summary
