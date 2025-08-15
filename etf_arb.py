import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from ib_async import IB, Stock


def filter_trading_hours(df, market_open_hour=9, market_close_hour=16, timezone='US/Eastern'):
    """
    Filter DataFrame to only include regular trading hours (9:30 AM - 4:00 PM ET).
    
    Args:
        df: DataFrame with 'datetime' column
        market_open_hour: Market open hour (default 9:30 AM ET)
        market_close_hour: Market close hour (default 4:00 PM ET)
        timezone: Timezone for market hours (default US/Eastern)
    
    Returns:
        Filtered DataFrame with only trading hours data
    """
    # Ensure datetime is timezone-aware
    if df['datetime'].dt.tz is None:
        # Assume UTC if no timezone info
        df = df.copy()
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    
    # Convert to Eastern time for US market hours
    df = df.copy()
    df['datetime_et'] = df['datetime'].dt.tz_convert('US/Eastern')
    
    # Filter for trading hours (9:30 AM - 4:00 PM ET, Monday-Friday)
    # Convert to minutes since midnight for easier comparison
    minutes_since_midnight = df['datetime_et'].dt.hour * 60 + df['datetime_et'].dt.minute
    
    trading_mask = (
        (minutes_since_midnight >= 9 * 60 + 30) &  # After 9:30 AM
        (minutes_since_midnight < 16 * 60) &       # Before 4:00 PM
        (df['datetime_et'].dt.dayofweek < 5)       # Monday-Friday
    )
    
    filtered_df = df[trading_mask].copy()
    filtered_df = filtered_df.drop('datetime_et', axis=1)
    
    return filtered_df


def detect_trading_hours_automatically(df, volatility_threshold=0.001, min_consecutive_periods=30):
    """
    Automatically detect trading vs non-trading hours based on spread volatility.
    
    Args:
        df: DataFrame with 'spread' column
        volatility_threshold: Minimum volatility to consider as trading hours
        min_consecutive_periods: Minimum consecutive periods to confirm trading hours
    
    Returns:
        Boolean mask for trading hours
    """
    # Calculate rolling volatility (standard deviation of spread changes)
    spread_changes = df['spread'].diff().abs()
    rolling_vol = spread_changes.rolling(window=10, min_periods=5).mean()
    
    # Identify periods with sufficient volatility
    volatile_periods = rolling_vol > volatility_threshold
    
    # Find consecutive periods of volatility (trading hours)
    trading_mask = volatile_periods.rolling(
        window=min_consecutive_periods, 
        min_periods=min_consecutive_periods
    ).sum() >= min_consecutive_periods
    
    return trading_mask








async def fetch_stock_minute_bars(
    ib: IB,
    ticker: str,
    exchange: str = "ARCA",  # ARCA is common for ETFs
    currency: str = "USD",
    end_datetime: dt.datetime = None,
    duration: str = "1 D",
    bar_size: str = "10 secs",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    format_date: int = 1
) -> pd.DataFrame:
    """
    Fetch 1-minute bars for a given stock/ETF ticker and return as a DataFrame.
    """
    if end_datetime is None:
        end_datetime = dt.datetime.now()

    end_str = end_datetime.strftime("%Y%m%d %H:%M:%S")
    contract = Stock(ticker, exchange=exchange, currency=currency)

    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime=end_str,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=format_date
    )

    if not bars:
        return pd.DataFrame()

    return pd.DataFrame([{
        'datetime': pd.to_datetime(bar.date),
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    } for bar in bars])








def fetch_pair_data(ib, sym1, sym2, duration, end_datetime):
    df1 = fetch_stock_minute_bars(ib, sym1, duration=duration, end_datetime=end_datetime)
    df2 = fetch_stock_minute_bars(ib, sym2, duration=duration, end_datetime=end_datetime)
    df = pd.merge(
        df1[['datetime', 'close']],
        df2[['datetime', 'close']],
        on='datetime',
        how='inner',
        suffixes=(f'_{sym1}', f'_{sym2}')
    )
    return df.dropna()


def compute_spread(df, sym1, sym2):
    df['spread'] = df[f'close_{sym1}'] - df[f'close_{sym2}']
    return df


def compute_rolling_stats(df, window=60):
    df['rolling_mean'] = df['spread'].rolling(window=window, min_periods=window).mean()
    df['rolling_std'] = df['spread'].rolling(window=window, min_periods=window).std()
    df['zscore'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']
    df['upper'] = df['rolling_mean'] + df['rolling_std']
    df['lower'] = df['rolling_mean'] - df['rolling_std']
    return df


def generate_signals(df, entry_threshold=1.0, exit_threshold=0.1):
    df['long_entry'] = df['zscore'] < -entry_threshold
    df['short_entry'] = df['zscore'] > entry_threshold
    df['exit'] = df['zscore'].abs() < exit_threshold

    position = 0
    positions = []
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


def compute_pnl(df, shares_per_leg=100, slippage=0.01, fee=0.005):
    df['spread_change'] = df['spread'].diff()
    df['shifted_position'] = df['position'].shift()
    df['raw_pnl'] = df['shifted_position'] * df['spread_change'] * shares_per_leg
    df['trade'] = df['position'].diff().fillna(0).abs() > 0
    df['cost'] = df['trade'] * 2 * (slippage + fee) * shares_per_leg
    df['net_pnl'] = df['raw_pnl'] - df['cost']
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    return df


def plot_spread(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['datetime'], df['spread'], label='Spread')
    plt.plot(df['datetime'], df['rolling_mean'], color='red', linestyle='--', label='Rolling Mean')
    plt.plot(df['datetime'], df['upper'], color='green', linestyle='--', label='+1σ Band')
    plt.plot(df['datetime'], df['lower'], color='orange', linestyle='--', label='-1σ Band')
    plt.title("Spread with Rolling Mean and Bands")
    plt.xlabel("Time")
    plt.ylabel("Price Difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pnl(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['datetime'], df['cumulative_pnl'], label='Cumulative PnL')
    plt.title("Cumulative PnL")
    plt.xlabel("Time")
    plt.ylabel("PnL ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_cointegration_tests(df, sym1, sym2):
    score, pvalue, _ = coint(df[f'close_{sym1}'], df[f'close_{sym2}'])
    adf_stat, adf_pval, _, _, crit_vals, _ = adfuller(df['spread'])
    return {
        "coint_score": score,
        "coint_pval": pvalue,
        "adf_stat": adf_stat,
        "adf_pval": adf_pval,
        "crit_vals": crit_vals
    }
















import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
# from tools import fetch_stock_minute_bars  # Commented out since we have the function in this file


class ZScoreStatArbStrategy:
    def __init__(self, ib, sym1, sym2, window=60, entry_threshold=1.0, exit_threshold=0.1,
                 slippage=0.01, fee=0.005, shares_per_leg=100, filter_trading_hours=True,
                 auto_detect_hours=False, volatility_threshold=0.001, initial_capital=10000):
        self.ib = ib
        self.sym1 = sym1
        self.sym2 = sym2
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.slippage = slippage
        self.fee = fee
        self.shares_per_leg = shares_per_leg
        self.filter_trading_hours = filter_trading_hours
        self.auto_detect_hours = auto_detect_hours
        self.volatility_threshold = volatility_threshold
        self.initial_capital = initial_capital

    async def fetch_data(self, duration, end_datetime):
        # Fetch data with use_rth=False to get all hours, then filter ourselves
        df1 = await fetch_stock_minute_bars(self.ib, self.sym1, duration=duration, end_datetime=end_datetime, use_rth=False)
        df2 = await fetch_stock_minute_bars(self.ib, self.sym2, duration=duration, end_datetime=end_datetime, use_rth=False)
        df = pd.merge(
            df1[['datetime', 'close']],
            df2[['datetime', 'close']],
            on='datetime',
            how='inner',
            suffixes=(f'_{self.sym1}', f'_{self.sym2}')
        )
        df.dropna(inplace=True)
        
        # Filter trading hours if enabled
        if self.filter_trading_hours:
            if self.auto_detect_hours:
                # First compute spread to enable automatic detection
                df = self.compute_spread_and_zscore(df)
                trading_mask = detect_trading_hours_automatically(df, self.volatility_threshold)
                df = df[trading_mask].copy()
                # Recompute spread and zscore with filtered data
                df = self.compute_spread_and_zscore(df)
            else:
                df = filter_trading_hours(df)
        
        return df

    def compute_spread_and_zscore(self, df):
        df['spread'] = df[f'close_{self.sym1}'] - df[f'close_{self.sym2}']
        df['rolling_mean'] = df['spread'].rolling(window=self.window, min_periods=self.window).mean()
        df['rolling_std'] = df['spread'].rolling(window=self.window, min_periods=self.window).std()
        df['zscore'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']
        df['upper'] = df['rolling_mean'] + df['rolling_std']
        df['lower'] = df['rolling_mean'] - df['rolling_std']
        return df

    def generate_signals(self, df):
        df['long_entry'] = df['zscore'] < -self.entry_threshold
        df['short_entry'] = df['zscore'] > self.entry_threshold
        df['exit'] = df['zscore'].abs() < self.exit_threshold

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
        constrained_shares = []
        portfolio_value = self.initial_capital
        
        for i, (pos, price1, price2) in enumerate(zip(positions, df[f'close_{self.sym1}'], df[f'close_{self.sym2}'])):
            current_shares = 0
            
            if pos == 1:  # Long position
                # For long positions: buy SPY, sell VOO
                # We need capital for the SPY purchase (price1)
                max_affordable_shares = int(portfolio_value / price1)
                current_shares = min(self.shares_per_leg, max_affordable_shares)
                
            elif pos == -1:  # Short position
                # For short positions: sell SPY, buy VOO  
                # We need capital for the VOO purchase (price2)
                max_affordable_shares = int(portfolio_value / price2)
                current_shares = min(self.shares_per_leg, max_affordable_shares)
                
            # If we can't afford the position, set position to 0
            if current_shares == 0 and pos != 0:
                positions[i] = 0
                
            constrained_shares.append(current_shares)
        
        df['position'] = positions
        df['shares'] = constrained_shares
        return df

    def compute_pnl(self, df):
        df['spread_change'] = df['spread'].diff()
        df['shifted_position'] = df['position'].shift()
        df['shifted_shares'] = df['shares'].shift()
        
        # Calculate PnL using dynamic shares
        df['raw_pnl'] = df['shifted_position'] * df['spread_change'] * df['shifted_shares']
        df['trade'] = df['position'].diff().fillna(0).abs() > 0
        df['cost'] = df['trade'] * 2 * (self.slippage + self.fee) * df['shares']
        df['net_pnl'] = df['raw_pnl'] - df['cost']
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        
        # Calculate portfolio value over time
        df['portfolio_value'] = self.initial_capital + df['cumulative_pnl']
        
        return df

    def run_cointegration_tests(self, df):
        score, pvalue, _ = coint(df[f'close_{self.sym1}'], df[f'close_{self.sym2}'])
        adf_stat, adf_pval, _, _, crit_vals, _ = adfuller(df['spread'])
        return {
            "coint_score": score,
            "coint_pval": pvalue,
            "adf_stat": adf_stat,
            "adf_pval": adf_pval,
            "crit_vals": crit_vals
        }

    def plot_spread(self, df):
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

    def plot_pnl(self, df):
        plt.figure(figsize=(15, 6))
        
        # Create a continuous time index for better visualization
        # This will compress the gaps but still show the data clearly
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



    async def backtest(self, duration, end_datetime, progress_callback=None):
        # Step 1: Fetch data (20% of progress)
        if progress_callback:
            progress_callback(10)
        df = await self.fetch_data(duration, end_datetime)
        
        if progress_callback:
            progress_callback(30)
        
        # Step 2: Compute spread and zscore (30% of progress)
        if not (self.filter_trading_hours and self.auto_detect_hours):
            df = self.compute_spread_and_zscore(df)
        
        if progress_callback:
            progress_callback(60)
            
        # Step 3: Generate signals (20% of progress)
        df = self.generate_signals(df)
        
        if progress_callback:
            progress_callback(80)
            
        # Step 4: Compute PnL and stats (20% of progress)
        df = self.compute_pnl(df)
        stats = self.run_cointegration_tests(df)
        
        if progress_callback:
            progress_callback(100)
        
        return df, stats













from ib_async import IB, util
import asyncio
# from zscore_stat_arb_strategy import ZScoreStatArbStrategy

async def main():
    ib = IB()
    await ib.connectAsync("127.0.0.1", 4002, clientId=11)
    util.startLoop()
    

    # Run strategy with optimized parameters
    strat = ZScoreStatArbStrategy(
        ib, sym1="SPY", sym2="VOO", 
        window=90,           # More stable signals
        entry_threshold=1.5,  # Fewer, higher-quality trades
        exit_threshold=0.2,   # Hold positions longer
        slippage=0.01,
        fee=0.005,
        shares_per_leg=200,   # Larger position size
        filter_trading_hours=True, 
        auto_detect_hours=False  # Use fixed trading hours for US markets
    )
    df, stats = await strat.backtest(duration="1 D", end_datetime=dt.datetime(2025, 8, 14, 16))
    
    strat.plot_spread(df)
    strat.plot_pnl(df)
    
    print(f"Data points: {len(df)}")
    print(f"Trading period: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Final PnL: ${df['cumulative_pnl'].iloc[-1]:.2f}")
    
    print("\nCointegration Test Results:")
    print(f"  Statistic = {stats['coint_score']:.4f}")
    print(f"  p-value   = {stats['coint_pval']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
