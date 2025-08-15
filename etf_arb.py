import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
from ib_async import IB, Stock
from backtest_analysis import BacktestAnalyzer
from risk_free_rate import RiskFreeRateFetcher

# Global configuration for trading frequency and bar sizes
TRADING_CONFIG = {
    'bar_size': '1 min',  # Options: '1 secs', '5 secs', '10 secs', '30 secs', '1 min', '5 mins', '15 mins', '30 mins', '1 hour', '1 day'
    'signal_frequency': '1 min',  # How often to generate signals (should match bar_size for most cases)
    'trading_hours': {
        'start': '09:30',  # Market open
        'end': '16:00',    # Market close
        'timezone': 'US/Eastern'
    }
}

def validate_trading_config():
    """
    Validate the trading configuration and print warnings for potential issues.
    """
    valid_bar_sizes = ['1 secs', '5 secs', '10 secs', '30 secs', '1 min', '5 mins', '15 mins', '30 mins', '1 hour', '1 day']
    
    if TRADING_CONFIG['bar_size'] not in valid_bar_sizes:
        print(f"WARNING: Invalid bar_size '{TRADING_CONFIG['bar_size']}'. Valid options: {valid_bar_sizes}")
    
    if TRADING_CONFIG['bar_size'] != TRADING_CONFIG['signal_frequency']:
        print(f"WARNING: bar_size ({TRADING_CONFIG['bar_size']}) differs from signal_frequency ({TRADING_CONFIG['signal_frequency']})")
    
    print(f"✓ Using bar size: {TRADING_CONFIG['bar_size']}")
    print(f"✓ Signal frequency: {TRADING_CONFIG['signal_frequency']}")
    print(f"✓ Trading hours: {TRADING_CONFIG['trading_hours']['start']} - {TRADING_CONFIG['trading_hours']['end']} {TRADING_CONFIG['trading_hours']['timezone']}")


def filter_trading_hours(df, market_open_time=None, market_close_time=None, timezone=None):
    """
    Filter DataFrame to only include regular trading hours.
    Uses global TRADING_CONFIG for default values.
    
    Args:
        df: DataFrame with 'datetime' column
        market_open_time: Market open time (default from TRADING_CONFIG)
        market_close_time: Market close time (default from TRADING_CONFIG)
        timezone: Timezone for market hours (default from TRADING_CONFIG)
    
    Returns:
        Filtered DataFrame with only trading hours data
    """
    # Use config defaults if not specified
    if market_open_time is None:
        market_open_time = TRADING_CONFIG['trading_hours']['start']
    if market_close_time is None:
        market_close_time = TRADING_CONFIG['trading_hours']['end']
    if timezone is None:
        timezone = TRADING_CONFIG['trading_hours']['timezone']
    
    # Parse time strings to hours and minutes
    open_hour, open_minute = map(int, market_open_time.split(':'))
    close_hour, close_minute = map(int, market_close_time.split(':'))
    
    # Ensure datetime is timezone-aware
    if df['datetime'].dt.tz is None:
        # Assume UTC if no timezone info
        df = df.copy()
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    
    # Convert to specified timezone for market hours
    df = df.copy()
    df['datetime_local'] = df['datetime'].dt.tz_convert(timezone)
    
    # Filter for trading hours, Monday-Friday
    # Convert to minutes since midnight for easier comparison
    minutes_since_midnight = df['datetime_local'].dt.hour * 60 + df['datetime_local'].dt.minute
    open_minutes = open_hour * 60 + open_minute
    close_minutes = close_hour * 60 + close_minute
    
    trading_mask = (
        (minutes_since_midnight >= open_minutes) &   # After market open
        (minutes_since_midnight < close_minutes) &   # Before market close
        (df['datetime_local'].dt.dayofweek < 5)     # Monday-Friday
    )
    
    filtered_df = df[trading_mask].copy()
    filtered_df = filtered_df.drop('datetime_local', axis=1)
    
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








async def fetch_stock_bars(
    ib: IB,
    ticker: str,
    exchange: str = "ARCA",  # ARCA is common for ETFs
    currency: str = "USD",
    end_datetime: dt.datetime = None,
    duration: str = "1 D",
    bar_size: str = None,
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    format_date: int = 1
) -> pd.DataFrame:
    """
    Fetch bars for a given stock/ETF ticker and return as a DataFrame.
    Uses global TRADING_CONFIG for bar_size if not specified.
    """
    if end_datetime is None:
        end_datetime = dt.datetime.now()
    
    # Use config bar_size if not specified
    if bar_size is None:
        bar_size = TRADING_CONFIG['bar_size']

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
    df1 = fetch_stock_bars(ib, sym1, duration=duration, end_datetime=end_datetime)
    df2 = fetch_stock_bars(ib, sym2, duration=duration, end_datetime=end_datetime)
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
                 auto_detect_hours=False, volatility_threshold=0.001, initial_capital=10000,
                 use_dynamic_thresholds=True, confidence_level=0.95, distribution_type='t'):
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
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.confidence_level = confidence_level
        self.distribution_type = distribution_type

    def fit_spread_distribution(self, spread_series):
        """
        Fit the spread data to a specified distribution.
        
        Args:
            spread_series: Series of spread values
            
        Returns:
            tuple: Distribution parameters and fitted distribution object
        """
        if self.distribution_type == 't':
            # Fit t-distribution (more robust to fat tails)
            params = stats.t.fit(spread_series)
            fitted_dist = stats.t(*params)
        elif self.distribution_type == 'normal':
            # Fit normal distribution
            params = stats.norm.fit(spread_series)
            fitted_dist = stats.norm(*params)
        elif self.distribution_type == 'skewnorm':
            # Fit skewed normal distribution
            params = stats.skewnorm.fit(spread_series)
            fitted_dist = stats.skewnorm(*params)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        return params, fitted_dist

    def calculate_dynamic_thresholds(self, spread_series):
        """
        Calculate dynamic entry and exit thresholds based on fitted distribution.
        
        Args:
            spread_series: Series of spread values
            
        Returns:
            tuple: (entry_threshold, exit_threshold) in terms of z-scores
        """
        if len(spread_series) < self.window:
            # Fall back to fixed thresholds if not enough data
            return self.entry_threshold, self.exit_threshold
        
        try:
            params, fitted_dist = self.fit_spread_distribution(spread_series)
            
            # Calculate percentiles for entry and exit
            entry_percentile = 1 - self.confidence_level  # e.g., 0.05 for 95% confidence
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
            return self.entry_threshold, self.exit_threshold

    def compute_dynamic_zscore(self, df):
        """
        Compute z-scores using fitted distribution instead of normal assumption.
        
        Args:
            df: DataFrame with 'spread' column
            
        Returns:
            DataFrame with additional 'dynamic_zscore' column
        """
        dynamic_zscores = []
        entry_thresholds = []
        exit_thresholds = []
        
        for i in range(self.window, len(df)):
            window_spread = df['spread'].iloc[i-self.window:i]
            
            try:
                # Fit distribution to rolling window
                params, fitted_dist = self.fit_spread_distribution(window_spread)
                
                # Calculate current spread's percentile in fitted distribution
                current_spread = df['spread'].iloc[i]
                percentile = fitted_dist.cdf(current_spread)
                
                # Convert to z-score (inverse normal CDF of percentile)
                dynamic_zscore = stats.norm.ppf(percentile)
                
                # Calculate dynamic thresholds for this window
                entry_thresh, exit_thresh = self.calculate_dynamic_thresholds(window_spread)
                
            except Exception as e:
                # Fall back to traditional z-score calculation
                mean = window_spread.mean()
                std = window_spread.std()
                dynamic_zscore = (current_spread - mean) / std if std > 0 else 0
                entry_thresh, exit_thresh = self.entry_threshold, self.exit_threshold
            
            dynamic_zscores.append(dynamic_zscore)
            entry_thresholds.append(entry_thresh)
            exit_thresholds.append(exit_thresh)
        
        # Pad the beginning with NaN values
        df['dynamic_zscore'] = [np.nan] * self.window + dynamic_zscores
        df['dynamic_entry_threshold'] = [np.nan] * self.window + entry_thresholds
        df['dynamic_exit_threshold'] = [np.nan] * self.window + exit_thresholds
        
        return df

    async def fetch_data(self, duration, end_datetime):
        # Fetch data with use_rth=False to get all hours, then filter ourselves
        df1 = await fetch_stock_bars(self.ib, self.sym1, duration=duration, end_datetime=end_datetime, use_rth=False)
        df2 = await fetch_stock_bars(self.ib, self.sym2, duration=duration, end_datetime=end_datetime, use_rth=False)
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
        
        # Add dynamic z-score calculation if enabled
        if self.use_dynamic_thresholds:
            df = self.compute_dynamic_zscore(df)
            print(f"Using dynamic thresholds (distribution: {self.distribution_type}, confidence: {self.confidence_level})")
        else:
            print(f"Using fixed thresholds (entry: {self.entry_threshold}, exit: {self.exit_threshold})")
        
        return df

    def generate_signals(self, df):
        if self.use_dynamic_thresholds and 'dynamic_zscore' in df.columns:
            # Use dynamic thresholds
            df['long_entry'] = df['dynamic_zscore'] < -df['dynamic_entry_threshold']
            df['short_entry'] = df['dynamic_zscore'] > df['dynamic_entry_threshold']
            df['exit'] = df['dynamic_zscore'].abs() < df['dynamic_exit_threshold']
        else:
            # Use fixed thresholds
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

    def get_config_summary(self):
        """
        Get a summary of the current trading configuration.
        """
        return {
            'bar_size': TRADING_CONFIG['bar_size'],
            'signal_frequency': TRADING_CONFIG['signal_frequency'],
            'trading_hours': TRADING_CONFIG['trading_hours'],
            'window': self.window,
            'use_dynamic_thresholds': self.use_dynamic_thresholds,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'distribution_type': self.distribution_type if self.use_dynamic_thresholds else 'N/A',
            'confidence_level': self.confidence_level if self.use_dynamic_thresholds else 'N/A'
        }

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

    def plot_distribution_analysis(self, df):
        """
        Plot the fitted distribution vs normal assumption for the spread.
        """
        if not self.use_dynamic_thresholds or 'dynamic_zscore' not in df.columns:
            print("Dynamic thresholds not enabled or not computed yet.")
            return
        
        # Get recent spread data for analysis
        recent_spread = df['spread'].tail(self.window).dropna()
        
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

    def plot_dynamic_thresholds(self, df):
        """
        Plot the dynamic thresholds over time.
        """
        if not self.use_dynamic_thresholds or 'dynamic_entry_threshold' not in df.columns:
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
    
    # Validate trading configuration
    validate_trading_config()
    print()

    # Configuration: Choose between fixed and dynamic thresholds
    USE_DYNAMIC_THRESHOLDS = False  # Set to False to use fixed thresholds
    
    # Run strategy with configurable thresholds
    strat = ZScoreStatArbStrategy(
        ib, sym1="SPY", sym2="VOO", 
        window=90,           # More stable signals
        entry_threshold=1.5,  # Fixed threshold (used when dynamic is disabled)
        exit_threshold=0.2,   # Fixed threshold (used when dynamic is disabled)
        slippage=0.01,
        fee=0.005,
        shares_per_leg=200,   # Larger position size
        filter_trading_hours=True, 
        auto_detect_hours=False,  # Use fixed trading hours for US markets
        use_dynamic_thresholds=USE_DYNAMIC_THRESHOLDS,  # Toggle dynamic thresholds
        confidence_level=0.95,  # 95% confidence level for entry signals (dynamic only)
        distribution_type='t'  # Use t-distribution (more robust to fat tails) (dynamic only)
    )
    df, stats = await strat.backtest(duration="1 D", end_datetime=dt.datetime(2025, 8, 14, 16))
    
    # Display configuration summary
    config = strat.get_config_summary()
    print("\n" + "="*60)
    print("TRADING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Bar Size: {config['bar_size']}")
    print(f"Signal Frequency: {config['signal_frequency']}")
    print(f"Trading Hours: {config['trading_hours']['start']} - {config['trading_hours']['end']} {config['trading_hours']['timezone']}")
    print(f"Rolling Window: {config['window']} periods")
    print(f"Dynamic Thresholds: {config['use_dynamic_thresholds']}")
    if config['use_dynamic_thresholds']:
        print(f"Distribution Type: {config['distribution_type']}")
        print(f"Confidence Level: {config['confidence_level']}")
    else:
        print(f"Entry Threshold: {config['entry_threshold']}")
        print(f"Exit Threshold: {config['exit_threshold']}")
    print("="*60)
    
    # Comprehensive backtest analysis (no risk-free rates required)
    analyzer = BacktestAnalyzer()
    
    df_analyzed, analysis_summary = analyzer.analyze_backtest(
        df, 
        initial_capital=strat.initial_capital
    )
    
    # Print comprehensive performance summary
    analyzer.print_summary(analysis_summary)
    
    # Debug return calculations
    analyzer.debug_returns(df_analyzed)
    
    # Plot strategy-specific results
    strat.plot_spread(df)
    strat.plot_pnl(df)
    
    # Show distribution analysis only when using dynamic thresholds
    if USE_DYNAMIC_THRESHOLDS:
        strat.plot_distribution_analysis(df)
        strat.plot_dynamic_thresholds(df)
    
    # Plot comprehensive performance analysis
    analyzer.plot_performance_analysis(df_analyzed, analysis_summary)
    
    print("\nCointegration Test Results:")
    print(f"  Statistic = {stats['coint_score']:.4f}")
    print(f"  p-value   = {stats['coint_pval']:.4f}")
    
    # Print dynamic threshold statistics only when using dynamic thresholds
    if USE_DYNAMIC_THRESHOLDS and strat.use_dynamic_thresholds and 'dynamic_entry_threshold' in df.columns:
        entry_thresholds = df['dynamic_entry_threshold'].dropna()
        exit_thresholds = df['dynamic_exit_threshold'].dropna()
        
        print(f"\nDynamic Threshold Statistics:")
        print(f"  Entry threshold - Mean: {entry_thresholds.mean():.3f}, Std: {entry_thresholds.std():.3f}")
        print(f"  Exit threshold - Mean: {exit_thresholds.mean():.3f}, Std: {exit_thresholds.std():.3f}")
        print(f"  Entry threshold range: [{entry_thresholds.min():.3f}, {entry_thresholds.max():.3f}]")
        print(f"  Exit threshold range: [{exit_thresholds.min():.3f}, {exit_thresholds.max():.3f}]")
    elif not USE_DYNAMIC_THRESHOLDS:
        print(f"\nUsing Fixed Thresholds:")
        print(f"  Entry threshold: {strat.entry_threshold}")
        print(f"  Exit threshold: {strat.exit_threshold}")

if __name__ == "__main__":
    asyncio.run(main())
