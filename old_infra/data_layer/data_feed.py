"""
Generic data fetching utilities for the trading system.
Provides clean, flexible interfaces for various IB API data requests.
"""

import datetime as dt
import pandas as pd
import numpy as np
import asyncio
from typing import Optional, Dict, List, Union, Any
from ib_async import IB
from ib_async.contract import Stock, Contract, Future, Option, Crypto
from infrastructure.config import config


# ============================================================================
# CONTRACT MANAGEMENT
# ============================================================================

async def qualify_contract(
    ib: IB,
    contract: Contract
) -> Optional[Contract]:
    """
    Qualify a contract to get full contract details.
    
    Args:
        ib: IB connection instance
        contract: Contract to qualify
        
    Returns:
        Qualified contract or None if qualification fails
    """
    try:
        qualified_contracts = await ib.qualifyContractsAsync(contract)
        return qualified_contracts[0] if qualified_contracts else None
    except Exception as e:
        print(f"Error qualifying contract {contract.symbol}: {e}")
        return None


def create_stock_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD"
) -> Stock:
    """Create a stock contract."""
    return Stock(symbol, exchange, currency)


def create_future_contract(
    symbol: str,
    exchange: str = "GLOBEX",
    currency: str = "USD",
    last_trading_day: str = None,
    multiplier: str = None
) -> Future:
    """Create a future contract."""
    return Future(
        symbol, 
        lastTradeDateOrContractMonth=last_trading_day or "",
        exchange=exchange, 
        currency=currency,
        multiplier=multiplier or ""
    )


def create_option_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
    last_trading_day: str = None,
    strike: float = None,
    right: str = None,
    multiplier: str = None
) -> Option:
    """Create an option contract."""
    return Option(
        symbol,
        lastTradeDateOrContractMonth=last_trading_day or "",
        exchange=exchange,
        currency=currency,
        strike=strike or 0.0,
        right=right or "",
        multiplier=multiplier or ""
    )


def create_crypto_contract(
    symbol: str,
    exchange: str = "ZEROHASH",
    currency: str = "USD"
) -> Crypto:
    """Create a crypto contract."""
    return Crypto(
        symbol,
        exchange=exchange,
        currency=currency
    )


# ============================================================================
# HISTORICAL DATA
# ============================================================================

async def fetch_historical_bars(
    ib: IB,
    contract: Contract,
    end_datetime: Optional[dt.datetime] = None,
    duration: str = "1 D",
    bar_size: str = None,
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    format_date: int = 1
) -> pd.DataFrame:
    """
    Fetch historical bars for any contract type.
    
    Args:
        ib: IB connection instance
        contract: Contract to fetch data for
        end_datetime: End time for data request (defaults to now)
        duration: Duration string (e.g., "1 D", "5 D", "1 M")
        bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour")
        what_to_show: Data type ("TRADES", "BID", "ASK", "MIDPOINT", etc.)
        use_rth: Use regular trading hours only
        format_date: Date format (1 for string, 2 for seconds)
        
    Returns:
        DataFrame with OHLCV data
    """
    if end_datetime is None:
        end_datetime = dt.datetime.now()
    
    # Use config bar_size if not specified
    if bar_size is None:
        bar_size = config.DATA_CONFIG['bar_size']

    end_str = end_datetime.strftime("%Y%m%d %H:%M:%S")

    try:
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
            'volume': bar.volume,
            'wap': getattr(bar, 'wap', None),
            'bar_count': getattr(bar, 'barCount', None)
        } for bar in bars])
        
    except Exception as e:
        print(f"Error fetching historical bars for {contract.symbol}: {e}")
        return pd.DataFrame()





# ============================================================================
# REAL-TIME DATA
# ============================================================================

def start_real_time_bars(
    ib: IB,
    contract: Contract,
    last_bar: Dict[str, Any],
    bar_size: int = 5,
    what_to_show: str = "TRADES",
    use_rth: bool = True
) -> Any:
    """
    Start a persistent real-time bars subscription.
    
    This creates a continuous subscription that updates the last_bar reference
    every 5 seconds. The subscription must be cancelled manually using cancel_real_time_bars().
    
    Args:
        ib: IB connection instance
        contract: Contract to fetch data for
        last_bar: Dictionary reference that will be updated with the latest bar
        bar_size: Bar size in seconds (must be 5 for real-time bars)
        what_to_show: Data type ("TRADES", "BID", "ASK", "MIDPOINT", etc.)
        use_rth: Use regular trading hours only
        
    Returns:
        RealTimeBarsList object (needed for cancellation)
    """
    if bar_size != 5:
        raise ValueError("Real-time bars only support 5-second intervals")
    
    try:
        # Start the real-time bars subscription
        bars_container = ib.reqRealTimeBars(
            contract,
            barSize=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth
        )
        
        # Set up callback to update last_bar whenever a new bar arrives
        def on_bar_update(bars, has_new_bar):
            if has_new_bar and len(bars) > 0:
                latest_bar = bars[-1]  # Get the most recent bar
                last_bar.update({
                    'datetime': pd.to_datetime(latest_bar.time),
                    'open': latest_bar.open,
                    'high': latest_bar.high,
                    'low': latest_bar.low,
                    'close': latest_bar.close,
                    'volume': latest_bar.volume,
                    'wap': getattr(latest_bar, 'wap', None),
                    'bar_count': getattr(latest_bar, 'barCount', None),
                    'symbol': contract.symbol,
                    'timestamp': pd.Timestamp.now()
                })
                print(f"Updated {contract.symbol} bar: {latest_bar.close} at {latest_bar.time}")
        
        # Connect the callback
        bars_container.updateEvent += on_bar_update
        
        print(f"Started real-time bars subscription for {contract.symbol}")
        print(f"Bars will be updated every {bar_size} seconds in the last_bar reference")
        
        return bars_container
        
    except Exception as e:
        print(f"Error starting real-time bars for {contract.symbol}: {e}")
        return None


def cancel_real_time_bars(ib: IB, bars_container: Any, contract: Contract = None) -> bool:
    """
    Cancel a real-time bars subscription.
    
    Args:
        ib: IB connection instance
        bars_container: The RealTimeBarsList object returned by start_real_time_bars
        contract: Optional contract (for logging only)
        
    Returns:
        True if successfully cancelled, False otherwise
    """
    try:
        if bars_container is not None:
            ib.cancelRealTimeBars(bars_container)
            symbol = contract.symbol if contract else "unknown"
            print(f"Cancelled real-time bars subscription for {symbol}")
            return True
    except Exception as e:
        symbol = contract.symbol if contract else "unknown"
        print(f"Error cancelling real-time bars subscription for {symbol}: {e}")
    
    return False








async def fetch_market_data(
    ib: IB,
    contract: Contract,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """
    Fetch real-time market data for any contract type.
    
    Args:
        ib: IB connection instance
        contract: Contract to fetch data for
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with market data fields
    """
    try:
        ticker = ib.reqMktData(contract)
        
        # Wait for data with timeout
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.1)
            if ticker.last or ticker.close or ticker.marketPrice():
                break
        
        data = {
            'symbol': contract.symbol,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'last': ticker.last,
            'close': ticker.close,
            'high': ticker.high,
            'low': ticker.low,
            'open': ticker.open,
            'volume': ticker.volume,
            'market_price': ticker.marketPrice(),
            'timestamp': pd.Timestamp.now()
        }
        
        # Calculate derived fields
        if data['bid'] and data['ask']:
            data['spread'] = data['ask'] - data['bid']
            data['spread_pct'] = data['spread'] / data['bid'] * 100
            data['mid_price'] = (data['bid'] + data['ask']) / 2
        
        # Cancel market data subscription
        ib.cancelMktData(contract)
        
        return data
        
    except Exception as e:
        print(f"Error fetching market data for {contract.symbol}: {e}")
        return {}





# ============================================================================
# CONTRACT DETAILS
# ============================================================================

async def fetch_contract_details(
    ib: IB,
    contract: Contract
) -> List[Dict[str, Any]]:
    """
    Fetch detailed contract information.
    
    Args:
        ib: IB connection instance
        contract: Contract to get details for
        
    Returns:
        List of contract details dictionaries
    """
    try:
        details = await ib.reqContractDetailsAsync(contract)
        return [{
            'contract': detail.contract,
            'market_name': detail.marketName,
            'min_tick': detail.minTick,
            'order_types': detail.orderTypes,
            'valid_exchanges': detail.validExchanges,
            'price_magnifier': detail.priceMagnifier,
            'under_conid': detail.underConId,
            'long_name': detail.longName,
            'contract_month': detail.contractMonth,
            'industry': detail.industry,
            'category': detail.category,
            'subcategory': detail.subcategory,
            'time_zone_id': detail.timeZoneId,
            'trading_hours': detail.tradingHours,
            'liquid_hours': detail.liquidHours,
            'ev_rule': detail.evRule,
            'ev_multiplier': detail.evMultiplier,
            'md_size_multiplier': detail.mdSizeMultiplier,
            'agg_group': detail.aggGroup,
            'under_symbol': detail.underSymbol,
            'under_sec_type': detail.underSecType,
            'market_rule_ids': detail.marketRuleIds,
            'sec_id_list': detail.secIdList,
            'real_expiration_date': detail.realExpirationDate,
            'last_trading_day': detail.lastTradingDay,
            'stock_type': detail.stockType,
            'min_size': detail.minSize,
            'size_increment': detail.sizeIncrement,
            'suggested_size_increment': detail.suggestedSizeIncrement
        } for detail in details]
        
    except Exception as e:
        print(f"Error fetching contract details for {contract.symbol}: {e}")
        return []


# ============================================================================
# ACCOUNT DATA
# ============================================================================

async def fetch_account_summary(
    ib: IB,
    tags: List[str] = None
) -> Dict[str, Any]:
    """
    Fetch account summary information.
    
    Args:
        ib: IB connection instance
        tags: List of account tags to fetch (defaults to common tags)
        
    Returns:
        Dictionary with account summary data
    """
    if tags is None:
        tags = [
            'NetLiquidation',
            'TotalCashValue',
            'SettledCash',
            'AccruedCash',
            'BuyingPower',
            'EquityWithLoanValue',
            'PreviousDayEquityWithLoanValue',
            'GrossPositionValue',
            'RegTMargin',
            'InitialMargin',
            'MaintenanceMargin',
            'AvailableFunds',
            'ExcessLiquidity',
            'Cushion',
            'FullInitMarginReq',
            'FullMaintMarginReq',
            'FullAvailableFunds',
            'FullExcessLiquidity',
            'Currency',
            'TotalCashValue',
            'NetLiquidation'
        ]
    
    try:
        summary = await ib.reqAccountSummaryAsync(tags)
        return {item.tag: item.value for item in summary}
        
    except Exception as e:
        print(f"Error fetching account summary: {e}")
        return {}


async def fetch_positions(
    ib: IB
) -> List[Dict[str, Any]]:
    """
    Fetch current positions.
    
    Args:
        ib: IB connection instance
        
    Returns:
        List of position dictionaries
    """
    try:
        positions = await ib.reqPositionsAsync()
        return [{
            'account': position.account,
            'contract': position.contract,
            'position': position.position,
            'avg_cost': position.avgCost,
            'symbol': position.contract.symbol,
            'sec_type': position.contract.secType,
            'exchange': position.contract.exchange,
            'currency': position.contract.currency
        } for position in positions]
        
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def fetch_pair_data(
    ib: IB,
    contract1: Contract,
    contract2: Contract,
    **kwargs
) -> pd.DataFrame:
    """
    Fetch data for a pair of contracts and merge them.
    
    Args:
        ib: IB connection instance
        contract1: First contract
        contract2: Second contract
        **kwargs: Additional arguments passed to fetch_historical_bars
        
    Returns:
        DataFrame with merged data
    """
    df1 = await fetch_historical_bars(ib, contract1, **kwargs)
    df2 = await fetch_historical_bars(ib, contract2, **kwargs)
    
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    df = pd.merge(
        df1[['datetime', 'close']],
        df2[['datetime', 'close']],
        on='datetime',
        how='inner',
        suffixes=(f'_{contract1.symbol}', f'_{contract2.symbol}')
    )
    return df.dropna()


def filter_trading_hours(
    df: pd.DataFrame,
    market_open_time: str = None,
    market_close_time: str = None,
    timezone: str = None
) -> pd.DataFrame:
    """
    Filter DataFrame to only include regular trading hours.
    Uses global config for default values.
    """
    # Use config defaults if not specified
    if market_open_time is None:
        market_open_time = config.DATA_CONFIG['trading_hours']['start']
    if market_close_time is None:
        market_close_time = config.DATA_CONFIG['trading_hours']['end']
    if timezone is None:
        timezone = config.DATA_CONFIG['trading_hours']['timezone']
    
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


def detect_trading_hours_automatically(
    df: pd.DataFrame,
    volatility_threshold: float = 0.001,
    min_consecutive_periods: int = 30
) -> pd.Series:
    """
    Automatically detect trading vs non-trading hours based on spread volatility.
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


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# Keep the old function name for backward compatibility
async def fetch_stock_bars(
    ib: IB,
    ticker: str,
    exchange: str = "ARCA",
    currency: str = "USD",
    end_datetime: dt.datetime = None,
    duration: str = "1 D",
    bar_size: str = None,
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    format_date: int = 1
) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    Use fetch_historical_bars_simple instead.
    """
    return await fetch_historical_bars_simple(
        ib, ticker, "stock", exchange, currency,
        end_datetime=end_datetime,
        duration=duration,
        bar_size=bar_size,
        what_to_show=what_to_show,
        use_rth=use_rth,
        format_date=format_date
    )
