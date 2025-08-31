"""
Mock data feed for testing purposes.
Provides mock implementations of all data_feed functions.
"""

import datetime as dt
import pandas as pd
import numpy as np
import asyncio
from typing import Optional, Dict, List, Union, Any
from ib_async.contract import Stock, Contract, Future, Option, Crypto


# ============================================================================
# MOCK CONTRACT CREATION
# ============================================================================

def create_mock_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    """Create a mock stock contract."""
    return Stock(symbol, exchange, currency)


def create_mock_future_contract(
    symbol: str,
    exchange: str = "GLOBEX",
    currency: str = "USD",
    last_trading_day: str = None,
    multiplier: str = None
) -> Future:
    """Create a mock future contract."""
    return Future(
        symbol, 
        lastTradeDateOrContractMonth=last_trading_day or "",
        exchange=exchange, 
        currency=currency,
        multiplier=multiplier or ""
    )


def create_mock_option_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
    last_trading_day: str = None,
    strike: float = None,
    right: str = None,
    multiplier: str = None
) -> Option:
    """Create a mock option contract."""
    return Option(
        symbol,
        lastTradeDateOrContractMonth=last_trading_day or "",
        exchange=exchange,
        currency=currency,
        strike=strike or 0.0,
        right=right or "",
        multiplier=multiplier or ""
    )


def create_mock_crypto_contract(symbol: str, exchange: str = "PAXOS", currency: str = "USD") -> Crypto:
    """Create a mock crypto contract."""
    return Crypto(symbol, exchange=exchange, currency=currency)


# ============================================================================
# MOCK CONTRACT MANAGEMENT
# ============================================================================

async def qualify_contract_mock(ib: Any, contract: Contract) -> Optional[Contract]:
    """Mock contract qualification."""
    # Simulate async delay
    await asyncio.sleep(0.1)
    
    # Return the same contract with some mock details
    if hasattr(contract, 'symbol'):
        print(f"Mock: Qualified contract {contract.symbol}")
    return contract


# ============================================================================
# MOCK HISTORICAL DATA
# ============================================================================

def generate_mock_ohlcv_data(
    start_time: dt.datetime,
    periods: int = 100,
    base_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Generate realistic mock OHLCV data."""
    timestamps = pd.date_range(start=start_time, periods=periods, freq='5min')
    
    # Generate price movements
    returns = np.random.normal(0, volatility, periods)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))  # Ensure positive prices
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close price
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else close
        
        # Ensure OHLC relationship
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = int(np.random.exponential(1000))
        wap = (high + low + close) / 3
        
        data.append({
            'datetime': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'wap': round(wap, 2),
            'bar_count': np.random.randint(50, 200)
        })
    
    return pd.DataFrame(data)


async def fetch_historical_bars_mock(
    ib: Any,
    contract: Contract,
    end_datetime: Optional[dt.datetime] = None,
    duration: str = "1 D",
    bar_size: str = "5 mins",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    format_date: int = 1
) -> pd.DataFrame:
    """Mock historical bars data."""
    # Simulate async delay
    await asyncio.sleep(0.1)
    
    if end_datetime is None:
        end_datetime = dt.datetime.now()
    
    # Parse duration to get number of periods
    duration_map = {
        "1 D": 288,  # 24 hours * 12 (5-min bars per hour)
        "5 D": 1440,
        "1 M": 8640,
        "3 M": 25920
    }
    periods = duration_map.get(duration, 288)
    
    # Adjust periods based on bar size
    if "min" in bar_size:
        mins = int(bar_size.split()[0])
        periods = int(periods * 5 / mins)
    elif "hour" in bar_size:
        hours = int(bar_size.split()[0])
        periods = int(periods * 5 / (hours * 60))
    
    # Generate mock data
    base_price = 100.0
    if hasattr(contract, 'symbol'):
        # Use symbol to create some variation in base price
        base_price = 50.0 + hash(contract.symbol) % 200
    
    df = generate_mock_ohlcv_data(end_datetime - dt.timedelta(minutes=periods*5), periods, base_price)
    
    print(f"Mock: Generated {len(df)} historical bars for {getattr(contract, 'symbol', 'unknown')}")
    return df


# ============================================================================
# MOCK REAL-TIME DATA
# ============================================================================

class MockRealTimeBarsContainer:
    """Mock container for real-time bars."""
    def __init__(self, contract: Contract):
        self.contract = contract
        self.updateEvent = None
        self._running = True
        self._last_bar = None
    
    def start_mock_updates(self, last_bar: Dict[str, Any], callback):
        """Start mock real-time updates."""
        self.updateEvent = callback
        
        async def update_loop():
            while self._running:
                await asyncio.sleep(5)  # 5-second intervals
                if self._running:
                    # Generate new mock bar
                    mock_bar = self._generate_mock_bar()
                    last_bar.update(mock_bar)
                    
                    # Call the callback
                    if self.updateEvent:
                        self.updateEvent([mock_bar], True)
        
        # Start the update loop
        asyncio.create_task(update_loop())
    
    def _generate_mock_bar(self) -> Dict[str, Any]:
        """Generate a mock real-time bar."""
        base_price = 100.0
        if hasattr(self.contract, 'symbol'):
            base_price = 50.0 + hash(self.contract.symbol) % 200
        
        # Generate realistic price movement
        price_change = np.random.normal(0, 0.01)
        close = base_price * (1 + price_change)
        
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = self._last_bar['close'] if self._last_bar else close
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        bar = {
            'datetime': pd.to_datetime(dt.datetime.now()),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': int(np.random.exponential(1000)),
            'wap': round((high + low + close) / 3, 2),
            'bar_count': np.random.randint(50, 200),
            'symbol': getattr(self.contract, 'symbol', 'unknown'),
            'timestamp': pd.Timestamp.now()
        }
        
        self._last_bar = bar
        return bar
    
    def stop(self):
        """Stop the mock updates."""
        self._running = False


def start_real_time_bars_mock(
    ib: Any,
    contract: Contract,
    last_bar: Dict[str, Any],
    bar_size: int = 5,
    what_to_show: str = "TRADES",
    use_rth: bool = True
) -> MockRealTimeBarsContainer:
    """Mock real-time bars subscription."""
    if bar_size != 5:
        raise ValueError("Real-time bars only support 5-second intervals")
    
    container = MockRealTimeBarsContainer(contract)
    
    def mock_callback(bars, has_new_bar):
        if has_new_bar and len(bars) > 0:
            latest_bar = bars[-1]
            print(f"Mock: Updated {contract.symbol} bar: {latest_bar['close']} at {latest_bar['datetime']}")
    
    container.start_mock_updates(last_bar, mock_callback)
    print(f"Mock: Started real-time bars subscription for {getattr(contract, 'symbol', 'unknown')}")
    
    return container


def cancel_real_time_bars_mock(ib: Any, bars_container: MockRealTimeBarsContainer, contract: Contract = None) -> bool:
    """Mock cancel real-time bars."""
    try:
        if bars_container is not None:
            bars_container.stop()
            symbol = getattr(contract, 'symbol', 'unknown') if contract else "unknown"
            print(f"Mock: Cancelled real-time bars subscription for {symbol}")
            return True
    except Exception as e:
        symbol = getattr(contract, 'symbol', 'unknown') if contract else "unknown"
        print(f"Mock: Error cancelling real-time bars subscription for {symbol}: {e}")
    
    return False


async def fetch_market_data_mock(
    ib: Any,
    contract: Contract,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """Mock market data."""
    # Simulate async delay
    await asyncio.sleep(0.1)
    
    base_price = 100.0
    if hasattr(contract, 'symbol'):
        base_price = 50.0 + hash(contract.symbol) % 200
    
    # Generate realistic market data
    spread = base_price * 0.001  # 0.1% spread
    bid = base_price - spread / 2
    ask = base_price + spread / 2
    
    data = {
        'symbol': getattr(contract, 'symbol', 'unknown'),
        'bid': round(bid, 2),
        'ask': round(ask, 2),
        'last': round(base_price, 2),
        'close': round(base_price * 0.99, 2),
        'high': round(base_price * 1.02, 2),
        'low': round(base_price * 0.98, 2),
        'open': round(base_price * 1.01, 2),
        'volume': int(np.random.exponential(10000)),
        'market_price': round(base_price, 2),
        'timestamp': pd.Timestamp.now(),
        'spread': round(spread, 2),
        'spread_pct': round(spread / bid * 100, 2),
        'mid_price': round((bid + ask) / 2, 2)
    }
    
    print(f"Mock: Fetched market data for {data['symbol']}")
    return data


# ============================================================================
# MOCK CONTRACT DETAILS
# ============================================================================

async def fetch_contract_details_mock(ib: Any, contract: Contract) -> List[Dict[str, Any]]:
    """Mock contract details."""
    # Simulate async delay
    await asyncio.sleep(0.1)
    
    symbol = getattr(contract, 'symbol', 'unknown')
    
    details = [{
        'contract': contract,
        'market_name': 'Mock Market',
        'min_tick': 0.01,
        'order_types': 'LMT,MKT,STP,STP LMT',
        'valid_exchanges': 'SMART,ARCA,EDGX',
        'price_magnifier': 1,
        'under_conid': 0,
        'long_name': f'Mock {symbol} Stock',
        'contract_month': '',
        'industry': 'Technology',
        'category': 'Stock',
        'subcategory': 'Common Stock',
        'time_zone_id': 'US/Eastern',
        'trading_hours': '09:30-16:00',
        'liquid_hours': '09:30-16:00',
        'ev_rule': '',
        'ev_multiplier': 1,
        'md_size_multiplier': 1,
        'agg_group': 0,
        'under_symbol': symbol,
        'under_sec_type': 'STK',
        'market_rule_ids': '1,2,3',
        'sec_id_list': [],
        'real_expiration_date': '',
        'last_trading_day': '',
        'stock_type': 'Common',
        'min_size': 1,
        'size_increment': 1,
        'suggested_size_increment': 1
    }]
    
    print(f"Mock: Fetched contract details for {symbol}")
    return details


# ============================================================================
# MOCK ACCOUNT DATA
# ============================================================================

async def fetch_account_summary_mock(ib: Any, tags: List[str] = None) -> Dict[str, Any]:
    """Mock account summary."""
    # Simulate async delay
    await asyncio.sleep(0.1)
    
    if tags is None:
        tags = [
            'NetLiquidation', 'TotalCashValue', 'SettledCash', 'AccruedCash',
            'BuyingPower', 'EquityWithLoanValue', 'PreviousDayEquityWithLoanValue',
            'GrossPositionValue', 'RegTMargin', 'InitialMargin', 'MaintenanceMargin',
            'AvailableFunds', 'ExcessLiquidity', 'Cushion', 'FullInitMarginReq',
            'FullMaintMarginReq', 'FullAvailableFunds', 'FullExcessLiquidity',
            'Currency', 'TotalCashValue', 'NetLiquidation'
        ]
    
    # Generate realistic account data
    base_value = 100000.0
    
    summary = {}
    for tag in tags:
        if 'Cash' in tag or 'Funds' in tag:
            summary[tag] = str(round(base_value * np.random.uniform(0.8, 1.2), 2))
        elif 'Margin' in tag:
            summary[tag] = str(round(base_value * np.random.uniform(0.1, 0.3), 2))
        elif 'Power' in tag:
            summary[tag] = str(round(base_value * np.random.uniform(1.5, 2.0), 2))
        elif 'Liquidation' in tag or 'Equity' in tag:
            summary[tag] = str(round(base_value * np.random.uniform(0.9, 1.1), 2))
        elif tag == 'Currency':
            summary[tag] = 'USD'
        else:
            summary[tag] = str(round(base_value * np.random.uniform(0.5, 1.5), 2))
    
    print("Mock: Fetched account summary")
    return summary


async def fetch_positions_mock(ib: Any) -> List[Dict[str, Any]]:
    """Mock positions data."""
    # Simulate async delay
    await asyncio.sleep(0.1)
    
    # Generate mock positions
    mock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    positions = []
    
    for i, symbol in enumerate(mock_symbols):
        if np.random.random() > 0.3:  # 70% chance of having a position
            position = {
                'account': 'DU123456',
                'contract': create_mock_stock_contract(symbol),
                'position': np.random.randint(-100, 100),
                'avg_cost': round(100 + np.random.uniform(-20, 20), 2),
                'symbol': symbol,
                'sec_type': 'STK',
                'exchange': 'SMART',
                'currency': 'USD'
            }
            positions.append(position)
    
    print(f"Mock: Fetched {len(positions)} positions")
    return positions


# ============================================================================
# MOCK UTILITY FUNCTIONS
# ============================================================================

async def fetch_pair_data_mock(
    ib: Any,
    contract1: Contract,
    contract2: Contract,
    **kwargs
) -> pd.DataFrame:
    """Mock pair data fetching."""
    # Simulate async delay
    await asyncio.sleep(0.1)
    
    df1 = await fetch_historical_bars_mock(ib, contract1, **kwargs)
    df2 = await fetch_historical_bars_mock(ib, contract2, **kwargs)
    
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    # Merge the data
    df = pd.merge(
        df1[['datetime', 'close']],
        df2[['datetime', 'close']],
        on='datetime',
        how='inner',
        suffixes=(f'_{getattr(contract1, "symbol", "contract1")}', 
                 f'_{getattr(contract2, "symbol", "contract2")}')
    )
    
    print(f"Mock: Generated pair data for {getattr(contract1, 'symbol', 'contract1')} and {getattr(contract2, 'symbol', 'contract2')}")
    return df.dropna()


def filter_trading_hours_mock(
    df: pd.DataFrame,
    market_open_time: str = "09:30",
    market_close_time: str = "16:00",
    timezone: str = "US/Eastern"
) -> pd.DataFrame:
    """Mock trading hours filter."""
    if df.empty:
        return df
    
    # Simple mock implementation - just return a subset of the data
    # In a real implementation, this would filter by actual trading hours
    filtered_df = df.iloc[::2].copy()  # Take every other row as mock filtering
    
    print(f"Mock: Filtered {len(df)} rows to {len(filtered_df)} trading hours rows")
    return filtered_df


def detect_trading_hours_automatically_mock(
    df: pd.DataFrame,
    volatility_threshold: float = 0.001,
    min_consecutive_periods: int = 30
) -> pd.Series:
    """Mock automatic trading hours detection."""
    if df.empty:
        return pd.Series(dtype=bool)
    
    # Mock implementation - randomly assign trading hours
    # In a real implementation, this would analyze volatility patterns
    trading_mask = pd.Series(np.random.choice([True, False], size=len(df), p=[0.7, 0.3]), index=df.index)
    
    print(f"Mock: Detected trading hours for {trading_mask.sum()} out of {len(df)} periods")
    return trading_mask


# ============================================================================
# MOCK LEGACY COMPATIBILITY
# ============================================================================

async def fetch_stock_bars_mock(
    ib: Any,
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
    """Mock legacy stock bars function."""
    contract = create_mock_stock_contract(ticker, exchange, currency)
    return await fetch_historical_bars_mock(
        ib, contract,
        end_datetime=end_datetime,
        duration=duration,
        bar_size=bar_size,
        what_to_show=what_to_show,
        use_rth=use_rth,
        format_date=format_date
    )
    






if __name__ == "__main__":
    barz = []
    start_real_time_bars_mock(
        None,
        None,
        barz,
    )
    