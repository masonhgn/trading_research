import datetime as dt
import pandas as pd
from ib_async import IB, Stock

# Import the trading config from etf_arb.py
try:
    from etf_arb import TRADING_CONFIG
except ImportError:
    # Fallback config if etf_arb.py is not available
    TRADING_CONFIG = {
        'bar_size': '1 min',
        'signal_frequency': '1 min',
        'trading_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    }

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
