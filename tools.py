import datetime as dt
import pandas as pd
from ib_async import IB, Stock

async def fetch_stock_minute_bars(
    ib: IB,
    ticker: str,
    exchange: str = "ARCA",  # ARCA is common for ETFs
    currency: str = "USD",
    end_datetime: dt.datetime = None,
    duration: str = "1 D",
    bar_size: str = "1 min",
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
