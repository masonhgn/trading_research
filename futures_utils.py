import datetime as dt
import asyncio
from typing import List, Dict, Optional
from ib_async import IB, Future

# ----------------- tiny utils (sync, unchanged except naming) -----------------

def ib_expiry_to_datetime(exp_str: str) -> dt.datetime:
    if len(exp_str) == 8:  # YYYYMMDD
        y, m, d = int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8])
        return dt.datetime(y, m, d, 23, 59, 59)
    elif len(exp_str) == 6:  # YYYYMM
        y, m = int(exp_str[:4]), int(exp_str[4:6])
        nxt = dt.date(y + (m == 12), 1 if m == 12 else m + 1, 1)
        last = nxt - dt.timedelta(days=1)
        return dt.datetime(last.year, last.month, last.day, 23, 59, 59)
    raise ValueError(f"Unrecognized expiry format: {exp_str}")

def _bar_date_to_date(bardate) -> dt.date:
    if isinstance(bardate, str):
        return dt.datetime.strptime(bardate[:8], "%Y%m%d").date()
    elif isinstance(bardate, dt.datetime):
        return bardate.date()
    elif isinstance(bardate, dt.date):
        return bardate
    else:
        raise TypeError(f"Unsupported bar date type: {type(bardate)}")

# ----------------- contract-chain filter: next N expiries -----------------

async def list_next_expiries(
    ib: IB,
    symbol: str,
    exchange: str,
    currency: str,
    *,
    anchor_day: dt.date,
    next_n: int = 6,
    min_days_to_expiry: int = 0,
) -> List[Dict]:
    """
    Return metadata for the next `next_n` expiries after `anchor_day` (and >= min DTE).
    Output: [{"contract", "expiry", "days_to_expiry"}, ...] sorted by expiry.
    """
    base = Future(symbol, exchange=exchange, currency=currency)
    details = await ib.reqContractDetailsAsync(base)
    if not details:
        return []

    rows = []
    for cd in details:
        c = cd.contract
        if c.exchange != exchange or c.currency != currency or not c.lastTradeDateOrContractMonth:
            continue
        exp = ib_expiry_to_datetime(c.lastTradeDateOrContractMonth)
        dte = (exp.date() - anchor_day).days
        if dte < min_days_to_expiry:
            continue
        if exp.date() < anchor_day:   # already expired relative to the day we care about
            continue
        rows.append({"contract": c, "expiry": exp, "days_to_expiry": dte})

    rows.sort(key=lambda r: r["expiry"])
    return rows[:next_n]

# ----------------- traded-on-day (only for the next N expiries) -----------------

async def list_traded_futures_on_day(
    ib: IB,
    symbol: str,
    exchange: str,
    currency: str,
    day: dt.date,
    *,
    next_n: int = 6,                 # e.g., only check the next 6 expiries
    min_days_to_expiry: int = 0,     # optional DTE filter
    concurrent: int = 3,             # modest concurrency to respect pacing
    what_to_show: str = "TRADES",
    pause: float = 0.12              # gentle pacing between calls
) -> List[Dict]:
    """
    For the next N expiries after `day`, fetch that day's daily bar and return
    contracts that traded (>0 volume). Output sorted by expiry.
    """
    cand = await list_next_expiries(
        ib, symbol, exchange, currency,
        anchor_day=day, next_n=next_n, min_days_to_expiry=min_days_to_expiry
    )
    if not cand:
        return []

    end_dt_str = (dt.datetime.combine(day, dt.time(23, 59, 59)) + dt.timedelta(days=1)).strftime("%Y%m%d %H:%M:%S")
    sem = asyncio.Semaphore(concurrent)

    async def fetch_one(meta):
        c, exp, dte = meta["contract"], meta["expiry"], meta["days_to_expiry"]
        async with sem:
            try:
                bars = await ib.reqHistoricalDataAsync(
                    c,
                    endDateTime=end_dt_str,
                    durationStr="2 D",
                    barSizeSetting="1 day",
                    whatToShow=what_to_show,
                    useRTH=False,
                    formatDate=1,
                )
                vol = 0
                for b in bars or []:
                    if _bar_date_to_date(b.date) == day:
                        vol = int(getattr(b, "volume", 0) or 0)
                        break
                return {"contract": c, "expiry": exp, "day": day, "volume": vol, "days_to_expiry": dte}
            finally:
                await asyncio.sleep(pause)

    results = await asyncio.gather(*(fetch_one(m) for m in cand))
    traded = [r for r in results if r and r["volume"] > 0]
    traded.sort(key=lambda x: x["expiry"])
    return traded

# ----------------- top-k (sync helper; unchanged) -----------------

def top_k_by_volume(
    contracts_info: List[Dict],
    k: int,
    min_days_to_expiry: Optional[int] = None
) -> List[Dict]:
    if min_days_to_expiry is not None:
        contracts_info = [x for x in contracts_info if x["days_to_expiry"] >= min_days_to_expiry]
    if not contracts_info:
        return []
    return sorted(contracts_info, key=lambda x: (x["volume"], -x["days_to_expiry"]), reverse=True)[:k]









async def fetch_minute_bars(
    ib: IB,
    contract: Future,
    end_datetime: Optional[dt.datetime] = None,
    duration: str = "1 D",
    bar_size: str = "1 min",
    what_to_show: str = "TRADES",
    use_rth: bool = False,
    format_date: int = 1
):
    """
    Fetch 1-minute bars for a given futures contract.

    Args:
        ib: Connected IB instance.
        contract: The IB `Future` contract object.
        end_datetime: Optional datetime for the end of the query (defaults to now).
        duration: IB duration string (e.g. "1 D", "3 H", etc.)
        bar_size: IB bar size (e.g. "1 min").
        what_to_show: "TRADES", "MIDPOINT", etc.
        use_rth: Whether to only use regular trading hours.
        format_date: Whether to return bars with datetime objects (1) or raw strings (2).

    Returns:
        List of bar objects (open, high, low, close, volume, etc.)
    """
    if end_datetime is None:
        end_datetime = dt.datetime.now()

    end_str = end_datetime.strftime("%Y%m%d %H:%M:%S")

    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime=end_str,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=format_date
    )

    return bars or []
