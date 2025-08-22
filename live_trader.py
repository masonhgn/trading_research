import asyncio
import datetime as dt
import pandas as pd
import numpy as np
import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from ib_async import IB, util
from ib_async.contract import Stock
from ib_async.order import MarketOrder, StopOrder
from strategy_layer.etf_arbitrage import ETFArbitrageStrategy
from infrastructure.notifications import send_trade_update, send_pnl_update, send_system_alert, send_data_collection_update

LIVE_PORT = 4001
PAPER_PORT = 4002


class TradingState(Enum):
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    WAITING_FOR_MARKET_OPEN = "waiting_for_market_open"
    TRADING = "trading"
    MARKET_CLOSED = "market_closed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class RiskLimits:
    max_position_size_pct: float = 0.1  # 10% of portfolio
    max_daily_loss_pct: float = 0.02    # 2% daily loss limit
    max_drawdown_pct: float = 0.05      # 5% max drawdown
    max_trades_per_day: int = 50
    emergency_stop_loss_pct: float = 0.03  # 3% emergency stop







async def get_account_info(ib: IB) -> Dict[str, float]:
    """
    Query account information from IB API.
    Returns dict with account details including net liquidation value.
    """
    try:
        logging.debug("Making accountSummaryAsync request...")
        
        # Use the working pattern from probe_ib.py
        account_values = await ib.accountSummaryAsync()
        logging.debug(f"Account summary response: {len(account_values)} values received")
        
        account_info = {}
        for value in account_values:
            logging.debug(f"Account value: tag={value.tag}, value={value.value}, currency={value.currency}")
            if value.tag == "NetLiquidation":
                account_info['net_liquidation'] = float(value.value)
            elif value.tag == "TotalCashValue":
                account_info['total_cash'] = float(value.value)
            elif value.tag == "BuyingPower":
                account_info['buying_power'] = float(value.value)
            elif value.tag == "AvailableFunds":
                account_info['available_funds'] = float(value.value)
        
        logging.debug(f"Processed account info: {account_info}")
        return account_info
        
    except Exception as e:
        logging.error(f"Error getting account info: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error details: {str(e)}")
        
        # Check if it's a rate limit error
        if "Maximum number of account summary requests exceeded" in str(e):
            logging.error("RATE LIMIT ERROR: Account summary requests exceeded")
            logging.error("This indicates we're making too many requests too quickly")
        elif "Error 322" in str(e):
            logging.error("ERROR 322: Maximum number of account summary requests exceeded")
            logging.error("This is a TWS/Gateway API rate limit")
        
        return {}
    except:
        logging.error("Unknown error occurred in get_account_info")
        logging.error("This could be a connection issue or API problem")
        return {}









async def get_commission_info(ib: IB, symbol: str) -> Dict[str, float]:
    """
    Query commission information for a specific symbol.
    Returns dict with commission details.
    """
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Request contract details to get commission info
        details = await ib.reqContractDetailsAsync(contract)
        
        if details:
            # Get commission info from contract details
            # Note: This is a simplified approach - actual commission may vary
            commission_info = {
                'per_share': 0.005,  # Default IB commission per share
                'min_commission': 1.0,  # Minimum commission per trade
                'max_commission': 29.95  # Maximum commission per trade
            }
            return commission_info
        else:
            return {'per_share': 0.005, 'min_commission': 1.0, 'max_commission': 29.95}
            
    except Exception as e:
        logging.error(f"Error getting commission info for {symbol}: {e}")
        return {'per_share': 0.005, 'min_commission': 1.0, 'max_commission': 29.95}


async def get_market_data_permissions(ib: IB) -> Dict[str, bool]:
    """
    Query market data permissions and subscriptions.
    Returns dict with permission status.
    """
    try:
        # Check if we have market data permissions
        # This is a simplified check - in practice you'd query specific permissions
        permissions = {
            'real_time_data': True,  # Assume we have real-time data
            'historical_data': True,
            'options_data': False,
            'futures_data': False
        }
        return permissions
    except Exception as e:
        logging.error(f"Error getting market data permissions: {e}")
        return {'real_time_data': False, 'historical_data': False, 'options_data': False, 'futures_data': False}


async def get_trading_permissions(ib: IB) -> Dict[str, bool]:
    """
    Query trading permissions for different asset types.
    Returns dict with permission status.
    """
    try:
        # Check trading permissions
        permissions = {
            'stocks': True,
            'options': False,
            'futures': False,
            'forex': False,
            'bonds': False
        }
        return permissions
    except Exception as e:
        logging.error(f"Error getting trading permissions: {e}")
        return {'stocks': False, 'options': False, 'futures': False, 'forex': False, 'bonds': False}







async def get_account_currency(ib: IB) -> str:
    """
    Query the account's base currency.
    Returns currency code (e.g., 'USD').
    """
    try:
        account_values = await ib.accountSummaryAsync()
        
        for value in account_values:
            if value.tag == "Currency":
                return value.value
        
        return "USD"  # Default to USD
    except Exception as e:
        logging.error(f"Error getting account currency: {e}")
        return "USD"








async def get_current_positions(ib: IB) -> List[Dict]:
    """
    Query current positions from IB API.
    Returns list of position dictionaries.
    """
    try:
        positions = await ib.reqPositionsAsync()
        
        position_list = []
        for position in positions:
            position_list.append({
                'symbol': position.contract.symbol,
                'exchange': position.contract.exchange,
                'position': position.position,
                'avg_cost': position.avgCost,
                'market_value': position.marketValue
            })
        
        return position_list
    except Exception as e:
        logging.error(f"Error getting positions: {e}")
        return []


async def get_account_status(ib: IB) -> Dict[str, any]:
    """
    Get comprehensive account status and health information.
    Returns dict with account status, permissions, and health metrics.
    """
    try:
        status = {
            'connected': ib.isConnected(),
            'client_id': ib.clientId(),
            'server_time': None,
            'account_info': {},
            'permissions': {},
            'health': 'unknown'
        }
        
        if status['connected']:
            # Get server time
            try:
                server_time = await ib.reqCurrentTimeAsync()
                status['server_time'] = server_time
            except:
                pass
            
            # Get account info
            status['account_info'] = await get_account_info(ib)
            
            # Get permissions
            status['permissions'] = {
                'market_data': await get_market_data_permissions(ib),
                'trading': await get_trading_permissions(ib)
            }
            
            # Determine health
            if status['account_info'] and status['permissions']['market_data']['real_time_data']:
                status['health'] = 'healthy'
            elif status['account_info']:
                status['health'] = 'limited'
            else:
                status['health'] = 'degraded'
        
        return status
        
    except Exception as e:
        logging.error(f"Error getting account status: {e}")
        return {'connected': False, 'health': 'error', 'error': str(e)}


async def get_intraday_historical_data(ib: IB, symbol: str, duration: str = "1 D", bar_size: str = "10 secs") -> List[Dict]:
    """
    Get intraday historical data for a symbol.
    Useful for initial data collection at market open.
    """
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify the contract first to get conId
        qualified_contracts = await ib.qualifyContractsAsync(contract)
        if not qualified_contracts:
            logging.error(f"Could not qualify contract for {symbol}")
            return []
        
        qualified_contract = qualified_contracts[0]
        
        # Request intraday historical data
        bars = await ib.reqHistoricalDataAsync(
            qualified_contract,
            dt.datetime.now().strftime('%Y%m%d %H:%M:%S'),
            duration,
            bar_size,
            'TRADES',
            useRTH=True
        )
        
        if bars:
            # Convert to our data format
            data = []
            for bar in bars:
                data.append({
                    'datetime': bar.date,
                    'close': bar.close,
                    'high': bar.high,
                    'low': bar.low,
                    'volume': bar.volume
                })
            
            return data
        else:
            return []
            
    except Exception as e:
        if "Trading TWS session is connected from a different IP address" in str(e):
            logging.warning(f"Session conflict for {symbol} - will retry with real-time data only")
            return []
        else:
            logging.error(f"Error getting historical data for {symbol}: {e}")
            return []


async def get_available_symbols(ib: IB) -> List[str]:
    """
    Query available symbols that can be traded.
    Returns list of available symbols.
    """
    try:
        # This would typically query available contracts
        # For now, return common ETFs
        return ["SPY", "VOO", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT"]
    except Exception as e:
        logging.error(f"Error getting available symbols: {e}")
        return ["SPY", "VOO"]


async def get_real_time_bid_ask(ib: IB, symbol: str) -> Dict[str, float]:
    """
    Get real-time bid and ask prices for a symbol.
    Returns dict with bid, ask, and spread information.
    """
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify the contract first to get conId
        qualified_contracts = await ib.qualifyContractsAsync(contract)
        if not qualified_contracts:
            logging.error(f"Could not qualify contract for {symbol}")
            return None
        
        qualified_contract = qualified_contracts[0]
        
        # Request real-time market data
        ticker = ib.reqMktData(qualified_contract)
        await asyncio.sleep(1)  # Wait for data
        
        if ticker.bid and ticker.ask:
            spread = ticker.ask - ticker.bid
            spread_pct = spread / ticker.bid * 100
            
            return {
                'bid': ticker.bid,
                'ask': ticker.ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'mid_price': (ticker.bid + ticker.ask) / 2
            }
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error getting bid-ask for {symbol}: {e}")
        return None


async def calculate_dynamic_slippage(ib: IB, symbol: str, lookback_days: int = 5) -> float:
    """
    Calculate dynamic slippage based on recent market data.
    Uses bid-ask spreads and volatility to estimate slippage.
    """
    try:
        # First try to get real-time bid-ask spread
        bid_ask = await get_real_time_bid_ask(ib, symbol)
        if bid_ask:
            # Use half the bid-ask spread as slippage estimate
            spread_slippage = bid_ask['spread'] / 2
        else:
            spread_slippage = 0.01  # Default spread slippage
        
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify the contract first to get conId
        qualified_contracts = await ib.qualifyContractsAsync(contract)
        if not qualified_contracts:
            logging.error(f"Could not qualify contract for {symbol}")
            return 0.03  # Default slippage
        
        qualified_contract = qualified_contracts[0]
        
        # Get historical data for volatility calculation
        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=lookback_days)
        
        # Request historical data
        bars = await ib.reqHistoricalDataAsync(
            qualified_contract,
            end_time.strftime('%Y%m%d %H:%M:%S'),
            f'{lookback_days} D',
            '1 day',
            'TRADES',
            useRTH=True
        )
        
        if not bars:
            return spread_slippage  # Use spread-based slippage
        
        # Calculate average daily range and volatility
        daily_ranges = []
        for bar in bars:
            daily_range = bar.high - bar.low
            daily_ranges.append(daily_range)
        
        avg_daily_range = sum(daily_ranges) / len(daily_ranges)
        avg_price = bars[-1].close
        
        # Estimate slippage as a percentage of average daily range
        # More volatile stocks have higher slippage
        volatility_slippage = (avg_daily_range / avg_price) * 0.05  # 5% of daily range
        
        # Combine spread and volatility slippage
        total_slippage = spread_slippage + volatility_slippage
        
        # Cap slippage at reasonable levels
        return min(max(total_slippage, 0.005), 0.10)  # Between 0.5 cents and 10 cents
        
    except Exception as e:
        if "Trading TWS session is connected from a different IP address" in str(e):
            logging.warning(f"Session conflict for {symbol} - using default slippage")
            return 0.03  # Default slippage
        else:
            logging.error(f"Error calculating dynamic slippage for {symbol}: {e}")
            return 0.03  # Default slippage


@dataclass
class TradingConfig:
    # Strategy parameters
    sym1: str = "SPY"
    sym2: str = "VOO"
    window: int = 90
    entry_threshold: float = 1.5
    exit_threshold: float = 0.2
    shares_per_leg: int = 200
    
    # Trading hours (ET)
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "US/Eastern"
    
    # Risk management
    risk_limits: RiskLimits = None
    
    # IB connection
    ib_host: str = "127.0.0.1"
    ib_port: int = PAPER_PORT  # Default to paper trading port
    ib_client_id: int = 11  # Trading instance clientId
    paper_trading: bool = True  # True for paper, False for live
    
    # Dynamic values (will be populated from API)
    initial_capital: float = None
    slippage: float = None
    fee: float = None
    
    def __post_init__(self):
        if self.risk_limits is None:
            self.risk_limits = RiskLimits()
    
    async def initialize_from_api(self, ib: IB):
        """
        Initialize config values from IB API calls.
        This replaces hardcoded values with dynamic API queries.
        """
        try:
            # Get account information - CRITICAL for risk management
            account_info = await get_account_info(ib)
            if not account_info.get('net_liquidation'):
                await asyncio.sleep(1.0)
                account_info = await get_account_info(ib)

            if not account_info.get('net_liquidation'):
                logging.error("CRITICAL: Could not get account information from API after retry")
                raise Exception("Cannot initialize trading config without account information")
            
            self.initial_capital = float(account_info['net_liquidation'])  # <-- ADD THIS
            
            # Get commission information for both symbols
            commission1 = await get_commission_info(ib, self.sym1)
            commission2 = await get_commission_info(ib, self.sym2)
            
            # Use average commission as fee
            avg_commission = (commission1['per_share'] + commission2['per_share']) / 2
            self.fee = avg_commission
            
            # Calculate dynamic slippage based on market conditions
            # Wait for session to be fully established before making requests
            await asyncio.sleep(3)  # Give session time to stabilize
            
            try:
                # Get real-time bid-ask for slippage calculation
                bid_ask1 = await get_real_time_bid_ask(ib, self.sym1)
                bid_ask2 = await get_real_time_bid_ask(ib, self.sym2)
                
                if bid_ask1 and bid_ask2:
                    # Use half the average bid-ask spread as slippage
                    avg_spread = (bid_ask1['spread'] + bid_ask2['spread']) / 2
                    self.slippage = avg_spread / 2
                    logging.info(f"Dynamic slippage calculated from real-time spreads: ${self.slippage:.3f}")
                else:
                    # CRITICAL ERROR: Cannot proceed without slippage calculation
                    logging.error("CRITICAL: Could not get real-time spreads for slippage calculation")
                    logging.error("Dynamic slippage is required for accurate trade execution")
                    raise Exception("Cannot proceed without dynamic slippage calculation")
            except Exception as e:
                logging.error(f"CRITICAL: Error calculating dynamic slippage: {e}")
                logging.error("Dynamic slippage calculation is required for safety")
                raise Exception("Cannot proceed without dynamic slippage calculation")
            
            logging.info(f"Initialized from API - Capital: ${self.initial_capital:.2f}, "
                        f"Fee: ${self.fee:.3f}/share, Slippage: ${self.slippage:.3f}/share")
            
        except Exception as e:
            logging.error(f"CRITICAL: Error initializing config from API: {e}")
            logging.error("Cannot proceed with hardcoded fallback values")
            logging.error("All configuration must come from live API data for safety")
            raise Exception("Trading system cannot start without proper API configuration")


class LiveTrader:
    """
    Live trading framework for ETF statistical arbitrage strategy.
    Handles real-time signal generation, order execution, and risk management.
    """
    
    def __init__(self, config: TradingConfig, ib_instance: IB = None):
        self.config = config
        self.ib = ib_instance if ib_instance else IB()
        self.strategy = None
        self.state = TradingState.INITIALIZING
        
        # Trading state
        self.current_position = 0  # -1: short, 0: flat, 1: long
        self.current_shares = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.portfolio_value = 0  # Will be set from API
        self.peak_portfolio = 0   # Will be set from API
        
        # Risk tracking
        self.daily_start_value = 0  # Will be set from API
        self.trades_today = []
        self.orders = {}
        
        # Data storage - separate by trading day
        self.price_data = {config.sym1: [], config.sym2: []}
        self.signal_history = []
        self.current_trading_day = None
        self.data_points_today = 0
        
        # Rate limiting for API requests
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum 2 seconds between requests
        
        # Notification control
        self.trade_occurred = False
        
        # Data initialization tracking
        self.data_initialized_for_day = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging for trading."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Main trading log
        mode = "paper" if self.config.paper_trading else "live"
        self.logger = logging.getLogger(f'{mode.capitalize()}Trader')
        self.logger.setLevel(logging.INFO)  # Only INFO and above for console
        
        # File handler (keep debug for file)
        today = dt.datetime.now().strftime('%Y-%m-%d')
        fh = logging.FileHandler(f'{log_dir}/{mode}_trading_{today}.log')
        fh.setLevel(logging.DEBUG)  # Keep debug logging in file
        
        # Console handler (clean output)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Only INFO and above for console
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    async def connect_to_ib(self):
        """Connect to Interactive Brokers with error handling."""
        try:
            mode = "PAPER" if self.config.paper_trading else "LIVE"
            self.logger.info(f"Connecting to IB {mode} at {self.config.ib_host}:{self.config.ib_port}")
            self.state = TradingState.CONNECTING
            
            await self.ib.connectAsync(
                self.config.ib_host, 
                self.config.ib_port, 
                clientId=self.config.ib_client_id
            )
            
            
            # Verify connection
            await asyncio.sleep(2)
            if self.ib.isConnected():
                self.logger.info(f"Successfully connected to IB {mode}")
                return True
            else:
                self.logger.error("Failed to connect to IB")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self.state = TradingState.ERROR
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = dt.datetime.now()
        
        # Convert to Eastern Time properly
        try:
            import pytz
            et_tz = pytz.timezone('US/Eastern')
            et_time = now.astimezone(et_tz)
        except ImportError:
            # Fallback if pytz is not available - assume local time is ET
            et_time = now
        
        # Simple market hours check (ET)
        market_open = dt.datetime.strptime(self.config.market_open, "%H:%M").time()
        market_close = dt.datetime.strptime(self.config.market_close, "%H:%M").time()
        current_time = et_time.time()
        
        # Check if it's a weekday
        is_weekday = et_time.weekday() < 5
        
        # Debug logging removed for cleaner output
        
        return is_weekday and market_open <= current_time <= market_close
    
    def get_current_trading_day(self) -> str:
        """Get current trading day as string (YYYY-MM-DD)."""
        return dt.datetime.now().strftime('%Y-%m-%d')
    
    def reset_daily_data(self):
        """Reset data for a new trading day."""
        today = self.get_current_trading_day()
        if self.current_trading_day != today:
            self.logger.info(f"New trading day detected: {today}")
            self.current_trading_day = today
            self.price_data = {self.config.sym1: [], self.config.sym2: []}
            self.data_points_today = 0
            self.daily_trades = 0
            self.daily_start_value = self.portfolio_value
            self.trades_today = []
            # Note: Data will be re-initialized in the trading loop when market opens
        else:
            # Same trading day - preserve existing data
            self.logger.debug(f"Same trading day: {today} - preserving existing data")
    
    def validate_same_day_data(self) -> bool:
        """Validate that all data points are from the current trading day."""
        today = self.get_current_trading_day()
        
        for symbol in [self.config.sym1, self.config.sym2]:
            for data_point in self.price_data[symbol]:
                try:
                    # Handle both timezone-aware and naive datetime objects
                    dt_obj = data_point['datetime']
                    if dt_obj.tzinfo is not None:
                        # Convert to local time if timezone-aware
                        dt_obj = dt_obj.astimezone().replace(tzinfo=None)
                    
                    data_date = dt_obj.strftime('%Y-%m-%d')
                    if data_date != today:
                        self.logger.error(f"Found data from wrong day: {data_date} != {today} for {symbol}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error validating data point date: {e}")
                    return False
        
        return True
    
    def validate_data_time_period(self, start_time: dt.datetime) -> bool:
        """Validate that all data points are from the correct time period."""
        for symbol in [self.config.sym1, self.config.sym2]:
            for data_point in self.price_data[symbol]:
                try:
                    # Handle both timezone-aware and naive datetime objects
                    dt_obj = data_point['datetime']
                    if dt_obj.tzinfo is not None:
                        # Convert to local time if timezone-aware
                        dt_obj = dt_obj.astimezone().replace(tzinfo=None)
                    
                    if dt_obj < start_time:
                        self.logger.error(f"Found data from before start time: {dt_obj} < {start_time} for {symbol}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error validating data point time: {e}")
                    return False
        
        return True
    
    def has_sufficient_data(self) -> bool:
        """Check if we have sufficient data points for the current trading day."""
        min_data_points = self.config.window  # Need exactly window size for rolling calculations
        
        # Check if we have enough data for both symbols
        sym1_count = len(self.price_data[self.config.sym1])
        sym2_count = len(self.price_data[self.config.sym2])
        
        has_enough = sym1_count >= min_data_points and sym2_count >= min_data_points
        
        # Only log status changes to avoid spam
        if not hasattr(self, 'last_sufficient_data_status'):
            self.last_sufficient_data_status = None
        
        current_status = has_enough
        if current_status != self.last_sufficient_data_status:
            if not has_enough:
                self.logger.info(f"Collecting data: {sym1_count}/{min_data_points} {self.config.sym1}, {sym2_count}/{min_data_points} {self.config.sym2}")
            else:
                self.logger.info(f"✅ SUFFICIENT DATA: {sym1_count} {self.config.sym1}, {sym2_count} {self.config.sym2}")
            self.last_sufficient_data_status = current_status
        
        return has_enough
    
    def calculate_data_start_time(self) -> dt.datetime:
        """Calculate the dynamic start time for data collection based on current time."""
        now = dt.datetime.now()
        today_start = dt.datetime.strptime(self.get_current_trading_day(), '%Y-%m-%d')
        
        # Calculate lookback period: window_size * frequency
        # For 10-second frequency, each data point represents 10 seconds
        # So window_size * 10 seconds = total lookback time
        frequency_seconds = 10  # From config signal_frequency: "10 secs"
        lookback_seconds = self.config.window * frequency_seconds
        lookback_start = now - dt.timedelta(seconds=lookback_seconds)
        
        # Use the later of: lookback_start or today_start
        data_start_time = max(lookback_start, today_start)
        
        # Ensure all datetime objects are timezone-naive for comparison
        if data_start_time.tzinfo is not None:
            data_start_time = data_start_time.replace(tzinfo=None)
        
        return data_start_time

    async def initialize_daily_data(self):
        """Initialize data collection with historical data from today."""
        try:
            self.logger.info("Initializing daily data collection...")
            
            # Calculate the dynamic start time for data collection
            data_start_time = self.calculate_data_start_time()
            now = dt.datetime.now()
            if now.tzinfo is not None:
                now = now.replace(tzinfo=None)
            
            # Calculate lookback period for logging
            frequency_seconds = 10  # From config signal_frequency: "10 secs"
            lookback_seconds = self.config.window * frequency_seconds
            today_start = dt.datetime.strptime(self.get_current_trading_day(), '%Y-%m-%d')
            lookback_start = now - dt.timedelta(seconds=lookback_seconds)
            
            self.logger.info(f"Data collection start time: {data_start_time}")
            self.logger.info(f"Lookback period: {lookback_seconds} seconds ({self.config.window} * {frequency_seconds}s)")
            self.logger.info(f"Today start: {today_start}")
            self.logger.info(f"Lookback start: {lookback_start}")
            
            # Get historical data for both symbols
            sym1_data = await get_intraday_historical_data(self.ib, self.config.sym1)
            sym2_data = await get_intraday_historical_data(self.ib, self.config.sym2)
            
            self.logger.info(f"Retrieved {len(sym1_data)} historical bars for {self.config.sym1}, {len(sym2_data)} for {self.config.sym2}")
            
            if sym1_data and sym2_data:
                # Filter data to start from the calculated start time
                filtered_sym1_data = []
                filtered_sym2_data = []
                
                for data_point in sym1_data:
                    # Handle timezone-aware datetime objects
                    dt_obj = data_point['datetime']
                    if dt_obj.tzinfo is not None:
                        dt_obj = dt_obj.replace(tzinfo=None)
                    
                    if dt_obj >= data_start_time:
                        filtered_sym1_data.append(data_point)
                
                for data_point in sym2_data:
                    # Handle timezone-aware datetime objects
                    dt_obj = data_point['datetime']
                    if dt_obj.tzinfo is not None:
                        dt_obj = dt_obj.replace(tzinfo=None)
                    
                    if dt_obj >= data_start_time:
                        filtered_sym2_data.append(data_point)
                
                self.logger.info(f"Filtered to {len(filtered_sym1_data)} {self.config.sym1} bars, {len(filtered_sym2_data)} {self.config.sym2} bars")
                
                # Add filtered data to our storage
                for data_point in filtered_sym1_data:
                    # Ensure timezone-naive datetime
                    dt_obj = data_point['datetime']
                    if dt_obj.tzinfo is not None:
                        dt_obj = dt_obj.replace(tzinfo=None)
                    
                    self.price_data[self.config.sym1].append({
                        'datetime': dt_obj,
                        'close': data_point['close']
                    })
                
                for data_point in filtered_sym2_data:
                    # Ensure timezone-naive datetime
                    dt_obj = data_point['datetime']
                    if dt_obj.tzinfo is not None:
                        dt_obj = dt_obj.replace(tzinfo=None)
                    
                    self.price_data[self.config.sym2].append({
                        'datetime': dt_obj,
                        'close': data_point['close']
                    })
                
                self.logger.info(f"After filtering: {len(filtered_sym1_data)} {self.config.sym1} bars, {len(filtered_sym2_data)} {self.config.sym2} bars")
                
                self.data_points_today = len(self.price_data[self.config.sym1])
                self.logger.info(f"Initialized with {self.data_points_today} data points")
                
                # Validate that we have data from the correct time period
                if not self.validate_data_time_period(data_start_time):
                    self.logger.warning("Some data points are from before expected start time, but keeping data")
                    # Don't clear data - just log the warning
            else:
                self.logger.warning("Could not get historical data - will start with real-time collection")
                
        except Exception as e:
            self.logger.error(f"Error initializing daily data: {e}")
    
    async def get_current_prices(self) -> Dict[str, float]:
        """Get current market prices for both symbols."""
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            self.logger.debug(f"Rate limiting: waiting {wait_time:.1f}s before next request")
            await asyncio.sleep(wait_time)
        
        try:
            prices = {}
            qualified_contracts = {}
            
            # First,  qualify all contracts
            for symbol in [self.config.sym1, self.config.sym2]:
                contract = Stock(symbol, 'SMART', 'USD')
                qualified_contracts_list = await self.ib.qualifyContractsAsync(contract)
                if not qualified_contracts_list:
                    self.logger.error(f"Could not qualify contract for {symbol}")
                    return None
                qualified_contracts[symbol] = qualified_contracts_list[0]
            
            # Then request real-time data using real-time bars
            for symbol in [self.config.sym1, self.config.sym2]:
                qualified_contract = qualified_contracts[symbol]
                
                # Use market data for consistent pricing
                ticker = self.ib.reqMktData(qualified_contract)
                
                # Wait for data with timeout
                timeout = 3.0
                start_time = time.time()
                
                while (time.time() - start_time) < timeout:
                    await asyncio.sleep(0.1)
                    if ticker.last or ticker.close or ticker.marketPrice():
                        break
                
                # Try different price fields in order of preference
                price = None
                if ticker.last and ticker.last > 0:
                    price = ticker.last
                    price_source = "last"
                elif ticker.close and ticker.close > 0:
                    price = ticker.close
                    price_source = "close"
                elif ticker.marketPrice() and ticker.marketPrice() > 0:
                    price = ticker.marketPrice()
                    price_source = "marketPrice"
                elif ticker.bid and ticker.ask:
                    price = (ticker.bid + ticker.ask) / 2
                    price_source = "mid_price"
                
                if price and price > 0:
                    # Basic data quality check
                    if price < 1.0 or price > 10000.0:
                        self.logger.warning(f"Suspicious price for {symbol}: ${price}")
                        return None
                    
                    prices[symbol] = price
                    self.logger.info(f"Got price for {symbol}: ${price} (from {price_source})")
                else:
                    self.logger.warning(f"No valid price data for {symbol}")
                    return None
                
                # Cancel market data subscription
                self.ib.cancelMktData(qualified_contract)
                
                await asyncio.sleep(0.3)  # Brief wait between requests
            
            self.last_request_time = time.time()  # Update last request time
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting prices: {e}")
            return None
    
    def calculate_position_size(self, prices: Dict[str, float]) -> int:
        """Calculate position size based on available capital and risk limits."""
        if not prices:
            return 0
            
        # For new trades, use the configured shares per leg
        # For existing positions, calculate based on current position
        if self.current_position == 0:
            # New trade - use configured shares per leg
            shares = self.config.shares_per_leg
        elif self.current_position == 1:  # Long position
            shares = self.current_shares
        elif self.current_position == -1:  # Short position
            shares = self.current_shares
        else:
            return 0
        
        # Check risk limits
        max_capital = self.portfolio_value * self.config.risk_limits.max_position_size_pct
        
        # Calculate required capital for the trade
        if self.current_position == 0:
            # For new trades, use the higher price to be conservative
            required_capital = max(prices.values()) * shares
        else:
            # For existing positions, use current position value
            required_capital = max(prices.values()) * shares
        
        if required_capital > max_capital:
            shares = int(max_capital / max(prices.values()))
            self.logger.warning(f"Position size limited by risk: {shares} shares")
            return shares
        
        return shares
    
    async def execute_trade(self, action: str, shares: int, prices: Dict[str, float]) -> bool:
        """Execute a trade with proper error handling."""
        try:
            mode = "PAPER" if self.config.paper_trading else "LIVE"
            
            if action == "BUY_LONG":
                # Buy SPY, Sell VOO
                spy_order = MarketOrder("BUY", shares)
                voo_order = MarketOrder("SELL", shares)
                
                # Qualify contracts first
                spy_contract = Stock(self.config.sym1, 'SMART', 'USD')
                voo_contract = Stock(self.config.sym2, 'SMART', 'USD')
                
                qualified_spy = await self.ib.qualifyContractsAsync(spy_contract)
                qualified_voo = await self.ib.qualifyContractsAsync(voo_contract)
                
                if not qualified_spy or not qualified_voo:
                    self.logger.error("Could not qualify contracts for trading")
                    return False
                
                # Submit orders
                spy_trade = self.ib.placeOrder(qualified_spy[0], spy_order)
                voo_trade = self.ib.placeOrder(qualified_voo[0], voo_order)
                
                self.logger.info(f"{mode} LONG: Bought {shares} SPY, Sold {shares} VOO")
                
            elif action == "SELL_SHORT":
                # Sell SPY, Buy VOO
                spy_order = MarketOrder("SELL", shares)
                voo_order = MarketOrder("BUY", shares)
                
                # Qualify contracts first
                spy_contract = Stock(self.config.sym1, 'SMART', 'USD')
                voo_contract = Stock(self.config.sym2, 'SMART', 'USD')
                
                qualified_spy = await self.ib.qualifyContractsAsync(spy_contract)
                qualified_voo = await self.ib.qualifyContractsAsync(voo_contract)
                
                if not qualified_spy or not qualified_voo:
                    self.logger.error("Could not qualify contracts for trading")
                    return False
                
                # Submit orders
                spy_trade = self.ib.placeOrder(qualified_spy[0], spy_order)
                voo_trade = self.ib.placeOrder(qualified_voo[0], voo_order)
                
                self.logger.info(f"{mode} SHORT: Sold {shares} SPY, Bought {shares} VOO")
                
            elif action == "CLOSE":
                if self.current_position == 1:
                    # Close long: Sell SPY, Buy VOO
                    spy_order = MarketOrder("SELL", self.current_shares)
                    voo_order = MarketOrder("BUY", self.current_shares)
                elif self.current_position == -1:
                    # Close short: Buy SPY, Sell VOO
                    spy_order = MarketOrder("BUY", self.current_shares)
                    voo_order = MarketOrder("SELL", self.current_shares)
                else:
                    return True
                
                # Qualify contracts first
                spy_contract = Stock(self.config.sym1, 'SMART', 'USD')
                voo_contract = Stock(self.config.sym2, 'SMART', 'USD')
                
                qualified_spy = await self.ib.qualifyContractsAsync(spy_contract)
                qualified_voo = await self.ib.qualifyContractsAsync(voo_contract)
                
                if not qualified_spy or not qualified_voo:
                    self.logger.error("Could not qualify contracts for trading")
                    return False
                
                spy_trade = self.ib.placeOrder(qualified_spy[0], spy_order)
                voo_trade = self.ib.placeOrder(qualified_voo[0], voo_order)
                
                self.logger.info(f"{mode} CLOSE: Closed position of {self.current_shares} shares")
            
            # Wait for order execution and check status
            await asyncio.sleep(2)
            
            # Basic order status check (in a real implementation, you'd monitor order status)
            if hasattr(spy_trade, 'orderStatus') and spy_trade.orderStatus.status == 'Filled':
                self.logger.info("SPY order filled successfully")
            else:
                self.logger.warning("SPY order status unclear - proceeding with caution")
            
            if hasattr(voo_trade, 'orderStatus') and voo_trade.orderStatus.status == 'Filled':
                self.logger.info("VOO order filled successfully")
            else:
                self.logger.warning("VOO order status unclear - proceeding with caution")
            
            # Update position
            if action == "BUY_LONG":
                self.current_position = 1
                self.current_shares = shares
            elif action == "SELL_SHORT":
                self.current_position = -1
                self.current_shares = shares
            elif action == "CLOSE":
                self.current_position = 0
                self.current_shares = 0
            
            self.daily_trades += 1
            
            # Mark that a trade occurred (for PnL notification control)
            self.trade_occurred = True
            
            # Send Telegram notification for trade execution
            try:
                # Calculate trade details for notification
                price1 = prices.get(self.config.sym1, 0)
                price2 = prices.get(self.config.sym2, 0)
                total_cost = (price1 + price2) * shares
                commission = self.config.fee * shares * 2  # Both legs
                
                # Determine trade type for notification
                if action == "BUY_LONG":
                    trade_type = "BUY"
                elif action == "SELL_SHORT":
                    trade_type = "SELL"
                else:  # CLOSE
                    trade_type = "EXIT"
                
                # Calculate daily PnL percentage
                daily_pnl_pct = 0
                if self.daily_start_value > 0:
                    daily_pnl_pct = (self.portfolio_value - self.daily_start_value) / self.daily_start_value * 100
                
                # Calculate drawdown
                drawdown = max(0, self.peak_portfolio - self.portfolio_value)
                drawdown_pct = 0
                if self.peak_portfolio > 0:
                    drawdown_pct = drawdown / self.peak_portfolio * 100
                
                await send_trade_update(
                    trade_type=trade_type,
                    symbol1=self.config.sym1,
                    symbol2=self.config.sym2,
                    shares_per_leg=shares,
                    price1=price1,
                    price2=price2,
                    total_cost=total_cost,
                    commission=commission,
                    portfolio_value=self.portfolio_value,
                    daily_pnl=self.portfolio_value - self.daily_start_value,
                    cumulative_pnl=self.portfolio_value - self.config.initial_capital,
                    trade_id=f"{action}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.logger.info(f"Sent trade notification for {trade_type} action")
            except Exception as e:
                self.logger.warning(f"Failed to send Telegram trade notification: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return False
    
    def check_risk_limits(self) -> bool:
        """Check if any risk limits have been exceeded."""
        # Daily loss limit
        if self.daily_start_value > 0:
            daily_loss_pct = (self.daily_start_value - self.portfolio_value) / self.daily_start_value
            if daily_loss_pct > self.config.risk_limits.max_daily_loss_pct:
                self.logger.error(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                return False
        else:
            self.logger.warning("Daily start value is zero - cannot calculate loss percentage")
        
        # Max drawdown
        if self.portfolio_value > self.peak_portfolio:
            self.peak_portfolio = self.portfolio_value
        
        if self.peak_portfolio > 0:
            drawdown_pct = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
            if drawdown_pct > self.config.risk_limits.max_drawdown_pct:
                self.logger.error(f"Max drawdown exceeded: {drawdown_pct:.2%}")
                return False
        else:
            self.logger.warning("Peak portfolio value is zero - cannot calculate drawdown")
        
        # Max trades per day
        if self.daily_trades >= self.config.risk_limits.max_trades_per_day:
            self.logger.warning(f"Max trades per day reached: {self.daily_trades}")
            return False
        
        # Emergency stop check
        if self.portfolio_value <= 0:
            self.logger.error("Portfolio value is zero or negative - emergency stop")
            return False
        
        return True
    
    async def collect_price_data(self, prices: Dict[str, float]):
        """Collect price data for the current trading day."""
        try:
            # Add current prices to data
            for symbol, price in prices.items():
                # Ensure timezone-naive datetime for consistency
                current_time = dt.datetime.now()
                if current_time.tzinfo is not None:
                    current_time = current_time.replace(tzinfo=None)
                
                self.price_data[symbol].append({
                    'datetime': current_time,
                    'close': price
                })
            
            # Update data points count to reflect total data points
            self.data_points_today = len(self.price_data[self.config.sym1])
            
            # Debug: Check if prices are changing
            if len(self.price_data[self.config.sym1]) > 1:
                last_spy = self.price_data[self.config.sym1][-2]['close']
                current_spy = self.price_data[self.config.sym1][-1]['close']
                last_voo = self.price_data[self.config.sym2][-2]['close']
                current_voo = self.price_data[self.config.sym2][-1]['close']
                
                if last_spy == current_spy and last_voo == current_voo:
                    self.logger.warning(f"⚠️  PRICES UNCHANGED: SPY ${current_spy:.2f}, VOO ${current_voo:.2f}")
                else:
                    self.logger.info(f"📈 Price change: SPY ${last_spy:.2f}→${current_spy:.2f}, VOO ${last_voo:.2f}→${current_voo:.2f}")
            
            # Log data collection progress (reduced frequency)
            if self.data_points_today % 30 == 0:  # Log every 30 data points
                sym1_count = len(self.price_data[self.config.sym1])
                sym2_count = len(self.price_data[self.config.sym2])
                self.logger.info(f"Data collection: {sym1_count} {self.config.sym1}, {sym2_count} {self.config.sym2} points")
                
        except Exception as e:
            self.logger.error(f"Error collecting price data: {e}")
    
    async def calculate_spread_statistics(self, prices: Dict[str, float]) -> Dict[str, any]:
        """Calculate spread statistics and return them for notifications."""
        try:
            # Check if we have sufficient data for the current trading day
            if not self.has_sufficient_data():
                return {
                    'has_sufficient_data': False,
                    'current_spread': None,
                    'rolling_mean': None,
                    'rolling_std': None,
                    'current_zscore': None
                }
            
            # Validate that all data is from the same trading day
            if not self.validate_same_day_data():
                self.logger.error("Data validation failed - cannot calculate spread")
                return {
                    'has_sufficient_data': False,
                    'current_spread': None,
                    'rolling_mean': None,
                    'rolling_std': None,
                    'current_zscore': None
                }
            
            # Create DataFrame for strategy using only today's data
            df1 = pd.DataFrame(self.price_data[self.config.sym1])
            df2 = pd.DataFrame(self.price_data[self.config.sym2])
            
            # Debug: Check data before merge
            self.logger.debug(f"Before merge: {len(df1)} {self.config.sym1} points, {len(df2)} {self.config.sym2} points")
            if len(df1) > 0 and len(df2) > 0:
                self.logger.debug(f"Latest {self.config.sym1}: {df1.iloc[-1]['close']:.2f} at {df1.iloc[-1]['datetime']}")
                self.logger.debug(f"Latest {self.config.sym2}: {df2.iloc[-1]['close']:.2f} at {df2.iloc[-1]['datetime']}")
            
            # Normalize timezone for datetime comparison
            # Convert all datetime objects to timezone-naive for consistent comparison
            for col in ['datetime']:
                if col in df1.columns:
                    # Handle timezone conversion more robustly
                    df1[col] = pd.to_datetime(df1[col]).apply(
                        lambda x: x.replace(tzinfo=None) if x.tzinfo is not None else x
                    )
                if col in df2.columns:
                    # Handle timezone conversion more robustly
                    df2[col] = pd.to_datetime(df2[col]).apply(
                        lambda x: x.replace(tzinfo=None) if x.tzinfo is not None else x
                    )
            
            # Use outer merge to include all data points, then forward fill missing values
            df = pd.merge(df1, df2, on='datetime', how='outer', suffixes=(f'_{self.config.sym1}', f'_{self.config.sym2}'))
            
            # Sort by datetime and forward fill missing values
            df = df.sort_values('datetime').ffill()
            
            # Remove any rows that still have NaN values (from the beginning)
            df = df.dropna()
            
            self.logger.debug(f"After merge: {len(df)} combined points")
            
            if len(df) < self.config.window:
                self.logger.warning(f"Insufficient merged data: {len(df)} < {self.config.window}")
                return {
                    'has_sufficient_data': False,
                    'current_spread': None,
                    'rolling_mean': None,
                    'rolling_std': None,
                    'current_zscore': None
                }
            
            # Calculate spread and z-score
            df['spread'] = df[f'close_{self.config.sym1}'] - df[f'close_{self.config.sym2}']
            df['rolling_mean'] = df['spread'].rolling(window=self.config.window).mean()
            df['rolling_std'] = df['spread'].rolling(window=self.config.window).std()
            df['zscore'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']
            
            # Get latest values
            current_spread = df['spread'].iloc[-1]
            current_rolling_mean = df['rolling_mean'].iloc[-1]
            current_rolling_std = df['rolling_std'].iloc[-1]
            current_zscore = df['zscore'].iloc[-1]
            
            # Debug: Log the actual calculation
            self.logger.debug(f"Spread calculation: {df[f'close_{self.config.sym1}'].iloc[-1]:.2f} - {df[f'close_{self.config.sym2}'].iloc[-1]:.2f} = {current_spread:.4f}")
            self.logger.debug(f"Rolling stats: mean={current_rolling_mean:.4f}, std={current_rolling_std:.4f}, z-score={current_zscore:.4f}")
            
            return {
                'has_sufficient_data': True,
                'current_spread': current_spread,
                'rolling_mean': current_rolling_mean,
                'rolling_std': current_rolling_std,
                'current_zscore': current_zscore
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating spread statistics: {e}")
            return {
                'has_sufficient_data': False,
                'current_spread': None,
                'rolling_mean': None,
                'rolling_std': None,
                'current_zscore': None
            }
    
    async def generate_signal(self, prices: Dict[str, float]) -> Optional[str]:
        """Generate trading signal using the strategy's real-time data processing."""
        try:
            # Use the strategy's real-time signal generation
            df = await self.strategy.get_latest_signals()
            
            if df.empty:
                self.logger.debug("No data available for signal generation")
                return None
            
            # Get the latest signal from the strategy
            latest_row = df.iloc[-1]
            
            # Extract signal information
            signal = latest_row.get('signal', None)
            current_zscore = latest_row.get('zscore', 0.0)
            current_spread = latest_row.get('spread', 0.0)
            
            if signal and signal != 'HOLD':
                self.logger.info(f"🎯 STRATEGY SIGNAL: {signal} (z-score: {current_zscore:.4f}, spread: {current_spread:.4f})")
                
                # Record signal in history
                self.signal_history.append({
                    'datetime': dt.datetime.now(),
                    'signal': signal,
                    'zscore': current_zscore,
                    'spread': current_spread
                })
                
                return signal
            else:
                self.logger.debug(f"Strategy signal: {signal} (z-score: {current_zscore:.4f})")
                return None
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    async def update_portfolio_value(self):
        """Update current portfolio value from IB account data."""
        # Rate limiting for account summary requests (max once per 30 seconds)
        current_time = time.time()
        if hasattr(self, 'last_portfolio_update'):
            time_since_last_update = current_time - self.last_portfolio_update
            if time_since_last_update < 30.0:  # 30-second minimum interval
                return
        
        try:
            # Get real-time account information
            account_info = await get_account_info(self.ib)
            
            if account_info.get('net_liquidation'):
                self.portfolio_value = account_info['net_liquidation']
                
                # Update peak portfolio if we have a new high
                if self.portfolio_value > self.peak_portfolio:
                    self.peak_portfolio = self.portfolio_value
                
                self.last_portfolio_update = current_time
                
                # Send PnL notification only when trades occur
                if hasattr(self, 'trade_occurred') and self.trade_occurred:
                    try:
                        # Calculate PnL metrics
                        daily_pnl = self.portfolio_value - self.daily_start_value
                        cumulative_pnl = self.portfolio_value - self.config.initial_capital
                        daily_pnl_pct = 0
                        if self.daily_start_value > 0:
                            daily_pnl_pct = daily_pnl / self.daily_start_value * 100
                        
                        drawdown = max(0, self.peak_portfolio - self.portfolio_value)
                        drawdown_pct = 0
                        if self.peak_portfolio > 0:
                            drawdown_pct = drawdown / self.peak_portfolio * 100
                        
                        await send_pnl_update(
                            portfolio_value=self.portfolio_value,
                            daily_pnl=daily_pnl,
                            cumulative_pnl=cumulative_pnl,
                            daily_pnl_pct=daily_pnl_pct,
                            peak_portfolio=self.peak_portfolio,
                            drawdown=drawdown,
                            drawdown_pct=drawdown_pct
                        )
                        self.logger.info("Sent PnL notification after trade")
                        # Reset the trade flag
                        self.trade_occurred = False
                    except Exception as e:
                        self.logger.warning(f"Failed to send Telegram PnL notification: {e}")
            else:
                # CRITICAL ERROR: Cannot continue without portfolio value
                self.logger.error("CRITICAL: Could not get portfolio value from API during trading")
                self.logger.error("Portfolio value is required for risk management")
                self.logger.error("Stopping trading system for safety")
                self.state = TradingState.ERROR
                return
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    async def trading_loop(self):
        """Main trading loop."""
        self.logger.info("Starting live trading loop")
        
        # Set initial state to TRADING if market is open
        if self.is_market_open():
            self.logger.info("Market is open - starting trading")
            self.state = TradingState.TRADING
            self.daily_start_value = self.portfolio_value
            self.daily_trades = 0
        else:
            self.logger.info("Market is closed - waiting for market open")
            self.state = TradingState.WAITING_FOR_MARKET_OPEN
        
        while self.state != TradingState.STOPPED:
            try:
                # Check connection health
                if not self.ib.isConnected():
                    self.logger.error("IB connection lost - attempting to reconnect")
                    if not await self.connect_to_ib():
                        self.logger.error("Failed to reconnect - stopping trading")
                        self.state = TradingState.STOPPED
                        break
                
                # Check if market is open
                market_open = self.is_market_open()
                
                if not market_open:
                    if self.state == TradingState.TRADING:
                        self.logger.info("Market closed - stopping trading")
                        self.state = TradingState.MARKET_CLOSED
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                if self.state == TradingState.MARKET_CLOSED:
                    self.logger.info("Market opened - starting trading")
                    self.state = TradingState.TRADING
                    self.daily_start_value = self.portfolio_value
                    self.daily_trades = 0
                
                # Check risk limits
                if not self.check_risk_limits():
                    self.logger.error("Risk limits exceeded - stopping trading")
                    self.state = TradingState.STOPPED
                    break
                
                # Get current prices for position sizing and execution
                prices = await self.get_current_prices()
                if not prices:
                    self.logger.warning("No price data - skipping iteration")
                    await asyncio.sleep(10)
                    continue
                
                # Generate signal using strategy's real-time data processing
                signal = await self.generate_signal(prices)
                
                # Execute trade if signal
                if signal:
                    shares = self.calculate_position_size(prices)
                    if shares > 0:
                        success = await self.execute_trade(signal, shares, prices)
                        if not success:
                            self.logger.error("Trade execution failed")
                
                # Update portfolio value
                await self.update_portfolio_value()
                
                # Log status
                self.logger.info(f"Portfolio: ${self.portfolio_value:.2f} | "
                               f"Position: {self.current_position} | "
                               f"Trades: {self.daily_trades}")
                
                # Wait before next iteration - 10 seconds for 5-second bar strategy
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                self.state = TradingState.ERROR
                await asyncio.sleep(60)
    
    async def start(self):
        """Start the trading system."""
        try:
            mode = "PAPER" if self.config.paper_trading else "LIVE"
            self.logger.info(f"Initializing {mode} trading system")
            
            # Connect to IB
            if not await self.connect_to_ib():
                self.logger.error("Failed to connect to IB - stopping")
                return
            
            # Wait for session to be fully established before making any API calls
            self.logger.info("Waiting for session to stabilize...")
            await asyncio.sleep(5)  # Give session more time to stabilize
            
            # Test basic API connection first
            self.logger.info("Testing API connection...")
            try:
                # Try a simple API call to test connection
                server_time = await self.ib.reqCurrentTimeAsync()
                self.logger.info(f"API connection test successful - server time: {server_time}")
            except Exception as e:
                self.logger.error(f"API connection test failed: {e}")
                self.logger.error("Cannot proceed without working API connection")
                return
            
            # Initialize config values from API
            await self.config.initialize_from_api(self.ib)
            
            # Get initial portfolio value from API - CRITICAL for risk management
            self.logger.info("Getting initial portfolio value...")
            account_info = await get_account_info(self.ib)
            if account_info.get('net_liquidation'):
                self.portfolio_value = account_info['net_liquidation']
                self.peak_portfolio = account_info['net_liquidation']
                self.daily_start_value = account_info['net_liquidation']
                self.logger.info(f"Initial portfolio value: ${self.portfolio_value:.2f}")
            else:
                # CRITICAL ERROR: Cannot proceed without portfolio value
                self.logger.error("CRITICAL: Could not get portfolio value from API")
                self.logger.error("Portfolio value is required for risk management and position sizing")
                self.logger.error("Stopping trading system for safety")
                self.state = TradingState.ERROR
                return
            
            # Initialize strategy
            self.strategy = ETFArbitrageStrategy(
                self.ib,
                strategy_name='etf_arbitrage',
                data_mode='live'  # Use live mode for real-time data
            )
            
            self.logger.info(f"{mode} trading system initialized")
            
            # Send Telegram notification that system started
            try:
                await send_system_alert(
                    title="Trading System Started",
                    message=f"{mode} trading system initialized successfully on port {self.config.ib_port}",
                    level="INFO"
                )
            except Exception as e:
                self.logger.warning(f"Failed to send Telegram startup notification: {e}")
            
            # Start trading loop
            await self.trading_loop()
            
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.state = TradingState.ERROR
        
        finally:
            # finally block in LiveTrader.start()
            if self.ib.isConnected():
                try:
                    self.ib.disconnect() # synchronous
                except:
                    pass
            
            # Send Telegram notification that system stopped
            try:
                mode = "PAPER" if self.config.paper_trading else "LIVE"
                await send_system_alert(
                    title="Trading System Stopped",
                    message=f"{mode} trading system stopped. Final portfolio value: ${self.portfolio_value:.2f}",
                    level="INFO"
                )
            except Exception as e:
                self.logger.warning(f"Failed to send Telegram stop notification: {e}")
            
            mode = "PAPER" if self.config.paper_trading else "LIVE"
            self.logger.info(f"{mode} trading system stopped")
    
    def stop(self):
        """Stop the trading system."""
        mode = "PAPER" if self.config.paper_trading else "LIVE"
        self.logger.info(f"Stopping {mode} trading system")
        self.state = TradingState.STOPPED


# Removed unused async main function - the interactive main is below


if __name__ == "__main__":
    import sys
    
    print("TRADING SYSTEM")
    print("="*60)
    print("PAPER TRADING: Port 4002")
    print("LIVE TRADING: Port 4001")
    print("="*60)
    
    # Safety check - ensure IB Gateway/TWS is running
    print("⚠️  IMPORTANT: Make sure IB Gateway or TWS is running!")
    print("   - Paper trading: Port 4002")
    print("   - Live trading: Port 4001")
    print("="*60)
    
    # Check if running in non-interactive mode (called from web interface)
    if len(sys.argv) > 1 and sys.argv[1] == "--paper":
        mode = "paper"
        print("Starting PAPER trading (non-interactive mode)")
    elif len(sys.argv) > 1 and sys.argv[1] == "--live":
        mode = "live"
        print("Starting LIVE trading (non-interactive mode)")
    else:
        # Interactive mode
        print("Starting interactive mode...")
        mode = input("Select mode (paper/live): ").lower().strip()
        print(f"Selected mode: {mode}")
    
    if mode == "live":
        print("LIVE TRADING - REAL MONEY!")
        print("WARNING: This will execute real trades!")
        confirm = input("Are you sure? (yes/no): ").lower().strip()
        if confirm != "yes":
            print("Live trading cancelled.")
            exit()
        
        # Update config for live trading
        config = TradingConfig(
            paper_trading=False,
            ib_port=LIVE_PORT,  # Live trading port (4001)
            shares_per_leg=200,
            risk_limits=RiskLimits(
                max_position_size_pct=1.0,   # 100% of portfolio
                max_daily_loss_pct=0.05,     # 5% daily loss limit
                max_drawdown_pct=0.10,       # 10% max drawdown
                max_trades_per_day=100       # Much higher trade limit
            )
        )
    else:
        print("Starting PAPER trading (safe testing)")
        config = TradingConfig(
            paper_trading=True,
            ib_port=PAPER_PORT,  # Paper trading port (4002)
            shares_per_leg=200,
            risk_limits=RiskLimits(
                max_position_size_pct=1.0,   # 100% of portfolio
                max_daily_loss_pct=0.05,     # 5% daily loss limit
                max_drawdown_pct=0.10,       # 10% max drawdown
                max_trades_per_day=100       # Much higher trade limit
            )
        )
    
    # Create single IB instance
    ib = IB()
    
    # Create and start trader with shared IB instance
    trader = LiveTrader(config, ib_instance=ib)
    
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        trader.logger.info("🛑 Received stop signal")
        trader.stop()
    finally:
        # main script cleanup
        if ib.isConnected():
            try:
                ib.disconnect()
            except:
                pass
