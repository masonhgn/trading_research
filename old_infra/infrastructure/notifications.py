import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "7309059762:AAEHeeoo6VCCqjaHAqSY3Eo-xSBmEHnC8RI"
TELEGRAM_CHAT_ID = "7528514870"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"

@dataclass
class TradeNotification:
    """Data structure for trade notifications"""
    timestamp: datetime
    trade_type: TradeType
    symbol1: str
    symbol2: str
    shares_per_leg: int
    price1: float
    price2: float
    total_cost: float
    commission: float
    portfolio_value: float
    daily_pnl: float
    cumulative_pnl: float
    trade_id: Optional[str] = None

@dataclass
class PnLNotification:
    """Data structure for PnL notifications"""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    cumulative_pnl: float
    daily_pnl_pct: float
    peak_portfolio: float
    drawdown: float
    drawdown_pct: float

@dataclass
class DataCollectionNotification:
    """Data structure for data collection status notifications"""
    timestamp: datetime
    symbol1: str
    symbol2: str
    data_points_sym1: int
    data_points_sym2: int
    window_size: int
    has_sufficient_data: bool
    current_spread: Optional[float] = None
    rolling_mean: Optional[float] = None
    rolling_std: Optional[float] = None
    current_zscore: Optional[float] = None
    entry_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    trading_day: Optional[str] = None

class TelegramNotifier:
    """
    Standalone Telegram notification system for trading updates.
    Sends real-time notifications for trades and PnL changes.
    """
    
    def __init__(self, bot_token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_pnl_notification = 0  # Track last PnL notification time
        self.pnl_notification_interval = 300  # 5 minutes between PnL updates
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.
        
        Args:
            message: The message to send
            parse_mode: HTML or Markdown formatting
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.session:
            logger.error("Telegram session not initialized")
            return False
            
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            async with self.session.post(f"{self.api_url}/sendMessage", json=payload) as response:
                if response.status == 200:
                    logger.debug("Telegram message sent successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Telegram API error: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def send_trade_notification(self, trade: TradeNotification) -> bool:
        """
        Send a trade notification to Telegram.
        
        Args:
            trade: TradeNotification object with trade details
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Format the trade message
        message = self._format_trade_message(trade)
        
        # Add emoji based on trade type
        if trade.trade_type == TradeType.BUY:
            emoji = "🟢"
        elif trade.trade_type == TradeType.SELL:
            emoji = "🔴"
        else:  # EXIT
            emoji = "⚪"
        
        full_message = f"{emoji} <b>TRADE EXECUTED</b>\n\n{message}"
        
        return await self.send_message(full_message)
    
    async def send_pnl_notification(self, pnl: PnLNotification, force: bool = False) -> bool:
        """
        Send a PnL notification to Telegram.
        
        Args:
            pnl: PnLNotification object with PnL details
            force: Force send even if within rate limit
            
        Returns:
            bool: True if successful, False otherwise
        """
        current_time = datetime.now().timestamp()
        
        # Rate limiting for PnL notifications (unless forced)
        if not force and (current_time - self.last_pnl_notification) < self.pnl_notification_interval:
            logger.debug("PnL notification rate limited")
            return True
        
        # Format the PnL message
        message = self._format_pnl_message(pnl)
        
        # Add emoji based on PnL
        if pnl.daily_pnl > 0:
            emoji = "📈"
        elif pnl.daily_pnl < 0:
            emoji = "📉"
        else:
            emoji = "➡️"
        
        full_message = f"{emoji} <b>PORTFOLIO UPDATE</b>\n\n{message}"
        
        success = await self.send_message(full_message)
        if success:
            self.last_pnl_notification = current_time
        
        return success
    
    async def send_system_notification(self, title: str, message: str, level: str = "INFO") -> bool:
        """
        Send a system notification to Telegram.
        
        Args:
            title: Notification title
            message: Notification message
            level: INFO, WARNING, ERROR
            
        Returns:
            bool: True if successful, False otherwise
        """
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️", 
            "ERROR": "🚨"
        }
        
        emoji = emoji_map.get(level, "ℹ️")
        full_message = f"{emoji} <b>{title}</b>\n\n{message}"
        
        return await self.send_message(full_message)
    
    async def send_data_collection_notification(self, data: DataCollectionNotification) -> bool:
        """
        Send a data collection status notification to Telegram.
        
        Args:
            data: DataCollectionNotification object with data collection details
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Format the data collection message
        message = self._format_data_collection_message(data)
        
        # Add emoji based on data status
        if data.has_sufficient_data:
            emoji = "📊"
        else:
            emoji = "📈"
        
        full_message = f"{emoji} <b>DATA COLLECTION STATUS</b>\n\n{message}"
        
        return await self.send_message(full_message)
    
    def _format_trade_message(self, trade: TradeNotification) -> str:
        """Format trade notification message"""
        timestamp = trade.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format prices with 2 decimal places
        price1_str = f"${trade.price1:.2f}"
        price2_str = f"${trade.price2:.2f}"
        total_cost_str = f"${trade.total_cost:.2f}"
        commission_str = f"${trade.commission:.2f}"
        portfolio_str = f"${trade.portfolio_value:.2f}"
        daily_pnl_str = f"${trade.daily_pnl:.2f}"
        cumulative_pnl_str = f"${trade.cumulative_pnl:.2f}"
        
        message = f"""
<b>Time:</b> {timestamp}
<b>Action:</b> {trade.trade_type.value}
<b>Symbols:</b> {trade.symbol1} / {trade.symbol2}
<b>Shares:</b> {trade.shares_per_leg} per leg

<b>Prices:</b>
• {trade.symbol1}: {price1_str}
• {trade.symbol2}: {price2_str}

<b>Costs:</b>
• Total Cost: {total_cost_str}
• Commission: {commission_str}

<b>Portfolio:</b>
• Value: {portfolio_str}
• Daily PnL: {daily_pnl_str}
• Cumulative PnL: {cumulative_pnl_str}
"""
        
        if trade.trade_id:
            message += f"\n<b>Trade ID:</b> {trade.trade_id}"
        
        return message
    
    def _format_pnl_message(self, pnl: PnLNotification) -> str:
        """Format PnL notification message"""
        timestamp = pnl.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format values with 2 decimal places
        portfolio_str = f"${pnl.portfolio_value:.2f}"
        daily_pnl_str = f"${pnl.daily_pnl:.2f}"
        cumulative_pnl_str = f"${pnl.cumulative_pnl:.2f}"
        daily_pnl_pct_str = f"{pnl.daily_pnl_pct:.2f}%"
        peak_str = f"${pnl.peak_portfolio:.2f}"
        drawdown_str = f"${pnl.drawdown:.2f}"
        drawdown_pct_str = f"{pnl.drawdown_pct:.2f}%"
        
        message = f"""
<b>Time:</b> {timestamp}

<b>Portfolio Value:</b> {portfolio_str}
<b>Daily PnL:</b> {daily_pnl_str} ({daily_pnl_pct_str})
<b>Cumulative PnL:</b> {cumulative_pnl_str}

<b>Risk Metrics:</b>
• Peak Portfolio: {peak_str}
• Current Drawdown: {drawdown_str} ({drawdown_pct_str})
"""
        
        return message
    
    def _format_data_collection_message(self, data: DataCollectionNotification) -> str:
        """Format data collection notification message"""
        timestamp = data.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format data collection status
        status_emoji = "✅" if data.has_sufficient_data else "⏳"
        status_text = "READY FOR TRADING" if data.has_sufficient_data else "COLLECTING DATA"
        
        message = f"""
<b>Time:</b> {timestamp}
<b>Status:</b> {status_emoji} {status_text}
<b>Trading Day:</b> {data.trading_day or 'Unknown'}

<b>Data Points:</b>
• {data.symbol1}: {data.data_points_sym1}/{data.window_size}
• {data.symbol2}: {data.data_points_sym2}/{data.window_size}

<b>Window Size:</b> {data.window_size} data points
"""
        
        # Add spread information if available
        if data.current_spread is not None:
            spread_str = f"${data.current_spread:.4f}"
            message += f"\n<b>Current Spread ({data.symbol1} - {data.symbol2}):</b> {spread_str}"
            
            if data.rolling_mean is not None and data.rolling_std is not None:
                mean_str = f"${data.rolling_mean:.4f}"
                std_str = f"${data.rolling_std:.4f}"
                message += f"\n<b>Rolling Mean:</b> {mean_str}"
                message += f"\n<b>Rolling Std Dev:</b> {std_str}"
                
                if data.current_zscore is not None:
                    zscore_str = f"{data.current_zscore:.2f}"
                    message += f"\n<b>Current Z-Score:</b> {zscore_str}"
                    
                    # Add threshold information
                    if data.entry_threshold is not None and data.exit_threshold is not None:
                        entry_str = f"{data.entry_threshold:.2f}"
                        exit_str = f"{data.exit_threshold:.2f}"
                        
                        # Determine proximity to thresholds
                        abs_zscore = abs(data.current_zscore)
                        if abs_zscore >= data.entry_threshold:
                            proximity = "🔴 ABOVE ENTRY THRESHOLD"
                        elif abs_zscore >= data.exit_threshold:
                            proximity = "🟡 BETWEEN THRESHOLDS"
                        else:
                            proximity = "🟢 BELOW EXIT THRESHOLD"
                        
                        message += f"\n<b>Thresholds:</b> Entry: ±{entry_str}, Exit: ±{exit_str}"
                        message += f"\n<b>Status:</b> {proximity}"
        
        return message

# Standalone functions for easy integration
async def send_trade_update(
    trade_type: str,
    symbol1: str,
    symbol2: str,
    shares_per_leg: int,
    price1: float,
    price2: float,
    total_cost: float,
    commission: float,
    portfolio_value: float,
    daily_pnl: float,
    cumulative_pnl: float,
    trade_id: Optional[str] = None
) -> bool:
    """
    Standalone function to send trade notifications.
    
    Args:
        All trade parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    async with TelegramNotifier() as notifier:
        trade = TradeNotification(
            timestamp=datetime.now(),
            trade_type=TradeType(trade_type.upper()),
            symbol1=symbol1,
            symbol2=symbol2,
            shares_per_leg=shares_per_leg,
            price1=price1,
            price2=price2,
            total_cost=total_cost,
            commission=commission,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            trade_id=trade_id
        )
        return await notifier.send_trade_notification(trade)

async def send_pnl_update(
    portfolio_value: float,
    daily_pnl: float,
    cumulative_pnl: float,
    daily_pnl_pct: float,
    peak_portfolio: float,
    drawdown: float,
    drawdown_pct: float,
    force: bool = False
) -> bool:
    """
    Standalone function to send PnL notifications.
    
    Args:
        All PnL parameters
        force: Force send even if within rate limit
        
    Returns:
        bool: True if successful, False otherwise
    """
    async with TelegramNotifier() as notifier:
        pnl = PnLNotification(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            daily_pnl_pct=daily_pnl_pct,
            peak_portfolio=peak_portfolio,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct
        )
        return await notifier.send_pnl_notification(pnl, force=force)

async def send_system_alert(title: str, message: str, level: str = "INFO") -> bool:
    """
    Standalone function to send system alerts.
    
    Args:
        title: Alert title
        message: Alert message
        level: INFO, WARNING, ERROR
        
    Returns:
        bool: True if successful, False otherwise
    """
    async with TelegramNotifier() as notifier:
        return await notifier.send_system_notification(title, message, level)

async def send_data_collection_update(
    symbol1: str,
    symbol2: str,
    data_points_sym1: int,
    data_points_sym2: int,
    window_size: int,
    has_sufficient_data: bool,
    current_spread: Optional[float] = None,
    rolling_mean: Optional[float] = None,
    rolling_std: Optional[float] = None,
    current_zscore: Optional[float] = None,
    entry_threshold: Optional[float] = None,
    exit_threshold: Optional[float] = None,
    trading_day: Optional[str] = None
) -> bool:
    """
    Standalone function to send data collection status notifications.
    
    Args:
        All data collection parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    async with TelegramNotifier() as notifier:
        data = DataCollectionNotification(
            timestamp=datetime.now(),
            symbol1=symbol1,
            symbol2=symbol2,
            data_points_sym1=data_points_sym1,
            data_points_sym2=data_points_sym2,
            window_size=window_size,
            has_sufficient_data=has_sufficient_data,
            current_spread=current_spread,
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            current_zscore=current_zscore,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            trading_day=trading_day
        )
        return await notifier.send_data_collection_notification(data)

# Test function
async def test_telegram_notifications():
    """Test all notification types"""
    print("Testing Telegram notifications...")
    
    # Test trade notification
    print("Sending trade notification...")
    success = await send_trade_update(
        trade_type="BUY",
        symbol1="SPY",
        symbol2="VOO", 
        shares_per_leg=100,
        price1=450.25,
        price2=420.75,
        total_cost=87100.00,
        commission=2.50,
        portfolio_value=1001685.60,
        daily_pnl=1250.50,
        cumulative_pnl=1685.60
    )
    print(f"Trade notification: {'✅' if success else '❌'}")
    
    await asyncio.sleep(2)
    
    # Test PnL notification
    print("Sending PnL notification...")
    success = await send_pnl_update(
        portfolio_value=1001685.60,
        daily_pnl=1250.50,
        cumulative_pnl=1685.60,
        daily_pnl_pct=0.125,
        peak_portfolio=1002000.00,
        drawdown=314.40,
        drawdown_pct=0.031
    )
    print(f"PnL notification: {'✅' if success else '❌'}")
    
    await asyncio.sleep(2)
    
    # Test system notification
    print("Sending system notification...")
    success = await send_system_alert(
        title="Trading System Started",
        message="Paper trading system initialized successfully on port 4002",
        level="INFO"
    )
    print(f"System notification: {'✅' if success else '❌'}")
    
    await asyncio.sleep(2)
    
    # Test data collection notification
    print("Sending data collection notification...")
    success = await send_data_collection_update(
        symbol1="SPY",
        symbol2="VOO",
        data_points_sym1=45,
        data_points_sym2=45,
        window_size=90,
        has_sufficient_data=False,
        current_spread=51.96,
        rolling_mean=52.15,
        rolling_std=0.85,
        current_zscore=-0.22,
        entry_threshold=1.5,
        exit_threshold=0.2,
        trading_day="2025-08-15"
    )
    print(f"Data collection notification: {'✅' if success else '❌'}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_telegram_notifications())
