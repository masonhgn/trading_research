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
from etf_arb import ZScoreStatArbStrategy


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


@dataclass
class TradingConfig:
    # Strategy parameters
    sym1: str = "SPY"
    sym2: str = "VOO"
    window: int = 90
    entry_threshold: float = 1.5
    exit_threshold: float = 0.2
    shares_per_leg: int = 200
    initial_capital: float = 10000
    
    # Transaction costs
    slippage: float = 0.03
    fee: float = 0.015
    
    # Trading hours (ET)
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "US/Eastern"
    
    # Risk management
    risk_limits: RiskLimits = None
    
    # IB connection
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001  # 4001 for paper, 4000 for live
    ib_client_id: int = 40
    paper_trading: bool = True  # True for paper, False for live
    
    def __post_init__(self):
        if self.risk_limits is None:
            self.risk_limits = RiskLimits()


class LiveTrader:
    """
    Live trading framework for ETF statistical arbitrage strategy.
    Handles real-time signal generation, order execution, and risk management.
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ib = IB()
        self.strategy = None
        self.state = TradingState.INITIALIZING
        
        # Trading state
        self.current_position = 0  # -1: short, 0: flat, 1: long
        self.current_shares = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.portfolio_value = config.initial_capital
        self.peak_portfolio = config.initial_capital
        
        # Risk tracking
        self.daily_start_value = config.initial_capital
        self.trades_today = []
        self.orders = {}
        
        # Data storage
        self.price_data = {config.sym1: [], config.sym2: []}
        self.signal_history = []
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging for trading."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Main trading log
        mode = "paper" if self.config.paper_trading else "live"
        self.logger = logging.getLogger(f'{mode.capitalize()}Trader')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        today = dt.datetime.now().strftime('%Y-%m-%d')
        fh = logging.FileHandler(f'{log_dir}/{mode}_trading_{today}.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
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
            util.startLoop()
            
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
        et_time = now.astimezone(dt.timezone.utc).replace(tzinfo=None)
        
        # Simple market hours check (ET)
        market_open = dt.datetime.strptime(self.config.market_open, "%H:%M").time()
        market_close = dt.datetime.strptime(self.config.market_close, "%H:%M").time()
        current_time = et_time.time()
        
        # Check if it's a weekday
        is_weekday = et_time.weekday() < 5
        
        return is_weekday and market_open <= current_time <= market_close
    
    async def get_current_prices(self) -> Dict[str, float]:
        """Get current market prices for both symbols."""
        try:
            prices = {}
            
            for symbol in [self.config.sym1, self.config.sym2]:
                contract = Stock(symbol, 'SMART', 'USD')
                
                # Request real-time data
                ticker = await self.ib.reqMktDataAsync(contract)
                await asyncio.sleep(0.5)  # Wait for data
                
                if ticker.last:
                    prices[symbol] = ticker.last
                    self.logger.debug(f"Current {symbol} price: ${ticker.last}")
                else:
                    self.logger.warning(f"No price data for {symbol}")
                    return None
                
                # Cancel market data subscription
                self.ib.cancelMktData(contract)
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting prices: {e}")
            return None
    
    def calculate_position_size(self, prices: Dict[str, float]) -> int:
        """Calculate position size based on available capital and risk limits."""
        if not prices:
            return 0
            
        # Calculate required capital for position
        if self.current_position == 1:  # Long position
            required_capital = prices[self.config.sym1] * self.config.shares_per_leg
        elif self.current_position == -1:  # Short position
            required_capital = prices[self.config.sym2] * self.config.shares_per_leg
        else:
            return 0
        
        # Check risk limits
        max_capital = self.portfolio_value * self.config.risk_limits.max_position_size_pct
        
        if required_capital > max_capital:
            shares = int(max_capital / max(prices.values()))
            self.logger.warning(f"Position size limited by risk: {shares} shares")
            return shares
        
        return self.config.shares_per_leg
    
    async def execute_trade(self, action: str, shares: int, prices: Dict[str, float]) -> bool:
        """Execute a trade with proper error handling."""
        try:
            mode = "PAPER" if self.config.paper_trading else "LIVE"
            
            if action == "BUY_LONG":
                # Buy SPY, Sell VOO
                spy_order = MarketOrder("BUY", shares)
                voo_order = MarketOrder("SELL", shares)
                
                spy_contract = Stock(self.config.sym1, 'SMART', 'USD')
                voo_contract = Stock(self.config.sym2, 'SMART', 'USD')
                
                # Submit orders
                spy_trade = await self.ib.placeOrderAsync(spy_contract, spy_order)
                voo_trade = await self.ib.placeOrderAsync(voo_contract, voo_order)
                
                self.logger.info(f"{mode} LONG: Bought {shares} SPY, Sold {shares} VOO")
                
            elif action == "SELL_SHORT":
                # Sell SPY, Buy VOO
                spy_order = MarketOrder("SELL", shares)
                voo_order = MarketOrder("BUY", shares)
                
                spy_contract = Stock(self.config.sym1, 'SMART', 'USD')
                voo_contract = Stock(self.config.sym2, 'SMART', 'USD')
                
                # Submit orders
                spy_trade = await self.ib.placeOrderAsync(spy_contract, spy_order)
                voo_trade = await self.ib.placeOrderAsync(voo_contract, voo_order)
                
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
                
                spy_contract = Stock(self.config.sym1, 'SMART', 'USD')
                voo_contract = Stock(self.config.sym2, 'SMART', 'USD')
                
                spy_trade = await self.ib.placeOrderAsync(spy_contract, spy_order)
                voo_trade = await self.ib.placeOrderAsync(voo_contract, voo_order)
                
                self.logger.info(f"{mode} CLOSE: Closed position of {self.current_shares} shares")
            
            # Wait for order execution
            await asyncio.sleep(2)
            
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
            return True
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return False
    
    def check_risk_limits(self) -> bool:
        """Check if any risk limits have been exceeded."""
        # Daily loss limit
        daily_loss_pct = (self.daily_start_value - self.portfolio_value) / self.daily_start_value
        if daily_loss_pct > self.config.risk_limits.max_daily_loss_pct:
            self.logger.error(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
            return False
        
        # Max drawdown
        if self.portfolio_value > self.peak_portfolio:
            self.peak_portfolio = self.portfolio_value
        
        drawdown_pct = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
        if drawdown_pct > self.config.risk_limits.max_drawdown_pct:
            self.logger.error(f"Max drawdown exceeded: {drawdown_pct:.2%}")
            return False
        
        # Max trades per day
        if self.daily_trades >= self.config.risk_limits.max_trades_per_day:
            self.logger.warning(f"Max trades per day reached: {self.daily_trades}")
            return False
        
        return True
    
    async def generate_signal(self, prices: Dict[str, float]) -> Optional[str]:
        """Generate trading signal based on current market data."""
        try:
            # Add current prices to data
            for symbol, price in prices.items():
                self.price_data[symbol].append({
                    'datetime': dt.datetime.now(),
                    'close': price
                })
            
            # Keep only recent data (last 200 points)
            for symbol in self.price_data:
                if len(self.price_data[symbol]) > 200:
                    self.price_data[symbol] = self.price_data[symbol][-200:]
            
            # Check if we have enough data
            if len(self.price_data[self.config.sym1]) < self.config.window:
                return None
            
            # Create DataFrame for strategy
            df1 = pd.DataFrame(self.price_data[self.config.sym1])
            df2 = pd.DataFrame(self.price_data[self.config.sym2])
            
            # Merge data
            df = pd.merge(df1, df2, on='datetime', suffixes=(f'_{self.config.sym1}', f'_{self.config.sym2}'))
            
            if len(df) < self.config.window:
                return None
            
            # Calculate spread and z-score
            df['spread'] = df[f'close_{self.config.sym1}'] - df[f'close_{self.config.sym2}']
            df['rolling_mean'] = df['spread'].rolling(window=self.config.window).mean()
            df['rolling_std'] = df['spread'].rolling(window=self.config.window).std()
            df['zscore'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']
            
            # Get latest z-score
            current_zscore = df['zscore'].iloc[-1]
            
            # Generate signal
            if pd.isna(current_zscore):
                return None
            
            signal = None
            
            if self.current_position == 0:  # No position
                if current_zscore < -self.config.entry_threshold:
                    signal = "BUY_LONG"
                elif current_zscore > self.config.entry_threshold:
                    signal = "SELL_SHORT"
            else:  # Have position
                if abs(current_zscore) < self.config.exit_threshold:
                    signal = "CLOSE"
            
            if signal:
                self.logger.info(f"Signal: {signal} (z-score: {current_zscore:.2f})")
                self.signal_history.append({
                    'datetime': dt.datetime.now(),
                    'signal': signal,
                    'zscore': current_zscore,
                    'spread': df['spread'].iloc[-1]
                })
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    async def update_portfolio_value(self):
        """Update current portfolio value."""
        try:
            # This is a simplified calculation
            # In practice, you'd get this from IB account data
            if self.current_position != 0:
                prices = await self.get_current_prices()
                if prices:
                    # Simplified PnL calculation
                    # In reality, you'd track actual position values
                    pass
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    async def trading_loop(self):
        """Main trading loop."""
        self.logger.info("Starting live trading loop")
        
        while self.state != TradingState.STOPPED:
            try:
                # Check if market is open
                if not self.is_market_open():
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
                
                # Get current prices
                prices = await self.get_current_prices()
                if not prices:
                    self.logger.warning("No price data - skipping iteration")
                    await asyncio.sleep(10)
                    continue
                
                # Generate signal
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
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
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
            
            # Initialize strategy
            self.strategy = ZScoreStatArbStrategy(
                self.ib,
                sym1=self.config.sym1,
                sym2=self.config.sym2,
                window=self.config.window,
                entry_threshold=self.config.entry_threshold,
                exit_threshold=self.config.exit_threshold,
                shares_per_leg=self.config.shares_per_leg,
                initial_capital=self.config.initial_capital,
                slippage=self.config.slippage,
                fee=self.config.fee
            )
            
            self.logger.info(f"{mode} trading system initialized")
            
            # Start trading loop
            await self.trading_loop()
            
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.state = TradingState.ERROR
        
        finally:
            # Cleanup
            if self.ib.isConnected():
                try:
                    await self.ib.disconnectAsync()
                except:
                    pass  # Handle disconnect gracefully
            mode = "PAPER" if self.config.paper_trading else "LIVE"
            self.logger.info(f"{mode} trading system stopped")
    
    def stop(self):
        """Stop the trading system."""
        mode = "PAPER" if self.config.paper_trading else "LIVE"
        self.logger.info(f"Stopping {mode} trading system")
        self.state = TradingState.STOPPED


async def main():
    """Main function to start trading."""
    # Configuration
    config = TradingConfig(
        initial_capital=10000,
        shares_per_leg=200,  # Increased position size
        risk_limits=RiskLimits(
            max_position_size_pct=1.0,   # 100% of portfolio
            max_daily_loss_pct=0.05,     # 5% daily loss limit
            max_drawdown_pct=0.10,       # 10% max drawdown
            max_trades_per_day=100       # Much higher trade limit
        )
    )
    
    # Create and start trader
    trader = LiveTrader(config)
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        trader.logger.info("Received stop signal")
        trader.stop()


if __name__ == "__main__":
    print("UNIFIED TRADING SYSTEM")
    print("="*60)
    print("PAPER TRADING: Port 4001 (default)")
    print("LIVE TRADING: Port 4000")
    print("="*60)
    
    # Ask for trading mode
    mode = input("Select mode (paper/live): ").lower().strip()
    
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
            ib_port=4000,  # Live trading port
            initial_capital=10000,
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
            ib_port=4001,  # Paper trading port
            initial_capital=10000,
            shares_per_leg=200,
            risk_limits=RiskLimits(
                max_position_size_pct=1.0,   # 100% of portfolio
                max_daily_loss_pct=0.05,     # 5% daily loss limit
                max_drawdown_pct=0.10,       # 10% max drawdown
                max_trades_per_day=100       # Much higher trade limit
            )
        )
    
    # Create and start trader
    trader = LiveTrader(config)
    
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        trader.logger.info("🛑 Received stop signal")
        trader.stop()
