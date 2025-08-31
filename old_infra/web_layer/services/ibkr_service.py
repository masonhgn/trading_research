"""
Enhanced IBKR service for Flask web interface.
Provides real-time portfolio tracking, position monitoring, and trading controls.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from ib_async import IB, Stock
from ib_async.order import MarketOrder
import pandas as pd

from infrastructure.config import config


class IBKRService:
    """
    Enhanced IBKR service for real-time trading dashboard.
    """
    
    def __init__(self):
        self.ib = None
        self.connected = False
        self.last_portfolio_update = None
        self.last_position_update = None
        self.cached_portfolio = {}
        self.cached_positions = []
        self.cached_account_info = {}
        self.logger = logging.getLogger(__name__)
        
        # Real-time data tracking
        self.live_prices = {}
        self.spread_data = []
        self.zscore_data = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_portfolio = 0.0
        self.max_drawdown = 0.0
        
    async def connect(self) -> bool:
        """Connect to IBKR."""
        try:
            self.ib = IB()
            connection_config = config.EXECUTION_CONFIG['ib_connection']
            
            if config.is_paper_trading():
                port = connection_config['paper_port']
            else:
                port = connection_config['live_port']
            
            await self.ib.connectAsync(
                connection_config['host'],
                port,
                clientId=connection_config['client_id']
            )
            
            self.connected = True
            self.logger.info(f"Connected to IBKR on port {port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IBKR")
    
    async def get_real_time_portfolio_value(self) -> Dict[str, float]:
        """Get real-time portfolio value and PnL."""
        try:
            if not self.connected:
                return {}
            
            # Get account summary with timeout
            try:
                account_values = await asyncio.wait_for(self.ib.accountSummaryAsync(), timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.error("Timeout getting account summary")
                return self.cached_portfolio
            
            portfolio_data = {}
            for value in account_values:
                if value.tag == "NetLiquidation":
                    portfolio_data['net_liquidation'] = float(value.value)
                elif value.tag == "TotalCashValue":
                    portfolio_data['total_cash'] = float(value.value)
                elif value.tag == "BuyingPower":
                    portfolio_data['buying_power'] = float(value.value)
                elif value.tag == "AvailableFunds":
                    portfolio_data['available_funds'] = float(value.value)
                elif value.tag == "UnrealizedPnL":
                    portfolio_data['unrealized_pnl'] = float(value.value)
                elif value.tag == "RealizedPnL":
                    portfolio_data['realized_pnl'] = float(value.value)
            
            # Calculate additional metrics
            if 'net_liquidation' in portfolio_data:
                current_value = portfolio_data['net_liquidation']
                
                # Update peak portfolio
                if current_value > self.peak_portfolio:
                    self.peak_portfolio = current_value
                
                # Calculate drawdown
                if self.peak_portfolio > 0:
                    drawdown = (self.peak_portfolio - current_value) / self.peak_portfolio
                    portfolio_data['current_drawdown'] = drawdown
                    portfolio_data['max_drawdown'] = max(self.max_drawdown, drawdown)
                
                # Calculate total PnL (assuming initial capital from config)
                initial_capital = config.get('strategy.etf_arbitrage.initial_capital', 10000)
                portfolio_data['total_pnl'] = current_value - initial_capital
                portfolio_data['total_pnl_pct'] = (current_value - initial_capital) / initial_capital
            
            self.cached_portfolio = portfolio_data
            self.last_portfolio_update = datetime.now()
            
            return portfolio_data
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return self.cached_portfolio
    
    async def get_detailed_positions(self) -> List[Dict]:
        """Get detailed position information."""
        try:
            if not self.connected:
                return self.cached_positions
            
            try:
                positions = await asyncio.wait_for(self.ib.reqPositionsAsync(), timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.error("Timeout getting positions")
                return self.cached_positions
            
            position_list = []
            for position in positions:
                pos_data = {
                    'symbol': position.contract.symbol,
                    'exchange': position.contract.exchange,
                    'position': position.position,
                    'avg_cost': position.avgCost,
                    'market_value': position.marketValue,
                    'unrealized_pnl': position.unrealizedPnL,
                    'realized_pnl': position.realizedPnL,
                    'contract_id': position.contract.conId
                }
                position_list.append(pos_data)
            
            self.cached_positions = position_list
            self.last_position_update = datetime.now()
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return self.cached_positions
    
    async def get_live_spread_data(self) -> Dict[str, float]:
        """Get live spread data for SPY vs VOO."""
        try:
            if not self.connected:
                return {}
            
            symbols = config.get('strategy.etf_arbitrage.symbols', ['SPY', 'VOO'])
            sym1, sym2 = symbols[0], symbols[1]
            
            # Get current prices
            prices = await self.get_current_prices([sym1, sym2])
            
            if sym1 in prices and sym2 in prices:
                spread = prices[sym1] - prices[sym2]
                
                # Calculate z-score (simplified - would need historical data for proper calculation)
                zscore = 0.0  # Placeholder
                
                spread_data = {
                    'spread': spread,
                    'zscore': zscore,
                    'spy_price': prices[sym1],
                    'voo_price': prices[sym2],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store for charting
                self.spread_data.append(spread_data)
                if len(self.spread_data) > 1000:  # Keep last 1000 data points
                    self.spread_data = self.spread_data[-1000:]
                
                return spread_data
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting spread data: {e}")
            return {}
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current market prices for symbols."""
        try:
            if not self.connected:
                return self.live_prices
            
            prices = {}
            for symbol in symbols:
                try:
                    contract = Stock(symbol, 'SMART', 'USD')
                    qualified_contracts = await self.ib.qualifyContractsAsync(contract)
                    
                    if qualified_contracts:
                        # Request market data
                        ticker = self.ib.reqMktData(qualified_contracts[0])
                        
                        # Wait for data (with timeout)
                        timeout = 5.0
                        start_time = datetime.now()
                        
                        while not ticker.marketPrice() and (datetime.now() - start_time).seconds < timeout:
                            await asyncio.sleep(0.1)
                        
                        if ticker.marketPrice():
                            prices[symbol] = ticker.marketPrice()
                            self.live_prices[symbol] = ticker.marketPrice()
                        
                        # Cancel market data subscription
                        self.ib.cancelMktData(qualified_contracts[0])
                        
                except Exception as e:
                    self.logger.error(f"Error getting price for {symbol}: {e}")
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting current prices: {e}")
            return self.live_prices
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get comprehensive connection status."""
        try:
            status = {
                'connected': self.connected,
                'timestamp': datetime.now().isoformat(),
                'last_portfolio_update': self.last_portfolio_update.isoformat() if self.last_portfolio_update else None,
                'last_position_update': self.last_position_update.isoformat() if self.last_position_update else None
            }
            
            if self.connected and self.ib:
                status['client_id'] = getattr(self.ib, 'clientId', lambda: 'unknown')()
                try:
                    status['server_time'] = await asyncio.wait_for(self.ib.reqCurrentTimeAsync(), timeout=5.0)
                except asyncio.TimeoutError:
                    status['server_time'] = None
                
                # Check if we can get basic account info
                try:
                    account_info = await self.get_real_time_portfolio_value()
                    status['account_accessible'] = bool(account_info)
                except:
                    status['account_accessible'] = False
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting connection status: {e}")
            return {'connected': False, 'error': str(e)}
    
    def check_market_status_sync(self) -> Dict[str, Any]:
        """Synchronous version of market status check."""
        try:
            import pytz
            
            # Get current time in EST
            est_tz = pytz.timezone('US/Eastern')
            current_time_est = datetime.now(est_tz)
            
            # Check if it's a weekday
            is_weekday = current_time_est.weekday() < 5  # Monday = 0, Friday = 4
            
            # Check if it's within market hours (9:30 AM - 4:00 PM EST)
            market_open = current_time_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time_est.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_market_hours = market_open <= current_time_est <= market_close
            
            # Check if markets are open (weekday + market hours)
            markets_open = is_weekday and is_market_hours
            
            # Calculate time until market open/close
            if is_weekday:
                if current_time_est < market_open:
                    time_until_open = market_open - current_time_est
                    next_event = f"Market opens in {time_until_open}"
                elif current_time_est > market_close:
                    # Next trading day
                    next_trading_day = current_time_est + timedelta(days=1)
                    while next_trading_day.weekday() >= 5:  # Skip weekends
                        next_trading_day += timedelta(days=1)
                    next_market_open = next_trading_day.replace(hour=9, minute=30, second=0, microsecond=0)
                    time_until_next_open = next_market_open - current_time_est
                    next_event = f"Next market opens in {time_until_next_open}"
                else:
                    time_until_close = market_close - current_time_est
                    next_event = f"Market closes in {time_until_close}"
            else:
                # Weekend
                next_trading_day = current_time_est + timedelta(days=1)
                while next_trading_day.weekday() >= 5:  # Skip weekends
                        next_trading_day += timedelta(days=1)
                next_market_open = next_trading_day.replace(hour=9, minute=30, second=0, microsecond=0)
                time_until_next_open = next_market_open - current_time_est
                next_event = f"Next market opens in {time_until_next_open}"
            
            result = {
                'current_time_est': current_time_est.strftime('%Y-%m-%d %H:%M:%S EST'),
                'is_weekday': is_weekday,
                'is_market_hours': is_market_hours,
                'markets_open': markets_open,
                'market_open_time': '09:30 EST',
                'market_close_time': '16:00 EST',
                'next_event': next_event,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return {
                'current_time_est': 'Error',
                'markets_open': False,
                'error': str(e)
            }
    
    async def execute_manual_trade(self, action: str, shares: int) -> Dict[str, Any]:
        """Execute a manual trade."""
        try:
            if not self.connected:
                return {'success': False, 'error': 'Not connected to IBKR'}
            
            symbols = config.get('strategy.etf_arbitrage.symbols', ['SPY', 'VOO'])
            sym1, sym2 = symbols[0], symbols[1]
            
            # Qualify contracts
            spy_contract = Stock(sym1, 'SMART', 'USD')
            voo_contract = Stock(sym2, 'SMART', 'USD')
            
            qualified_spy = await self.ib.qualifyContractsAsync(spy_contract)
            qualified_voo = await self.ib.qualifyContractsAsync(voo_contract)
            
            if not qualified_spy or not qualified_voo:
                return {'success': False, 'error': 'Could not qualify contracts'}
            
            # Execute trade based on action
            if action == "BUY_LONG":
                # Buy SPY, Sell VOO
                spy_order = MarketOrder("BUY", shares)
                voo_order = MarketOrder("SELL", shares)
                
                spy_trade = self.ib.placeOrder(qualified_spy[0], spy_order)
                voo_trade = self.ib.placeOrder(qualified_voo[0], voo_order)
                
                return {
                    'success': True,
                    'action': 'BUY_LONG',
                    'shares': shares,
                    'message': f'Bought {shares} SPY, Sold {shares} VOO'
                }
                
            elif action == "SELL_SHORT":
                # Sell SPY, Buy VOO
                spy_order = MarketOrder("SELL", shares)
                voo_order = MarketOrder("BUY", shares)
                
                spy_trade = self.ib.placeOrder(qualified_spy[0], spy_order)
                voo_trade = self.ib.placeOrder(qualified_voo[0], voo_order)
                
                return {
                    'success': True,
                    'action': 'SELL_SHORT',
                    'shares': shares,
                    'message': f'Sold {shares} SPY, Bought {shares} VOO'
                }
                
            elif action == "CLOSE_ALL":
                # Close all positions
                positions = await self.get_detailed_positions()
                
                for position in positions:
                    if position['position'] != 0:
                        contract = Stock(position['symbol'], 'SMART', 'USD')
                        qualified = await self.ib.qualifyContractsAsync(contract)
                        
                        if qualified:
                            if position['position'] > 0:
                                order = MarketOrder("SELL", abs(position['position']))
                            else:
                                order = MarketOrder("BUY", abs(position['position']))
                            
                            self.ib.placeOrder(qualified[0], order)
                
                return {
                    'success': True,
                    'action': 'CLOSE_ALL',
                    'message': 'Closed all positions'
                }
            
            return {'success': False, 'error': 'Invalid action'}
            
        except Exception as e:
            self.logger.error(f"Error executing manual trade: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_trade_history(self, days: int = 30) -> List[Dict]:
        """Get trade history for the last N days."""
        try:
            if not self.connected:
                return []
            
            # Get executions
            executions = await self.ib.reqExecutionsAsync()
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_executions = []
            
            for execution in executions:
                if execution.time >= cutoff_date:
                    exec_data = {
                        'symbol': execution.contract.symbol,
                        'side': execution.order.action,
                        'shares': execution.order.totalQuantity,
                        'price': execution.execution.price,
                        'time': execution.time.isoformat(),
                        'commission': execution.commissionReport.commission if execution.commissionReport else 0
                    }
                    recent_executions.append(exec_data)
            
            return recent_executions
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_cached_data(self) -> Dict[str, Any]:
        """Get all cached data for dashboard."""
        return {
            'portfolio': self.cached_portfolio,
            'positions': self.cached_positions,
            'prices': self.live_prices,
            'spread_data': self.spread_data[-100:] if self.spread_data else [],  # Last 100 points
            'last_update': {
                'portfolio': self.last_portfolio_update.isoformat() if self.last_portfolio_update else None,
                'positions': self.last_position_update.isoformat() if self.last_position_update else None
            }
        }


# Global IBKR service instance
ibkr_service = IBKRService()
