"""
Tests for the data_feed module.
Tests REAL market data fetching functionality with crypto markets (24/7 availability).
NO MOCKING - Only real IB connections and real data.
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta

from ib_async import IB
from ib_async.contract import Stock, Future, Option

# Import the functions we want to test
import os
from data_layer.data_feed import (
    qualify_contract,
    create_stock_contract,
    create_future_contract,
    create_option_contract,
    create_crypto_contract,
    fetch_historical_bars,
    start_real_time_bars,
    cancel_real_time_bars,
    fetch_market_data,
    fetch_contract_details,
    fetch_account_summary,
    fetch_positions,
    fetch_pair_data,
    filter_trading_hours,
    detect_trading_hours_automatically
)


class TestContractCreation:
    """Test contract creation functions."""
    
    def test_create_stock_contract(self):
        """Test stock contract creation."""
        contract = create_stock_contract("SPY", "SMART", "USD")
        assert contract.symbol == "SPY"
        assert contract.secType == "STK"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
    
    def test_create_future_contract(self):
        """Test future contract creation."""
        contract = create_future_contract("ES", "GLOBEX", "USD", "20241220")
        assert contract.symbol == "ES"
        assert contract.secType == "FUT"
        assert contract.exchange == "GLOBEX"
        assert contract.currency == "USD"
        assert contract.lastTradeDateOrContractMonth == "20241220"
    
    def test_create_option_contract(self):
        """Test option contract creation."""
        contract = create_option_contract("SPY", "SMART", "USD", "20241220", 500.0, "C")
        assert contract.symbol == "SPY"
        assert contract.secType == "OPT"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
        assert contract.lastTradeDateOrContractMonth == "20241220"
        assert contract.strike == 500.0
        assert contract.right == "C"














class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_filter_trading_hours(self):
        """Test trading hours filtering."""
        # Create test data
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:00:00', periods=24, freq='1H'),
            'close': [100] * 24
        })
        
        # Test filtering
        filtered = filter_trading_hours(df)
        assert len(filtered) < len(df)  # Should filter out some hours
        
        # Test with custom hours
        filtered_custom = filter_trading_hours(
            df, market_open_time="10:00", market_close_time="15:00"
        )
        assert len(filtered_custom) < len(filtered)
    
    def test_detect_trading_hours_automatically(self):
        """Test automatic trading hours detection."""
        # Create test data with spread
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'spread': [0.1] * 50 + [0.001] * 50  # High volatility then low
        })
        
        # Test detection
        trading_mask = detect_trading_hours_automatically(df)
        assert len(trading_mask) == len(df)
        assert isinstance(trading_mask, pd.Series)


class TestRealTimeBars:
    """Test real-time bars functionality - requires IB connection."""
    
    @pytest.mark.asyncio
    async def test_real_time_bars(self):
        """Test real-time bars: start, wait 6 seconds, check last_bar, stop."""
        print("\n=== TESTING REAL-TIME BARS ===")
        
        # Connect to IB
        ib = IB()
        await ib.connectAsync('127.0.0.1', 4002, clientId=999)
        print("Connected to IB")
        
        # Test with BTC (crypto markets are 24/7)
        last_bar = {}
        
        # Create BTC contract and qualify it
        btc_contract = create_crypto_contract("BTC", "PAXOS", "USD")
        qualified_contract = await qualify_contract(ib, btc_contract)
        
        if not qualified_contract:
            raise Exception("Could not qualify BTC contract")
        
        print("Qualified BTC contract successfully")
        
        # Start subscription
        bars_container = start_real_time_bars(
            ib, qualified_contract, last_bar, use_rth=False
        )
        
        if bars_container is None:
            raise Exception("Could not start real-time bars subscription")
        
        print("Started real-time bars subscription")
        print("Waiting 6 seconds...")
        
        # Wait 6 seconds
        await asyncio.sleep(6)
        
        # Check if last_bar contains something
        print(f"last_bar contents: {last_bar}")
        
        # Verify we got real data
        assert last_bar, "Should have received at least one bar"
        assert 'close' in last_bar, "Bar should have close price"
        assert 'datetime' in last_bar, "Bar should have datetime"
        assert 'symbol' in last_bar, "Bar should have symbol"
        
        # Verify the data looks reasonable for BTC
        assert last_bar['close'] > 1000, f"BTC price should be > $1000, got {last_bar['close']}"
        assert last_bar['close'] < 200000, f"BTC price should be < $200000, got {last_bar['close']}"
        assert last_bar['symbol'] == 'BTC', f"Symbol should be BTC, got {last_bar['symbol']}"
        
        print(f"✓ Successfully received real-time bar: ${last_bar['close']}")
        
        # Cleanup
        if bars_container is not None:
            cancel_real_time_bars(ib, bars_container)
            print("Cancelled subscription")
        
        ib.disconnect()
        print("Disconnected from IB")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
