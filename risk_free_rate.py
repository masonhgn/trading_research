import asyncio
from ib_async import IB, Ticker
from typing import Dict, Optional
import datetime as dt


class RiskFreeRateFetcher:
    """
    Fetch current risk-free rates from Interactive Brokers.
    """
    
    def __init__(self, ib: IB):
        self.ib = ib
    
    async def get_treasury_bill_rates(self) -> Dict[str, float]:
        """
        Fetch current Treasury bill rates.
        
        Returns:
            Dictionary with Treasury bill yields
        """
        try:
            # Define Treasury bill contracts (1-month, 3-month, 6-month)
            tbill_contracts = {
                'tbill_1m': 'TB1M',  # 1-month T-bill
                'tbill_3m': 'TB3M',  # 3-month T-bill
                'tbill_6m': 'TB6M',  # 6-month T-bill
            }
            
            rates = {}
            
            for name, symbol in tbill_contracts.items():
                try:
                    # Request market data for Treasury bill
                    ticker = await self.ib.reqMktDataAsync(symbol)
                    
                    # Wait a moment for data to arrive
                    await asyncio.sleep(1)
                    
                    # Get the yield (bid/ask average or last price)
                    if hasattr(ticker, 'bid') and hasattr(ticker, 'ask'):
                        yield_rate = (ticker.bid + ticker.ask) / 2
                    elif hasattr(ticker, 'last'):
                        yield_rate = ticker.last
                    else:
                        yield_rate = None
                    
                    if yield_rate and yield_rate > 0:
                        rates[name] = yield_rate / 100  # Convert to decimal
                        print(f"✓ {name}: {yield_rate:.3f}%")
                    else:
                        print(f"✗ {name}: No data available")
                        
                except Exception as e:
                    print(f"✗ {name}: Error fetching data - {e}")
                    continue
            
            return rates
            
        except Exception as e:
            print(f"Error fetching Treasury bill rates: {e}")
            return {}
    
    async def get_federal_funds_rate(self) -> Optional[float]:
        """
        Fetch Federal Funds Rate (if available).
        
        Returns:
            Federal Funds Rate as decimal, or None if not available
        """
        try:
            # Try to get Fed Funds Rate from IB
            # Note: This might not be directly available, so we'll use a fallback
            print("Federal Funds Rate not directly available from IB")
            print("Using Treasury bill rates as proxy...")
            return None
            
        except Exception as e:
            print(f"Error fetching Federal Funds Rate: {e}")
            return None
    
    async def get_sofr_rate(self) -> Optional[float]:
        """
        Fetch SOFR (Secured Overnight Financing Rate).
        
        Returns:
            SOFR rate as decimal, or None if not available
        """
        try:
            # SOFR might be available through specific contracts
            # For now, we'll use Treasury bills as a proxy
            print("SOFR not directly available from IB")
            print("Using Treasury bill rates as proxy...")
            return None
            
        except Exception as e:
            print(f"Error fetching SOFR: {e}")
            return None
    
    async def get_current_risk_free_rate(self, preferred_type: str = 'tbill_3m') -> float:
        """
        Get the current risk-free rate for Sharpe ratio calculations.
        
        Args:
            preferred_type: Preferred rate type ('tbill_1m', 'tbill_3m', 'tbill_6m')
            
        Returns:
            Risk-free rate as decimal
        """
        print("Fetching current risk-free rates from Interactive Brokers...")
        
        # Get Treasury bill rates
        tbill_rates = await self.get_treasury_bill_rates()
        
        if not tbill_rates:
            print("No Treasury bill rates available, using default 2%")
            return 0.02
        
        # Use preferred rate if available, otherwise use 3-month as default
        if preferred_type in tbill_rates:
            rate = tbill_rates[preferred_type]
            print(f"Using {preferred_type}: {rate:.4f} ({rate*100:.2f}%)")
            return rate
        elif 'tbill_3m' in tbill_rates:
            rate = tbill_rates['tbill_3m']
            print(f"Using tbill_3m (fallback): {rate:.4f} ({rate*100:.2f}%)")
            return rate
        else:
            # Use the first available rate
            rate = list(tbill_rates.values())[0]
            print(f"Using first available rate: {rate:.4f} ({rate*100:.2f}%)")
            return rate
    
    def get_annualized_risk_free_rate(self, base_rate: float, periods_per_year: int = 252) -> float:
        """
        Convert annual risk-free rate to period-specific rate.
        
        Args:
            base_rate: Annual risk-free rate (e.g., 0.05 for 5%)
            periods_per_year: Number of periods per year
            
        Returns:
            Period-specific risk-free rate
        """
        if periods_per_year == 252:  # Daily
            return (1 + base_rate) ** (1/252) - 1
        elif periods_per_year > 252:  # Intraday
            # For intraday, use a very small fraction of the annual rate
            return base_rate / periods_per_year
        else:
            return (1 + base_rate) ** (1/periods_per_year) - 1


async def quick_risk_free_rate_check(ib: IB) -> float:
    """
    Quick function to get current risk-free rate.
    
    Args:
        ib: Interactive Brokers connection
        
    Returns:
        Current risk-free rate as decimal
    """
    fetcher = RiskFreeRateFetcher(ib)
    return await fetcher.get_current_risk_free_rate()


# Example usage and testing
async def test_risk_free_rate():
    """
    Test function to demonstrate risk-free rate fetching.
    """
    ib = IB()
    try:
        await ib.connectAsync("127.0.0.1", 4002, clientId=12)
        
        fetcher = RiskFreeRateFetcher(ib)
        
        # Get current risk-free rate
        rate = await fetcher.get_current_risk_free_rate()
        print(f"\nCurrent risk-free rate: {rate:.4f} ({rate*100:.2f}%)")
        
        # Test different period annualizations
        periods = [252, 1440, 252*6.5*60]  # Daily, hourly, 1-minute
        for period in periods:
            period_rate = fetcher.get_annualized_risk_free_rate(rate, period)
            print(f"Period rate for {period} periods/year: {period_rate:.8f}")
        
    except Exception as e:
        print(f"Error in test: {e}")
    finally:
        await ib.disconnectAsync()


if __name__ == "__main__":
    asyncio.run(test_risk_free_rate())
