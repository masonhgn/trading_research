# ETF Arbitrage Strategy: A Complete Guide for New Traders

## What is Statistical Arbitrage?

Statistical arbitrage (stat arb) is a trading strategy that looks for temporary price differences between related financial instruments. Think of it like finding a sale at the grocery store - when two similar products have different prices, you buy the cheaper one and sell the more expensive one, expecting them to eventually trade at similar prices.

In the world of ETFs (Exchange-Traded Funds), this strategy works because many ETFs track the same underlying assets but may have slight price differences due to:
- Different trading volumes
- Market maker spreads
- Temporary supply and demand imbalances
- Different expense ratios affecting pricing

## The Core Concept: Mean Reversion

The fundamental idea behind this strategy is **mean reversion** - the belief that when two related assets temporarily diverge in price, they will eventually return to their historical relationship.

### Simple Analogy
Imagine you have two identical twins who usually walk side by side. If one twin suddenly walks ahead of the other, you expect them to eventually walk together again. In trading, we're betting that when two similar ETFs temporarily separate in price, they'll come back together.

## The ETF Pair: SPY vs VOO

Our strategy focuses on two popular ETFs:
- **SPY** (SPDR S&P 500 ETF Trust)
- **VOO** (Vanguard S&P 500 ETF)

Both ETFs track the exact same index (S&P 500) but are managed by different companies. They should theoretically trade at very similar prices, but sometimes small differences occur.

## Understanding the Spread

### What is a Spread?
The "spread" is simply the difference between the prices of our two ETFs:
```
Spread = SPY Price - VOO Price
```

For example:
- If SPY is trading at $500.00 and VOO is at $499.50
- The spread is: $500.00 - $499.50 = $0.50

### Why Does the Spread Matter?
The spread tells us how far apart the two ETFs are trading. When the spread is unusually large (positive or negative), it might indicate an opportunity to profit from mean reversion.

## The Z-Score: Measuring "Unusualness"

### What is a Z-Score?
A z-score measures how unusual a current value is compared to historical values. Think of it like a "weirdness meter."

### How Z-Score Works
1. **Calculate the average spread** over the past 90 periods (our "window")
2. **Calculate how much the spread varies** (standard deviation)
3. **Compare current spread to the average**:
   ```
   Z-Score = (Current Spread - Average Spread) / Standard Deviation
   ```

### Z-Score Interpretation
- **Z-Score = 0**: Current spread is exactly at the average (normal)
- **Z-Score = +1**: Current spread is 1 standard deviation above average (somewhat unusual)
- **Z-Score = +2**: Current spread is 2 standard deviations above average (very unusual)
- **Z-Score = -1**: Current spread is 1 standard deviation below average (somewhat unusual)
- **Z-Score = -2**: Current spread is 2 standard deviations below average (very unusual)

## Trading Signals: When to Enter and Exit

### Entry Signals (When to Start Trading)
We enter a trade when the z-score indicates the spread is unusually large:

**Long Position (Buy SPY, Sell VOO):**
- When z-score < -1.5 (spread is unusually negative)
- This means SPY is trading much lower than VOO relative to history
- We bet that SPY will rise relative to VOO

**Short Position (Sell SPY, Buy VOO):**
- When z-score > +1.5 (spread is unusually positive)
- This means SPY is trading much higher than VOO relative to history
- We bet that SPY will fall relative to VOO

### Exit Signals (When to Close the Trade)
We exit when the spread returns to more normal levels:
- When |z-score| < 0.2 (spread is close to average)
- This means the ETFs are trading at their normal relationship

## Risk Management: Protecting Your Capital

### Position Sizing
- We trade 200 shares of each ETF per position
- This is a moderate size that balances profit potential with risk

### Stop Losses
- If the spread continues to move against us, we have built-in risk controls
- The strategy automatically exits when the spread normalizes

### Diversification
- This is just one strategy among many possible approaches
- Never put all your money in one strategy

## Transaction Costs: The Hidden Enemy

### What Are Transaction Costs?
Every trade costs money:
- **Commission**: Fee paid to your broker (e.g., $1 per trade)
- **Slippage**: Difference between expected and actual execution price
- **Bid-Ask Spread**: Difference between buying and selling prices

### Impact on Profits
In our strategy:
- Commission: $0.005 per share
- Slippage: $0.01 per share
- Total cost per round trip: $0.03 per share

For a 200-share position, that's $6.00 in costs per trade. We need the spread to move more than this to be profitable.

## Dynamic vs Fixed Thresholds

### Fixed Thresholds (Current Setting)
- Entry threshold: 1.5 (enter when z-score > 1.5 or < -1.5)
- Exit threshold: 0.2 (exit when |z-score| < 0.2)
- These are set manually and don't change

### Dynamic Thresholds (Advanced Feature)
- Thresholds automatically adjust based on market conditions
- Uses statistical distributions (t-distribution) instead of normal distribution
- More sophisticated but requires more data

## Data Requirements: What We Need

### Historical Data
- We need at least 90 periods of price data to calculate averages
- More data = more reliable statistics
- We use 10-second bars for high-frequency trading

### Real-Time Data
- Live price feeds from Interactive Brokers
- Continuous monitoring of both ETFs
- Automatic signal generation

## Trading Hours and Market Conditions

### Regular Trading Hours
- Strategy operates during normal market hours (9:30 AM - 4:00 PM ET)
- Markets are most liquid during these hours
- Spreads are typically tighter

### After-Hours Trading
- Less liquid, wider spreads
- Strategy may not work as well
- Higher risk of poor execution

## Performance Metrics: How We Measure Success

### Total Return
- Percentage gain or loss over the entire period
- Includes all profits and losses

### Sharpe Ratio
- Risk-adjusted return measure
- Higher is better (more return per unit of risk)
- Above 1.0 is considered good

### Maximum Drawdown
- Largest peak-to-trough decline
- Measures worst-case scenario
- Lower is better

### Win Rate
- Percentage of profitable trades
- Higher is better, but not the only measure

## Common Pitfalls and Challenges

### 1. Overfitting
- Using too much historical data to optimize parameters
- Strategy may not work in future markets
- Solution: Use out-of-sample testing

### 2. Market Regime Changes
- Relationships between assets can change
- What worked yesterday may not work tomorrow
- Solution: Monitor performance and adjust

### 3. Liquidity Issues
- ETFs may become illiquid during stress
- Hard to enter or exit positions
- Solution: Monitor trading volumes

### 4. Transaction Costs
- High-frequency trading can generate many small losses
- Need sufficient spread movement to overcome costs
- Solution: Optimize entry/exit thresholds

## Backtesting: Testing Before Trading

### What is Backtesting?
Backtesting is like a flight simulator for trading strategies. We test our strategy on historical data to see how it would have performed.

### Backtesting Process
1. **Data Collection**: Gather historical prices for SPY and VOO
2. **Signal Generation**: Apply our strategy rules to historical data
3. **Trade Simulation**: Simulate buying and selling based on signals
4. **Performance Analysis**: Calculate returns, risk metrics, and other statistics

### Limitations of Backtesting
- Past performance doesn't guarantee future results
- Doesn't account for all real-world costs
- May not capture extreme market events

## Live Trading Considerations

### Paper Trading First
- Test the strategy with virtual money
- Learn the mechanics without financial risk
- Validate that the strategy works in real-time

### Gradual Implementation
- Start with small position sizes
- Increase gradually as you gain confidence
- Monitor performance closely

### Technology Requirements
- Reliable internet connection
- Fast execution platform
- Real-time data feeds
- Automated trading system (optional)

## Advanced Features

### Dynamic Z-Score Calculation
Instead of assuming normal distribution, we can fit the actual distribution of spreads:
- **T-Distribution**: Better for financial data with fat tails
- **Skewed Normal**: Accounts for asymmetric distributions
- **Confidence Levels**: Adjust thresholds based on statistical confidence

### Automatic Trading Hours Detection
- Algorithm automatically detects when markets are most active
- Adjusts strategy parameters based on volatility
- Reduces false signals during low-liquidity periods

### Portfolio Constraints
- Respects maximum position sizes
- Manages overall portfolio risk
- Prevents over-leveraging

## Code Structure Overview

The strategy is implemented in several Python modules:

### `ETFArbitrageStrategy` Class
- Main strategy logic
- Handles data fetching and processing
- Generates trading signals
- Manages risk controls

### Key Methods:
- `fetch_data()`: Gets historical price data
- `process_data()`: Calculates spreads and z-scores
- `generate_signals()`: Creates buy/sell signals
- `compute_pnl()`: Calculates profits and losses

### Data Processing Pipeline:
1. **Raw Data** → Price feeds from IB
2. **Spread Calculation** → SPY - VOO difference
3. **Rolling Statistics** → Moving averages and standard deviations
4. **Z-Score Calculation** → Measure of current vs historical spread
5. **Signal Generation** → Buy/sell decisions
6. **Position Management** → Track open positions
7. **PnL Calculation** → Profit/loss tracking

## Configuration Parameters

The strategy behavior is controlled by these key parameters:

```yaml
strategy:
  etf_arbitrage:
    symbols: ["SPY", "VOO"]          # ETF pair to trade
    window: 90                       # Periods for rolling statistics
    entry_threshold: 1.5             # Z-score for trade entry
    exit_threshold: 0.2              # Z-score for trade exit
    shares_per_leg: 200              # Number of shares per position
    initial_capital: 10000           # Starting capital
    slippage: 0.01                   # Expected slippage per share
    fee: 0.005                       # Commission per share
```

## Getting Started

### Prerequisites
1. Interactive Brokers account (paper or live)
2. Python programming environment
3. Required packages: pandas, numpy, ib_async, etc.

### Setup Steps
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Configure IB Connection**: Set up TWS or IB Gateway
3. **Update Configuration**: Modify `config.yaml` as needed
4. **Run Backtest**: Test strategy on historical data
5. **Paper Trade**: Test with virtual money
6. **Live Trading**: Start with small positions

### Running the Strategy
```bash
# Run backtest
python scripts/backtest_runner.py --duration "5 D"

# Run live trading
python live_trader.py
```

## Conclusion

ETF arbitrage is a sophisticated strategy that requires understanding of:
- Statistical concepts (z-scores, mean reversion)
- Market microstructure (spreads, liquidity)
- Risk management (position sizing, stop losses)
- Technology (data feeds, execution)

While the concept is simple (buy low, sell high), successful implementation requires:
- Careful parameter tuning
- Robust risk management
- Reliable technology infrastructure
- Continuous monitoring and adjustment

Remember: All trading involves risk. This strategy is educational and should be thoroughly tested before using real money. Start small, learn continuously, and never risk more than you can afford to lose.

## Further Reading

- "Pairs Trading: Quantitative Methods and Analysis" by Ganapathy Vidyamurthy
- "Statistical Arbitrage: Algorithmic Trading Insights and Techniques" by Andrew Pole
- Interactive Brokers API documentation
- Financial mathematics and statistics textbooks

---

*Disclaimer: This guide is for educational purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss and is not suitable for all investors.*
