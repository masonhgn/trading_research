# Quantitative Trading Primer: From Zero to Alpha

## What is Quantitative Trading?

Quantitative trading (quant trading) is the use of mathematical models, statistical analysis, and computer algorithms to identify and execute trading opportunities. Instead of relying on gut feelings or traditional analysis, quant traders use data, mathematics, and systematic approaches to make trading decisions.

## Understanding US Equities

### What Are Equities?
Equities (stocks) represent ownership in a company. When you buy a share of stock, you own a small piece of that company.

### US Equity Markets
The United States has several major stock exchanges:
- **NYSE (New York Stock Exchange)**: Largest exchange, home to many blue-chip companies
- **NASDAQ**: Technology-focused exchange, many growth companies
- **AMEX**: Smaller exchange, many ETFs and options

## Long vs Short Positions

### Long Position (Going Long)
**Definition**: Buying an asset with the expectation that its price will rise.

**How it Works**:
1. Buy shares at current market price
2. Hold the shares
3. Sell later at a higher price (hopefully)
4. Profit = Selling Price - Buying Price

**Example**:
- Buy 100 shares of AAPL at $150 per share = $15,000 investment
- AAPL rises to $160 per share
- Sell 100 shares = $16,000
- Profit = $16,000 - $15,000 = $1,000 (6.67% return)

**Risk**: If the price falls, you lose money
- If AAPL falls to $140, you lose $1,000

### Short Position (Going Short)
**Definition**: Selling an asset you don't own, with the expectation that its price will fall.

**How it Works**:
1. Borrow shares from your broker
2. Sell the borrowed shares at current market price
3. Buy back the shares later at a lower price (hopefully)
4. Return the borrowed shares to your broker
5. Profit = Selling Price - Buying Price

**Example**:
- Borrow 100 shares of TSLA from broker
- Sell 100 shares at $200 per share = $20,000 cash
- TSLA falls to $180 per share
- Buy back 100 shares = $18,000
- Return borrowed shares to broker
- Profit = $20,000 - $18,000 = $2,000 (10% return)

**Risk**: If the price rises, you lose money (potentially unlimited)
- If TSLA rises to $250, you lose $5,000

### Key Differences

| Aspect | Long Position | Short Position |
|--------|---------------|----------------|
| **Direction** | Bet on price increase | Bet on price decrease |
| **Maximum Loss** | Limited to initial investment | Potentially unlimited |
| **Maximum Gain** | Unlimited | Limited to 100% (if price goes to zero) |
| **Borrowing** | Not required | Must borrow shares |
| **Dividends** | Receive dividends | Pay dividends to lender |
| **Margin Requirements** | Usually 50% | Usually 150% |

### Why Short Selling Matters
Short selling is crucial for:
- **Market Efficiency**: Helps correct overvalued stocks
- **Hedging**: Protect against market declines
- **Arbitrage**: Exploit price differences
- **Alpha Generation**: Profit from both rising and falling markets

## Opening and Closing Positions

### The Four Position Types
When trading, there are four fundamental actions you can take:

#### 1. Open to Buy (Long)
**Action**: Buy shares you don't own
**Result**: Creates a long position
**Example**: Buy 100 shares of AAPL at $150
**Risk**: Price could fall

#### 2. Open to Sell (Short)
**Action**: Sell shares you don't own (borrowed)
**Result**: Creates a short position
**Example**: Sell 100 shares of TSLA at $200 (borrowed)
**Risk**: Price could rise

#### 3. Close to Buy
**Action**: Buy shares to close an existing short position
**Result**: Eliminates a short position
**Example**: Buy back 100 shares of TSLA at $180 to return borrowed shares
**Profit**: $20 per share if price fell

#### 4. Close to Sell
**Action**: Sell shares you own
**Result**: Eliminates a long position
**Example**: Sell 100 shares of AAPL at $160
**Profit**: $10 per share if price rose

### Position Lifecycle
```
Open to Buy → [Hold Long Position] → Close to Sell
Open to Sell → [Hold Short Position] → Close to Buy
```

## Statistical Arbitrage Strategy

### What is Statistical Arbitrage?
Statistical arbitrage (stat arb) is a market-neutral strategy that exploits temporary price inefficiencies between related assets. Unlike traditional strategies that bet on market direction, stat arb bets on the relationship between assets.

### The Core Concept: Spread Trading
Instead of betting that the market will go up or down, we bet that the **spread** between two related assets will converge or diverge from its historical average.

### How It Works
1. **Identify Related Assets**: Find two assets that historically move together (e.g., SPY and VOO)
2. **Calculate the Spread**: Spread = Asset A Price - Asset B Price
3. **Measure Deviation**: How far is the current spread from its historical average?
4. **Trade the Spread**: 
   - If spread is unusually wide: Short the expensive asset, long the cheap asset
   - If spread is unusually narrow: Long the expensive asset, short the cheap asset
5. **Wait for Convergence**: Profit when the spread returns to normal

### Market-Neutral Approach
By simultaneously opening both a long and short position:
- **Long Position**: Buy the "cheap" asset
- **Short Position**: Sell the "expensive" asset
- **Net Market Exposure**: Approximately zero

This means we're not betting on market direction - we're betting on the relationship between assets.

### Example: SPY vs VOO Strategy
**Scenario**: SPY is trading at $500, VOO at $499.50
- **Spread**: $500 - $499.50 = $0.50
- **Historical Average Spread**: $0.20
- **Current Spread**: $0.50 (wider than average)

**Trade**:
- **Open to Sell SPY**: Short SPY at $500 (bet it will fall relative to VOO)
- **Open to Buy VOO**: Long VOO at $499.50 (bet it will rise relative to SPY)

**Outcome**: If spread converges to $0.20:
- SPY falls to $499.70, VOO rises to $499.50
- **Profit**: $0.30 per share on both positions
- **Total Profit**: $0.60 per share pair

### Why This Works
1. **Market Neutral**: Profits regardless of overall market direction
2. **Mean Reversion**: Spreads tend to return to historical averages
3. **Diversification**: Reduces exposure to market-wide risks
4. **Consistent Returns**: Can generate alpha in both bull and bear markets

### Key Advantages
- **Direction Independent**: Works whether market goes up, down, or sideways
- **Lower Risk**: Market-neutral approach reduces systematic risk
- **Consistent Alpha**: Exploits persistent market inefficiencies
- **Scalable**: Can be applied to many asset pairs simultaneously

## Trading Strategies: The Foundation

### What is a Trading Strategy?
A trading strategy is a systematic approach to buying and selling financial instruments based on specific rules, patterns, or signals.

### Simple Signal Example: MACD

MACD (Moving Average Convergence Divergence) is a popular technical indicator that generates trading signals.

**How MACD Works**:
1. Calculate 12-day exponential moving average (EMA)
2. Calculate 26-day exponential moving average (EMA)
3. MACD Line = 12-day EMA - 26-day EMA
4. Signal Line = 9-day EMA of MACD Line
5. Histogram = MACD Line - Signal Line

**Trading Rules**:
- **Buy Signal**: MACD Line crosses above Signal Line
- **Sell Signal**: MACD Line crosses below Signal Line

**Example**:
```python
# Simple MACD Strategy
def macd_strategy(prices):
    # Calculate EMAs
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    
    # Calculate MACD
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    
    # Generate signals
    signals = pd.Series(0, index=prices.index)
    signals[macd_line > signal_line] = 1  # Buy
    signals[macd_line < signal_line] = -1  # Sell
    
    return signals
```

## Strategy Categories

### 1. Momentum Strategies
**Concept**: Buy assets that have performed well recently, expecting continued performance.

**Core Idea**: "The trend is your friend" - assets that are moving in a direction tend to continue moving in that direction.

**Examples**:
- **Relative Strength**: Buy stocks that have outperformed the market
- **Moving Average Crossover**: Buy when short-term MA crosses above long-term MA
- **Breakout Trading**: Buy when price breaks above resistance levels

**Pros**: Can capture strong trends and large moves
**Cons**: Vulnerable to reversals, can lag market changes

### 2. Mean Reversion Strategies
**Concept**: Buy assets that have fallen below their average, sell assets that have risen above their average.

**Core Idea**: "What goes up must come down" - prices tend to return to their historical averages.

**Examples**:
- **Bollinger Bands**: Buy when price touches lower band, sell when it touches upper band
- **Pairs Trading**: Buy underperforming stock, sell outperforming stock in same sector
- **Statistical Arbitrage**: Trade when z-score indicates extreme deviation from mean

**Pros**: Works well in sideways markets, can be market neutral
**Cons**: Can lose money in strong trends, requires statistical sophistication

## Performance Metrics: Measuring Success

### Return Metrics

#### 1. Total Return
**Formula**: `(Ending Value - Beginning Value) / Beginning Value`

**Example**: 
- Start: $100,000
- End: $110,000
- Total Return = (110,000 - 100,000) / 100,000 = 10%

#### 2. Annualized Return
**Formula**: `(1 + Total Return)^(252/Trading Days) - 1`

**Example**:
- 6-month return: 5%
- Trading days: 126
- Annualized Return = (1.05)^(252/126) - 1 = 10.25%

### Risk Metrics

#### 1. Volatility (Standard Deviation)
**Formula**: `√(Σ(Return - Mean Return)² / (n-1))`

**Interpretation**: 
- Higher volatility = higher risk
- Usually annualized by multiplying by √252

**Example**:
- Daily returns: [1%, -2%, 3%, -1%, 2%]
- Mean return: 0.6%
- Volatility = √(Σ(return - 0.6%)² / 4) = 2.1%

#### 2. Maximum Drawdown
**Formula**: `(Peak Value - Trough Value) / Peak Value`

**Interpretation**: 
- Largest peak-to-trough decline
- Measures worst-case scenario

**Example**:
- Peak: $120,000
- Trough: $100,000
- Max Drawdown = (120,000 - 100,000) / 120,000 = 16.7%

### Risk-Adjusted Return Metrics

#### 1. Sharpe Ratio
**Formula**: `(Return - Risk-Free Rate) / Volatility`

**Components**:
- **Return**: Strategy's average return
- **Risk-Free Rate**: Return on risk-free assets (e.g., Treasury bills)
- **Volatility**: Standard deviation of returns

**Interpretation**:
- Higher is better
- Above 1.0 is considered good
- Above 2.0 is considered excellent
- Negative means the strategy underperforms risk-free assets

**Example**:
- Strategy return: 12%
- Risk-free rate: 2%
- Volatility: 15%
- Sharpe Ratio = (12% - 2%) / 15% = 0.67

**Why Sharpe Ratio Matters**:
- Measures return per unit of risk
- Allows comparison of strategies with different risk levels
- Standard metric in the investment industry

## Alpha: The Goal

### What is Alpha?
Alpha is the excess return of a strategy compared to a benchmark, adjusted for risk. It represents the "skill" component of returns.

### Alpha Formula
**Basic Alpha**: `Strategy Return - Benchmark Return`

**Risk-Adjusted Alpha**: `Strategy Return - (Risk-Free Rate + Beta × (Market Return - Risk-Free Rate))`

### Sources of Alpha
- **Market Inefficiencies**: Temporary price dislocations
- **Behavioral Biases**: Investors making systematic errors
- **Statistical Relationships**: Mean reversion, momentum, correlations
- **Alternative Data**: Non-traditional information sources

## Conclusion

Quantitative trading combines mathematics, computer science, and finance to systematically identify and exploit trading opportunities. The key is developing strategies that generate consistent alpha while managing risk effectively.

Key takeaways:
1. **Long positions** profit from price increases, **short positions** profit from price decreases
2. **Momentum strategies** follow trends, **mean reversion strategies** bet on price returns to average
3. **Sharpe ratio** measures risk-adjusted returns and is the standard performance metric
4. **Alpha** represents the skill-based excess return above the market

Success in quantitative trading requires:
- Understanding the mathematical foundations
- Developing robust, testable strategies
- Managing risk effectively
- Continuous research and improvement

---

*Disclaimer: This primer is for educational purposes only. Quantitative trading involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results.*
