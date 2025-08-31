# STATISTICAL ARBITRAGE SYSTEM IMPLEMENTATION

We need several different modules that will work together:

1. data collection

2. data processing and analysis

3. trading strategy logic

4. backtesting engine

5. live execution

6. visualization


For each of these modules, I will specify the following:

- what the module is responsible for at a high level
- what classes and functions are necessary
- what data structures are necessary



## Data Collection Module

This module will be responsible for collecting data from exchanges.

### Responsibilities
This module will do the following:
- connect to one or more brokerages (IBKR)
- collect different types of data such as price data, account/balance data, etc.
- parse this data into a format that can be used by the rest of this system.
- store this data in a database for fast queries of repeat fetches.


### Classes/Functions

**BrokerageWrapper**
- manages connection to a brokerage.
- cleanly organizes and simplifies all of the different functions to collect data from a brokerage.
- for example, in IBKR, we need to create a Contract object, and then call a specific function depending on the frequency we need. This will abstract that away.

**DatabaseWrapper**
- manages connection to a database.
- this will abstract away all of the SQL/database logic needed to collect or store data within the database.
- the output and input will contain preprocessed data.

**DataRouter**
- this will act as the intermediary between the data source wrappers and their users, and its main responsibility is to take as input a data request, and query the database for it.
- if this data is not in the database, we will query the brokerage for it, preprocess it, and store it in the database.

**Preprocessor**
- takes in raw data from the Collector submodule and parses it into a format that is easily usable by our system.

### Data Structures

**SecurityType** (enum)
- can be one of {EQUITY, CRYPTO, OPTION, FUTURE, etc.}

**Bar**
- represents the price of one security at one timestamp
- contains a SecurityType, timestamp, ticker, price, and other metadata.












