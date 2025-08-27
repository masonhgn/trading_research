from ib_async import *
import time
import sys
import signal

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

# Define contracts

contract1 = Crypto(
    symbol = "BTC",
    #secType = "CRYPTO",
    currency = "USD",
    exchange = "ZEROHASH",
)

contract2 = Crypto(
    symbol = "ETH",
    #secType = "CRYPTO",
    currency = "USD",
    exchange = "ZEROHASH",
)


# Request market data
ticker1 = ib.reqMktData(contract1, '', False, False)
ticker2 = ib.reqMktData(contract2, '', False, False)




def handler(signal, frame):
    print('sigint caught, exiting...')
    ib.disconnect()
    print('ib disconnected')
    sys.exit(0)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, handler)

    # Poll every 10 seconds
    while True:
        ticker1 = ib.reqMktData(contract1, '', False, False)
        ticker2 = ib.reqMktData(contract2, '', False, False)
        ib.sleep(2)
        print(f"btc last price: {ticker1.last}")
        print(f"eth last price: {ticker2.last}")





