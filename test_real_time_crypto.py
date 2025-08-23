import asyncio
import datetime as dt
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ib_async import IB
from data_layer.data_feed import (
    create_crypto_contract,
    start_real_time_bars,
    cancel_real_time_bars,
    qualify_contract
)


async def test_crypto_real_time_bars():
    """
    Test real-time bars with crypto (BTC/USD) using PAXOS.
    """
    print("🔄 Starting test for real-time bars (crypto)...")

    ib = IB()
    try:
        print("🔌 Connecting to IB Gateway...")
        await ib.connectAsync("127.0.0.1", 4002, clientId=99)
        ib.run()  # Start event loop for subscriptions

        print("✅ Connected to IB Gateway")

        # Create and qualify crypto contract
        crypto_contract = create_crypto_contract("BTC", exchange="ZEROHASH", currency="USD")
        crypto_contract = await qualify_contract(ib, crypto_contract)
        if not crypto_contract:
            print("❌ Failed to qualify crypto contract")
            return

        print(f"📄 Qualified contract: {crypto_contract.symbol} on {crypto_contract.exchange}")

        # Dictionary to store last bar
        last_bar = {}

        print("📡 Requesting real-time bar subscription...")
        bars_container = start_real_time_bars(
            ib=ib,
            contract=crypto_contract,
            last_bar=last_bar,
            bar_size=5,
            what_to_show="TRADES",
            use_rth=False
        )

        if bars_container is None:
            print("❌ Real-time bar subscription failed")
            return

        print("✅ Real-time bar subscription started")
        print("⏳ Waiting for real-time bars (30 seconds)...")

        start_time = dt.datetime.now()
        while (dt.datetime.now() - start_time).total_seconds() < 30:
            await asyncio.sleep(1)

            if last_bar:
                print(f"🕒 {dt.datetime.now().strftime('%H:%M:%S')} | {crypto_contract.symbol} real-time bar:")
                print(f"  Open: {last_bar.get('open')}")
                print(f"  High: {last_bar.get('high')}")
                print(f"  Low: {last_bar.get('low')}")
                print(f"  Close: {last_bar.get('close')}")
                print(f"  Volume: {last_bar.get('volume')}")
                print(f"  WAP: {last_bar.get('wap')}")
                print(f"  Bar Count: {last_bar.get('bar_count')}")
                print(f"  Timestamp: {last_bar.get('timestamp')}")
                print("-" * 60)
            else:
                print("⏳ Still waiting for first bar...")

        print("🛑 Cancelling real-time bar subscription...")
        cancel_real_time_bars(ib, bars_container, crypto_contract)

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if ib.isConnected():
            ib.disconnect()
            print("🔌 Disconnected from IB Gateway")


if __name__ == "__main__":
    asyncio.run(test_crypto_real_time_bars())
