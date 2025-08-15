import asyncio
from ib_async import IB

HOST = "127.0.0.1"
PORT = 4002
CLIENT_ID = 42

async def main():
    ib = IB()
    await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID)

    # Pull the snapshot
    rows = await ib.accountSummaryAsync()
    for row in rows:
        print(row.tag, row.value, row.currency)

    ib.disconnect()

asyncio.run(main())