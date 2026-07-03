
import asyncio
import time

async def func1():
    print("Start func1")
    await asyncio.sleep(5)
    print("End func1")
    return "func1 complete"

async def func2():
    print("Start func2")
    await asyncio.sleep(3)
    print("End func2")
    return "func2 complete"

async def main():
    start_time = time.time()
    co_routines = asyncio.gather(func1(), func2())
    a, b = await co_routines
    print(a, b)
    end_time = time.time()
    print("time", end_time-start_time)

asyncio.run(main())
