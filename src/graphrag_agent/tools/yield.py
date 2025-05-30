import asyncio


async def func_a():
    for i in range(3):
        await asyncio.sleep(0.1)  # simulate async work
        yield f"func_a: {i}"


async def func_b():
    async for val in func_a():
        print(f"func_b received: {val}")
        yield f"func_b processed: {val}"


async def main():
    async for result in func_b():
        print(f"main got: {result}")


# Run the async main
asyncio.run(main())
