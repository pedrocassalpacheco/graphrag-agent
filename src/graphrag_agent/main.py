import asyncio
from graphrag_agent.tools.crawler import AsyncWebCrawler, AsyncFileSystemCrawler

from openai import AsyncOpenAI
import os
from dotenv import load_dotenv


async def test_web_crawler():
    """
    Test the AsyncWebCrawler class.
    """
    url = "https://docs.langflow.org/"
    crawler = AsyncWebCrawler(base_path=url)
    queue = asyncio.Queue()

    # Run the crawler
    await crawler.run(current_path=url, depth=10, output=queue)

    # Check if the queue is not empty
    assert not queue.empty()
    dump_queue(queue)

    # Check if the crawler found resources
    # assert len(crawler.found_resources) > 0


def dump_queue(queue: asyncio.Queue):
    """
    Dump the contents of a queue to a file.
    """
    with open("queue_dump.txt", "w") as f:
        while not queue.empty():
            item = queue.get_nowait()
            f.write(f"{item}\n")


if __name__ == "__main__":

    asyncio.run(test_web_crawler())
