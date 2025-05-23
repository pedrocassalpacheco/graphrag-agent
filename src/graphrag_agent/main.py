import asyncio
from graphrag_agent.tools.crawler import AsyncWebCrawler, AsyncFileSystemCrawler
from graphrag_agent.tools.content_parser import AsyncMarkdownParser

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
    await dump_queue(queue)

    # Check if the crawler found resources
    # assert len(crawler.found_resources) > 0


async def test_file_crawler():
    """
    Test the AsyncWebCrawler class.
    """
    path = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/components"
    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["py"])
    queue = asyncio.Queue()
    files = []
    # Run the crawler
    await crawler.run(output=queue)
    dump_queue(queue)


async def test_markdown_parser():
    path = "/Users/pedropacheco/Projects/dev//langflow.current/docs/docs/Components"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["md"])
    parser = AsyncMarkdownParser()

    crawler_task = asyncio.create_task(crawler.run(output=crawler_queue))
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task)
    await dump_queue(parser_queue, "markdown_parser.txt")


async def dump_queue(queue: asyncio.Queue, file_path: str = "queue_dump.txt"):
    """
    Dump the contents of a queue to a file.
    """
    with open(file_path, "w") as f:
        while not queue.empty():
            item = await queue.get()
            if item is None:
                break
            else:
                f.write(f"{item}\n")


if __name__ == "__main__":

    asyncio.run(test_file_crawler())
    asyncio.run(test_markdown_parser())
