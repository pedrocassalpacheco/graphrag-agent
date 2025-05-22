import asyncio
from graphrag_agent.tools.crawler import AsyncWebCrawler, AsyncFileSystemCrawler

from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


def dump_queue(queue: asyncio.Queue):
    """
    Dump the contents of a queue to a file.
    """
    with open("queue_dump.txt", "w") as f:
        while not queue.empty():
            item = queue.get_nowait()
            f.write(f"{item}\n")


if __name__ == "__main__":

    async def run(self, queue: asyncio.Queue):
        logger.info(f"Starting file scan from {self.base_path}")
        await self._run(self.base_path, 0, queue)
        await queue.put(None)
        logger.info(f"File scan completed. Found {len(self.found_resources)} files")
        return self.found_resources
