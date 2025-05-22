import asyncio
import io
import json
import os
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from typing import override  # Python 3.12+
except ImportError:

    def override(func):  # fallback for Python <3.12
        return func


import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseAsyncCrawler(ABC):
    """
    Abstract base class for all asynchronous processors that work with queues.
    Provides common queue processing logic while delegating specific processing to subclasses.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.found_resources = set()

    async def run(
        self,
        *,
        current_path: Optional[str] = None,
        depth: int = 0,
        output: asyncio.Queue,
    ) -> Set[str]:
        """
        Main entry point for the file crawler.

        Args:
            queue: Queue to put file paths or file contents

        Returns:
            Set[str]: A set of all file paths that were visited
        """
        logger.info(f"Starting scan from {current_path}")

        await self._run(current_path=current_path, depth=depth, output=output)

        # Signal that crawling is complete
        await output.put(None)

        logger.info(f"File scan completed. Found {len(self.visited) - 1} files")
        return self.found_resources

    @abstractmethod
    async def _run(
        self, current_path: Optional[str], depth: int, output: asyncio.Queue
    ) -> Set[str]:
        """
        Abstract method to be implemented by subclasses for specific crawling logic.
        Args:
            current_path: The current path being scanned
            depth: The current depth of the scan
            queue: Queue to put file paths or file contents
        """
        pass


class AsyncWebCrawler(BaseAsyncCrawler):
    """Asynchronous web crawler for extracting links and page content.
    This class crawls a given base URL, extracts links from the pages, and
    fetches their content concurrently. It can be configured to include or exclude
    external links and to limit the depth of the crawl.
    Attributes:
        base_url (str): The base URL to start crawling from.
        max_depth (int): The maximum depth to crawl. Default is 2.
        include_external (bool): Whether to include external links. Default is False.
        delay (float): Delay in seconds between requests to avoid overwhelming the server.
        visited (Set[str]): A set of URLs that have already been visited.
        lock (asyncio.Lock): An asyncio lock for thread-safe access to the visited set.
    """

    def __init__(
        self,
        base_path: str,
        max_depth: int = 2,
        delay: float = 0.001,
    ):
        super().__init__(base_path)
        self.max_depth = max_depth
        self.delay = delay
        self.visited = set()
        self.include_external = False
        self.lock = asyncio.Lock()

    async def _fetch(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; GraphRagCrawler/1.0; +https://github.com/yourusername/graphrag)"
            }
            response = await self.client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            return None

    async def _extract_links(self, url: str) -> List[str]:
        links = []
        html = await self._fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        logger.debug(f"Extracting links from {url}")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            joined_url = urljoin(url, href)
            defragmented_url, _ = urldefrag(
                joined_url
            )  # Remove the fragment part (e.g., #section1)
            parsed_url = urlparse(defragmented_url)
            if not self.include_external and parsed_url.netloc != urlparse(url).netloc:
                continue
            if defragmented_url not in links:  # Avoid duplicates
                links.append(defragmented_url)
        return links

    async def _crawl(self, url: str, depth: int, queue: asyncio.Queue):
        logger.debug(f"Crawling {url} at depth {depth}")
        try:
            async with self.lock:  # Ensure thread-safe access to self.visited
                if depth > self.max_depth or url in self.visited:
                    logger.info(
                        f"Skipping {url} - {'max depth reached' if depth > self.max_depth else 'already visited'}"
                    )
                    return
                self.visited.add(url)
                await queue.put(url)  # Add the URL to the queue

            # Fetch links from the page
            links = await self._extract_links(url)
            if links:
                logger.debug(f"Found {len(links)} links on {url}")

                # Crawl the extracted links
                tasks = [
                    self._crawl(link, depth + 1, queue)
                    for link in links
                    if link not in self.visited
                ]
                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            traceback.print_exc()

    async def _run(
        self,
        *,
        current_path: Optional[str] = None,
        depth: int = 0,
        output: asyncio.Queue,
    ) -> Set[str]:
        self.client = httpx.AsyncClient()
        logger.info(f"Starting crawl from {current_path}")
        await self._crawl(current_path, 0, output)


class AsyncFileSystemCrawler(BaseAsyncCrawler):
    """Asynchronous file system crawler for finding files with specific extensions."""

    def __init__(
        self,
        base_path: str,
        max_depth: int = 2,
        extensions: List[str] = None,
        delay: float = 0.001,
    ):
        super().__init__(base_path)
        self.max_depth = max_depth
        self.extensions = extensions or []
        self.delay = delay
        self.visited = set()

    def _should_include_file(self, filename: str) -> bool:
        """
        Determines whether a file should be included based on its extension.
        Args:
            filename: The name of the file to check
        Returns:
            bool: True if the file should be included, False otherwise
        """
        logger.debug(
            f"Checking if file should be included: {filename} {self.extensions}"
        )
        return any(filename.endswith(ext) for ext in self.extensions)

    @override
    async def _run(
        self,
        *,
        current_path: Optional[str] = None,
        depth: int = 0,
        output: asyncio.Queue,
    ) -> Set[str]:

        # Starts at the base path if not provided
        if current_path is None:
            current_path = self.base_path

        # Recursively scan a directory for files with matching extensions.
        if depth > self.max_depth or current_path in self.visited:
            logger.debug(
                "Skipping directory: %s (max depth reached or already visited)",
                current_path,
            )
            return

        # Mark this directory as visited
        self.visited.add(current_path)
        logger.debug(
            f"Scanning directory: {current_path} (depth: {depth}/{self.max_depth})"
        )

        try:
            # Get directory entries asynchronously
            entries = await asyncio.to_thread(os.scandir, current_path)

            subdirs = []
            for entry in entries:
                # Small delay to prevent resource exhaustion
                await asyncio.sleep(self.delay)

                if entry.is_file() and self._should_include_file(entry.name):
                    file_path = os.path.abspath(entry.path)
                    logger.debug(f"Found matching file: {file_path}")
                    self.found_resources.add(file_path)
                    await output.put(file_path)

                elif entry.is_dir() and depth < self.max_depth:
                    subdirs.append(entry.path)

            # Process subdirectories concurrently
            tasks = [
                self._run(current_path=subdir, depth=depth + 1, output=output)
                for subdir in subdirs
            ]
            await asyncio.gather(*tasks)

        except PermissionError:
            logger.warning(f"Permission denied: {current_path}")
        except Exception as e:
            logger.error(f"Error scanning {current_path}: {str(e)}")
