import asyncio
import httpx
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from typing import List, Set, Optional
from graphrag_agent.utils.logging_config import get_logger  # Import the shared logger
import os
from pathlib import Path

logger = get_logger(__name__)  # Get a logger for this module

class AsyncWebCrawler:
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
        base_url: str,
        max_depth: int = 2,
        include_external: bool = False,
        delay: float = 1.0,
    ):
        self.base_url = base_url
        self.max_depth = max_depth
        self.include_external = include_external
        self.delay = delay  # Delay between requests in seconds
        self.visited: Set[str] = set()
        self.lock = asyncio.Lock()

    async def fetch(self, url: str, client: httpx.AsyncClient) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; GraphRagCrawler/1.0; +https://github.com/yourusername/graphrag)"
            }
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            return ""

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        links = []
        logger.debug(f"Extracting links from {current_url}")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            joined_url = urljoin(current_url, href)
            defragmented_url, _ = urldefrag(
                joined_url
            )  # Remove the fragment part (e.g., #section1)
            parsed_url = urlparse(defragmented_url)
            if (
                not self.include_external
                and parsed_url.netloc != urlparse(self.base_url).netloc
            ):
                continue
            if defragmented_url not in links:  # Avoid duplicates
                links.append(defragmented_url)
        return links

    async def crawl(
        self, url: str, depth: int, client: httpx.AsyncClient, queue: asyncio.Queue
    ):
        logger.debug(f"Crawling {url} at depth {depth}")

        async with self.lock:  # Ensure thread-safe access to self.visited
            if depth > self.max_depth or url in self.visited:
                logger.info(
                    f"Skipping {url} - {'max depth reached' if depth > self.max_depth else 'already visited'}"
                )
                return
            self.visited.add(url)

        # Add delay between requests
        await asyncio.sleep(self.delay)

        html = await self.fetch(url, client)
        if not html:
            logger.warning(f"Failed to fetch content from {url}")
            return

        soup = BeautifulSoup(html, "html.parser")

        links = self.extract_links(soup, url)
        logger.debug(f"Found {len(links)} links on {url}")

        # Add the current URL to the queue for parsing
        await queue.put(url)

        # Crawl the extracted links
        tasks = [
            self.crawl(link, depth + 1, client, queue)
            for link in links
            if link not in self.visited
        ]
        await asyncio.gather(*tasks)

    async def run(self, queue: asyncio.Queue):
        """
        Main entry point for the crawler. Initializes an async HTTP client and starts
        the crawling process from the base URL.

        Returns:
            Set[str]: A set of all URLs that were visited during the crawl.
        """
        async with httpx.AsyncClient() as client:
            logger.info(f"Starting crawl from {self.base_url}")
            await self.crawl(self.base_url, 0, client, queue)  # Start from depth 0
        await queue.put(None)  # Signal that crawling is complete
        logger.info(f"Crawl completed. Visited {len(self.visited)} pages")
        await queue.put(None)
        return self.visited


class AsyncFileSystemCrawler:
    """Asynchronous file system crawler for finding files with specific extensions.
    
    This class scans a directory and its subdirectories to find files matching 
    specified extensions. It can be configured to limit the depth of recursion
    and to include or exclude specific patterns.
    
    Attributes:
        base_path (str): The base directory path to start crawling from.
        max_depth (int): The maximum depth to crawl. Default is 2.
        extensions (List[str]): List of file extensions to include (e.g., ['.py', '.md'])
        delay (float): Small delay between operations to prevent resource exhaustion.
        visited (Set[str]): A set of paths that have already been visited.
    """

    def __init__(
        self,
        base_path: str,
        max_depth: int = 2,
        extensions: List[str] = None,
        delay: float = 0.001,
    ):
        self.base_path = os.path.abspath(base_path)
        self.max_depth = max_depth
        self.extensions = extensions or []  # Default extensions
        self.delay = delay
        self.visited: Set[str] = set()
        self.found_files: Set[str] = set()
    
    def _should_include_file(self, filename: str) -> bool:
        """Check if a file should be included based on its extension."""
        return any(filename.endswith(ext) for ext in self.extensions)
    
    
    async def scan_directory(
        self, 
        current_path: str, 
        depth: int, 
        queue: asyncio.Queue
    ) -> None:
        """Recursively scan a directory for files with matching extensions."""
        if depth > self.max_depth or current_path in self.visited:
            return
        
        # Mark this directory as visited
        self.visited.add(current_path)
        logger.debug(f"Scanning directory: {current_path} (depth: {depth})")
        
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
                    self.found_files.add(file_path)
                    await queue.put(file_path)
                
                elif entry.is_dir() and depth < self.max_depth:
                    subdirs.append(entry.path)
            
            # Process subdirectories concurrently
            tasks = [
                self.scan_directory(subdir, depth + 1, queue)
                for subdir in subdirs
            ]
            await asyncio.gather(*tasks)
                
        except PermissionError:
            logger.warning(f"Permission denied: {current_path}")
        except Exception as e:
            logger.error(f"Error scanning {current_path}: {str(e)}")
    
    async def run(self, queue: asyncio.Queue) -> Set[str]:
        """
        Main entry point for the file crawler.
        
        Args:
            queue: Queue to put file paths or file contents
            
        Returns:
            Set[str]: A set of all file paths that were visited
        """
        logger.info(f"Starting file scan from {self.base_path}")
        logger.info(f"Looking for files with extensions: {self.extensions}")
        
        await self.scan_directory(self.base_path, 0, queue)
        
        # Signal that crawling is complete
        await queue.put(None)
        
        logger.info(f"File scan completed. Found {len(self.visited) - 1} files")
        return self.found_files