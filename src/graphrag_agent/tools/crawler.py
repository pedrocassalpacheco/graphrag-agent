import asyncio
import httpx
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from typing import List, Set, Dict
from utils.logging_config import get_logger  # Import the shared logger

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
    def __init__(self, base_url: str, max_depth: int = 2, include_external: bool = False, delay: float = 1.0):
        self.base_url = base_url
        self.max_depth = max_depth
        self.include_external = include_external
        self.delay = delay  # Delay between requests in seconds
        self.visited: Set[str] = set()
        self.lock = asyncio.Lock()

    async def fetch(self, url: str, client: httpx.AsyncClient) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; GraphRagCrawler/1.0; +https://github.com/yourusername/graphrag)'
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
            defragmented_url, _ = urldefrag(joined_url)  # Remove the fragment part (e.g., #section1)
            parsed_url = urlparse(defragmented_url)
            if not self.include_external and parsed_url.netloc != urlparse(self.base_url).netloc:
                continue
            if defragmented_url not in links:  # Avoid duplicates
                links.append(defragmented_url)
        return links

    async def crawl(self, url: str, depth: int, client: httpx.AsyncClient, queue: asyncio.Queue):
        logger.debug(f"Crawling {url} at depth {depth}")
        
        async with self.lock:  # Ensure thread-safe access to self.visited
            if depth > self.max_depth or url in self.visited:
                logger.info(f"Skipping {url} - {'max depth reached' if depth > self.max_depth else 'already visited'}")
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
        tasks = [self.crawl(link, depth + 1, client, queue) for link in links if link not in self.visited]
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
            await self.crawl(self.base_url,  0 , client, queue)  # Start from depth 0
        await queue.put(None)  # Signal that crawling is complete
        logger.info(f"Crawl completed. Visited {len(self.visited)} pages")
        return self.visited