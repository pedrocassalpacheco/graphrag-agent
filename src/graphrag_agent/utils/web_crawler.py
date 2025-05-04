import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from typing import List, Dict, Set, Optional
import logging
from dataclasses import dataclass
import time

@dataclass
class PageContent:
    url: str
    title: str
    content: str

class WebCrawler:
    def __init__(
        self,
        base_url: str,
        follow_external: bool = False,
        max_pages: int = 100,
        delay: float = 1.0,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.follow_external = follow_external
        self.max_pages = max_pages
        self.delay = delay
        self.user_agent = user_agent
        self.visited_urls: Set[str] = set()
        self.results: List[PageContent] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def is_internal_url(self, url: str) -> bool:
        """Check if the URL belongs to the same domain as the base URL."""
        return urlparse(url).netloc == self.base_domain

    def get_page_content(self, url: str) -> Optional[PageContent]:
        """Fetch and parse a single page."""
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.title.string if soup.title else "No Title"
            
            # Get main content (this is a simple implementation - you might want to customize this)
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            content = soup.get_text(separator=' ', strip=True)
            
            return PageContent(url=url, title=title, content=content)
            
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> Set[str]:
        """Extract all links from the page."""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(current_url, href)
            # Only add if it's a valid URL and matches our criteria
            if absolute_url.startswith(('http://', 'https://')):
                if self.follow_external or self.is_internal_url(absolute_url):
                    links.add(absolute_url)
        return links

    def crawl(self) -> List[Dict]:
        """Crawl the website starting from the base URL."""
        urls_to_visit = {self.base_url}
        pages_crawled = 0
        
        self.logger.info(f"Starting crawl of {self.base_url}")
        self.logger.info(f"Max pages to crawl: {self.max_pages}")
        self.logger.info(f"Follow external links: {self.follow_external}")
        
        while urls_to_visit and pages_crawled < self.max_pages:
            current_url = urls_to_visit.pop()
            
            if current_url in self.visited_urls:
                self.logger.debug(f"Skipping already visited URL: {current_url}")
                continue
                
            self.logger.info(f"Crawling page {pages_crawled + 1}/{self.max_pages}: {current_url}")
            
            # Respect the delay between requests
            time.sleep(self.delay)
            
            try:
                headers = {"User-Agent": self.user_agent}
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get page content
                page_content = self.get_page_content(current_url)
                if page_content:
                    self.results.append(page_content)
                    self.logger.info(f"Successfully crawled: {current_url}")
                    self.logger.info(f"Title: {page_content.title}")
                    self.logger.info(f"Content length: {len(page_content.content)} characters")
                
                # Extract new links
                new_links = self.extract_links(soup, current_url)
                # Only add links that haven't been visited yet
                new_links = new_links - self.visited_urls
                urls_to_visit.update(new_links)
                
                # Mark current URL as visited
                self.visited_urls.add(current_url)
                pages_crawled += 1
                
                self.logger.info(f"Found {len(new_links)} new links to crawl")
                self.logger.info(f"Total URLs in queue: {len(urls_to_visit)}")
                self.logger.info(f"Total URLs visited: {len(self.visited_urls)}")
                
            except Exception as e:
                self.logger.error(f"Error processing {current_url}: {str(e)}")
                # Mark as visited even if there was an error to avoid retrying
                self.visited_urls.add(current_url)
                continue
        
        self.logger.info(f"Crawl completed. Crawled {pages_crawled} pages.")
        self.logger.info(f"Total unique URLs visited: {len(self.visited_urls)}")
        self.logger.info(f"Total pages with content: {len(self.results)}")
        
        return self.to_json()

    def to_json(self) -> List[Dict]:
        """Convert the results to a JSON-serializable format."""
        return [
            {
                "url": result.url,
                "title": result.title,
                "content": result.content
            }
            for result in self.results
        ]

    def save_to_file(self, filename: str):
        """Save the results to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2) 