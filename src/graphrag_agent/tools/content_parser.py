import asyncio
import httpx
from bs4 import BeautifulSoup
from typing import Dict, List, Union
import re
import io
import json

from utils.logging_config import get_logger  # Import the shared logger

logger = get_logger(__name__)  # Get a logger for this module


class AsyncPageContentParser:
    """Asynchronous parser for web page content.
    This class fetches and parses web pages concurrently, extracting structured content
    from HTML elements such as headings and paragraphs.
    Attributes:
        delay (float): Delay in seconds between requests to avoid overwhelming the server.
    """

    def __init__(self, delay: float = 1.0):
        self.delay = delay

    def clean_text(self, text: str) -> str:
        """
        Cleans unwanted control characters and other unnecessary symbols from the text.
        """
        return re.sub(r"[\u0000-\u001F\u007F-\u009F\u00b6]", "", text)

    async def fetch_page(self, url: str, client: httpx.AsyncClient) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; GraphRagParser/1.0; +https://github.com/yourusername/graphrag)"
            }
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""
        except Exception as e:
            print(f"Unexpected error fetching {url}: {str(e)}")
            return ""

    async def parse_content(self, html: str) -> Dict[str, List[str]]:
        soup = BeautifulSoup(html, "html.parser")
        content = {}
        current_title = None

        # Ugly but effective way to parse the content
        for element in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "ul",
                "ol",
                "table",
                "blockquote",
                "pre",
                "code",
                "img",
            ]
        ):
            logger.debug(
                f"Parsing element: {element.name} with text: {element.get_text(strip=True)}"
            )
            if element.name.startswith("h"):  # Titles and subtitles
                current_title = self.clean_text(element.get_text(strip=True))
                content[current_title] = []
            elif current_title and element.name == "p":  # Paragraphs
                content[current_title].append(
                    self.clean_text(element.get_text(strip=True))
                )
            elif current_title and element.name in ["ul", "ol"]:  # Lists
                list_items = [
                    self.clean_text(li.get_text(strip=True))
                    for li in element.find_all("li")
                ]
                content[current_title].extend(list_items)
            elif current_title and element.name == "table":  # Tables
                rows = []
                for row in element.find_all("tr"):
                    cells = [
                        self.clean_text(cell.get_text(strip=True))
                        for cell in row.find_all(["td", "th"])
                    ]
                    rows.append(cells)
                content[current_title].append({"table": rows})
            elif current_title and element.name == "blockquote":  # Blockquotes
                content[current_title].append(
                    self.clean_text(element.get_text(strip=True))
                )
            elif current_title and element.name in ["pre", "code"]:  # Code blocks
                content[current_title].append(
                    self.clean_text(element.get_text(strip=True))
                )
            elif current_title and element.name == "img":  # Images
                alt_text = self.clean_text(element.get("alt", "No description"))
                src = self.clean_text(element.get("src", "No source"))
                content[current_title].append({"image": {"alt": alt_text, "src": src}})

        await asyncio.sleep(0)  # Yield control to the event loop
        return content

    async def parse(
        self,
        input: asyncio.Queue,
        output: Union[
            asyncio.Queue, Dict[str, Dict[str, List[str]]], io.TextIOWrapper
        ] = None,
        output_file: io.TextIOWrapper = None,
    ) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                url = await input.get()
                if url is None:  # Crawling is complete
                    break
                logger.info(f"Parsing page {url}")
                cleaned_url = self.clean_text(url)  # Clean the URL
                html = await self.fetch_page(cleaned_url, client)
                if html:
                    parsed_content = await self.parse_content(html)
                    # Handle different output types
                    if isinstance(output, asyncio.Queue):
                        await output.put(
                            {"url": cleaned_url, "parsed_content": parsed_content}
                        )
                    elif isinstance(output, dict):
                        output[cleaned_url] = parsed_content
                    elif isinstance(output, io.TextIOWrapper):
                        for title, content in parsed_content.items():
                            json_line = {
                                "url": cleaned_url,
                                "title": title,
                                "content": content,
                            }
                            output.write(json.dumps(json_line) + "\n")
