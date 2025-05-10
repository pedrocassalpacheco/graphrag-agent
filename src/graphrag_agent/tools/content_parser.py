import ast
import asyncio
import httpx
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from graphrag_agent.utils.logging_config import get_logger


logger = get_logger(__name__)


class BaseAsyncParser:
    """Base class for asynchronous content parsers.
    
    Provides common functionality for parsing different types of content.
    
    Attributes:
        delay (float): Delay between operations to prevent resource exhaustion.
    """
    
    def __init__(self, delay: float):
        self.delay = delay
    
    def clean_text(self, text: str) -> str:
        """
        Cleans unwanted control characters and other unnecessary symbols from text.
        """
        if not text:
            return ""
        # Clean control characters and normalize whitespace
        cleaned = re.sub(r"[\u0000-\u001F\u007F-\u009F\u00b6]", "", text)
        return re.sub(r"\s+", " ", cleaned).strip()
        
    async def parse(
        self,
        input: asyncio.Queue,
        output: Union[
            asyncio.Queue, Dict[str, Dict[str, List[str]]], io.TextIOWrapper
        ] = None,
    ) -> None:
        """
        Process content from the input queue and write structured content to the output.
        """
        while True:
            source = await input.get()
            
            if source is None:  # Processing is complete
                logger.info(f"{self.__class__.__name__} received termination signal, forwarding it")
                if isinstance(output, asyncio.Queue):
                    await output.put(None)
                break

            logger.info(f"Parsing {source}")
            
            # Get content - implemented by subclasses
            content = await self._get_content(source)
            
            if content:
                # Parse the content - implemented by subclasses
                parsed_content = await self._parse_content(content)
                
                # Process each section
                for title, section_content in parsed_content.items():
                    if not section_content:
                        continue
                        
                    logger.debug(f"Parsed section: {title} from {source}")
                    
                    item = {
                        "url": source,  # Using source as the identifier
                        "title": title,
                        "content": section_content,
                    }
                    
                    # Handle different output types
                    if isinstance(output, asyncio.Queue):
                        await output.put(item)
                    elif isinstance(output, dict):
                        if source not in output:
                            output[source] = []
                        output[source].append(item)
                    elif isinstance(output, io.TextIOWrapper):
                        output.write(json.dumps(item) + "\n")
            
            # Mark the item as processed
            input.task_done()
    
    async def _get_content(self, source: str) -> str:
        """Abstract method to get content from a source."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def _parse_content(self, content: str) -> Dict[str, List[str]]:
        """Abstract method to parse content."""
        raise NotImplementedError("Subclasses must implement this method")


class AsyncPageContentParser(BaseAsyncParser):
    """Asynchronous parser for web page content."""

    def __init__(self, delay: float = 1.0):
        super().__init__(delay)

    async def fetch_page(self, url: str, client: httpx.AsyncClient) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; GraphRagParser/1.0)"
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

    async def _get_content(self, url: str) -> str:
        """Fetch HTML content from a URL."""
        async with httpx.AsyncClient() as client:
            return await self.fetch_page(url, client)

    async def _parse_content(self, html: str) -> Dict[str, List[str]]:
        """Parse HTML content into a structured format."""
        soup = BeautifulSoup(html, "html.parser")
        content = {}
        current_title = None

        # Ugly but effective way to parse the content
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "blockquote", "pre", "code", "img"]
        ):
            logger.debug(f"Parsing element: {element.name}")
            
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


class AsyncMarkdownParser(BaseAsyncParser):
    """Asynchronous parser for local markdown files."""

    def __init__(self, delay: float = 0.01):
        super().__init__(delay)

    async def read_file(self, file_path: str) -> Optional[str]:
        """Read the content of a markdown file asynchronously."""
        try:
            content = await asyncio.to_thread(Path(file_path).read_text, encoding='utf-8')
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    async def _get_content(self, file_path: str) -> Optional[str]:
        """Read content from a local file."""
        return await self.read_file(file_path)

    async def _parse_content(self, content: str) -> Dict[str, List[str]]:
        """Parse markdown content directly without converting to HTML."""
        if not content:
            return {}
        
        structured_content = {}
        md = MarkdownIt('gfm-like')
        tokens = md.parse(content)

        structured_content = {}
        current_title = None
        current_content = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == 'heading_open':
                # Save previous section
                if current_title:
                    structured_content[current_title] = '\n'.join(current_content).strip()
                    current_content = []

                # Get the heading content
                i += 1
                if i < len(tokens) and tokens[i].type == 'inline':
                    current_title = tokens[i].content
                i += 1  # Skip heading_close
                continue

            elif token.type == 'paragraph_open':
                i += 1
                if i < len(tokens) and tokens[i].type == 'inline':
                    current_content.append(tokens[i].content)
                i += 1  # Skip paragraph_close

            elif token.type == 'inline' and token.children:
                # Look for inline image tokens
                for child in token.children:
                    if child.type == 'image':
                        alt = child.attrs.get('alt', '')
                        src = child.attrs.get('src', '')
                        img_text = f"![{alt}]({src})"
                        current_content.append(img_text)
                i += 1

            elif token.type == 'image':
                # Standalone image token
                alt = token.attrs.get('alt', '')
                src = token.attrs.get('src', '')
                img_text = f"![{alt}]({src})"
                current_content.append(img_text)
                i += 1

            elif token.type == 'table_open':
                table_lines = []
                while token.type != 'table_close':
                    if token.type == 'inline':
                        table_lines.append(token.content)
                    i += 1
                    if i < len(tokens):
                        token = tokens[i]
                    else:
                        break
                i += 1  # Skip table_close
                table_str = '\n'.join(table_lines)
                current_content.append(table_str)
                continue

            else:
                i += 1

        if current_title and current_content:
            structured_content[current_title] = '\n'.join(current_content).strip()
        
        return structured_content
    
class AsyncPythonSourceParser(BaseAsyncParser):
    """
    Asynchronous tool to parse Python source code files into chunks of functions and classes.
    """
    
    def __init__(self, delay: float = 0.0):
        super().__init__(delay)
        self.parsed_count = 0
    
    async def _get_content(self, path_str: str) -> Optional[str]:
        """Validate and return the python file path object from string."""
        path = Path(path_str)
        if not path.exists() or not path.is_file() or path.suffix != ".py":
            logger.error(f"Invalid Python file path: {path_str}")
            return ""
        else:
            return path
    
    async def _parse_content(self, path: Path) -> Dict[str, List[str]]:
        """Parse Python source code into a structured format using a file object."""
        structured_content = {}
        try:
            # Open the file and pass the file object to ast.parse()
            with path.open("r", encoding="utf-8") as file:
                tree = ast.parse(file.read(), filename=str(path))

            # Extract module docstring if available
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                structured_content[f"{path.name}::module"] = [module_docstring]

            # Parse functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, "end_lineno") else node.body[-1].lineno
                    with path.open("r", encoding="utf-8") as file:
                        content = file.read()
                    chunk = "\n".join(content.splitlines()[start_line:end_line])

                    section_title = f"{path.name}::{node.name}"
                    structured_content[section_title] = [chunk]

                elif isinstance(node, ast.ClassDef):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, "end_lineno") else node.body[-1].lineno
                    with path.open("r", encoding="utf-8") as file:
                        content = file.read()
                    chunk = "\n".join(content.splitlines()[start_line:end_line])

                    section_title = f"{path.name}::{node.name}"
                    structured_content[section_title] = [chunk]

            self.parsed_count += len(structured_content)
            logger.info(f"Parsed {len(structured_content)} Python code sections from {path.name}")

        except Exception as e:
            logger.error(f"Error parsing Python code from {path}: {e}")

        return structured_content