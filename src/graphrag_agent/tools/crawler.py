from abc import ABC, abstractmethod
import asyncio
import io
from typing import Dict, List, Set, Union, Any, Optional, override
import traceback
import os

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag

from graphrag_agent.tools.async_processor import BaseAsyncProcessor
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
        self.find_count = 0

    async def run(
        self,
        *,
        current_path: Optional[str] = None,
        depth: int = 0,
        output: Union[asyncio.Queue, list[Any], io.TextIOWrapper],
    ):
        """
        Main entry point for the file crawler.

        Args:
            queue: Queue to put file paths or file contents

        Returns:
            Set[str]: A set of all file paths that were visited
        """
        logger.info(f"Starting scan from {self.base_path} at depth {depth}")

        await self._crawl(current_path=self.base_path, depth=depth, output=output)

        # Signal that crawling is complete
        if isinstance(output, asyncio.Queue):
            await output.put(None)  # Forward termination
        if isinstance(output, io.TextIOWrapper):
            output.close()

        logger.info(f"File scan completed. Found {self.find_count} files")

    async def _update_output(
        self, file_path: str, output: Union[asyncio.Queue, List[Any], io.TextIOWrapper]
    ):
        """
        Update the output with the found file path.
        Args:
            file_path: The path of the file to be added
            output: The output destination (queue or list)
        """
        if isinstance(output, asyncio.Queue):
            await output.put(file_path)
        elif isinstance(output, list):
            logger.debug(f"Added to list {output}")
            output.append(file_path)
        elif isinstance(output, io.TextIOWrapper):
            output.write(f"{file_path}\n")
        else:
            raise ValueError("Output must be a Queue, List, or TextIOWrapper")

        # Next ...

        self.find_count += 1

    @abstractmethod
    async def _crawl(
        self,
        *,
        current_path: Optional[str],
        depth: int,
        output: Union[asyncio.Queue, List[Any], io.TextIOWrapper],
    ):
        """
        Abstract method to be implemented by subclasses for specific crawling logic.
        Args:
            current_path: The current path being scanned
            depth: The current depth of the scan
            queue: Queue to put file paths or file contents
        """
        pass


class AsyncFileSystemCrawler(BaseAsyncCrawler):
    """Asynchronous file system crawler for finding files with specific extensions."""

    def __init__(
        self,
        base_path: str,
        max_depth: int = 2,
        extensions: List[str] = [],
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
    async def _crawl(
        self,
        *,
        current_path: Optional[str] = None,
        depth: int = 0,
        output: Union[asyncio.Queue, List[Any], io.TextIOWrapper],
    ) -> Any:

        # Starts at the base path if not provided
        if current_path is None:
            current_path = self.base_path

        # Recursively scan a directory for files with matching extensions.
        if depth > self.max_depth or current_path in self.visited:
            logger.debug(
                "Skipping directory: %s (max depth reached or already visited)",
                current_path,
            )
            return None

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
                    await self._update_output(file_path, output)

                elif entry.is_dir() and depth < self.max_depth:
                    subdirs.append(entry.path)

            # Process subdirectories concurrently
            tasks = [
                self._crawl(current_path=subdir, depth=depth + 1, output=output)
                for subdir in subdirs
            ]
            await asyncio.gather(*tasks)

        except PermissionError:
            logger.warning(f"Permission denied: {current_path}")
        except Exception as e:
            logger.error(f"Error scanning {current_path}: {str(e)}")
            logger.debug(traceback.format_exc())
