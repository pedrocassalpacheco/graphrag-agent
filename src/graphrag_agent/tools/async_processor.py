import asyncio
import io
import json
import logging
import traceback
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pathlib import Path

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseAsyncProcessor(ABC):
    """
    Abstract base class for all asynchronous processors that work with queues.
    Provides common queue processing logic while delegating specific processing to subclasses.
    """

    def __init__(self):
        self.processed_count = 0

    @abstractmethod
    async def _process_item(self, item: Any) -> Any:
        """
        Process a single item from the queue. Must be implemented by subclasses.

        Args:
            item: The item retrieved from the queue

        Returns:
            Processed result or None if processing should be skipped
        """
        pass

    async def _cleanup(self):
        """
        Cleanup method to be called after processing is complete. Can be used for finalizing tasks.
        """
        pass

    async def run(
        self,
        input: asyncio.Queue,
        output: Union[asyncio.Queue, List[Any], io.TextIOWrapper] = None,
    ):
        """
        Process items from input queue, handle outputs, and manage queue state.

        Args:
            input_queue: Queue containing items to process
            output: Destination for processed results - can be:
                - asyncio.Queue: for async pipeline processing
                - List: to collect results in a list
                - io.TextIOWrapper: to write results to a file (JSON serialized)
        """
        concrete_class = self.__class__.__name__
        logger.debug(f"Starting {concrete_class} processor...")
        while True:
            # Get item from queue. No assumptions are made on what the item is.
            # It is up to the implementation to figure it out.
            item = await input.get()

            logger.debug(f"Processing {str(item)[:50]} ...")

            # Check for termination signal
            if item is None:
                logger.debug(
                    f"{concrete_class} received termination signal.Processed {self.processed_count} items."
                )
                # Cleanup any resources
                await self._cleanup()
                if isinstance(output, asyncio.Queue):
                    await output.put(None)  # Forward termination
                if isinstance(output, io.TextIOWrapper):
                    output.close()
                break

            try:
                # Process the item (implemented by subclasses)
                logger.debug
                result = await self._process_item(item)

                # Convert results to a list for uniform processing
                if result is None:
                    items_to_process = []
                elif isinstance(result, (list, set, tuple)):
                    items_to_process = list(result)
                else:
                    items_to_process = [result]

                # Process each item uniformly
                for result in items_to_process:
                    if result is not None:
                        # Handle different output types
                        if isinstance(output, asyncio.Queue):
                            await output.put(result)
                        elif isinstance(output, io.TextIOWrapper):
                            output.write(str(result) + "\n")
                        elif isinstance(output, list):
                            output.append(result)

                    self.processed_count += 1

            except Exception as e:
                logger.error(f"Error processing fron {concrete_class} item: {e}")
                logger.error(traceback.format_exc())
            finally:
                # Mark as done regardless of success/failure
                input.task_done()

    def _validate_file_path(self, file_path: str) -> Path:
        """Validate file path and return Path object."""
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if not os.access(path, os.R_OK):
            raise PermissionError(f"File is not readable: {file_path}")

        return path
