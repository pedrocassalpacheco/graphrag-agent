import asyncio
import io
import json
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

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
        while True:
            # Get item from queue
            item = await input.get()
            logger.debug(f"Processing item: {item}")

            # Check for termination signal
            if item is None:
                logger.debug(f"{self.__class__.__name__} received termination signal")
                # Cleanup any resources
                self._cleanup()
                if isinstance(output, asyncio.Queue):
                    await output.put(None)  # Forward termination
                if isinstance(output, io.TextIOWrapper):
                    output.close()
                break

            try:
                # Process the item (implemented by subclasses)
                result = await self._process_item(item)
                logger.debug(f"Result: {result}")
                # If result is None, skip output handling
                if result is not None:
                    # Handle different output types
                    if isinstance(output, asyncio.Queue):
                        await output.put(result)
                    elif isinstance(output, io.TextIOWrapper):
                        output.write(str(item) + "\n")
                    elif isinstance(output, list):
                        output.append(result)

                    self.processed_count += 1
                    logger.debug(f"Processed item {self.processed_count}")

            except Exception as e:
                logger.error(f"Error processing item: {e}")
                logger.error(traceback.format_exc())
            finally:
                # Mark as done regardless of success/failure
                input.task_done()
