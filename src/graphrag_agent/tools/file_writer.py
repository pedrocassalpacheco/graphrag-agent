import asyncio
import json
import os
from pathlib import Path
from dataclasses import asdict
from typing import IO, Optional, Dict, Any, TextIO, Union, override
from io import TextIOWrapper
from graphrag_agent.tools.parse_content import ParsedContent
from graphrag_agent.tools.async_processor import BaseAsyncProcessor
from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class AsyncJSONLWriter(BaseAsyncProcessor):
    """Async JSONL file writer that follows the BaseAsyncProcessor pattern."""

    def __init__(
        self, output_dir: str = "output", filename_prefix: str = "data", *args, **kwargs
    ):
        """
        Initialize the JSONL writer.

        Args:
            output_dir: Directory to write files to
            filename_prefix: Prefix for the output filename (e.g., 'docs', 'components', 'samples')
        """
        super().__init__(*args, **kwargs)
        self.output_dir = Path(output_dir)
        self.filename_prefix = filename_prefix
        self.output_dir.mkdir(exist_ok=True)
        self.output_file = self.output_dir / f"{self.filename_prefix}.jsonl"
        self.file_handle: Optional[IO[Any]] = None
        self.processed_count = 0
        self.mode: str = "w"

    async def _get_file_handle(self) -> IO[Any]:
        """Initialize the file handle if not already open."""
        if self.file_handle is None:
            # Open file in append mode to allow multiple runs
            self.file_handle = open(self.output_file, self.mode, encoding="utf-8")
            self.mode = "a"
            logger.info(f"Opened {self.output_file} for writing")
        return self.file_handle

    @override
    async def _process_item(self, item: ParsedContent) -> Dict[str, Any]:
        """
        Process a single item by writing it to the JSONL file.

        Args:
            item: The item to write to the file

        Returns:
            Dict with status information
        """
        try:
            file_handle = await self._get_file_handle()

            # Write item as JSON line
            item_dict = asdict(item)
            json_line = json.dumps(item_dict, ensure_ascii=False)
            file_handle.write(json_line + "\n")
            file_handle.flush()  # Ensure data is written immediately

            self.processed_count += 1
            item_id = (
                item.get("_id", f"item_{self.processed_count}")
                if isinstance(item, dict)
                else f"item_{self.processed_count}"
            )

            logger.debug(f"Written item {item_id} to {self.output_file}")

            return {
                "status": "written",
                "source": item.source,
                "section": item.section,
                "file": str(self.output_file),
            }

        except Exception as e:
            logger.error(f"Error writing item to JSONL file: {e}")
            return {"status": "error", "error": str(e), "file": str(self.output_file)}

    async def _cleanup(self):
        """Cleanup file handle resources."""
        try:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
                logger.info(
                    f"Closed {self.output_file} after writing {self.processed_count} items"
                )
        except Exception as e:
            logger.error(f"Error closing file handle: {str(e)}")
