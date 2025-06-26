import json
import tiktoken
from typing import Union, List, Dict, Optional
import asyncio
import os
from datetime import datetime

LOG_DIR = os.getenv("LOGS_DIR", "./logs")
# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


def print_pretty_json(json_data: Union[dict, str]) -> None:
    """Pretty print a JSON dictionary or string."""
    try:
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Parse string into a dictionary
        print(json.dumps(json_data, indent=4))  # Pretty print with indentation
    except (TypeError, ValueError) as e:
        print(f"Error printing JSON: {e}")


def count_tokens(
    messages: Optional[List[Dict]] = None,
    prompt: Optional[str] = None,
    model: str = "gpt-4-turbo",
) -> int:
    """
    Count tokens for either a complete messages array or a single prompt.

    Args:
        messages: List of message dictionaries
        prompt: Single string prompt
        model: Model name to use for tokenization

    Returns:
        int: Estimated token count
    """
    # Initialize tokenizer for the model
    encoding = tiktoken.encoding_for_model(model)

    if messages:
        # Count tokens in a full message list (more accurate for chat completions)
        num_tokens = 0
        for message in messages:
            # Add tokens for message role (system, user, assistant)
            num_tokens += 4  # Each message has a ~4 token overhead

            # Add tokens for each part of the message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                # Add tokens for the key name (role, content)
                num_tokens += 1

        # Every conversation has ~3 tokens of overhead
        num_tokens += 3
        return num_tokens

    elif prompt:
        # Simpler counting for a single string
        return len(encoding.encode(prompt))

    return 0


def dump_queue(queue: asyncio.Queue, max_items: int = 10):
    """Dump queue contents with better formatting to both stdout and log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"queue_dump_{timestamp}.log")

    header = f"\n=== Queue Contents ({queue.qsize()} items) ==="
    print(header)

    # Open file and process queue items in one loop
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(header + "\n")

            count = 0
            while not queue.empty() and count < max_items:
                item = queue.get_nowait()
                count += 1

                item_header = f"\n--- Item {count} ---"
                print(item_header)
                f.write(item_header + "\n")

                if hasattr(item, "source"):
                    source_line = f"Source: {item.source}"
                    section_line = f"Section: {item.section}"
                    content_line = f"Content: {item.content}"
                    type_line = f"Type: {item.content_type}"
                    parser_line = f"Parser: {item.metadata.get('parser', 'unknown')}"

                    # Print to stdout and write to file
                    print(source_line)
                    print(section_line)
                    print(content_line)
                    print(type_line)
                    print(parser_line)

                    f.write(source_line + "\n")
                    f.write(section_line + "\n")
                    f.write(content_line + "\n")
                    f.write(type_line + "\n")
                    f.write(parser_line + "\n")
                else:
                    raw_line = f"Raw item: {str(item)}"
                    print(raw_line)
                    f.write(raw_line + "\n")

            if queue.qsize() > 0:
                remaining_line = f"\n... and {queue.qsize()} more items"
                print(remaining_line)
                f.write(remaining_line + "\n")

            footer = f"\n=== End Queue Contents ===\n"
            print(footer)
            f.write(footer + "\n")

        print(f"Queue dump written to: {log_file}")

    except Exception as e:
        print(f"Error writing to log file {log_file}: {e}")


def dump_jsonl_file(file_path: str, max_items: int = 10):
    """Read JSONL file contents and dump ParsedContent objects to stdout only."""
    from graphrag_agent.tools.parse_content import ParsedContent

    try:
        # Read JSONL file into list of ParsedContent
        parsed_contents = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    # Convert dict back to ParsedContent if it has the right structure
                    if isinstance(data, dict) and "content" in data:
                        parsed_content = ParsedContent(
                            source=data.get("source", ""),
                            section=data.get("section", ""),
                            content=data.get("content", ""),
                            content_type=data.get("content_type", ""),
                            metadata=data.get("metadata", {}),
                        )
                        parsed_contents.append(parsed_content)
                    else:
                        # Handle raw dict objects
                        parsed_contents.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue

        total_items = len(parsed_contents)

        header = f"\n=== JSONL File Contents: {file_path} ({total_items} items) ==="
        print(header)

        count = 0
        for item in parsed_contents:
            count += 1
            item_header = f"\n--- Item {count} ---"
            print(item_header)

            source_line = f"Source: {item.source}"
            section_line = f"Section: {item.section}"
            content_with_newlines = item.content.replace("\\n", "\n")
            content_line = f"Content: {content_with_newlines}"
            type_line = f"Type: {item.content_type}"

            logger.info(source_line)
            logger.info(section_line)
            logger.info(content_line)
            logger.info(type_line)

        footer = f"\n=== End JSONL File Contents ===\n"
        print(footer)

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error reading JSONL file {file_path}: {e}")
