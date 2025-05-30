import json
import tiktoken
from typing import Union, List, Dict, Optional
import asyncio
import os
from datetime import datetime

LOG_DIR = os.getenv("LOGS_DIR", "./logs")
# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)


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


def dump_queue(queue: asyncio.Queue) -> List:
    """
    Dump the contents of an asyncio queue to a list and write them to a date-stamped log file.

    Args:
        queue: The asyncio queue to dump.

    Returns:
        List: A list containing the items in the queue.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{LOG_DIR}/conversation_dump_{timestamp}.txt"

    # Write conversation to file
    with open(filename, "w") as f:
        f.write("QEUE DUMP\n")
        f.write("=" * 80 + "\n")
        while not queue.empty():
            item = queue.get_nowait()
            f.write(f"{item}\n")
