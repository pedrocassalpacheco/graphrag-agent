import json
from typing import Union

def print_pretty_json(json_data: Union[dict, str]) -> None:
    """Pretty print a JSON dictionary or string."""
    try:
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Parse string into a dictionary
        print(json.dumps(json_data, indent=4))  # Pretty print with indentation
    except (TypeError, ValueError) as e:
        print(f"Error printing JSON: {e}")