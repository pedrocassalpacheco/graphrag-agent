import json
from typing import Dict, Any, List, Optional

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class FlowTools:
    """Tools for the FlowGenerator to use with OpenAI function calling."""

    def __init__(self, flow_generator):
        """Initialize with a reference to the FlowGenerator instance."""
        self.flow_generator = flow_generator

    async def dump_conversation_memory(self) -> str:
        """
        Retrieve the current conversation history in a simple format.
        """
        try:
            # Access messages through the flow generator reference
            messages = self.flow_generator.messages

            if not messages:
                return "Conversation history is empty."

            # Build a simple text representation
            output = [f"Conversation History ({len(messages)} messages):"]

            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Format each message as a numbered entry
                output.append(f"\n--- Message {i+1} ---")
                output.append(f"Role: {role}")
                output.append(f"Content: {content}")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Error dumping conversation memory: {e}")
            return f"Error retrieving conversation history: {str(e)}"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "dump_conversation_memory",
                    "description": "Retrieve the current conversation history between the user and assistant",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
            # Add more tool definitions here
        ]

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with the given arguments."""
        # Map tool names to instance methods
        tool_methods = {
            "dump_conversation_memory": self.dump_conversation_memory,
            # Add more tool mappings here
        }

        if tool_name not in tool_methods:
            return f"Error: Tool '{tool_name}' not found"

        try:
            # Call the method with the provided arguments
            return await tool_methods[tool_name](**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"
