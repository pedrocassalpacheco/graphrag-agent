async def dump_conversation_memory(self) -> str:
    """
    Retrieve the current conversation history in a simple format.

    Returns:
        A string containing the conversation history with each message's role and content.
    """
    try:
        # Direct access to self.messages
        messages = self.messages

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
