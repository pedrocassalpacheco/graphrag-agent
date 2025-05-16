import asyncio
from datetime import datetime
from rich.console import Console
from rich.panel import Panel


async def main():
    """Simple console chat with tab-indented responses."""
    # Initialize Rich console just for the welcome panel
    console = Console()

    # Display welcome message in a panel
    console.print(Panel("Chat Console Demo", title="Welcome"))

    # Add initial system message
    print("Hello! I'm your chat assistant. How can I help you today?")

    # Main chat loop
    while True:
        # Get user input
        user_input = await asyncio.to_thread(input, "\nYou: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\t\tGoodbye! Thanks for chatting!")
            break

        # Process and generate AI response
        if user_input.lower() == "code":
            # Demo code response with proper indentation for each line
            response = "def hello_world():\n\t\t    print('Hello from Chat!')\n\t\t    for i in range(3):\n\t\t        print(f'Message {i}')\n\t\t\n\t\thello_world()"
        elif "?" in user_input:
            response = "That's a great question! I'll do my best to answer it."
        else:
            # Echo with some personality
            responses = [
                f"I understand you said: {user_input}",
                f"Thanks for sharing that!",
                f"That's interesting. Tell me more.",
                f'I appreciate your message about "{user_input}"',
                f"Got it. What else would you like to discuss?",
            ]
            import random

            response = random.choice(responses)

        # Add a slight delay to simulate thinking
        await asyncio.sleep(0.7)

        # Print the response with tab indentation
        print(f">>> {response}")  # Multiple arrows (like code prompts)

    # Final goodbye
    print("\n\t\tThank you for trying the Chat Demo!")


if __name__ == "__main__":
    asyncio.run(main())
