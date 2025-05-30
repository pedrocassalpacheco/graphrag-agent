import asyncio
from datetime import datetime
from pathlib import Path
import os
import traceback
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

from graphrag_agent.tools.document_embedding import OpenAIEmbeddingGenerator
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository
from graphrag_agent.utils.logging_config import get_logger, get_file_logger
from graphrag_agent.tools.code_validation import validate_code
from graphrag_agent.utils.utils import count_tokens

# Add these imports at the top of your file
from selenium.webdriver.common.by import By
from graphrag_agent.tools.browser_automation import BrowserAutomation


# Load environment variables
load_dotenv()

# Keep globals as requested
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "langflow")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FLOW_DIR = os.getenv("OUTPUT_FLOW_DIR", "./flows")
LOG_DIR = os.getenv("LOGS_DIR", "./logs")

# Ensure directories exist
os.makedirs(FLOW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
logger = get_logger(__name__)
file_logger = get_file_logger(__name__, log_file=f"{LOG_DIR}/flow_gen.log")

# Templates and system messages
PROMPT_TEMPLATE = """Based on the following documentation, components, and sample code, generate a python function. Reply with code only. Do not provide any commentary:

### Documentation:
{documentation_section}

### Component Information
{components_section}

### Code Examples
{sample_code_section}

### Question
{query}
"""

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an AI assistant specialized in generating Python code for creating "
        "flows in langflow-ai. Use the provided context to generate "
        "accurate and efficient solutions. Flows are composed of components and connections. "
        "Each component has a set of inputs and outputs, along with mandatory attributes. "
        "Your job is to instantiate the components, populate their attributes, and connect them "
        "to other components. The output should strictly adhere to the provided context and consist "
        "only of valid Python code and necessary comments. When selecting the class name for components "
        "look at Component Information section and extract the class name. Remember that the flow must end "
        "with a  Graph(start=<first component>, end=<last component>) statement. You must infer which component is "
        "the first and which is the last. The first component is the one that has no inputs and the last "
        "component is the one that has no outputs. If you are not sure about the first and last component, "
        "ask the user for clarification. If you don't know how to answer, say 'I don't know'."
    ),
}

# Test flow examples
GOOD_FLOWS = [
    (
        "Create an agent that is able to search online using Tavily "
        "and has a prompt telling it to be thorough and focused on technical details"
    ),
    (
        "Generate an agent which can send or access Gmail and also create a latest news "
        "newsletter. Uses relevant tools or components to get the latest news and optimize "
        "the prompt to create a beautiful, well-articulated newsletter which could be sent via Gmail."
    ),
    ("Build a RAG flow that uses Weaviate as vector store and Claude as the LLM"),
    ("Make a flow that summarizes PDF documents using Mistral and returns key points"),
    (
        "Build a flow to generate synthetic data for an e-commerce store. "
        "Allow the user to input store type, description, and product count."
    ),
    (
        "Build an e-commerce agent that can answer questions about customers "
        "and purchases for an e-commerce store."
    ),
]


# Keep existing utility functions
def transform_documentation(results):
    """Format documentation content (already in string format)."""
    formatted_contents = [
        f"Component name: {result.get('title', '[No Title]')}\n\n"
        f"{result.get('content', '[Missing content]')}\n\n"
        for result in results
    ]
    return "" + "\n\n".join(formatted_contents)


def transform_components(results):
    """Format component information (structured dictionary format)."""
    formatted_contents = []
    for result in results:
        content = result.get("content", {})
        if not content:
            formatted_contents.append("[Missing content]")
            continue

        # Extract relevant component information
        class_name = content.get("class_name", "[No class name]")
        inputs = content.get("inputs", "[]")
        name = content.get("name", "[No name]")

        # Format as readable text
        component_info = f"Component: {class_name}\nName: {name}\nInputs: {inputs}"
        formatted_contents.append(component_info)

    return "" + "\n\n".join(formatted_contents)


def transform_sample_code(results, add_docstring=False):
    """Format sample code (extract code and docstring)."""
    formatted_contents = []
    for result in results:
        content = result.get("content", {})
        if not content:
            formatted_contents.append("[Missing content]")
            continue

        # Extract just the code and docstring as requested
        code = content.get("code", "[No code]")
        docstring = content.get("docstring", "[No docstring]")

        # Combine code and docstring
        sample = f"```python\n{code}\n```" + (
            f"\n\nDocstring:\n{docstring}" if add_docstring else ""
        )
        formatted_contents.append(sample)

    return "" + "\n\n".join(formatted_contents)


def transform_llm_response(llm_response: str):
    """
    Extract Python code from LLM response, handling both markdown code blocks and direct code.

    Args:
        llm_response: The raw response from the LLM

    Returns:
        str: The extracted Python code
    """
    # Check if empty or None
    if not llm_response:
        logger.warning("Empty response received")
        return ""

    # Case 1: Extract code from markdown blocks if present
    if "```" in llm_response:
        import re

        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", llm_response)
        if code_blocks:
            extracted_code = code_blocks[0].strip()
            logger.info("Extracted code from markdown")
            return extracted_code

    # Case 2: Check if the response is already Python code
    if any(
        python_indicator in llm_response
        for python_indicator in [
            "import ",
            "from ",
            "def ",
            "class ",
            "# ",
            "if __name__",
        ]
    ):
        logger.info("Response appears to be direct Python code")
        return llm_response.strip()

    # Case 3: Unknown format, return as is with warning
    logger.warning("Could not identify clear Python code in response")
    return llm_response.strip()


class FlowGenerator:
    """Langflow code flow generator with better state management."""

    def __init__(self):
        """Initialize the flow generator."""
        # Initialize clients to None (lazy loading)
        self.openai_client: Optional[AsyncOpenAI] = None
        self.embedding_generator: Optional[OpenAIEmbeddingGenerator] = None
        self.langflow_docs: Optional[AsyncAstraDBRepository] = None
        self.component_code: Optional[AsyncAstraDBRepository] = None
        self.sample_code: Optional[AsyncAstraDBRepository] = None
        self.current_flow: Optional[str] = None

        # Initialize message history with system message
        self.messages: List[Dict[str, str]] = [SYSTEM_MESSAGE]

    def reset_conversation(self):
        """Reset the conversation to initial state."""
        self.messages = [SYSTEM_MESSAGE]

    async def initialize_clients(self):
        """Initialize all API clients and vector stores if not already done."""
        # Initialize OpenAI client if needed
        if not self.openai_client:
            self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        # Initialize embedding generator if needed
        if not self.embedding_generator:
            self.embedding_generator = OpenAIEmbeddingGenerator(api_key=OPENAI_API_KEY)

        # Initialize vector stores if needed
        if not self.langflow_docs:
            logger.info("Connecting to datastores")

            # Create all repository instances with global configuration
            self.langflow_docs = AsyncAstraDBRepository(
                collection_name="langflow_docs",
                token=ASTRA_DB_TOKEN,
                endpoint=ASTRA_DB_ENDPOINT,
                keyspace=ASTRA_DB_KEYSPACE,
                vector_dimension=VECTOR_DIMENSION,
                truncate_collection=False,
                use_vectorize=False,
            )

            self.component_code = AsyncAstraDBRepository(
                collection_name="component_code",
                token=ASTRA_DB_TOKEN,
                endpoint=ASTRA_DB_ENDPOINT,
                keyspace=ASTRA_DB_KEYSPACE,
                vector_dimension=VECTOR_DIMENSION,
                truncate_collection=False,
                use_vectorize=False,
            )

            self.sample_code = AsyncAstraDBRepository(
                collection_name="sample_code",
                token=ASTRA_DB_TOKEN,
                endpoint=ASTRA_DB_ENDPOINT,
                keyspace=ASTRA_DB_KEYSPACE,
                vector_dimension=VECTOR_DIMENSION,
                truncate_collection=False,
                use_vectorize=False,
            )

    async def get_context_for_query(self, query):
        """Retrieve and format context information for a query."""
        # Ensure clients are initialized
        await self.initialize_clients()

        # Generate embedding for query
        logger.info("Generating embedding for query")
        embedding_vector = await self.embedding_generator._generate_embedding(query)

        # Retrieve results from each collection
        logger.info("Retrieving results from datastores")
        langflow_doc_results = await self.langflow_docs.retrieve(
            query=query, embedding_vector=embedding_vector, limit=10
        )
        component_code_results = await self.component_code.retrieve(
            query=query, embedding_vector=embedding_vector, limit=10
        )
        sample_code_results = await self.sample_code.retrieve(
            query=query, embedding_vector=embedding_vector, limit=10
        )

        # Format sections
        logger.info("Formating context for better prompting ...")
        documentation_section = transform_documentation(langflow_doc_results)
        components_section = transform_components(component_code_results)
        sample_code_section = transform_sample_code(sample_code_results)

        return documentation_section, components_section, sample_code_section

    async def generate_flow(self, prompt):
        """Send a prompt to OpenAI and get the response."""
        # Ensure clients are initialized
        await self.initialize_clients()

        # Add the current prompt to message history
        self.messages.append({"role": "user", "content": prompt})
        logger.debug(f"Sending {count_tokens(self.messages)} tokens to LLM")

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=self.messages,
                temperature=0.2,
            )

            # Get the response content
            response_content = response.choices[0].message.content

            # Isolate the code from the response
            flow_code = transform_llm_response(response_content)

            logger.debug(f"LLM response: {flow_code}")

            logger.info("Validating response ...")
            flow_code, _ = await self._validate_and_correct_code(
                flow_code, prompt, max_attempts=3
            )

            # Update conversation history
            self.messages.append({"role": "assistant", "content": flow_code})

            return flow_code

        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            logger.error(traceback.format_exc())

            # Log error and conversation
            file_logger.error("Error querying OpenAI:\n")
            file_logger.error(f"{e}\n")
            file_logger.error("Traceback:\n")
            file_logger.error(traceback.format_exc())
            file_logger.error("\n" + "=" * 80)
            file_logger.error("\nMESSAGE:\n")
            for i, msg in enumerate(self.messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                file_logger.error(f"\nRole #{i+1}: {role} \n")
                file_logger.error(f"{content}\n")

            raise

    def _dump_conversation_to_file(self):
        """
        Dump the current conversation messages to a file for debugging/analysis.

        Returns:
            str: Path to the saved file
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{LOG_DIR}/conversation_dump_{timestamp}.txt"

            # Write conversation to file
            with open(filename, "w") as f:
                f.write("CONVERSATION DUMP\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated at: {datetime.now().isoformat()}\n")
                f.write(f"Total messages: {len(self.messages)}\n")
                f.write("=" * 80 + "\n\n")

                for i, message in enumerate(self.messages):
                    role = message.get("role", "unknown")
                    content = message.get("content", "")

                    f.write(f"MESSAGE #{i+1}\n")
                    f.write(f"Role: {role}\n")
                    f.write(f"Tokens: {count_tokens([message])}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{content}\n")
                    f.write("\n" + "=" * 80 + "\n\n")

            logger.info(f"Conversation dumped to: {filename}")
            print(f"Conversation saved to: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error dumping conversation to file: {e}")
            print(f"Error saving conversation: {e}")
            return None

    def _save_flow_to_file(self, code: str):
        """
        Save generated code to a file.

        Args:
            code: The code to save
            query: The query that generated the code

        Returns:
            str: Path to the saved file
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Full path including directory
            filename = f"{FLOW_DIR}/generate_flow_{timestamp}.py"

            # Write to file
            with open(filename, "w") as f:
                f.write(code)

            logger.info(f"Flow saved to: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error saving flow to file: {e}")
            return None

    async def _validate_and_correct_code(
        self, code, query, max_attempts=3
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validate code and attempt to correct it by asking the LLM to fix errors.

        Args:
            code: The code to validate
            query: The original query (for file naming)
            max_attempts: Maximum number of correction attempts

        Returns:
            Tuple of (final_code, validation_result)
        """
        attempts = 0
        current_code = code

        for attempts in range(max_attempts):
            # Validate the code
            validation_result = await validate_code(
                current_code, check_imports=False, strict_mode=True
            )

            # If valid, return the code and result
            if validation_result["is_valid"]:
                logger.info(
                    f"Code validation passed on attempt {attempts + 1}/{max_attempts}"
                )

                # Save the valid code
                # filename = self._save_flow_to_file(current_code, query)
                # logger.info(f"Code validated and saved to: {filename}")

                return current_code, validation_result

            # Log validation errors
            logger.warning(
                f"Code validation failed on attempt {attempts + 1}/{max_attempts}:"
            )

            # Format error message for the LLM
            error_messages = "\n".join(
                [f"- {error}" for error in validation_result.get("errors", [])]
            )
            warning_messages = "\n".join(
                [f"- {warning}" for warning in validation_result.get("warnings", [])]
            )

            correction_prompt = (
                f"Your previous code has the following errors that need to be fixed:\n\n"
                f"{error_messages}\n\n"
                f"{'Additionally, there are some warnings you might want to address:' if warning_messages else ''}\n"
                f"{warning_messages if warning_messages else ''}\n\n"
                f"Please fix these issues and provide the corrected code. "
                f"Reply only with the complete corrected code, no explanations."
            )

            # Use the existing generate_flow method instead of direct API calls
            logger.info("Requesting code correction from LLM")
            try:
                # Get corrected code from LLM using existing method
                corrected_code = await self.generate_flow(correction_prompt)

                # Extract code from markdown if needed
                if "```python" in corrected_code:
                    import re

                    code_blocks = re.findall(
                        r"```(?:python)?\s*([\s\S]*?)```", corrected_code
                    )
                    if code_blocks:
                        corrected_code = code_blocks[0].strip()
                        logger.info("Extracted code from markdown")

                # Update the current code with the corrected version
                current_code = corrected_code

                # Increment attempt counter
                attempts += 1

            except Exception as e:
                logger.error(f"Error requesting correction: {e}")
                logger.error(traceback.format_exc())
                # Return the last code version with its validation result
                return current_code, validation_result

        # If we exceed max attempts, return the last code version and its validation
        validation_result = await validate_code(
            current_code, check_imports=False, strict=True
        )
        return current_code, validation_result

    def _title_screen(self):
        # Get initial query from user
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.text import Text
        from rich.box import Box, DOUBLE, ROUNDED, HEAVY
        from rich.console import Console
        from rich.align import Align

        console = Console()

        title_text = Text()
        title_text.append("✨ ", style="yellow")
        title_text.append("Chat Console ", style="bold cyan")
        title_text.append(
            "(Type 'exit' to end, 'reset' to get new context):",
            style="bold white on red",
        )
        title_text.append(" ✨", style="yellow")

        console.print(
            Panel(
                Align.center(title_text),
                box=ROUNDED,
                border_style="green",
                title="Welcome",
                subtitle="Version 1.0",
            )
        )

    def _input_query(self):
        # Get initial query from user
        from rich.prompt import Prompt

        while True:
            # Prompt for user input
            query = Prompt.ask(
                "What kind of flow would you like to build?",
                default="",
                show_default=False,
                show_choices=False,
            )

            # Check for exit or reset commands
            if query.lower() == "exit":
                logger.info("Goodbye!")
                import sys

                sys.exit(0)
            elif query.lower() == "reset":
                logger.info("Refreshing context...")
                self.reset_conversation()
            elif query.strip():
                return query

    async def _launch_browser_automation(self, flow_code=None):
        """
        Launch browser automation to visualize the flow in the Langflow UI.

        Args:
            flow_code: Optional flow code to visualize
        """
        logger.info("Launching browser automation...")

        # Create browser automation instance
        browser = BrowserAutomation(headless=False)  # Set to True to run invisibly

        try:
            # Start browser and navigate to localhost
            browser.start("http://localhost:3000/flows")

            # Wait for page to load
            time.sleep(5)

            # Click the "Create first flow" button
            browser.click_element("new-project-btn", By.ID)
            time.sleep(5)

            # Click the "blank flow" button
            browser.click_element("[data-testid='blank-flow']")
            time.sleep(5)

            # If flow_code is provided, you could potentially parse it to add appropriate components
            # For now, just add a Text Input component as an example
            browser.input_text("[data-testid='sidebar-search-input']", "TextInput")
            time.sleep(5)
            browser.click_element("[data-testid='add-component-button-text-input']")

            # Print instructions
            print("\nBrowser is open with Langflow UI.")
            print("The automation has added a TextInput component.")
            print("Press Enter to close the browser and return to the conversation.")

            # Wait for user to press Enter
            input()

        except Exception as e:
            logger.error(f"Error in browser automation: {e}")
            print(f"Error launching browser: {e}")
        finally:
            # Close the browser when done
            browser.close()

    async def single_turn(self, query):
        """Generate code flow based on query (non-interactive mode)."""
        # Get context information
        documentation_section, components_section, sample_code_section = (
            await self.get_context_for_query(query)
        )

        # Format prompt using global template
        prompt = PROMPT_TEMPLATE.format(
            documentation_section=documentation_section,
            components_section=components_section,
            sample_code_section=sample_code_section,
            query=query,
        )

        # Send to OpenAI
        logger.info(f"Prompting LLM for {query}")
        llm_response = await self.generate_flow(prompt)

        # Removes any additions made by the LLM that is not valid python code
        flow_code = transform_llm_response(llm_response)

        flow_code, _ = await self._validate_and_correct_code(flow_code, query)

        return flow_code

    async def multi_turn(self):
        self._title_screen()
        """Interactive flow generation with continuous feedback loop."""
        query = self._input_query()

        # Get initial context
        logger.info("\tBuilding context...")
        documentation_section, components_section, sample_code_section = (
            await self.get_context_for_query(query)
        )

        prompt = None
        while True:
            try:
                # Send to OpenAI
                logger.info(f"Building prompt ...")
                if not prompt:
                    prompt = PROMPT_TEMPLATE.format(
                        documentation_section=documentation_section,
                        components_section=components_section,
                        sample_code_section=sample_code_section,
                        query=query,
                    )
                else:
                    prompt = query

                logger.info(f"Thinking ...")
                flow_code = await self.generate_flow(prompt)

                logger.info("\n" + "=" * 80)
                logger.info("LLM RESPONSE:")
                logger.info("=" * 80)
                print(flow_code)
                logger.info("=" * 80)

                # Prompt for feedback or follow-up
                while True:
                    feedback = input(
                        "\nAny feedback or follow-up questions? (Type 'exit' to end): "
                    )

                    if feedback.lower() == "exit":
                        logger.info("Ending conversation. Goodbye!")
                        import sys

                        sys.exit(0)
                    elif feedback.lower().startswith("reset"):
                        logger.debug("Tool detection: reset context")
                        self.reset_conversation()
                        query = input("What's your new query for refreshed context? ")
                        # Get fresh context
                        (
                            documentation_section,
                            components_section,
                            sample_code_section,
                        ) = await self.get_context_for_query(query)
                        prompt = None
                        break
                    elif feedback.startswith("save"):
                        logger.debug("Tool detection: save")
                        self._save_flow_to_file(flow_code)
                    elif feedback.startswith("plot"):
                        logger.debug("Tool detection: broswer automation")
                        await self._launch_browser_automation(flow_code)
                    elif feedback.startswith("dump"):
                        logger.debug("Tool detection: dump conversation")
                        self._dump_conversation_to_file()
                    else:
                        # Use feedback directly as the next query
                        query = feedback
                        break

            except Exception as e:
                logger.error(f"Error during interactive session: {e}")
                logger.error(traceback.format_exc())
                feedback = input("\nAn error occurred. Try again? (y/n): ")
                if feedback.lower() != "y":
                    break
        # Save the final flow code to a file

        self._save_flow_to_file(flow_code)


# Updated main block using the class
if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run flow generation in single or multi-turn mode"
    )
    parser.add_argument(
        "mode",
        choices=["single_turn", "multi_turn"],
        help="Mode to run: single_turn or multi_turn",
    )
    parser.add_argument(
        "--query", type=str, help="Query for single_turn mode", default=None
    )
    args = parser.parse_args()

    # Create a new async main function
    async def run_flow_tests():
        # Create flow generator instance
        generator = FlowGenerator()

        if args.mode == "single_turn":
            if not args.query:
                print("Error: single_turn mode requires a --query argument")
                return
            flow_code = await generator.single_turn(args.query)
            print("\nGenerated flow code:")
            print(flow_code)
        else:  # multi_turn
            await generator.multi_turn()

    # Add this line to actually run the async function
    asyncio.run(run_flow_tests())
