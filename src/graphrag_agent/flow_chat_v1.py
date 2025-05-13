import asyncio
from graphrag_agent.tools.document_embedding import OpenAIEmbeddingGenerator
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository
from graphrag_agent.utils.utils import print_pretty_json
from graphrag_agent.utils.logging_config import get_logger, get_file_logger
from graphrag_agent.tools.code_validation import validate_code

from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import traceback
import random

# Load environment variables from .env file
load_dotenv()

# Lots' of globals
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "langflow")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FLOW_DIR = os.getenv("OUTPUT_FLOW_DIR", "./flows")
# Ensure the flow directory exists
if not os.path.exists(FLOW_DIR):
    os.makedirs(FLOW_DIR)
LOG_DIR = os.getenv("LOGS_DIR", "./logs")
# Ensure the flow directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Logging
logger = get_logger(__name__)
file_logger = get_file_logger(__name__, log_file=f"{LOG_DIR}/flow_gen.log")

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

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant specialized in generating Python code for creating "
            "flows in langflow-ai. Use the provided context and documentation to generate "
            "accurate and efficient solutions. Flows are composed of components and connections. "
            "Each component has a set of inputs and outputs, along with mandatory attributes. "
            "Your job is to instantiate the components, populate their attributes, and connect them "
            "to other components. The output should strictly adhere to the provided context and consist "
            "only of valid Python code and necessary comments. When selecting the class name for components "
            "look at Component Information section and extract the class name. Remember that the flow must end "
            "with a  Graph(start=<first component>, end=<last component>) statement. "
            "If you don't know how to answer, say 'I don't know'."
        ),
    }
]

GOOD_FLOWS = [
    "Create an agent that is able to search online using Tavily and has a prompt telling it to be thorough and focused on technical details",
    "Generate an agent which can send or access Gmail and also create a latest news news letter.uses relevant tools or components to get the latest news and optimize the prompt to create a beautiful well articulated news letter which could be send via Gmail.",
    "Build a RAG flow that uses Weaviate as vector store and Claude as the LLM",
    "Make a flow that summarizes PDF documents using Mistral and returns key points",
    "Create a flow that extracts tables from Excel files and generates charts based on the data",
    "Build an agent that can analyze sentiment in customer reviews and categorize them by topic",
    "Create a text-to-SQL converter that uses SQLite as the database",
    "Make a flow that can extract named entities from news articles and visualize relationships",
    "Build a chatbot that responds to questions about uploaded CSV data with statistics",
    "Create a flow that takes a website URL, scrapes the content, and generates an SEO report",
    "Build an agent that can explain complex code by breaking it down into simpler steps",
]

BAD_FLOWS = [
    "Create a flow that does not use any components or connections",
    "Make a flow that connects an oracle database to a XYZ model",
    "Build a flow that tries to connect unrelated components without any logic",
    "Create a flow that uses deprecated components without any explanation",
]

#################################
# UTILITY FUNCTIONS
#################################


async def initialize_vector_stores():
    """Create and return DB repository instances."""
    # Create instances of AsyncAstraDBRepository for each collection
    logger.info("Connecting to datastores")
    langflow_docs = AsyncAstraDBRepository(
        collection_name="langflow_docs",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=VECTOR_DIMENSION,
        truncate_collection=False,
        use_vectorize=False,
    )

    component_code = AsyncAstraDBRepository(
        collection_name="component_code",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=VECTOR_DIMENSION,
        truncate_collection=False,
        use_vectorize=False,
    )

    sample_code = AsyncAstraDBRepository(
        collection_name="sample_code",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=VECTOR_DIMENSION,
        truncate_collection=False,
        use_vectorize=False,
    )

    return langflow_docs, component_code, sample_code


def transform_documentation(results):
    """Format documentation content (already in string format)."""
    formatted_contents = [
        result.get("content", "[Missing content]") for result in results
    ]
    return "### Documentation\n" + "\n\n".join(formatted_contents)


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

    return "### Component Information\n" + "\n\n".join(formatted_contents)


def transform_sample_code(results):
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
        sample = f"```python\n{code}\n```\n\nDocstring:\n{docstring}"
        formatted_contents.append(sample)

    return "### Code Examples\n" + "\n\n".join(formatted_contents)


async def get_context_for_query(query):
    """Retrieve and format context information for a query."""
    # Create repository instances
    langflow_docs, component_code, sample_code = await initialize_vector_stores()

    # Embedding for initial questions
    logger.info("Generating embedding for query")
    embedding_generator = OpenAIEmbeddingGenerator(api_key=OPENAI_API_KEY)
    embedding_vector = await embedding_generator._generate_embedding(query)

    # Retrieve results from each collection
    logger.info("Retrieving results from datastores")
    langflow_doc_results = await langflow_docs.retrieve(
        query=query, embedding_vector=embedding_vector, limit=10
    )
    component_code_results = await component_code.retrieve(
        query=query, embedding_vector=embedding_vector, limit=10
    )
    sample_code_results = await sample_code.retrieve(
        query=query, embedding_vector=embedding_vector, limit=10
    )

    # Format sections
    logger.info("Formatting results")
    documentation_section = transform_documentation(langflow_doc_results)
    components_section = transform_components(component_code_results)
    sample_code_section = transform_sample_code(sample_code_results)

    return documentation_section, components_section, sample_code_section


async def generate_flow(prompt):
    """Send a prompt to OpenAI and get the response, maintaining conversation history."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Add the current prompt
    MESSAGES.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=MESSAGES,
            temperature=0.2,
        )

        # Get the response content
        response_content = response.choices[0].message.content

        # Update conversation history
        MESSAGES.append({"role": "assistant", "content": response_content})

        return response_content

    except Exception as e:
        logger.error(f"Error querying OpenAI: {e}")
        logger.error(traceback.format_exc())

        # Write error information to a log.err file
        with open("log.err", "a") as error_log:
            file_logger.error("Error querying OpenAI:\n")
            file_logger.error("{e}\n")
            file_logger.error("Traceback:\n")
            file_logger.error(traceback.format_exc())
            file_logger.error("\nMESSAGES:\n")
            for i, msg in enumerate(MESSAGES):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Write role header
                file_logger.error(f"\n--- MESSAGE #{i+1}: {role} ---\n")

                # Write content with original formatting preserved
                file_logger.error(f"{content}\n")
                file_logger.error("\n======================\n")


#################################
# USER-FACING FUNCTIONS
#################################


async def one_shot_flow_gen(query: str):
    """Generate code flow based on query (non-interactive mode)."""
    # Get context information
    documentation_section, components_section, sample_code_section = (
        await get_context_for_query(query)
    )

    prompt = PROMPT_TEMPLATE.format(
        documentation_section=documentation_section,
        components_section=components_section,
        sample_code_section=sample_code_section,
        query=query,
    )

    # Send to OpenAI
    logger.info("fPrompting LLM for {query}")
    llm_response = await generate_flow(prompt)

    logger.info("\n" + "=" * 80)
    logger.info("LLM RESPONSE:")
    logger.info("=" * 80)
    logger.info(llm_response)
    logger.info("=" * 80)

    return llm_response


async def interactive_flow_gen():
    """Interactive flow generation with continuous feedback loop."""
    # Initialize conversation history
    conversation_history = []

    # Get initial query from user
    query = input("What kind of flow would you like to build? ")
    logger.info("Thinking...")

    documentation_section, components_section, sample_code_section = (
        await get_context_for_query(query)
    )

    while True:
        try:
            # Process the query
            llm_response = await generate_flow(query)

            logger.info("\n" + "=" * 80)
            logger.info("LLM RESPONSE:")
            logger.info("=" * 80)
            logger.info(llm_response)
            logger.info("=" * 80)

            # Prompt for feedback or follow-up
            feedback = input(
                "\nAny feedback or follow-up questions? (Type 'exit' to end, 'refresh' to get new context): "
            )

            if feedback.lower() == "exit":
                logger.info("Ending conversation. Goodbye!")
                break
            elif feedback.lower() == "refresh":
                context_needed = True
                query = input("What's your new query for refreshed context? ")
            else:
                # Use feedback directly as the next query
                query = feedback

        except Exception as e:
            logger.error(f"Error during interactive session: {e}")
            logger.error(traceback.format_exc())
            feedback = input("\nAn error occurred. Try again? (y/n): ")
            if feedback.lower() != "y":
                break


if __name__ == "__main__":
    # Run the interactive version
    initial_message = MESSAGES

    for i in range(5):
        asyncio.run(one_shot_flow_gen(random.choice(GOOD_FLOWS)))
