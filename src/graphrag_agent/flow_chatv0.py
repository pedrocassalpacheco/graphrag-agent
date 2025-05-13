import asyncio
from graphrag_agent.tools.document_embedding import OpenAIEmbeddingGenerator
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository
from graphrag_agent.utils.utils import print_pretty_json
from graphrag_agent.utils.logging_config import get_logger
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import traceback

# Constants
logger = get_logger(__name__)
# Load environment variables from .env file
load_dotenv()

ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "langflow")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


async def create_repositories():
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


def format_documentation(results):
    """Format documentation content (already in string format)."""
    formatted_contents = [
        result.get("content", "[Missing content]") for result in results
    ]
    return "### Documentation\n" + "\n\n".join(formatted_contents)


def format_components(results):
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


def format_sample_code(results):
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


async def query_openai(prompt, conversation_history=None):
    """Send a prompt to OpenAI and get the response, maintaining conversation history."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Initialize conversation history if not provided
    if conversation_history is None:
        conversation_history = []

    # Create the messages list with system prompt and history
    messages = [
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

    # Add conversation history to messages
    messages.extend(conversation_history)

    # Add the current prompt
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.2,
        )

        # Get the response content
        response_content = response.choices[0].message.content

        # Update conversation history
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": response_content})

        return response_content, conversation_history

    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(f"Error querying OpenAI: {e}")
        print(traceback.format_exc())

        # Update conversation history even on error
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": error_message})

        return error_message, conversation_history


async def get_context_for_query(query):
    """Retrieve and format context information for a query."""
    # Create repository instances
    langflow_docs, component_code, sample_code = await create_repositories()

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
    documentation_section = format_documentation(langflow_doc_results)
    components_section = format_components(component_code_results)
    sample_code_section = format_sample_code(sample_code_results)

    return documentation_section, components_section, sample_code_section


async def one_shot_flow_gen(query:str):
    
    """Generate code flow based on query (non-interactive mode)."""
    # Get context information
    documentation_section, components_section, sample_code_section = (
        await get_context_for_query(query)
    )

    prompt = f"{PROMPT_TEMPLATE}".format(
    documentation_section=documentation_section,
    components_section=components_section,
    sample_code_section=sample_code_section,
    query=query
)

    # Send to OpenAI
    print("LLM PROMPT:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    llm_response, _ = await query_openai(prompt)  # Ignore conversation history

    print("\n" + "=" * 80)
    print("LLM RESPONSE:")
    print("=" * 80)
    print(llm_response)
    print("=" * 80)

    return llm_response


async def interactive_flow_gen():
    """Interactive flow generation with continuous feedback loop."""
    # Initialize conversation history
    conversation_history = []

    # Get initial query from user
    query = input("What kind of flow would you like to build? ")
    logger.info("Thinking...")

    context_needed = True


    while True:
        try:
            if context_needed:
                # Get context information
                documentation_section, components_section, sample_code_section = (
                    await get_context_for_query(query)
                )

                # Add context as a system message
                if not conversation_history:
                    # First message - add to new conversation
                    conversation_history = [
                        {"role": "system", "content": context_prompt}
                    ]
                else:
                    # Add as an update to existing conversation
                    conversation_history.append(
                        {"role": "system", "content": context_prompt}
                    )

                print("ADDING CONTEXT TO CONVERSATION")
                print("=" * 80)
                print(context_prompt)
                print("=" * 80)

                # Reset context flag
                context_needed = False

            # Process the query
            llm_response, conversation_history = await query_openai(
                query, conversation_history
            )

            # Stop the animation when done
            print("\n" + "=" * 80)
            print("LLM RESPONSE:")
            print("=" * 80)
            print(llm_response)
            print("=" * 80)

            # Prompt for feedback or follow-up
            feedback = input(
                "\nAny feedback or follow-up questions? (Type 'exit' to end, 'refresh' to get new context): "
            )

            if feedback.lower() == "exit":
                print("Ending conversation. Goodbye!")
                break
            elif feedback.lower() == "refresh":
                context_needed = True
                query = input("What's your new query for refreshed context? ")
            else:
                # Use feedback directly as the next query
                query = feedback

        except Exception as e:
            print(f"Error during interactive session: {e}")
            print(traceback.format_exc())
            feedback = input("\nAn error occurred. Try again? (y/n): ")
            if feedback.lower() != "y":
                break


if __name__ == "__main__":
    # Run the interactive version
    asyncio.run(interactive_flow_gen())
