import asyncio
from graphrag_agent.tools.crawler import AsyncWebCrawler, AsyncFileSystemCrawler
from graphrag_agent.tools.content_parser import AsyncPageContentParser
from graphrag_agent.tools.qa_generator import AsyncQuestionGenerator3
from graphrag_agent.tools.content_parser import AsyncMarkdownParser
from graphrag_agent.tools.document_embedding import OllamaEmbeddingGenerator
from graphrag_agent.tools.document_embedding import OpenAIEmbeddingGenerator
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository
from graphrag_agent.tools.content_parser import AsyncPythonSourceParser
from graphrag_agent.tools.content_parser import AsyncLangflowDocsMarkdownParser
from graphrag_agent.utils.logging_config import get_logger
from graphrag_agent.utils.utils import print_pretty_json


from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Constants
load_dotenv()

ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "langflow")
ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "langflow_docs")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY", "embedding")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def search_documents(collection, query):
    vector_db = AsyncAstraDBRepository(
        collection_name=collection,
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=3072,
        truncate_collection=False,
        use_vectorize=False,
    )

    print_pretty_json(await vector_db.sample())

    # Create the embedding generator
    embedding_generator = OpenAIEmbeddingGenerator(api_key=OPENAI_API_KEY)

    # Generate the embedding vector for your query
    embedding_vector = await embedding_generator._generate_embedding(query)

    # Use the generated vector for retrieval
    results = await vector_db.retrieve(
        query=query, embedding_vector=embedding_vector, limit=10
    )

    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Title: {doc.get('title', 'No title')}")
        print(f"Similarity: {doc.get('$similarity', 'No similarity')}")
        print("-" * 40)

    # Format results for the LLM prompt
    formatted_results = []
    for i, doc in enumerate(results):
        title = doc.get("title", "No title")
        content = doc.get("content", "")

        # Add document with preserved formatting
        formatted_results.append(f"Document {i+1}: {title}\n\n{content}")

    # Join all documents with clear separation
    context = "\n\n" + "=" * 50 + "\n\n".join(formatted_results)

    query = "What is the input parameter for the AstraDB vector component that expect a connection to an embedding model?"
    # Create prompt with the query and context
    prompt = f"""Based on the following documentation, please answer this question:
    
Question: {query}

Context:
{context}

Please provide a detailed answer based only on the information in the documents above.
"""

    # Send to OpenAI
    print("LLM PROMPT:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    llm_response = await query_openai(prompt)

    print("\n" + "=" * 80)
    print("LLM RESPONSE:")
    print("=" * 80)
    print(llm_response)
    print("=" * 80)

    return results


async def query_openai(prompt):
    """Send a prompt to OpenAI and get the response."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",  # You can use a different model like "gpt-3.5-turbo" if preferred
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides information about software components based on documentation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=1000,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    asyncio.run(
        search_documents(
            "component_code",
            "What is the input parameter for the AstraDB vector component that expect a connection to an embedding model?",
        )
    )
    asyncio.run(
        search_documents(
            "langflow_docs",
            "What is the input parameter for the AstraDB vector component that expect a connection to an embedding model?",
        )
    )
