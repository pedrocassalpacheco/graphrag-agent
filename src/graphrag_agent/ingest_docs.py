import asyncio
from graphrag_agent.tools.crawler import AsyncFileSystemCrawler
from graphrag_agent.tools.document_embedding import (
    OpenAIEmbedding,
)  # Use the correct class name if it exists
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository
from graphrag_agent.tools.content_parser import AsyncPythonComponentParser
from graphrag_agent.tools.content_parser import AsyncPythonSampleParser
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


async def ingest_docs():

    path = "/Users/pedropacheco/Projects/dev/langflow.current/docs/docs"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    embedding_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["py"])
    parser = AsyncPythonComponentParser()
    vector_db = AsyncAstraDBRepository(
        collection_name="component_code",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=3072,
        truncate_collection=True,
        use_vectorize=False,
    )
    embedding = OpenAIEmbeddingGenerator(
        api_key=OPENAI_API_KEY,
    )

    crawler_task = asyncio.create_task(crawler.run(crawler_queue))
    parser_task = asyncio.create_task(
        parser.parse(input=crawler_queue, output=parser_queue)
    )
    emdedding_task = asyncio.create_task(
        embedding.embed(input=parser_queue, output=embedding_queue)
    )
    writer_task = asyncio.create_task(vector_db.write(input=embedding_queue))

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, emdedding_task, writer_task)


if __name__ == "__main__":
    # asyncio.run(component_flow())
    asyncio.run(sample_flow())
